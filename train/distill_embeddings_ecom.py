#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distill google/embeddinggemma-300m -> thebajajra/RexBERT-base
Data: nomic-ai/nomic-embed-unsupervised-data (ONLY: amazonqa, amazon_reviews, wikipedia), streamed
Prompts: EmbeddingGemma task-specific prompts per split/role

Speedups (optional): multi-GPU (torchrun), grad checkpointing, FlashAttention-2, DeepSpeed ZeRO-3, Liger Kernel.

Install:
  pip install -U "sentence-transformers>=3.0.0" "transformers>=4.46.0" datasets accelerate
  # Optional:
  pip install flash-attn --no-build-isolation
  pip install deepspeed
  pip install liger-kernel
"""

import argparse, os, json
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from tqdm.auto import tqdm


# --------- Split → Prompt mapping (teacher) ----------
# amazonqa: question→answer (QA-style query; doc stays retrieval-style)
# amazon_reviews: title/body-like retrieval
# wikipedia: retrieval
def prompts_for_split(split: str) -> Tuple[str, str]:
    if split == "amazonqa":
        return "Question Answering", "Retrieval-document"
    # Default for these two: retrieval
    return "Retrieval-query", "Retrieval-document"


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Where to save checkpoints/final model.")
    p.add_argument("--teacher", default="google/embeddinggemma-300m")
    p.add_argument("--student", default="thebajajra/RexBERT-base")

    # Streaming + chunking (never materialize the whole split)
    p.add_argument("--chunk_pairs", type=int, default=200_000,
                   help="How many (query,doc) pairs per training chunk.")
    p.add_argument("--epochs_per_chunk", type=int, default=1)
    p.add_argument("--max_seq_length", type=int, default=256)

    # Optim/trainer knobs
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)

    # Features
    p.add_argument("--multi_gpu", action="store_true", help="Launch with torchrun to actually use DDP.")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn", action="store_true", help="Use attn_implementation=flash_attention_2 if backbone supports it.")
    p.add_argument("--use_deepspeed_zero3", action="store_true")
    p.add_argument("--deepspeed_config", type=str, default=None)
    p.add_argument("--use_liger_kernel", action="store_true")

    # Limit to these three by default; you can still override if needed.
    p.add_argument("--splits", nargs="*", default=["amazonqa","amazon_reviews","wikipedia"],
                   help="Dataset splits to train on (default: amazonqa, amazon_reviews, wikipedia)")
    return p


def teacher_encode(model: SentenceTransformer, texts: List[str], prompt_name: str, batch_size: int = 1024) -> List[List[float]]:
    with torch.inference_mode():
        return model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()


def stream_pairs(split: str, seed: int) -> Iterable[Dict]:
    """
    Yield {'text': str, 'role': 'query'|'document', 'split': split} from the split.
    Assumes 'query' and 'document' columns.
    """
    ds = load_dataset("nomic-ai/nomic-embed-unsupervised-data", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1_000_000)
    for ex in ds:
        q = ex.get("query"); d = ex.get("document")
        if q and d:
            yield {"text": q, "role": "query", "split": split}
            yield {"text": d, "role": "document", "split": split}


def distill_chunk(student: SentenceTransformer,
                  teacher: SentenceTransformer,
                  examples: List[Dict],
                  args,
                  ds_cfg,
                  chunk_idx: int):
    """
    Compute teacher embeddings with per-role prompt_names and train the student for one chunk.
    """
    # Group by prompt_name to minimize prompt switching
    by_prompt: Dict[str, List[int]] = {}
    for i, ex in enumerate(examples):
        q_prompt, d_prompt = prompts_for_split(ex["split"])
        p_name = q_prompt if ex["role"] == "query" else d_prompt
        by_prompt.setdefault(p_name, []).append(i)

    labels = [None] * len(examples)
    for p_name, idxs in by_prompt.items():
        batch_texts = [examples[i]["text"] for i in idxs]
        vecs = teacher_encode(teacher, batch_texts, prompt_name=p_name, batch_size=1024)
        for i, v in zip(idxs, vecs):
            labels[i] = v

    ds_chunk = Dataset.from_list([{"text": ex["text"], "labels": labels[i]} for i, ex in enumerate(examples)])
    loss = losses.MSELoss(student)

    # Derive steps robustly for streamed chunk sizes
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    eff_bsz = args.per_device_train_batch_size * world * max(1, args.grad_accum)
    steps = max(1, len(ds_chunk) // eff_bsz)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(args.output_dir, f"chunk_{chunk_idx:05d}"),
        num_train_epochs=args.epochs_per_chunk,
        max_steps=steps,  # works well with Iterable datasets
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        fp16=False,  # prefer bf16/fp32 with EmbeddingGemma
        logging_steps=args.logging_steps,
        save_steps=steps,             # save at end of chunk
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        deepspeed=ds_cfg,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["none"],
        use_liger_kernel=args.use_liger_kernel,  # ignored for unsupported archs; safe to pass
    )

    trainer = SentenceTransformerTrainer(
        model=student,
        args=train_args,
        train_dataset=ds_chunk,
        loss=loss,
    )
    print(f"[chunk {chunk_idx}] items={len(ds_chunk)} steps={steps}")
    trainer.train()
    student.save(train_args.output_dir)


def main():
    ap = build_argparser()
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.multi_gpu:
        print(f"[info] multi_gpu flag set; WORLD_SIZE={os.environ.get('WORLD_SIZE','1')} "
              f"(launch with torchrun to use DDP)")

    # Teacher (bf16 preferred if available)
    teacher_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    teacher = SentenceTransformer(
        args.teacher,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": teacher_dtype},
    )

    # Student
    student_kwargs = {"max_seqlen": args.max_seq_length}
    if args.flash_attn:
        # Only applied if backbone supports it (BERTs may ignore).
        student_kwargs["attn_implementation"] = "flash_attention_2"
    student = SentenceTransformer(
        args.student,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs=student_kwargs,
    )

    # Enable gradient checkpointing on the underlying HF model if available
    if args.gradient_checkpointing:
        try:
            backbone = getattr(student, "modules", [student])[0]
            auto_model = getattr(backbone, "auto_model", None)
            if auto_model and hasattr(auto_model, "gradient_checkpointing_enable"):
                auto_model.gradient_checkpointing_enable()
                print("[info] Enabled gradient checkpointing.")
        except Exception as e:
            print(f"[warn] gradient_checkpointing_enable failed: {e}")

    # DeepSpeed ZeRO-3 (optional)
    ds_cfg = None
    if args.use_deepspeed_zero3:
        if args.deepspeed_config and os.path.isfile(args.deepspeed_config):
            with open(args.deepspeed_config) as f:
                ds_cfg = json.load(f)
        else:
            ds_cfg = {
                "zero_optimization": {
                    "stage": 3,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                },
                "bf16": {"enabled": bool(args.bf16)},
                "gradient_accumulation_steps": "auto",
                "train_micro_batch_size_per_gpu": "auto",
            }

    # Stream → buffer into chunks → distill
    rng_seed = args.seed
    chunk_idx = 0
    buffer: List[Dict] = []
    for split in args.splits:
        print(f"[stream] split={split}")
        for ex in stream_pairs(split, seed=rng_seed):
            buffer.append(ex)
            if len(buffer) >= 2 * args.chunk_pairs:  # two entries per pair (q + d)
                distill_chunk(student, teacher, buffer, args, ds_cfg, chunk_idx)
                buffer.clear()
                chunk_idx += 1
        # flush leftovers from this split
        if buffer:
            distill_chunk(student, teacher, buffer, args, ds_cfg, chunk_idx)
            buffer.clear()
            chunk_idx += 1

    # Save final merged student
    final_dir = os.path.join(args.output_dir, "final")
    student.save(final_dir)
    with open(os.path.join(final_dir, "sbert_config.json"), "w") as f:
        json.dump({
            "student": args.student,
            "teacher": args.teacher,
            "data": "nomic-ai/nomic-embed-unsupervised-data (amazonqa, amazon_reviews, wikipedia)",
            "task": "text-embeddings (distilled)",
            "prompts": "Per-split per-role prompts (QA→Question Answering; others→Retrieval)",
            "max_seq_length": args.max_seq_length
        }, f, indent=2)
    print("[done] Saved distilled student to", final_dir)


if __name__ == "__main__":
    main()