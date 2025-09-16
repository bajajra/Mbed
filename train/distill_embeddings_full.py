#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-corpus distillation:
  Teacher: google/embeddinggemma-300m
  Student: thebajajra/RexBERT-base
  Data:    nomic-ai/nomic-embed-unsupervised-data (ALL splits, streaming)
Prompts:   Uses EmbeddingGemma task-specific prompts per split.

Speedups (optional): multi-GPU (torchrun), grad checkpointing, FlashAttention-2, DeepSpeed ZeRO-3, Liger Kernel.

pip install -U "sentence-transformers>=3.0.0" "transformers>=4.46.0" datasets accelerate
# Optional:
pip install flash-attn --no-build-isolation
pip install deepspeed
pip install liger-kernel
"""

import argparse, os, json, itertools
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

# ---------- Split → Prompt mapping ----------
# See EmbeddingGemma prompt names: Retrieval, Question Answering, STS, InstructionRetrieval (code).   [oai_citation:2‡Hugging Face](https://huggingface.co/blog/embeddinggemma)
QA_SPLITS = {
    "paq", "gooaq", "yahoo_qa", "squad", "eli5"
}
CODE_SPLITS = {"codesearch"}
STS_DUP_SPLITS = {
    "quora", "stackexchange_duplicate_questions", "wikianswers",
    "sentence_compression", "altlex", "simplewiki"
}
# Everything else defaults to retrieval-style (title/body, news, s2orc, etc.). 29 splits total.  [oai_citation:3‡Hugging Face](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data)

def prompts_for_split(split: str) -> Tuple[str, str]:
    """Return (query_prompt_name, document_prompt_name) for this split."""
    if split in CODE_SPLITS:
        return "InstructionRetrieval", "Retrieval-document"
    if split in QA_SPLITS:
        return "Question Answering", "Retrieval-document"
    if split in STS_DUP_SPLITS:
        # Pairs are paraphrases / duplicates: use symmetric STS prompts on both sides.
        return "STS", "STS"
    # Default: retrieval-style title→body, title→abstract, etc.
    return "Retrieval-query", "Retrieval-document"


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--teacher", default="google/embeddinggemma-300m")
    p.add_argument("--student", default="thebajajra/RexBERT-base")

    # Chunked streaming
    p.add_argument("--chunk_pairs", type=int, default=200_000,
                   help="Pairs per training chunk (query+document => 2 examples per pair).")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--epochs_per_chunk", type=int, default=1)

    # Trainer knobs
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
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--use_deepspeed_zero3", action="store_true")
    p.add_argument("--deepspeed_config", type=str, default=None)
    p.add_argument("--use_liger_kernel", action="store_true")

    # Control which splits to include; default is ALL advertised splits.
    p.add_argument("--splits", type=str, nargs="*", default=None,
                   help="Subset of dataset splits to train on (defaults to all).")
    return p


def teacher_encode(model: SentenceTransformer,
                   texts: List[str],
                   prompt_name: str,
                   batch_size: int = 1024) -> List[List[float]]:
    # Always use explicit prompt_name to match EmbeddingGemma's task instructions.  [oai_citation:4‡Hugging Face](https://huggingface.co/blog/embeddinggemma)
    with torch.inference_mode():
        return model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()


def stream_pairs(split: str, seed: int) -> Iterable[Dict]:
    """Yield {'text': str, 'role': 'query'|'document', 'split': split} for the given split."""
    ds = load_dataset("nomic-ai/nomic-embed-unsupervised-data",
                      split=split, streaming=True)  # huge; don't download fully
    # Samples expose 'query' and 'document' columns.  [oai_citation:5‡Hugging Face](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data)
    ds = ds.shuffle(seed=seed, buffer_size=1_000_000)
    for ex in ds:
        q = ex.get("query"); d = ex.get("document")
        if q and d:
            yield {"text": q, "role": "query", "split": split}
            yield {"text": d, "role": "document", "split": split}


def train_chunk(student: SentenceTransformer,
                teacher: SentenceTransformer,
                examples: List[Dict],
                args,
                ds_cfg,
                global_step_offset: int,
                chunk_idx: int):
    """Compute teacher labels for this chunk and train the student on it."""
    # Group by role/split to apply prompts
    # Prepare batches for teacher with the right prompt_name
    by_prompt: Dict[str, List[int]] = {}
    texts: List[str] = []
    for i, ex in enumerate(examples):
        q_prompt, d_prompt = prompts_for_split(ex["split"])
        p_name = q_prompt if ex["role"] == "query" else d_prompt
        key = p_name
        by_prompt.setdefault(key, []).append(i)
        texts.append(ex["text"])

    labels = [None] * len(examples)
    # Compute teacher embeddings per prompt group (saves prompt switching overhead)
    for p_name, idxs in by_prompt.items():
        batch_texts = [examples[i]["text"] for i in idxs]
        vecs = teacher_encode(teacher, batch_texts, prompt_name=p_name, batch_size=1024)
        for i, v in zip(idxs, vecs):
            labels[i] = v

    # Build HF Dataset for the chunk with SBERT-friendly fields
    ds_chunk = Dataset.from_list([{"text": ex["text"], "labels": labels[i]} for i, ex in enumerate(examples)])

    # Loss: student embedding -> teacher embedding (MSE).  [oai_citation:6‡Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/losses.html?utm_source=chatgpt.com)
    loss = losses.MSELoss(student)

    # Compute steps for this chunk
    num_items = len(ds_chunk)
    effective_bsz = args.per_device_train_batch_size * max(1, int(os.environ.get("WORLD_SIZE", "1"))) * max(1, args.grad_accum)
    steps = max(1, num_items // effective_bsz)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(args.output_dir, f"chunk_{chunk_idx:05d}"),
        num_train_epochs=args.epochs_per_chunk,
        max_steps=steps,  # robust for Iterable-like sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        fp16=False,  # EmbeddingGemma forbids fp16 activations; keep student in bf16/fp32.  [oai_citation:7‡Hugging Face](https://huggingface.co/google/embeddinggemma-300m)
        logging_steps=args.logging_steps,
        save_steps=steps,  # save at end of chunk
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        deepspeed=ds_cfg,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["none"],
        use_liger_kernel=args.use_liger_kernel  # ignored for unsupported archs; safe to pass. 
    )

    trainer = SentenceTransformerTrainer(
        model=student,
        args=train_args,
        train_dataset=ds_chunk,
        loss=loss,
    )
    print(f"[chunk {chunk_idx}] items={num_items}, steps={steps}")
    trainer.train()
    # Merge best weights into student object in-place
    student.save(train_args.output_dir)


def main():
    ap = build_argparser()
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.multi_gpu:
        print(f"[info] WORLD_SIZE={os.environ.get('WORLD_SIZE','1')} (launch with torchrun for DDP).")

    # Teacher (bfloat16 if available)   [oai_citation:8‡Hugging Face](https://huggingface.co/google/embeddinggemma-300m)
    teacher_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    teacher = SentenceTransformer(
        args.teacher,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": teacher_dtype},
    )

    # Student
    student_kwargs = {"max_seq_length": args.max_seq_length}
    if args.flash_attn:
        # Will be honored only if the backbone supports it (BERTs often don't; silently ignored).
        student_kwargs["attn_implementation"] = "flash_attention_2"
    student = SentenceTransformer(
        args.student,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs=student_kwargs,
    )

    # Try to enable gradient checkpointing on underlying HF model
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

    # Determine splits (default: all listed by the dataset viewer)   [oai_citation:9‡Hugging Face](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data)
    default_splits = [
        "reddit_title_body","amazon_reviews","paq","s2orc_citation_titles",
        "s2orc_title_abstract","s2orc_abstract_citation","s2orc_abstract_body",
        "wikianswers","wikipedia","gooaq","codesearch","yahoo_title_answer",
        "agnews","amazonqa","yahoo_qa","yahoo_title_question","ccnews","npr",
        "eli5","cnn","stackexchange_duplicate_questions","stackexchange_title_body",
        "stackexchange_body_body","sentence_compression","wikihow","altlex",
        "quora","simplewiki","squad"
    ]
    splits = args.splits or default_splits

    # Stream → Chunk → Distill
    rng_seed = args.seed
    chunk_idx = 0
    buffer: List[Dict] = []
    for split in splits:
        print(f"[stream] split={split}")
        for ex in stream_pairs(split, seed=rng_seed):
            buffer.append(ex)
            if len(buffer) >= 2 * args.chunk_pairs:  # 2 entries per pair (query+document)
                train_chunk(student, teacher, buffer, args, ds_cfg, 0, chunk_idx)
                buffer.clear()
                chunk_idx += 1
        # flush remainder per split
        if buffer:
            train_chunk(student, teacher, buffer, args, ds_cfg, 0, chunk_idx)
            buffer.clear()
            chunk_idx += 1

    # Save final merged student
    final_dir = os.path.join(args.output_dir, "final")
    student.save(final_dir)
    with open(os.path.join(final_dir, "sbert_config.json"), "w") as f:
        json.dump({
            "student": args.student,
            "teacher": args.teacher,
            "data": "nomic-ai/nomic-embed-unsupervised-data (ALL splits, streaming)",
            "task": "text-embeddings (distilled)",
            "prompts": "Per-split per-role prompts applied from EmbeddingGemma config",
            "max_seq_length": args.max_seq_length
        }, f, indent=2)
    print("[done] Saved distilled student to", final_dir)


if __name__ == "__main__":
    main()