#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distill google/embeddinggemma-300m -> thebajajra/RexBERT-base using Sentence Transformers.
- Data: nomic-ai/nomic-embed-unsupervised-data (streaming; sample N pairs).
- Loss: MSE between student embedding and teacher embedding (embedding-level distillation).
- Extras: multi-GPU, gradient checkpointing, FlashAttention 2, DeepSpeed ZeRO-3, Liger Kernel.

Requires:
  pip install -U "sentence-transformers>=3.0.0" "transformers>=4.46.0" datasets accelerate
  # Optional speedups:
  pip install flash-attn --no-build-isolation    # FlashAttention 2 (if your GPUs/driver support it)
  pip install deepspeed                          # DeepSpeed ZeRO-3
  pip install liger-kernel                       # Liger Kernel patches (mainly for Gemma/LLama/etc.)
"""

import argparse
import os
import json
import math
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from transformers import AutoConfig

# ---------------------------
# CLI
# ---------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints.")
    p.add_argument("--dataset_config", type=str, default="reddit_title_body",
                   help="Config/split name inside nomic-ai/nomic-embed-unsupervised-data.")
    p.add_argument("--max_train_pairs", type=int, default=200_000,
                   help="How many (query, document) pairs to sample from the streaming dataset.")
    p.add_argument("--max_eval_pairs", type=int, default=2_000, help="Eval sample size.")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Feature toggles
    p.add_argument("--multi_gpu", action="store_true",
                   help="Just a convenience flag to remind to launch with torchrun.")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn", action="store_true",
                   help="Try to enable FlashAttention 2 in the student (may be ignored if unsupported).")
    p.add_argument("--use_deepspeed_zero3", action="store_true")
    p.add_argument("--deepspeed_config", type=str, default=None,
                   help="Path to a DeepSpeed JSON config. If omitted with --use_deepspeed_zero3, a default ZeRO-3 config is used.")
    p.add_argument("--use_liger_kernel", action="store_true",
                   help="Request Liger kernel patching (works for Gemma/Llama/Mistral/Mixtral archs).")

    # Teacher/student
    p.add_argument("--teacher", type=str, default="google/embeddinggemma-300m")
    p.add_argument("--student", type=str, default="thebajajra/RexBERT-base")

    # Misc
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 training if available.")
    p.add_argument("--hub_private", action="store_true", help="If you push to Hub privately (requires HF auth).")
    return p


# ---------------------------
# Data prep (stream, sample, precompute teacher embeddings)
# ---------------------------
def sample_pairs(dataset_name: str, config: str, n_pairs: int, seed: int) -> List[Dict[str, str]]:
    """
    Stream the dataset split and materialize n_pairs (query, document).
    Expects columns 'query' and 'document' (as shown on the Hub preview).
    """
    ds_iter = load_dataset(dataset_name, config, split="train", streaming=True)
    # Shuffle buffer makes sampling less ordered
    ds_iter = ds_iter.shuffle(seed=seed, buffer_size=50_000)

    out = []
    for ex in ds_iter:
        if "query" in ex and "document" in ex and ex["query"] and ex["document"]:
            out.append({"text": ex["query"], "role": "query"})
            out.append({"text": ex["document"], "role": "document"})
            if len(out) >= 2 * n_pairs:
                break
    return out


def compute_teacher_embeddings(examples: List[Dict[str, Any]], teacher: SentenceTransformer,
                               batch_size: int = 1024) -> Dataset:
    """
    Compute teacher embeddings for each text.
    For EmbeddingGemma we use encode_query for role=='query' and encode_document for role=='document'.
    Otherwise we fall back to encode().
    """
    texts_q = [e["text"] for e in examples if e["role"] == "query"]
    texts_d = [e["text"] for e in examples if e["role"] == "document"]

    # Figure out if teacher exposes asymmetric encoders
    has_asym = all(hasattr(teacher, m) for m in ["encode_query", "encode_document"])

    with torch.inference_mode():
        if has_asym:
            q_emb = teacher.encode_query(texts_q, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
            d_emb = teacher.encode_document(texts_d, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        else:
            # Fallback to a single encoder
            q_emb = teacher.encode(texts_q, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
            d_emb = teacher.encode(texts_d, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

    # Reassemble into a flat list with labels
    labeled = []
    iq = id_ = 0
    for e in examples:
        if e["role"] == "query":
            vec = q_emb[iq]; iq += 1
        else:
            vec = d_emb[id_]; id_ += 1
        labeled.append({"text": e["text"], "labels": vec})  # SentenceTransformers MSELoss expects 'labels'
    return Dataset.from_list(labeled)


# ---------------------------
# Main
# ---------------------------
def main():
    args = build_argparser().parse_args()

    if args.multi_gpu:
        # You must launch with torchrun to actually get DDP:
        # torchrun --nproc_per_node=NUM_GPUS train_distill_embeddings.py --multi_gpu ...
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"[info] WORLD_SIZE={world_size}. Make sure you launched with torchrun for multi-GPU.")

    # -------------------
    # Load teacher (EmbeddingGemma) and student (RexBERT) models
    # Note: EmbeddingGemma doesn't support float16; prefer bf16 or fp32.  [oai_citation:2‡Hugging Face](https://huggingface.co/google/embeddinggemma-300m)
    # -------------------
    teacher_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    teacher = SentenceTransformer(
        args.teacher,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": teacher_dtype},
    )

    student_model_kwargs = {"max_seq_length": args.max_seq_length}
    if args.flash_attn:
        # Will be honored only if the underlying HF encoder supports it.
        # BERT-like encoders often don't; we'll just try and warn later.  [oai_citation:3‡GitHub](https://github.com/huggingface/transformers/issues/26424?utm_source=chatgpt.com)
        student_model_kwargs["attn_implementation"] = "flash_attention_2"

    student = SentenceTransformer(
        args.student,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs=student_model_kwargs,
    )

    # Try to enable gradient checkpointing on the underlying HF model
    if args.gradient_checkpointing:
        try:
            # First module is usually the Transformer backbone with .auto_model
            backbone = getattr(student, "modules", [student])[0]
            auto_model = getattr(backbone, "auto_model", None)
            if auto_model is not None and hasattr(auto_model, "gradient_checkpointing_enable"):
                auto_model.gradient_checkpointing_enable()
                print("[info] Enabled gradient checkpointing on student backbone.")
        except Exception as e:
            print(f"[warn] Could not enable gradient checkpointing explicitly: {e}")

    # Warn if Liger likely unsupported for student arch (Trainer will otherwise try/ignore).  [oai_citation:4‡Hugging Face](https://huggingface.co/docs/transformers/v4.46.2/trainer?utm_source=chatgpt.com)
    student_arch = None
    try:
        # Grab the underlying HF config to check model_type
        backbone = getattr(student, "modules", [student])[0]
        auto_model = getattr(backbone, "auto_model", None)
        if auto_model is not None and hasattr(auto_model, "config"):
            student_arch = auto_model.config.model_type
            if args.use_liger_kernel and student_arch not in {"gemma", "llama", "mistral", "mixtral"}:
                print(f"[warn] Liger Kernel requested, but student arch '{student_arch}' is not in the supported set "
                      "(gemma/llama/mistral/mixtral). Continuing without patching.")
    except Exception:
        pass

    # -------------------
    # Data
    # -------------------
    print(f"[info] Sampling {args.max_train_pairs} pairs from nomic-ai/nomic-embed-unsupervised-data/{args.dataset_config}")
    train_pairs = sample_pairs("nomic-ai/nomic-embed-unsupervised-data", args.dataset_config,
                               args.max_train_pairs, args.seed)
    eval_pairs = sample_pairs("nomic-ai/nomic-embed-unsupervised-data", args.dataset_config,
                              args.max_eval_pairs, args.seed + 1)

    print("[info] Computing teacher embeddings (this can take a while on first run)...")
    train_ds = compute_teacher_embeddings(train_pairs, teacher)
    eval_ds = compute_teacher_embeddings(eval_pairs, teacher)

    # -------------------
    # Loss: Embedding-level distillation (MSE student vs teacher embedding)
    # This is a standard KD setup in Sentence Transformers. 
    # -------------------
    loss = losses.MSELoss(student)

    # -------------------
    # DeepSpeed config (optional)
    # -------------------
    ds_cfg = None
    if args.use_deepspeed_zero3:
        if args.deepspeed_config and os.path.isfile(args.deepspeed_config):
            with open(args.deepspeed_config) as f:
                ds_cfg = json.load(f)
        else:
            # A compact ZeRO-3 config; Trainer will sync 'auto' with TrainingArguments. 
            ds_cfg = {
                "zero_optimization": {
                    "stage": 3,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto"
                },
                "bf16": {"enabled": bool(args.bf16)},
                "gradient_accumulation_steps": "auto",
                "train_micro_batch_size_per_gpu": "auto",
            }

    # -------------------
    # Training args (SentenceTransformerTrainer wraps HF Trainer)  [oai_citation:5‡Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html?utm_source=chatgpt.com)
    # -------------------
    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        fp16=False,  # EmbeddingGemma forbids fp16 activations; keep student bf16/fp32.  [oai_citation:6‡Hugging Face](https://huggingface.co/google/embeddinggemma-300m)
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=args.seed,
        deepspeed=ds_cfg,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["none"],  # change to ["tensorboard"] if desired
        # Liger: only makes sense for supported archs; we pass the user's choice through.  [oai_citation:7‡Hugging Face](https://huggingface.co/docs/transformers/v4.46.2/trainer?utm_source=chatgpt.com)
        use_liger_kernel=args.use_liger_kernel and (student_arch in {"gemma", "llama", "mistral", "mixtral"}),
    )

    # -------------------
    # Trainer
    # -------------------
    trainer = SentenceTransformerTrainer(
        model=student,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if args.eval_steps > 0 else None,
        loss=loss,
    )

    # Helpful log
    print(f"[info] Training with: epochs={args.epochs}, batch={args.per_device_train_batch_size}, "
          f"bf16={args.bf16}, grad_ckpt={args.gradient_checkpointing}, "
          f"flash_attn={args.flash_attn}, ds_zero3={args.use_deepspeed_zero3}, liger={args.use_liger_kernel}")

    trainer.train()
    trainer.save_model(args.output_dir)

    # Save a short card so SentenceTransformers can load it nicely
    card = {
        "student": args.student,
        "teacher": args.teacher,
        "data": f"nomic-ai/nomic-embed-unsupervised-data/{args.dataset_config}",
        "task": "text-embeddings",
        "distillation": "MSE teacher-embedding regression on queries+documents",
        "max_seq_length": args.max_seq_length,
    }
    with open(os.path.join(args.output_dir, "sbert_config.json"), "w") as f:
        json.dump(card, f, indent=2)

    print("[done] Distilled model saved to:", args.output_dir)


if __name__ == "__main__":
    main()