from sentence_transformers import SentenceTransformer, models, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, evaluation
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
import argparse
import os
import logging
from typing import List, Tuple
import torch

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="thebajajra/RexBERT-base")
    p.add_argument("--output_dir", required=True, help="Where to save checkpoints/final model.")
    p.add_argument("--teacher", default="google/embeddinggemma-300m")
    p.add_argument("--seq_len", default=2048, type=int, help="Max sequence length")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--mode", choices=["full", "global_layers", "projection"], default="full")
    p.add_argument("--ds_path", nargs="*", required=True,
                        help="Dataset paths to process")
    return p

if __name__ == "__main__":

    ap = build_argparser()
    args = ap.parse_args()
    mode = args.mode

    if  mode=="full":
        print("Fine-tuning full model")
        model = SentenceTransformer(args.student, device="cuda", model_kwargs={
        "attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16,
    })
        
    elif mode=="global_layers":
        print("Fine-tuning only global attention layers")
        word_embedding = models.Transformer(
        args.student,
        max_seq_length=args.seq_len,
    )

        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )

        model = SentenceTransformer(modules=[word_embedding, pooling], device='cuda' ,model_kwargs={
        "attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16
    })
        # Freeze everything in the Student backbone
        hf = word_embedding.auto_model  # This is a Hugging Face StudentModel
        for p in hf.parameters():
            p.requires_grad = False

        stride = int(getattr(hf.config, "global_attn_every_n_layers", 3))
        global_layers = []
        for i, layer in enumerate(hf.layers):           # ModernBertModel exposes `layers: ModuleList[...]`
            if i % stride == 0:                         # 0, n, 2n, ...
                for p in layer.parameters():
                    p.requires_grad = True
                global_layers.append(i)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Global-attention layers unfrozen: {global_layers}")
        print(f"Number of trainable tensors: {len(trainable)}")

    elif mode=="projection":
        print("Fine-tuning only the projection head")

    teacher_model = SentenceTransformer(args.teacher, device="cuda" ,model_kwargs={
        "attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16
    })

    query_ds = load_from_disk(args.ds_path[0])
    doc_ds = load_from_disk(args.ds_path[1])

    combined_ds = concatenate_datasets([query_ds, doc_ds])

    train_dataset = combined_ds.train_test_split(test_size=0.05, seed=args.seed)["train"]
    eval_dataset = combined_ds.train_test_split(test_size=0.05, seed=args.seed)["test"]

    train_loss = losses.MSELoss(model=model)

    eval_sentences = eval_dataset["sentence"]
    dev_evaluator_mse = evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model)

    model.max_seq_length = args.seq_len
    model.tokenizer.model_max_length = args.seq_len

    training_args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=args.output_dir,
    # Optional training parameters:
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    warmup_ratio=0.1,
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # metric_for_best_model="eval_sts-dev_spearman_cosine",
    learning_rate=args.lr,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=args.save_total_limit,
    gradient_accumulation_steps=args.grad_accum,
    logging_steps=args.logging_steps,
    run_name="{}-{}-nomic-unsupervised-mse".format(args.student.split("/")[-1], mode),
    seed=args.seed,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator_mse,
)

trainer.train()

final_output_dir = f"{args.output_dir}/final"
model.save(final_output_dir)