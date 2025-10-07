from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
    losses,
    evaluation,
)
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk, Value
import argparse
import os
import torch
from typing import List, Tuple


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
    p.add_argument("--ds_path", nargs="*", required=True, help="Dataset paths to process")
    return p

def convert_to_bfloat16(example):
    example["label"] = torch.tensor(example["label"], dtype=torch.bfloat16)
    return example

if __name__ == "__main__":
    ap = build_argparser()
    args = ap.parse_args()
    mode = args.mode

    torch._dynamo.config.disable = True

    # Initialize the student model
    if mode == "full":
        print("Fine-tuning full model")
        model = SentenceTransformer(
            args.student,
            device="cuda",
            model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        )
    elif mode == "global_layers":
        print("Fine-tuning only global attention layers")
        word_embedding = models.Transformer(
            args.student,
            max_seq_length=args.seq_len,
            model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        )
        pooling = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
        model = SentenceTransformer(modules=[word_embedding, pooling], device="cuda")

        # Freeze everything in the Student backbone
        hf = word_embedding.auto_model
        for p in hf.parameters():
            p.requires_grad = False

        stride = int(getattr(hf.config, "global_attn_every_n_layers", 3))
        global_layers = []
        for i, layer in enumerate(hf.layers):
            if i % stride == 0:
                for p in layer.parameters():
                    p.requires_grad = True
                global_layers.append(i)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Global-attention layers unfrozen: {global_layers}")
        print(f"Number of trainable tensors: {len(trainable)}")
    else: # mode == "projection"
        print("Fine-tuning only the projection head")
        # Add projection-only fine-tuning logic here if needed
        exit()

    # Initialize the teacher model
    teacher_model = SentenceTransformer(
        args.teacher,
        device="cuda",
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
    )

    # Load the dataset
    query_ds = load_from_disk(args.ds_path[0])
    doc_ds = load_from_disk(args.ds_path[1])
    combined_ds = concatenate_datasets([query_ds, doc_ds])
    combined_ds = combined_ds.select_columns(["sentence", "label"])
    combined_ds = combined_ds.map(convert_to_bfloat16, num_proc=64)
    split_ds = combined_ds.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]

    # Initialize the loss function
    train_loss = losses.MSELoss(model=model)

    # Create an evaluator
    eval_sentences = eval_dataset["sentence"]
    # dev_evaluator_mse = evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model)

    model.max_seq_length = args.seq_len
    if hasattr(model.tokenizer, "model_max_length"):
        model.tokenizer.model_max_length = args.seq_len

    # Define the training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_ratio=0.1,
        bf16=args.bf16,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        run_name=f"{args.student.split('/')[-1]}-{mode}-nomic-unsupervised-mse",
        seed=args.seed,
        report_to=["tensorboard"],
    )

    # Create the trainer
    # No custom data collator is needed. The trainer will use the default one
    # which works correctly with MSELoss.
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        # evaluator=dev_evaluator_mse,
    )

    # Train the model
    trainer.train()

    # Save the final model
    final_output_dir = f"{args.output_dir}/final"
    model.save(final_output_dir)