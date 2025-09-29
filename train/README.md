```bash
pip install deepspeed flash-attn --no-build-isolation
torchrun --nproc_per_node=8 distill_embeddings_full.py \
  --multi_gpu --use_deepspeed_zero3 \
  --flash_attn --bf16 --gradient_checkpointing \
  --output_dir ./rexbert-distilled-full \
  --chunk_pairs 300000 --epochs_per_chunk 1 \
  --per_device_train_batch_size 64 --grad_accum 2
```

```bash
python train/distill_embeddings.py \
  --output_dir ./models/rexbert-base-distilled-gemma300m-trial \
  --dataset_config amazonqa \
  --max_eval_pairs 1000 \
  --epochs 1 --per_device_train_batch_size 64 \
  --bf16 --gradient_checkpointing --flash_attn
```

```bash
pip install deepspeed flash-attn --no-build-isolation
torchrun --nproc_per_node=8 distill_embeddings_ecom.py \
  --multi_gpu --use_deepspeed_zero3 \
  --flash_attn --bf16 --gradient_checkpointing \
  --output_dir ./rexbert-distilled-3splits \
  --chunk_pairs 300000 --epochs_per_chunk 1 \
  --per_device_train_batch_size 64 --grad_accum 4
```