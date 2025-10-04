import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nomic-ai/nomic-embed-unsupervised-data",
                        help="HF dataset repo id or local path to load")
    parser.add_argument("--splits", nargs="*", default=["amazonqa", "amazon_reviews", "wikipedia"],
                        help="Dataset splits to process")
    parser.add_argument("--teacher", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name or path")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for encoding")
    parser.add_argument("--output-dir", default="data/embedded",
                        help="Directory where the combined dataset will be stored")
    parser.add_argument("--cache-dir", default=None,
                        help="Optional cache directory for dataset downloads")
    parser.add_argument("--bf16", action="store_true",
                        help="Load the model weights in bfloat16 when supported")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional upper bound on rows to encode (debug only)")
    parser.add_argument("--num-proc", type=int, default=None,
                        help="dataset.map worker processes (defaults to number of GPUs)")
    return parser


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(" ", "_")


def _model_kwargs(bf16: bool) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"attn_implementation": "flash_attention_2"}
    if bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


def _available_devices() -> List[str]:
    if not torch.cuda.is_available():
        return ["cpu"]
    device_count = torch.cuda.device_count()
    return [f"cuda:{idx}" for idx in range(device_count)] or ["cuda"]


def _resolve_num_proc(requested: Optional[int], devices: List[str]) -> Optional[int]:
    if requested is not None and requested > 0:
        return requested if requested > 1 else None
    if len(devices) > 1:
        return len(devices)
    return None


def _select_device(devices: List[str]) -> str:
    process_index = int(os.environ.get("DATASET_PROCESS_INDEX", "0"))
    return devices[process_index % len(devices)]


def _get_or_init_model(device: str, teacher: str, bf16: bool) -> SentenceTransformer:
    cache_key = f"{teacher}|{device}|{'bf16' if bf16 else 'fp32'}"
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = SentenceTransformer(teacher, device=device, model_kwargs=_model_kwargs(bf16))
        _MODEL_CACHE[cache_key] = model
    return model


def _embed_batch(batch: Dict[str, List[str]], *, devices: List[str], teacher: str,
                 bf16: bool, batch_size: int) -> Dict[str, List[List[float]]]:
    device = _select_device(devices)
    model = _get_or_init_model(device, teacher, bf16)

    query_embeddings = model.encode(batch["query"], batch_size=batch_size,
                                    convert_to_numpy=True, show_progress_bar=False)
    document_embeddings = model.encode(batch["document"], batch_size=batch_size,
                                       convert_to_numpy=True, show_progress_bar=False)

    return {
        "query_embedding": np.asarray(query_embeddings).tolist(),
        "document_embedding": np.asarray(document_embeddings).tolist(),
    }


def _process_split(split: str, args: argparse.Namespace, devices: List[str]) -> Dataset:
    dataset = load_dataset(args.dataset, split=split, cache_dir=args.cache_dir)

    if args.limit is not None:
        effective_limit = min(len(dataset), args.limit)
        dataset = dataset.select(range(effective_limit))

    required_columns = {"query", "document"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Split '{split}' missing required columns: {', '.join(sorted(missing))}")

    num_proc = _resolve_num_proc(args.num_proc, devices)
    dataset = dataset.map(
        _embed_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        fn_kwargs={
            "devices": devices,
            "teacher": args.teacher,
            "bf16": args.bf16,
            "batch_size": args.batch_size,
        },
        desc=f"Embedding split '{split}'",
    )

    dataset = dataset.add_column("split", [split] * len(dataset))
    return dataset


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    devices = _available_devices()

    processed_splits: List[Dataset] = []
    for split in args.splits:
        processed_splits.append(_process_split(split, args, devices))

    combined = processed_splits[0] if len(processed_splits) == 1 else concatenate_datasets(processed_splits)

    model_id = _sanitize_model_name(args.teacher)
    output_dir = os.path.join(args.output_dir, model_id)
    _ensure_dir(output_dir)

    combined.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
