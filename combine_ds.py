from datasets import load_from_disk, Dataset, concatenate_datasets
import argparse, os, json

def process_ds(dataset: Dataset, field: str) -> Dataset:
    """
    The function chooses the field from dataset. Renames the field column to 'sentence' and {field}_embedding to 'label'.
    There can be two possible fields: 'query' or 'document'.
    The other field and {field}_embedding is removed from dataset 
    """
    possible_fields = {"query", "document"}
    if field not in possible_fields:
        raise ValueError(f"Field '{field}' is not valid. Must be one of: {', '.join(sorted(possible_fields))}")

    field_embedding = f"{field}_embedding"
    if field not in dataset.column_names:
        raise ValueError(f"Field '{field}' not found in dataset columns: {', '.join(sorted(dataset.column_names))}")
    if field_embedding not in dataset.column_names:
        raise ValueError(f"Field embedding '{field_embedding}' not found in dataset columns: {', '.join(sorted(dataset.column_names))}")

    # Rename the columns
    dataset = dataset.rename_column(field, "sentence")
    dataset = dataset.rename_column(field_embedding, "label")

    # Remove the other field and its embedding
    other_field = (possible_fields - {field}).pop()
    other_field_embedding = f"{other_field}_embedding"
    dataset = dataset.remove_columns([other_field, other_field_embedding])

    return dataset

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", nargs="*", required=True,
                        help="Dataset splits to process")
    
    parser.add_argument("--output_dir", required=True,
                        help="Directory where the combined dataset will be stored")
    parser.add_argument("--field", default=None, required=True,
                        help="the field to be processed")
    return parser

if __name__ == "__main__":

    ap = build_argparser()
    args = ap.parse_args()

    for path in args.ds_path:
        if not os.path.exists(path):
            raise ValueError(f"Dataset path {path} does not exist")
    _datasets = [load_from_disk(path) for path in args.ds_path]
    dataset: Dataset = concatenate_datasets(_datasets)

    if args.field_type not in dataset.column_names:
        raise ValueError(f"Field '{args.field_type}' not found in dataset columns: {', '.join(sorted(dataset.column_names))}")

    dataset = process_ds(dataset, args.field)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset.save_to_disk(output_dir)
    print(f"Combined dataset saved to {output_dir}")


