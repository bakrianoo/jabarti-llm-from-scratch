"""
01 — Download, Filter, and Clean the Corpus
============================================

Step 1 — download every pretrain shard from HuggingFace (~5 GB, cached).
Step 2 — measure whether Type-A cleaning still earns its place.
Step 3 — demo the Type-B normalizer (unchanged, still the golden rule).
Step 4 — write a SAMPLED corpus for the tokenizer to train on.

"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from cleaning import clean_dataframe, normalize_text, prepare_document
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "bakrianoo/jabarti-llm-dataset"

TRAIN_GLOB = "train-*.parquet"
EVAL_GLOB = "eval-*.parquet"

KEEP_COLUMNS = ["text", "language", "article_id"]

DEFAULT_TOKENIZER_DOCS = 400_000

MAX_CHUNKS = 20

FINETUNE_FILES = {
    "ft_train": "finetune/train-00000-of-00001.parquet",
    "ft_eval":  "finetune/eval-00000-of-00001.parquet",
}

TRAIN_SPLITS = {"phase1_train", "phase2_train"}
EVAL_SPLITS = {"phase1_eval", "phase2_eval"}

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_CORPUS = OUT_DIR / "tokenizer_corpus.parquet"

def shard_patterns(max_shards: int=None):
    if max_shards is None:
        return ["pretrain/train-*.parquet", "pretrain/eval-*.parquet"]

    train_pattern = [ f"pretrain/train-{i:05d}-of-*.parquet" for i in range(max_shards) ]
    eval_pattern = ["pretrain/eval-*.parquet"]

    return train_pattern + eval_pattern

def download_pretrain(max_shards: int=None):
    """Download the pretrain shards we need, and return where they landed."""
    print("STEP 1: DOWNLOAD THE PRETRAIN SHARDS")

    patterns = shard_patterns(max_shards) + ["finetune/*.parquet"]
    root = snapshot_download(
        repo_id=REPO, repo_type="dataset", allow_patterns=patterns,
    )

    pretrain_folder = Path(root) / "pretrain"
    pretrain_train_shards = sorted(pretrain_folder.glob(TRAIN_GLOB))
    pretrain_eval_shards = sorted(pretrain_folder.glob(EVAL_GLOB))

    return Path(root), pretrain_train_shards, pretrain_eval_shards

def build_tokenizer_corpus(train_shards, target_docs : int):
    """Sample evenly across shards and write the tokenizer's training text."""
    parts = []
    for i, shard in enumerate(train_shards):
        parts.append(pd.read_parquet(shard, columns=KEEP_COLUMNS))

    corpus = pd.concat(parts, ignore_index=True)
    corpus = corpus.iloc[: target_docs]

    corpus = clean_dataframe(corpus)

    corpus.to_parquet(TOKENIZER_CORPUS, index=False)
    return corpus

def write_finetune_splits(root):
    # download finetune splits
    for name, filename in FINETUNE_FILES.items():
        frame = pd.read_parquet(root / filename)
        out = OUT_DIR / f"{name}_filtered.parquet"
        frame.to_parquet(out, index=False)

def demo_normalizer():
    print("TYPE B NORMALIZER (embeddable, runs at inference too)")

    samples = [
        "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة",      # heavy diacritics
        "أحمد إبراهيم آدم علىّ",                   # alef + ya variants
        "The   United    States",                 # extra spaces
    ]

    for s in samples:
        print(f"  raw       : {s}")
        print(f"  normalized: {normalize_text(s)}")
        print("="*30)

def download_and_filter():
    """Download raw HF splits, apply hard cap to train splits, write *_filtered.parquet."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, filename in HF_FILES.items():

        print(f"Load: {name}")

        local = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=filename,
        )

        df = pd.read_parquet(local)

        if name in TRAIN_SPLITS:
            before_filter = len(df)

            df = (
                df.sort_values(["article_id", "chunk_index"])
                  .groupby("article_id", sort=False)
                  .head(MAX_CHUNKS)
                  .reset_index(drop=True)
            )

            after_filter = len(df)
            print(f"df: {name} => {before_filter} filtered to be {after_filter}")

        output_path = OUT_DIR / f"{name}_filtered.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")

def clean_phase(phase, input_path, output_path):

    print(f"==== clean_phase: {phase} ====")

    df = pd.read_parquet(input_path)

    print(f"  Source        : {input_path}")
    print(f"  Rows          : {len(df):,}")

    cleaned_df = clean_dataframe(df)

    removed_rows = len(df) - len(cleaned_df)
    print(f"  Rows kept        : {len(cleaned_df):,}  (dropped {removed_rows:,} empty rows)")

    cleaned_df.to_parquet(output_path, index=False)

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--max-shards",  type=int, default=None,
                        help="use only the first N train shards (classroom runs)")

    parser.add_argument("--tokenizer-docs", type=int, default=DEFAULT_TOKENIZER_DOCS,
                        help="documents to sample for tokenizer training")

    args = parser.parse_args()

    root, pretrain_train_shards, pretrain_eval_shards = download_pretrain(args.max_shards)
    write_finetune_splits(root)

    if args.max_shards:
        pretrain_train_shards = pretrain_train_shards[: args.max_shards]

    _ = build_tokenizer_corpus(pretrain_train_shards, target_docs=args.tokenizer_docs)

if __name__ == "__main__":
    main()
    
