"""
03 — Pack the Corpus into a Token Binary
================================================================

A corpus is one long stream of token IDs; a training example is a window
into it.

Tokenizing happens here, once, instead of at the start of every training
run. The result is a flat uint16 file that PackedDataset memory-maps, so
training never loads the corpus into RAM.


Step 1 — special tokens are added per DOCUMENT, before concatenation
--------------------------------------------------------------------
Each document is encoded and then wrapped in [BOS] (id 2) and [EOS] (id 3).
The real ids from this course's tokenizer:

    "The Nile flows north."  encode->  [7776, 10245, 29003, 29146]
                             wrap  ->  [2, 7776, 10245, 29003, 29146, 3]
                                        ^                            ^
                                      [BOS]                        [EOS]

    "مصر دولة عربية."       encode->  [7954, 13322, 18518, 8297]
                             wrap  ->  [2, 7954, 13322, 18518, 8297, 3]

Wrapping has to happen BEFORE the documents are joined. Once they are one
stream there is no boundary left to mark, and [EOS] is the only thing that
tells the model "this document ended; what follows is unrelated."

Step 2 — concatenate into one stream, then cut fixed windows
-------------------------------------------------------------
    train.bin   [2, 7776, 10245, 29003, 29146, 3, 2, 7954, 13322, 18518, ...]
                 |<------- document 1 ------->| |<------ document 2 ----- ...
    example i   tokens[i * 1025 : (i + 1) * 1025]

1025 is max_seq_len + 1. The extra token is what makes the next-token shift
possible: collate.py takes the first 1024 as the input and the last 1024 as
the target. A window may begin or end mid-document -- that is expected, and
the [EOS] seams inside it are how the model learns where the joins are.


Run:
    python scripts/ch03-build-tokenizer/03_pack_corpus.py
    python scripts/ch03-build-tokenizer/03_pack_corpus.py --max-shards 2

Output (to output/): train.bin, eval.bin, meta.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from jabarti_llm.tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "bakrianoo/jabarti-llm-dataset"
OUT_DIR = Path(__file__).parent / "output"
DTYPE = np.uint16          # vocab is 32,000, so every ID fits in two bytes
SHUFFLE_SEED = 1234        # documents are shuffled before packing

def shard_patterns(max_shards=None):
    """allow_patterns for snapshot_download: every shard, or just the first N."""
    if max_shards is None:
        return ["pretrain/train-*.parquet", "pretrain/eval-*.parquet"]
    return ([f"pretrain/train-{i:05d}-of-*.parquet" for i in range(max_shards)]
            + ["pretrain/eval-*.parquet"])


def pack_split(shards, tokenizer, out_path):
    """Encode every document in `shards` and append them to one flat binary."""
    total_tokens = total_docs = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One shard is encoded, appended and released before the next is read,
    # so peak memory is one shard rather than one corpus.
    with open(out_path, "wb") as handle:
        for i, shard in enumerate(shards, 1):
            texts = (pd.read_parquet(shard, columns=["text"])["text"]
                     .sample(frac=1.0, random_state=SHUFFLE_SEED + i).tolist())

            # [BOS] ... [EOS] per document, before concatenation: the seam is
            # what tells the model where one document stops inside a window.
            stream = []
            for ids in tokenizer.encode_batch(texts):
                stream.append(tokenizer.BOS)
                stream.extend(ids)
                stream.append(tokenizer.EOS)

            np.asarray(stream, dtype=DTYPE).tofile(handle)
            total_tokens += len(stream)
            total_docs += len(texts)
            print(f"  [{i}/{len(shards)}] {shard.name}: {len(texts):,} docs, "
                  f"{total_tokens / 1e6:,.1f}M tokens")

    return {"documents": total_docs, "tokens": total_tokens}

def main():
    # RawDescription keeps the docstring's diagrams intact in --help; the
    # default formatter reflows them into one unreadable paragraph.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max-shards", type=int, default=None,
                        help="pack only the first N train shards")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    root = Path(snapshot_download(
        repo_id=REPO, repo_type="dataset",
        allow_patterns=shard_patterns(args.max_shards),
    )) / "pretrain"
    train_shards = sorted(root.glob("train-*.parquet"))
    eval_shards = sorted(root.glob("eval-*.parquet"))

    if args.max_shards:
        train_shards = train_shards[: args.max_shards]

    tokenizer = Tokenizer.from_file()

    print("packing train")
    train = pack_split(train_shards, tokenizer, args.out_dir / "train.bin")
    print("packing eval")
    evaluation = pack_split(eval_shards, tokenizer, args.out_dir / "eval.bin")

    # Provenance: how a checkpoint can say which corpus it was trained on.
    (args.out_dir / "meta.json").write_text(json.dumps({
        "repo": REPO,
        "dtype": DTYPE.__name__,
        "vocab_size": tokenizer.vocab_size,
        "shuffle_seed": SHUFFLE_SEED,
        "train": train,
        "eval": evaluation,
    }, indent=2), encoding="utf-8")

    print(f"\ntrain.bin: {train['tokens']:,} tokens from {train['documents']:,} docs")
    print(f"eval.bin : {evaluation['tokens']:,} tokens from {evaluation['documents']:,} docs")

if __name__ == "__main__":
    main()