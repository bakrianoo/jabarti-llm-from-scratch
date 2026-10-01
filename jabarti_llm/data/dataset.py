"""
dataset.py -- the corpus as training examples
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PackedDataset(Dataset):

    def __init__(self, path, config):

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"No packed corpus at {self.path}. Build it first:\n"
                f"  python scripts/ch03-build-tokenizer/03_pack_corpus.py"
            )

        self.tokens = np.memmap(self.path, dtype=np.uint16, mode='r')
        self.window = config.max_seq_len + 1

        self.num_windows = len(self.tokens) // self.window
        self.meta = None

        if self.num_windows == 0:
            raise ValueError(
                f"{self.path} holds {len(self.tokens):,} tokens, fewer than the "
                f"{self.window} needed for a single window."
            )

    @classmethod
    def from_meta(cls, path, config):
        path = Path(path)
        dataset = cls(path, config)
        meta_path = path.parent / "meta.json"

        dataset.meta = json.loads(
            meta_path.read_text(encoding="utf8")
        )

        return dataset

    @property
    def num_tokens(self):
        return len(self.tokens)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        start = index * self.window
        window = self.tokens[start : start + self.window]

        return {
            "input_ids": torch.from_numpy(
                window.astype(np.int64)
            )
        }

