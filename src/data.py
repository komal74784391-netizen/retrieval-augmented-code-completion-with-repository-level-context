"""Data pipeline for the paper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset


class JsonlDataset(IterableDataset):
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _records(self) -> Iterator[dict]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for rec in self._records():
            ids = self.tokenizer(rec["text"], truncation=True, max_length=self.max_len)
            tensor = torch.tensor(ids["input_ids"], dtype=torch.long)
            yield tensor, torch.zeros_like(tensor)
