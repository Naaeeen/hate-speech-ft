from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .tokenizer import FrozenDistilBertTokenizer


class HateXplainFrozenDistilBertDataset(Dataset):
    """Dataset wrapper for HateXplain text classification with DistilBERT inputs."""

    def __init__(
        self,
        records: list[dict],
        tokenizer: FrozenDistilBertTokenizer,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer.encode(record["text"])

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(int(record["label"]), dtype=torch.long),
        }
