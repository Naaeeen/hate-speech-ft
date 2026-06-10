"""PyTorch Dataset wrapper for BiLSTM records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .tokenizer import StandardBiLSTMTokenizer


class HateXplainBiLSTMDataset(Dataset):
    """Torch Dataset that encodes shared HateXplain text for the BiLSTM."""

    def __init__(
        self,
        records: list[dict],
        tokenizer: StandardBiLSTMTokenizer,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the number of preprocessed records in this split."""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Encode one record into tensors consumed by the BiLSTM loop."""

        record = self.records[index]
        encoded = self.tokenizer.encode(record["text"])
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "lengths": torch.tensor(encoded["length"], dtype=torch.long),
            "labels": torch.tensor(int(record["label"]), dtype=torch.long),
        }
