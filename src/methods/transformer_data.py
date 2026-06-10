"""Dataset/tokenization helpers for Hugging Face Transformer methods.

These helpers keep the HateXplain split policy consistent across Full FT,
Frozen, LoRA, LP-FT, and Efficient-Head. The size fields are intentionally
verbose because they let us manually compare a new run with the old aggregate
CSVs later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.label_policy import LABEL_ID_TO_NAME, LABEL_NAME_TO_ID
from src.data.preprocessing import (
    preprocess_hatexplain_split,
    select_data_fraction,
    tokenize_preprocessed_record,
)


@dataclass(frozen=True)
class TokenizedSplit:
    """Tokenized split plus the accounting numbers we save in result JSON."""

    dataset: list[dict[str, Any]]
    records: list[dict[str, Any]]
    raw_size: int | None
    preprocessed_size: int
    dropped_no_majority_count: int | None


def find_split_name(ds_dict, candidates):
    """Return the first available split name from a candidate list."""

    for name in candidates:
        if name in ds_dict:
            return name
    return None


def resolve_eval_split_name(ds_dict, test_split_name: str = "test") -> str:
    """Resolve validation split while refusing to reuse the test split."""

    eval_split = find_split_name(ds_dict, ["validation", "valid"])
    if eval_split is None:
        raise ValueError(
            "No validation/valid split found. The test split is never used as "
            f"validation. Available splits: {list(ds_dict.keys())}"
        )
    if eval_split == test_split_name:
        raise ValueError(
            f"Evaluation split '{eval_split}' would alias the configured test split. "
            "Use distinct validation and test splits."
        )
    return eval_split


def maybe_select_subset(
    records: list[dict[str, Any]],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Apply a simple first-N sample cap for smoke/debug runs."""

    if max_samples is None:
        return records
    return records[: min(max_samples, len(records))]


def build_fixed_label_maps():
    """Return the fixed HateXplain label maps and number of labels."""

    return dict(LABEL_ID_TO_NAME), dict(LABEL_NAME_TO_ID), len(LABEL_ID_TO_NAME)


def build_tokenized_dataset_with_stats(
    examples,
    tokenizer,
    max_length: int,
    data_fraction: float | None = None,
    fraction_seed: int = 42,
    max_samples: int | None = None,
) -> TokenizedSplit:
    """Preprocess, optionally subset, tokenize, and keep split-size evidence.

    `raw_size` is what the Hugging Face split exposed. `preprocessed_size` is
    after strict-majority filtering. `dataset` is the final selected/tokenized
    list that training actually sees.
    """

    raw_size = len(examples) if hasattr(examples, "__len__") else None
    records = preprocess_hatexplain_split(examples)
    preprocessed_size = len(records)
    dropped_no_majority_count = (
        raw_size - preprocessed_size if raw_size is not None else None
    )
    if data_fraction is not None:
        records = select_data_fraction(records, data_fraction, seed=fraction_seed)
    records = maybe_select_subset(records, max_samples)
    tokenized = [
        tokenize_preprocessed_record(record, tokenizer, max_length=max_length)
        for record in records
    ]
    return TokenizedSplit(
        dataset=tokenized,
        records=records,
        raw_size=raw_size,
        preprocessed_size=preprocessed_size,
        dropped_no_majority_count=dropped_no_majority_count,
    )
