from __future__ import annotations

from typing import Any

from src.data.label_policy import LABEL_ID_TO_NAME, LABEL_NAME_TO_ID
from src.data.preprocessing import (
    preprocess_hatexplain_split,
    select_data_fraction,
    tokenize_preprocessed_record,
)


def find_split_name(ds_dict, candidates):
    for name in candidates:
        if name in ds_dict:
            return name
    return None


def maybe_select_subset(records: list[dict[str, Any]], max_samples: int | None):
    if max_samples is None:
        return records
    return records[: min(max_samples, len(records))]


def build_fixed_label_maps():
    return dict(LABEL_ID_TO_NAME), dict(LABEL_NAME_TO_ID), len(LABEL_ID_TO_NAME)


def build_tokenized_dataset_with_count(
    examples,
    tokenizer,
    max_length: int,
    data_fraction: float | None = None,
    fraction_seed: int = 42,
    max_samples: int | None = None,
):
    records = preprocess_hatexplain_split(examples)
    full_count = len(records)
    if data_fraction is not None:
        records = select_data_fraction(records, data_fraction, seed=fraction_seed)
    records = maybe_select_subset(records, max_samples)
    tokenized = [
        tokenize_preprocessed_record(record, tokenizer, max_length=max_length)
        for record in records
    ]
    return tokenized, full_count


def build_tokenized_dataset(
    examples,
    tokenizer,
    max_length: int,
    max_samples: int | None = None,
):
    tokenized, _ = build_tokenized_dataset_with_count(
        examples,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
    )
    return tokenized
