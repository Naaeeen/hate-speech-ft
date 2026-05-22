from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.preprocessing import preprocess_hatexplain_split, select_data_fraction
from src.methods.transformer_data import find_split_name, maybe_select_subset, resolve_eval_split_name


@dataclass(frozen=True)
class BiLSTMSplit:
    records: list[dict[str, Any]]
    raw_size: int | None
    preprocessed_size: int
    dropped_no_majority_count: int | None


def load_dataset_library():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Bi-LSTM requires the 'datasets' package. In Colab run "
            "`pip install -r requirements-colab.txt` first."
        ) from exc
    return load_dataset


def _build_split(
    examples,
    *,
    data_fraction: float | None = None,
    fraction_seed: int = 42,
    max_samples: int | None = None,
) -> BiLSTMSplit:
    raw_size = len(examples) if hasattr(examples, "__len__") else None
    records = preprocess_hatexplain_split(examples)
    preprocessed_size = len(records)
    dropped_no_majority_count = (
        raw_size - preprocessed_size if raw_size is not None else None
    )
    if data_fraction is not None:
        records = select_data_fraction(records, data_fraction, seed=fraction_seed)
    records = maybe_select_subset(records, max_samples)
    return BiLSTMSplit(
        records=records,
        raw_size=raw_size,
        preprocessed_size=preprocessed_size,
        dropped_no_majority_count=dropped_no_majority_count,
    )


def resolve_bilstm_split_names(dataset, args) -> tuple[str, str, str | None]:
    train_split = find_split_name(dataset, ["train"])
    if train_split is None:
        raise ValueError(f"No train split found. Available splits: {list(dataset.keys())}")
    eval_split = resolve_eval_split_name(dataset, test_split_name=args.test_split_name)
    test_split = find_split_name(dataset, [args.test_split_name])
    if args.run_test and test_split is None:
        raise ValueError(
            f"No test split named '{args.test_split_name}' found. "
            f"Available splits: {list(dataset.keys())}"
        )
    if args.run_test and eval_split == test_split:
        raise ValueError(
            f"Evaluation split '{eval_split}' and test split '{test_split}' must differ."
        )
    return train_split, eval_split, test_split


def build_bilstm_data_splits(dataset, args, *, train_split: str, eval_split: str, test_split: str | None):
    train_data = _build_split(
        dataset[train_split],
        data_fraction=args.data_fraction,
        fraction_seed=args.data_fraction_seed,
        max_samples=args.max_train_samples,
    )
    eval_data = _build_split(dataset[eval_split], max_samples=args.max_eval_samples)
    test_data = (
        _build_split(dataset[test_split], max_samples=args.max_test_samples)
        if args.run_test and test_split is not None
        else None
    )
    return train_data, eval_data, test_data


def print_split_summary(
    *,
    train_split: str,
    eval_split: str,
    test_split: str | None,
    train_data: BiLSTMSplit,
    eval_data: BiLSTMSplit,
    test_data: BiLSTMSplit | None,
) -> None:
    print(
        f"Train split: {train_split}, size={len(train_data.records)} "
        f"(preprocessed full={train_data.preprocessed_size})"
    )
    print(
        f"Eval split: {eval_split}, size={len(eval_data.records)} "
        f"(preprocessed full={eval_data.preprocessed_size})"
    )
    print(
        "Strict-majority dropped: "
        f"train={train_data.dropped_no_majority_count}, "
        f"eval={eval_data.dropped_no_majority_count}"
    )
    if test_data is not None:
        print(
            f"Test split: {test_split}, size={len(test_data.records)} "
            f"(preprocessed full={test_data.preprocessed_size})"
        )
        print(f"Strict-majority dropped: test={test_data.dropped_no_majority_count}")
