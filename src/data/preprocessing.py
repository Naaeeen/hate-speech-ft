"""Shared dataset-level preprocessing for HateXplain experiments.

This module is the common layer every experiment should run before entering a
method-specific pipeline. It standardizes text construction, label construction,
optional no-majority filtering, deterministic data fractions, and transformer
tokenization wrappers.
"""

from collections.abc import Iterable, Mapping
from random import Random
from typing import Any

from src.data.label_policy import (
    LABEL_ID_TO_NAME,
    UNDECIDED_LABEL_ID,
    UNDECIDED_LABEL_NAME,
    extract_annotator_label_ids,
    extract_annotator_label_names,
    majority_vote_label_id,
)
from src.data.text_field_policy import TextTokenizer, build_text_from_post_tokens


HATEXPLAIN_PREPROCESSING_POLICY = """
Dataset-level preprocessing policy
==================================

Use the official train/validation/test splits from Hugging Face. Do not create
new random splits for the main experiments.

For text, use only `post_tokens` and construct:

    text = " ".join(example["post_tokens"])

No extra dataset-level cleaning is applied, and rationales are not model input
for the main classification experiments. Metadata such as post id, targets, or
annotator ids must not be concatenated into the text.

For labels, use strict majority vote over the three annotator labels using the
Hugging Face ClassLabel order: 0=hatespeech, 1=normal, 2=offensive. Samples
with no majority vote are excluded from the main classification dataset by
default because they do not have a stable gold label.

For data-fraction experiments, sample deterministically with a shared seed and
preserve the selected records in original split order.

For transformer methods, apply model-specific tokenization after this shared
record is built:

    tokenizer(record["text"], truncation=True, max_length=max_length)

For TF-IDF, Logistic Regression, and Bi-LSTM baselines, use the same `text`
field and then apply the method-specific vectorizer or vocabulary pipeline.
""".strip()


def _extract_example_id(example: Mapping[str, Any]) -> str:
    if "id" in example:
        return str(example["id"])
    if "post_id" in example:
        return str(example["post_id"])
    raise KeyError("HateXplain example is missing required field 'id' or 'post_id'.")


def preprocess_hatexplain_example(
    example: Mapping[str, Any],
    include_undecided: bool = False,
) -> dict[str, Any] | None:
    """Convert one HateXplain example into a shared model-ready record.

    Args:
        example: One row/sample from the Hugging Face dataset or raw JSON.
        include_undecided: If False, return None for samples where the three
            annotators have no strict majority. If True, keep them with
            `label=-1` and `label_name="undecided"` for auditing only.

    Returns:
        A dictionary with stable fields shared by all experiment methods, or
        None when the sample has no majority label and `include_undecided` is
        False.
    """

    text = build_text_from_post_tokens(example)
    annotator_label_ids = extract_annotator_label_ids(example)
    annotator_label_names = extract_annotator_label_names(example)
    label_id = majority_vote_label_id(annotator_label_ids)
    has_majority_label = label_id is not None

    if label_id is None:
        if not include_undecided:
            return None
        label_id = UNDECIDED_LABEL_ID
        label_name = UNDECIDED_LABEL_NAME
    else:
        label_name = LABEL_ID_TO_NAME[label_id]

    return {
        "id": _extract_example_id(example),
        "text": text,
        "label": label_id,
        "label_name": label_name,
        "annotator_label_ids": annotator_label_ids,
        "annotator_label_names": annotator_label_names,
        "token_count": len(example["post_tokens"]),
        "has_majority_label": has_majority_label,
    }


def preprocess_hatexplain_split(
    examples: Iterable[Mapping[str, Any]],
    include_undecided: bool = False,
) -> list[dict[str, Any]]:
    """Preprocess a split and drop no-majority samples by default."""

    records = []
    for example in examples:
        record = preprocess_hatexplain_example(
            example,
            include_undecided=include_undecided,
        )
        if record is not None:
            records.append(record)
    return records


def select_data_fraction(
    records: list[dict[str, Any]],
    fraction: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a deterministic data fraction while preserving split order."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and less than or equal to 1.")

    if not records or fraction == 1:
        return list(records)

    selected_count = max(1, int(len(records) * fraction))
    indices = list(range(len(records)))
    Random(seed).shuffle(indices)
    selected_indices = sorted(indices[:selected_count])
    return [records[index] for index in selected_indices]


def tokenize_preprocessed_record(
    record: Mapping[str, Any],
    tokenizer: TextTokenizer,
    max_length: int,
    **tokenizer_kwargs: Any,
) -> dict[str, Any]:
    """Tokenize a shared preprocessed record for Hugging Face Trainer."""

    options = {"truncation": True, "max_length": max_length}
    options.update(tokenizer_kwargs)
    tokenized = dict(tokenizer(str(record["text"]), **options))
    tokenized["labels"] = int(record["label"])
    return tokenized
