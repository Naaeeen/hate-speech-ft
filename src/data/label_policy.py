"""Shared HateXplain label construction policy.

HateXplain has three annotator labels per post in the classification task.
For the main 3-class experiments, we use strict majority vote and exclude
samples where all three annotators disagree.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


LABEL_ID_TO_NAME = {
    0: "hatespeech",
    1: "normal",
    2: "offensive",
}
LABEL_NAME_TO_ID = {name: label_id for label_id, name in LABEL_ID_TO_NAME.items()}

UNDECIDED_LABEL_ID = -1
UNDECIDED_LABEL_NAME = "undecided"

LABEL_POLICY = (
    "Use the HateXplain 3-class labels in the Hugging Face ClassLabel order: "
    "0=hatespeech, 1=normal, 2=offensive. Build the final training label by "
    "strict majority vote over the three annotator labels. If no strict "
    "majority exists, exclude the sample from the main classification training "
    "and evaluation data unless an audit explicitly asks to keep undecided "
    "examples."
)


def normalize_label_id(label: Any) -> int:
    """Convert a raw or Hugging Face HateXplain label to the shared id."""

    if isinstance(label, bool):
        raise TypeError("Boolean values are not valid HateXplain labels.")

    if isinstance(label, int):
        if label not in LABEL_ID_TO_NAME:
            raise ValueError(f"Unknown HateXplain label id: {label!r}")
        return label

    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized not in LABEL_NAME_TO_ID:
            raise ValueError(f"Unknown HateXplain label name: {label!r}")
        return LABEL_NAME_TO_ID[normalized]

    raise TypeError(f"Unsupported HateXplain label type: {type(label).__name__}")


def _extract_raw_annotator_labels(example: Mapping[str, Any]) -> list[Any]:
    if "annotators" not in example:
        raise KeyError("HateXplain example is missing required field 'annotators'.")

    annotators = example["annotators"]

    # Hugging Face Sequence({label, annotator_id, target}) may appear as a
    # dictionary of lists when materialized by datasets.
    if isinstance(annotators, Mapping):
        if "label" not in annotators:
            raise KeyError("HateXplain annotators field is missing 'label'.")
        labels = annotators["label"]
        if isinstance(labels, str) or not isinstance(labels, Sequence):
            raise TypeError("'annotators.label' must be a sequence of labels.")
        return list(labels)

    # The raw JSON stores annotators as a list of dictionaries.
    if isinstance(annotators, Sequence) and not isinstance(annotators, str):
        labels = []
        for annotator in annotators:
            if not isinstance(annotator, Mapping) or "label" not in annotator:
                raise TypeError("Each annotator must be a mapping with a 'label'.")
            labels.append(annotator["label"])
        return labels

    raise TypeError("'annotators' must be a mapping or a sequence of mappings.")


def extract_annotator_label_ids(example: Mapping[str, Any]) -> list[int]:
    """Return annotator labels as shared integer ids."""

    raw_labels = _extract_raw_annotator_labels(example)
    return [normalize_label_id(label) for label in raw_labels]


def extract_annotator_label_names(example: Mapping[str, Any]) -> list[str]:
    """Return annotator labels as shared string names."""

    return [LABEL_ID_TO_NAME[label_id] for label_id in extract_annotator_label_ids(example)]


def majority_vote_label_id(label_ids: Sequence[int]) -> int | None:
    """Return the strict majority label id, or None when no majority exists."""

    if not label_ids:
        raise ValueError("Cannot build a majority label from an empty label list.")

    normalized_ids = [normalize_label_id(label_id) for label_id in label_ids]
    label_counts = Counter(normalized_ids)
    label_id, count = label_counts.most_common(1)[0]

    if count > len(normalized_ids) / 2:
        return label_id
    return None


def build_label_from_annotators(
    example: Mapping[str, Any],
    on_no_majority: str = "raise",
) -> int | None:
    """Build the shared training label from HateXplain annotators.

    Args:
        example: One HateXplain row/sample containing `annotators`.
        on_no_majority: One of:
            - "raise": raise ValueError when annotators have no strict majority.
            - "return_none": return None for no-majority examples.
            - "undecided": return -1 for no-majority examples.

    Returns:
        Label id 0, 1, or 2. May return None or -1 depending on
        `on_no_majority`.
    """

    label_ids = extract_annotator_label_ids(example)
    majority_label = majority_vote_label_id(label_ids)
    if majority_label is not None:
        return majority_label

    if on_no_majority == "raise":
        raise ValueError("HateXplain annotator labels have no strict majority.")
    if on_no_majority == "return_none":
        return None
    if on_no_majority == "undecided":
        return UNDECIDED_LABEL_ID

    raise ValueError(
        "on_no_majority must be one of: 'raise', 'return_none', 'undecided'."
    )
