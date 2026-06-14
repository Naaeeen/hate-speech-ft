"""Flat classification metrics shared by non-Trainer methods.

TF-IDF and BiLSTM do not use Hugging Face's metric callback, so this file
creates the same flat keys: accuracy, per-class precision/recall/F1/support,
and macro averages. The flat shape works for local JSON and W&B logging.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from src.data.label_policy import LABEL_ID_TO_NAME


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _class_counts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_id: int,
) -> tuple[int, int, int, int]:
    true_positive = sum(
        1
        for gold, predicted in zip(y_true, y_pred)
        if gold == label_id and predicted == label_id
    )
    false_positive = sum(
        1
        for gold, predicted in zip(y_true, y_pred)
        if gold != label_id and predicted == label_id
    )
    false_negative = sum(
        1
        for gold, predicted in zip(y_true, y_pred)
        if gold == label_id and predicted != label_id
    )
    support = sum(1 for gold in y_true if gold == label_id)
    return true_positive, false_positive, false_negative, support


def build_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    prefix: str,
    label_id_to_name: Mapping[int, str] = LABEL_ID_TO_NAME,
) -> dict[str, float | int]:
    """Compute accuracy, per-class metrics, and macro metrics with a prefix."""

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Metric inputs must have the same length: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}."
        )
    if not y_true:
        raise ValueError(f"Cannot compute {prefix} metrics for an empty split.")

    label_ids = sorted(label_id_to_name)
    correct = sum(1 for gold, predicted in zip(y_true, y_pred) if gold == predicted)
    metrics: dict[str, float | int] = {
        f"{prefix}_accuracy": _safe_divide(correct, len(y_true))
    }
    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []

    for label_id in label_ids:
        true_positive, false_positive, false_negative, support = _class_counts(
            y_true,
            y_pred,
            label_id,
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        label_name = label_id_to_name[label_id]
        metrics[f"{prefix}_precision_{label_name}"] = precision
        metrics[f"{prefix}_recall_{label_name}"] = recall
        metrics[f"{prefix}_f1_{label_name}"] = f1
        metrics[f"{prefix}_support_{label_name}"] = support
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    metrics[f"{prefix}_precision_macro"] = sum(per_class_precision) / len(label_ids)
    metrics[f"{prefix}_recall_macro"] = sum(per_class_recall) / len(label_ids)
    metrics[f"{prefix}_f1_macro"] = sum(per_class_f1) / len(label_ids)
    return metrics
