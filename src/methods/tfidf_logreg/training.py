from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.experiments.results import write_json


def parse_ngram_range(value: str | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            parsed = json.loads(text)
        else:
            parsed = [item.strip() for item in text.split(",")]
    else:
        parsed = list(value)

    if len(parsed) != 2:
        raise ValueError(
            "ngram_range must contain exactly two integers, e.g. '1,2' or '[1,2]'."
        )
    try:
        lower, upper = (int(parsed[0]), int(parsed[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ngram_range must contain integer values, e.g. '1,2' or '[1,2]'."
        ) from exc
    if lower <= 0 or upper < lower:
        raise ValueError(
            "ngram_range must satisfy 0 < lower <= upper; "
            f"received ({lower}, {upper})."
        )
    return lower, upper


def load_libraries():
    try:
        from datasets import load_dataset
        from joblib import dump
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "TF-IDF + Logistic Regression requires datasets, scikit-learn, and "
            "joblib. In Colab run `pip install -r requirements-colab.txt` first."
        ) from exc
    return load_dataset, dump, TfidfVectorizer, LogisticRegression, Pipeline


def validate_classical_args(args: argparse.Namespace, ngram_range: tuple[int, int]) -> None:
    if args.min_df < 1:
        raise ValueError("--min_df must be >= 1.")
    if args.max_features is not None and args.max_features < 1:
        raise ValueError("--max_features must be >= 1.")
    if args.C <= 0:
        raise ValueError("--C must be > 0.")
    if args.mixed_precision != "none":
        raise ValueError("TF-IDF is CPU/classical and does not support mixed precision.")
    if args.gradient_checkpointing:
        raise ValueError("TF-IDF is CPU/classical and does not support gradient checkpointing.")
    if ngram_range[0] > ngram_range[1]:
        raise ValueError("Invalid --ngram_range.")
    for option_name in ("max_train_samples", "max_eval_samples", "max_test_samples"):
        value = getattr(args, option_name)
        if value is not None and value < 1:
            raise ValueError(f"--{option_name} must be >= 1 when provided.")


def build_pipeline(
    *,
    TfidfVectorizer,
    LogisticRegression,
    Pipeline,
    args: argparse.Namespace,
    ngram_range: tuple[int, int],
):
    class_weight = "balanced" if args.class_weighting == "balanced" else None
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=args.min_df,
                    max_features=args.max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=args.C,
                    solver="liblinear",
                    random_state=args.seed,
                    max_iter=1000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def get_model_stats(pipeline) -> tuple[int, int, int]:
    classifier = pipeline.named_steps["clf"]
    vectorizer = pipeline.named_steps["tfidf"]
    coef_size = getattr(classifier.coef_, "size", None)
    if coef_size is None:
        coef_size = sum(len(row) if hasattr(row, "__len__") else 1 for row in classifier.coef_)
    intercept_size = getattr(classifier.intercept_, "size", None)
    if intercept_size is None:
        intercept_size = len(classifier.intercept_)
    trainable_params = int(coef_size + intercept_size)
    vocab_size = int(len(vectorizer.vocabulary_))
    return trainable_params, trainable_params, vocab_size


def _class_counts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_id: int,
) -> tuple[int, int, int, int]:
    true_positive = sum(
        1 for gold, predicted in zip(y_true, y_pred) if gold == label_id and predicted == label_id
    )
    false_positive = sum(
        1 for gold, predicted in zip(y_true, y_pred) if gold != label_id and predicted == label_id
    )
    false_negative = sum(
        1 for gold, predicted in zip(y_true, y_pred) if gold == label_id and predicted != label_id
    )
    support = sum(1 for gold in y_true if gold == label_id)
    return true_positive, false_positive, false_negative, support


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    prefix: str,
    label_id_to_name: Mapping[int, str],
) -> dict[str, float | int]:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Metric inputs must have the same length: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}."
        )
    if not y_true:
        raise ValueError("Cannot compute metrics for an empty split.")

    label_ids = sorted(label_id_to_name)
    correct = sum(1 for gold, predicted in zip(y_true, y_pred) if gold == predicted)
    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []
    metrics: dict[str, float | int] = {
        f"{prefix}_accuracy": _safe_divide(correct, len(y_true)),
    }

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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def save_classical_prediction_file(
    path: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    predicted_labels: Sequence[int],
    probabilities: Any,
    id2label: Mapping[int, str],
) -> Path:
    probability_rows = _as_list(probabilities)
    if len(records) != len(predicted_labels):
        raise ValueError(
            "Prediction output length does not match source records: "
            f"records={len(records)}, predictions={len(predicted_labels)}."
        )
    if probability_rows and len(records) != len(probability_rows):
        raise ValueError(
            "Probability output length does not match source records: "
            f"records={len(records)}, probabilities={len(probability_rows)}."
        )

    predictions = []
    for index, (record, predicted_label) in enumerate(zip(records, predicted_labels)):
        gold_label = int(record["label"])
        probabilities_for_record = (
            [float(value) for value in _as_list(probability_rows[index])]
            if probability_rows
            else []
        )
        predicted_label_id = int(predicted_label)
        predictions.append(
            {
                "id": record.get("id"),
                "text": record.get("text"),
                "label": gold_label,
                "label_name": id2label.get(gold_label),
                "predicted_label": predicted_label_id,
                "predicted_label_name": id2label.get(predicted_label_id),
                "probabilities": probabilities_for_record,
            }
        )

    return write_json(
        path,
        {
            "count": len(predictions),
            "predictions": predictions,
        },
    )
