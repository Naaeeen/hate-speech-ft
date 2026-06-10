"""Training helpers for the TF-IDF + Logistic Regression baseline.

The model is small enough that the "training loop" is one sklearn `.fit()`.
These helpers still produce stats and prediction files shaped like the neural
methods, so later manual tables do not need a totally separate TF-IDF format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.methods.classification_metrics import build_classification_metrics
from src.results import write_json
from src.results import validate_sample_selection_args


def parse_ngram_range(value: Sequence[int]) -> tuple[int, int]:
    """Validate the editable `ngram_range` list from manual config."""

    if isinstance(value, str):
        raise TypeError("ngram_range must be a Python list such as [1, 2].")
    parsed = list(value)
    if len(parsed) != 2:
        raise ValueError(
            "ngram_range must contain exactly two integers, e.g. [1, 2]."
        )
    try:
        lower, upper = (int(parsed[0]), int(parsed[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ngram_range must contain integer values, e.g. [1, 2]."
        ) from exc
    if lower <= 0 or upper < lower:
        raise ValueError(
            "ngram_range must satisfy 0 < lower <= upper; "
            f"received ({lower}, {upper})."
        )
    return lower, upper


def load_libraries():
    """Import sklearn/datasets/joblib lazily with a friendly Colab error."""

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


def validate_classical_args(args: Any, ngram_range: tuple[int, int]) -> None:
    """Validate TF-IDF/logreg manual settings before fitting sklearn."""

    validate_sample_selection_args(args)
    if args.min_df < 1:
        raise ValueError("min_df must be >= 1.")
    if args.max_df <= 0:
        raise ValueError("max_df must be > 0.")
    if args.max_features is not None and args.max_features < 1:
        raise ValueError("max_features must be >= 1.")
    if args.C <= 0:
        raise ValueError("C must be > 0.")
    if ngram_range[0] > ngram_range[1]:
        raise ValueError("Invalid ngram_range.")

def build_pipeline(
    *,
    TfidfVectorizer,
    LogisticRegression,
    Pipeline,
    args: Any,
    ngram_range: tuple[int, int],
):
    """Build the sklearn TF-IDF -> LogisticRegression pipeline."""

    class_weight = "balanced" if args.class_weighting == "balanced" else None
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=args.min_df,
                    max_df=args.max_df,
                    max_features=args.max_features,
                    sublinear_tf=args.sublinear_tf,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=args.C,
                    # liblinear is deterministic and handles this small
                    # one-vs-rest style baseline well enough for our comparison.
                    solver="liblinear",
                    random_state=args.seed,
                    max_iter=1000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def get_model_stats(pipeline) -> tuple[int, int, int]:
    """Return linear-model parameter counts and TF-IDF vocabulary size."""

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
    """Save prediction rows with probabilities, matching the neural JSON shape."""

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
