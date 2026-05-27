from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.experiments.results import write_json


DEFAULT_LABEL_ID_TO_NAME = {
    0: "hatespeech",
    1: "normal",
    2: "offensive",
}

CONFUSION_MATRIX_COLUMNS = (
    "method",
    "trial_id",
    "search_stage",
    "seed",
    "split",
    "summary_path",
    "prediction_path",
    "true_label",
    "true_label_name",
    "predicted_label",
    "predicted_label_name",
    "count",
)

ERROR_EXAMPLE_COLUMNS = (
    "method",
    "trial_id",
    "search_stage",
    "seed",
    "split",
    "summary_path",
    "prediction_path",
    "id",
    "text",
    "label",
    "label_name",
    "predicted_label",
    "predicted_label_name",
    "confidence",
)

AUROC_COLUMNS = (
    "method",
    "trial_id",
    "search_stage",
    "seed",
    "split",
    "summary_path",
    "prediction_path",
    "available",
    "score_source",
    "macro_ovr",
    "weighted_ovr",
    "reason",
)


def load_prediction_file(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Prediction file does not contain a predictions list: {path}")
    return predictions


def analyze_prediction_file(
    path: str | Path,
    *,
    split: str | None = None,
    max_error_examples: int = 50,
    label_id_to_name: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    prediction_path = Path(path)
    rows = load_prediction_file(prediction_path)
    labels = _label_entries(rows, label_id_to_name or DEFAULT_LABEL_ID_TO_NAME)
    label_ids = [entry["id"] for entry in labels]
    label_names = [entry["name"] for entry in labels]
    label_to_index = {label_id: index for index, label_id in enumerate(label_ids)}
    matrix = [[0 for _ in label_ids] for _ in label_ids]

    for row in rows:
        gold = _required_int(row, "label", prediction_path)
        predicted = _required_int(row, "predicted_label", prediction_path)
        if gold not in label_to_index:
            label_to_index[gold] = len(label_ids)
            label_ids.append(gold)
            label_names.append(str(gold))
            for matrix_row in matrix:
                matrix_row.append(0)
            matrix.append([0 for _ in label_ids])
        if predicted not in label_to_index:
            label_to_index[predicted] = len(label_ids)
            label_ids.append(predicted)
            label_names.append(str(predicted))
            for matrix_row in matrix:
                matrix_row.append(0)
            matrix.append([0 for _ in label_ids])
        matrix[label_to_index[gold]][label_to_index[predicted]] += 1

    error_analysis = _build_error_analysis(
        rows,
        max_error_examples=max_error_examples,
    )
    return {
        "split": split or _infer_split_from_path(prediction_path),
        "prediction_path": prediction_path.as_posix(),
        "count": len(rows),
        "labels": [
            {"id": label_id, "name": label_name}
            for label_id, label_name in zip(label_ids, label_names)
        ],
        "confusion_matrix": {
            "rows_true_labels": label_names,
            "columns_predicted_labels": label_names,
            "matrix": matrix,
        },
        "auroc": _compute_auroc(rows, label_ids),
        "error_analysis": error_analysis,
    }


def write_prediction_analysis_artifacts(
    output_dir: str | Path,
    report: dict[str, Any],
    *,
    splits: Iterable[str] = ("eval", "test"),
    max_error_examples: int = 50,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_names = tuple(splits)
    analyses = []
    skipped = []

    for record in report.get("runs") or []:
        for split in split_names:
            prediction_path = _prediction_path_for_split(record, split)
            if not prediction_path:
                continue
            resolved_path = _resolve_prediction_path(prediction_path, record)
            if not resolved_path.is_file():
                skipped.append(
                    {
                        **_run_identity(record),
                        "split": split,
                        "prediction_path": str(prediction_path),
                        "reason": "prediction file not found",
                    }
                )
                continue
            try:
                analysis = analyze_prediction_file(
                    resolved_path,
                    split=split,
                    max_error_examples=max_error_examples,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                skipped.append(
                    {
                        **_run_identity(record),
                        "split": split,
                        "prediction_path": resolved_path.as_posix(),
                        "reason": (
                            "analysis failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
                continue
            analyses.append(
                {
                    **_run_identity(record),
                    "split": split,
                    "prediction_path": resolved_path.as_posix(),
                    "analysis": analysis,
                }
            )

    analysis_json = write_json(
        output_path / "prediction_analysis.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analyzed_prediction_files": len(analyses),
            "skipped_prediction_files": len(skipped),
            "runs": analyses,
            "skipped": skipped,
        },
    )
    confusion_csv = _write_csv(
        output_path / "confusion_matrices.csv",
        _confusion_matrix_rows(analyses),
        CONFUSION_MATRIX_COLUMNS,
    )
    error_csv = _write_csv(
        output_path / "error_examples.csv",
        _error_example_rows(analyses),
        ERROR_EXAMPLE_COLUMNS,
    )
    auroc_csv = _write_csv(
        output_path / "auroc_summary.csv",
        _auroc_rows(analyses),
        AUROC_COLUMNS,
    )
    return {
        "analysis_json": analysis_json,
        "confusion_matrix_csv": confusion_csv,
        "error_examples_csv": error_csv,
        "auroc_csv": auroc_csv,
    }


def _label_entries(
    rows: Sequence[Mapping[str, Any]],
    defaults: Mapping[int, str],
) -> list[dict[str, Any]]:
    labels = {int(label_id): str(label_name) for label_id, label_name in defaults.items()}
    for row in rows:
        for id_key, name_key in (
            ("label", "label_name"),
            ("predicted_label", "predicted_label_name"),
        ):
            if id_key in row and row[id_key] is not None:
                label_id = int(row[id_key])
                labels.setdefault(label_id, str(row.get(name_key) or label_id))
    return [{"id": label_id, "name": labels[label_id]} for label_id in sorted(labels)]


def _required_int(row: Mapping[str, Any], key: str, path: Path) -> int:
    if key not in row:
        raise ValueError(f"Prediction row in {path} is missing required key: {key}")
    return int(row[key])


def _infer_split_from_path(path: Path) -> str:
    name = path.name
    if name.endswith("_predictions.json"):
        return name.removesuffix("_predictions.json")
    return "unknown"


def _score_vector(row: Mapping[str, Any]) -> tuple[str | None, list[float]]:
    probabilities = row.get("probabilities")
    if probabilities:
        return "probabilities", [float(value) for value in probabilities]
    logits = row.get("logits")
    if logits:
        return "logits_softmax", _softmax([float(value) for value in logits])
    return None, []


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _compute_auroc(
    rows: Sequence[Mapping[str, Any]],
    label_ids: Sequence[int],
) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "no predictions"}
    score_source, first_scores = _score_vector(rows[0])
    if not score_source or not first_scores:
        return {
            "available": False,
            "reason": "prediction rows do not include probabilities or logits",
        }

    y_true = []
    score_rows = []
    for row in rows:
        row_source, scores = _score_vector(row)
        if row_source != score_source or len(scores) != len(first_scores):
            return {
                "available": False,
                "reason": "prediction rows have inconsistent score vectors",
            }
        y_true.append(int(row["label"]))
        score_rows.append(scores)
    if len(first_scores) != len(label_ids):
        return {
            "available": False,
            "score_source": score_source,
            "reason": (
                "score vector length does not match the number of labels: "
                f"scores={len(first_scores)}, labels={len(label_ids)}"
            ),
        }

    if len(set(y_true)) < 2:
        return {
            "available": False,
            "score_source": score_source,
            "reason": "AUROC requires at least two classes in y_true",
        }

    try:
        per_class = _per_class_auroc(y_true, score_rows, label_ids)
        missing_labels = [
            label_id for label_id, value in per_class.items() if value is None
        ]
        if missing_labels:
            raise ValueError(
                "AUROC is not defined for every label; missing positive or "
                f"negative examples for labels: {missing_labels}"
            )
        available_items = [
            (label_id, value)
            for label_id, value in per_class.items()
            if value is not None
        ]
        if not available_items:
            raise ValueError("AUROC requires at least one one-vs-rest class split.")
        supports = Counter(y_true)
        macro_ovr = sum(value for _label_id, value in available_items) / len(
            available_items
        )
        weighted_denominator = sum(supports[int(label_id)] for label_id, _value in available_items)
        weighted_ovr = (
            sum(supports[int(label_id)] * value for label_id, value in available_items)
            / weighted_denominator
            if weighted_denominator
            else macro_ovr
        )
    except ValueError as exc:
        return {
            "available": False,
            "score_source": score_source,
            "reason": str(exc),
        }

    return {
        "available": True,
        "score_source": score_source,
        "macro_ovr": float(macro_ovr),
        "weighted_ovr": float(weighted_ovr),
        "per_class_ovr": {str(label_id): value for label_id, value in per_class.items()},
    }


def _per_class_auroc(
    y_true: Sequence[int],
    score_rows: Sequence[Sequence[float]],
    label_ids: Sequence[int],
) -> dict[int, float | None]:
    scores: dict[int, float | None] = {}
    for index, label_id in enumerate(label_ids):
        binary_true = [1 if label == label_id else 0 for label in y_true]
        if len(set(binary_true)) < 2:
            scores[int(label_id)] = None
            continue
        class_scores = [row[index] for row in score_rows]
        scores[int(label_id)] = _binary_auc(binary_true, class_scores)
    return scores


def _binary_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    positive_scores = [score for label, score in zip(y_true, scores) if label == 1]
    negative_scores = [score for label, score in zip(y_true, scores) if label == 0]
    if not positive_scores or not negative_scores:
        raise ValueError("AUROC requires both positive and negative examples.")

    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def _build_error_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_error_examples: int,
) -> dict[str, Any]:
    errors = [
        row for row in rows if int(row["label"]) != int(row["predicted_label"])
    ]
    pair_counts = Counter(
        (
            int(row["label"]),
            str(row.get("label_name") or row["label"]),
            int(row["predicted_label"]),
            str(row.get("predicted_label_name") or row["predicted_label"]),
        )
        for row in errors
    )
    return {
        "error_count": len(errors),
        "error_rate": len(errors) / len(rows) if rows else None,
        "confusion_pairs": [
            {
                "label": label,
                "label_name": label_name,
                "predicted_label": predicted,
                "predicted_label_name": predicted_name,
                "count": count,
            }
            for (label, label_name, predicted, predicted_name), count in sorted(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "examples": [
            _error_example(row)
            for row in errors[: max(0, int(max_error_examples))]
        ],
    }


def _error_example(row: Mapping[str, Any]) -> dict[str, Any]:
    _source, scores = _score_vector(row)
    predicted = int(row["predicted_label"])
    confidence = scores[predicted] if scores and predicted < len(scores) else None
    return {
        "id": row.get("id"),
        "text": row.get("text"),
        "label": int(row["label"]),
        "label_name": row.get("label_name"),
        "predicted_label": predicted,
        "predicted_label_name": row.get("predicted_label_name"),
        "confidence": confidence,
    }


def _prediction_path_for_split(record: Mapping[str, Any], split: str) -> Any:
    return record.get(f"{split}_predictions_path")


def _resolve_prediction_path(path_value: Any, record: Mapping[str, Any]) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or path.is_file():
        return path
    summary_path = record.get("summary_path")
    if summary_path:
        summary_parent = Path(str(summary_path)).parent
        for candidate in (summary_parent / path, summary_parent / path.name):
            if candidate.is_file():
                return candidate
    return path


def _run_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": record.get("method"),
        "trial_id": record.get("trial_id"),
        "search_stage": record.get("search_stage"),
        "seed": record.get("seed"),
        "summary_path": record.get("summary_path"),
    }


def _confusion_matrix_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in analyses:
        analysis = item["analysis"]
        labels = analysis["labels"]
        matrix = analysis["confusion_matrix"]["matrix"]
        for true_index, true_label in enumerate(labels):
            for pred_index, predicted_label in enumerate(labels):
                rows.append(
                    {
                        **_analysis_identity(item),
                        "true_label": true_label["id"],
                        "true_label_name": true_label["name"],
                        "predicted_label": predicted_label["id"],
                        "predicted_label_name": predicted_label["name"],
                        "count": matrix[true_index][pred_index],
                    }
                )
    return rows


def _error_example_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in analyses:
        for example in item["analysis"]["error_analysis"]["examples"]:
            rows.append({**_analysis_identity(item), **example})
    return rows


def _auroc_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in analyses:
        auroc = item["analysis"]["auroc"]
        rows.append(
            {
                **_analysis_identity(item),
                "available": bool(auroc.get("available")),
                "score_source": auroc.get("score_source"),
                "macro_ovr": auroc.get("macro_ovr"),
                "weighted_ovr": auroc.get("weighted_ovr"),
                "reason": auroc.get("reason"),
            }
        )
    return rows


def _analysis_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": item.get("method"),
        "trial_id": item.get("trial_id"),
        "search_stage": item.get("search_stage"),
        "seed": item.get("seed"),
        "split": item.get("split"),
        "summary_path": item.get("summary_path"),
        "prediction_path": item.get("prediction_path"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    column_names = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in column_names})
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value
