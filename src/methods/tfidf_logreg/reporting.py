from __future__ import annotations

from pathlib import Path
from typing import Any

from src.methods.tfidf_logreg.data import ClassicalSplit, records_to_xy
from src.methods.tfidf_logreg.training import save_classical_prediction_file


def write_final_prediction_files(
    *,
    output_dir: Path,
    args,
    pipeline,
    eval_data: ClassicalSplit,
    eval_predictions,
    test_data: ClassicalSplit | None,
    id2label: dict[int, str],
) -> dict[str, Path]:
    if args.search_stage != "final":
        return {}

    prediction_paths = {
        "eval": save_classical_prediction_file(
            output_dir / "eval_predictions.json",
            records=eval_data.records,
            predicted_labels=eval_predictions,
            probabilities=pipeline.predict_proba(records_to_xy(eval_data.records)[0]),
            id2label=id2label,
        )
    }
    if args.run_test:
        if test_data is None:
            raise ValueError("Cannot save final test predictions before loading test data.")
        x_test, _y_test = records_to_xy(test_data.records)
        prediction_paths["test"] = save_classical_prediction_file(
            output_dir / "test_predictions.json",
            records=test_data.records,
            predicted_labels=pipeline.predict(x_test),
            probabilities=pipeline.predict_proba(x_test),
            id2label=id2label,
        )
    return prediction_paths


def print_result_report(
    *,
    output_dir: Path,
    eval_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    runtime_metrics: dict[str, Any],
    result_paths: dict[str, Path],
    prediction_paths: dict[str, Path],
) -> None:
    print("\nFinal validation metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")
    if test_metrics is not None:
        print("\nFinal test metrics:")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")
    print("\nRuntime metrics:")
    for key, value in runtime_metrics.items():
        print(f"{key}: {value}")
    print("\nResult files:")
    for key, value in result_paths.items():
        print(f"{key}: {value}")
    if prediction_paths:
        print("\nPrediction files:")
        for key, value in prediction_paths.items():
            print(f"{key}: {value}")
    print(f"\nDone. Saved to: {output_dir}")
