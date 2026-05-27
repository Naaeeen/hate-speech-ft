import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.experiments.prediction_analysis import (
    analyze_prediction_file,
    write_prediction_analysis_artifacts,
)


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"count": len(rows), "predictions": rows}),
        encoding="utf-8",
    )


class PredictionAnalysisTests(unittest.TestCase):
    def test_analyze_prediction_file_computes_confusion_auroc_and_errors(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_predictions.json"
            write_predictions(
                path,
                [
                    {
                        "id": "a",
                        "text": "true hate",
                        "label": 0,
                        "label_name": "hatespeech",
                        "predicted_label": 0,
                        "predicted_label_name": "hatespeech",
                        "probabilities": [0.9, 0.05, 0.05],
                    },
                    {
                        "id": "b",
                        "text": "hate confused as normal",
                        "label": 0,
                        "label_name": "hatespeech",
                        "predicted_label": 1,
                        "predicted_label_name": "normal",
                        "probabilities": [0.2, 0.7, 0.1],
                    },
                    {
                        "id": "c",
                        "text": "true normal",
                        "label": 1,
                        "label_name": "normal",
                        "predicted_label": 1,
                        "predicted_label_name": "normal",
                        "probabilities": [0.05, 0.9, 0.05],
                    },
                    {
                        "id": "d",
                        "text": "normal confused as offensive",
                        "label": 1,
                        "label_name": "normal",
                        "predicted_label": 2,
                        "predicted_label_name": "offensive",
                        "probabilities": [0.05, 0.2, 0.75],
                    },
                    {
                        "id": "e",
                        "text": "true offensive",
                        "label": 2,
                        "label_name": "offensive",
                        "predicted_label": 2,
                        "predicted_label_name": "offensive",
                        "probabilities": [0.05, 0.1, 0.85],
                    },
                    {
                        "id": "f",
                        "text": "true offensive again",
                        "label": 2,
                        "label_name": "offensive",
                        "predicted_label": 2,
                        "predicted_label_name": "offensive",
                        "probabilities": [0.1, 0.05, 0.85],
                    },
                ],
            )

            analysis = analyze_prediction_file(path, split="test", max_error_examples=1)

            self.assertEqual(analysis["split"], "test")
            self.assertEqual(analysis["count"], 6)
            self.assertEqual(
                analysis["confusion_matrix"]["matrix"],
                [
                    [1, 1, 0],
                    [0, 1, 1],
                    [0, 0, 2],
                ],
            )
            self.assertEqual(analysis["error_analysis"]["error_count"], 2)
            self.assertEqual(len(analysis["error_analysis"]["examples"]), 1)
            self.assertTrue(analysis["auroc"]["available"])
            self.assertEqual(analysis["auroc"]["score_source"], "probabilities")
            self.assertIn("macro_ovr", analysis["auroc"])

    def test_analyze_prediction_file_marks_auroc_unavailable_without_scores(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eval_predictions.json"
            write_predictions(
                path,
                [
                    {"id": "a", "label": 0, "predicted_label": 0},
                    {"id": "b", "label": 1, "predicted_label": 2},
                    {"id": "c", "label": 2, "predicted_label": 2},
                ],
            )

            analysis = analyze_prediction_file(path, split="eval")

            self.assertFalse(analysis["auroc"]["available"])
            self.assertIn("probabilities or logits", analysis["auroc"]["reason"])
            self.assertEqual(analysis["error_analysis"]["error_count"], 1)

    def test_analyze_prediction_file_computes_auroc_from_hf_logits(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_predictions.json"
            write_predictions(
                path,
                [
                    {"id": "a", "label": 0, "predicted_label": 0, "logits": [3, 0, 0]},
                    {"id": "b", "label": 0, "predicted_label": 1, "logits": [0, 3, 0]},
                    {"id": "c", "label": 1, "predicted_label": 1, "logits": [0, 3, 0]},
                    {"id": "d", "label": 1, "predicted_label": 2, "logits": [0, 0, 3]},
                    {"id": "e", "label": 2, "predicted_label": 2, "logits": [0, 0, 3]},
                    {"id": "f", "label": 2, "predicted_label": 2, "logits": [0, 0, 3]},
                ],
            )

            analysis = analyze_prediction_file(path, split="test")

            self.assertTrue(analysis["auroc"]["available"])
            self.assertEqual(analysis["auroc"]["score_source"], "logits_softmax")

    def test_analyze_prediction_file_marks_auroc_unavailable_when_a_class_is_missing(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_predictions.json"
            write_predictions(
                path,
                [
                    {
                        "id": "a",
                        "label": 0,
                        "predicted_label": 0,
                        "probabilities": [0.8, 0.1, 0.1],
                    },
                    {
                        "id": "b",
                        "label": 1,
                        "predicted_label": 1,
                        "probabilities": [0.1, 0.8, 0.1],
                    },
                ],
            )

            analysis = analyze_prediction_file(path, split="test")

            self.assertFalse(analysis["auroc"]["available"])
            self.assertIn("not defined for every label", analysis["auroc"]["reason"])

    def test_write_prediction_analysis_artifacts_resolves_nested_relative_paths(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            prediction_path = run_dir / "predictions" / "test_predictions.json"
            write_predictions(
                prediction_path,
                [
                    {"id": "a", "label": 0, "predicted_label": 0},
                    {"id": "b", "label": 1, "predicted_label": 1},
                    {"id": "c", "label": 2, "predicted_label": 2},
                ],
            )
            report = {
                "runs": [
                    {
                        "method": "full-ft",
                        "trial_id": "nested_path_seed42",
                        "search_stage": "final",
                        "seed": 42,
                        "summary_path": (run_dir / "result_summary.json").as_posix(),
                        "test_predictions_path": "predictions/test_predictions.json",
                    }
                ]
            }

            outputs = write_prediction_analysis_artifacts(root / "analysis", report)

            payload = json.loads(outputs["analysis_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["analyzed_prediction_files"], 1)
            self.assertEqual(payload["skipped_prediction_files"], 0)

    def test_write_prediction_analysis_artifacts_skips_malformed_prediction_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_path = root / "run" / "test_predictions.json"
            prediction_path.parent.mkdir(parents=True)
            prediction_path.write_text('{"not_predictions": []}', encoding="utf-8")
            report = {
                "runs": [
                    {
                        "method": "full-ft",
                        "trial_id": "bad_predictions_seed42",
                        "search_stage": "final",
                        "seed": 42,
                        "summary_path": (root / "run" / "result_summary.json").as_posix(),
                        "test_predictions_path": prediction_path.as_posix(),
                    }
                ]
            }

            outputs = write_prediction_analysis_artifacts(root / "analysis", report)

            payload = json.loads(outputs["analysis_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["analyzed_prediction_files"], 0)
            self.assertEqual(payload["skipped_prediction_files"], 1)
            self.assertIn("analysis failed", payload["skipped"][0]["reason"])


    def test_write_prediction_analysis_artifacts_exports_json_and_csvs(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_path = root / "run" / "test_predictions.json"
            write_predictions(
                prediction_path,
                [
                    {
                        "id": "a",
                        "label": 0,
                        "label_name": "hatespeech",
                        "predicted_label": 0,
                        "predicted_label_name": "hatespeech",
                        "probabilities": [0.8, 0.1, 0.1],
                    },
                    {
                        "id": "b",
                        "label": 1,
                        "label_name": "normal",
                        "predicted_label": 2,
                        "predicted_label_name": "offensive",
                        "probabilities": [0.1, 0.2, 0.7],
                    },
                    {
                        "id": "c",
                        "label": 2,
                        "label_name": "offensive",
                        "predicted_label": 2,
                        "predicted_label_name": "offensive",
                        "probabilities": [0.1, 0.1, 0.8],
                    },
                ],
            )
            report = {
                "runs": [
                    {
                        "method": "tfidf-logreg",
                        "trial_id": "tfidf_final_seed42",
                        "search_stage": "final",
                        "seed": 42,
                        "summary_path": (root / "run" / "result_summary.json").as_posix(),
                        "test_predictions_path": prediction_path.as_posix(),
                    }
                ]
            }

            outputs = write_prediction_analysis_artifacts(root / "analysis", report)

            self.assertTrue(outputs["analysis_json"].is_file())
            self.assertTrue(outputs["confusion_matrix_csv"].is_file())
            self.assertTrue(outputs["error_examples_csv"].is_file())
            self.assertTrue(outputs["auroc_csv"].is_file())

            payload = json.loads(outputs["analysis_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["analyzed_prediction_files"], 1)
            self.assertEqual(payload["runs"][0]["split"], "test")

            with outputs["confusion_matrix_csv"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            hate_hate = [
                row
                for row in rows
                if row["true_label_name"] == "hatespeech"
                and row["predicted_label_name"] == "hatespeech"
            ][0]
            self.assertEqual(hate_hate["count"], "1")


if __name__ == "__main__":
    unittest.main()
