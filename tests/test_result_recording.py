import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

from src.results import (
    prepare_output_dir_for_run,
    write_resolved_config,
    write_result_files,
)


class ResultRecordingTests(unittest.TestCase):
    def test_writes_config_metrics_runtime_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = {"method": "full-ft", "hyperparameters": {"learning_rate": 2e-5}}
            eval_metrics = {"eval_f1_macro": 0.5}
            test_metrics = {"test_f1_macro": 0.4}
            runtime_metrics = {"training_time_sec": 12.3, "gpu_type": "T4"}

            config_path = write_resolved_config(output_dir, config)
            paths = write_result_files(
                output_dir,
                config=config,
                eval_metrics=eval_metrics,
                test_metrics=test_metrics,
                runtime_metrics=runtime_metrics,
                prediction_paths={"test": output_dir / "test_predictions.json"},
            )

            self.assertTrue(config_path.exists())
            self.assertTrue(paths["metrics"].exists())
            self.assertTrue(paths["runtime"].exists())
            self.assertTrue(paths["summary"].exists())

            metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(metrics["eval"]["eval_f1_macro"], 0.5)
            self.assertEqual(metrics["test"]["test_f1_macro"], 0.4)
            self.assertNotIn("status", summary)
            self.assertEqual(summary["config"]["method"], "full-ft")
            self.assertEqual(summary["model_selection"], {})
            self.assertEqual(
                summary["artifacts"]["predictions"]["test"],
                (output_dir / "test_predictions.json").as_posix(),
            )

    def test_json_writer_serializes_paths_and_dataclasses(self):
        @dataclass
        class Payload:
            path: Path

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            paths = write_result_files(
                output_dir,
                config={"payload": Payload(path=Path("outputs/example"))},
                eval_metrics={"array_like": [1, 2, 3]},
                runtime_metrics={},
                model_selection={"checkpoint": Path("checkpoint-1")},
            )

            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["config"]["payload"]["path"], "outputs/example")
            self.assertEqual(summary["model_selection"]["checkpoint"], "checkpoint-1")

    def test_output_dir_rejects_existing_run_artifacts_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "result_summary.json").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already contains run artifacts"):
                prepare_output_dir_for_run(output_dir, overwrite=False)

            self.assertTrue((output_dir / "result_summary.json").exists())

    def test_output_dir_overwrite_clears_existing_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "result_summary.json").write_text("old", encoding="utf-8")
            (output_dir / "checkpoint-1").mkdir()
            (output_dir / "notes.txt").write_text("keep", encoding="utf-8")

            prepare_output_dir_for_run(output_dir, overwrite=True)

            self.assertFalse((output_dir / "result_summary.json").exists())
            self.assertFalse((output_dir / "checkpoint-1").exists())
            self.assertTrue((output_dir / "notes.txt").exists())


if __name__ == "__main__":
    unittest.main()
