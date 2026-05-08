import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

from src.experiments.results import (
    write_failure_file,
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
            )

            self.assertTrue(config_path.exists())
            self.assertTrue(paths["metrics"].exists())
            self.assertTrue(paths["runtime"].exists())
            self.assertTrue(paths["summary"].exists())

            metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(metrics["eval"]["eval_f1_macro"], 0.5)
            self.assertEqual(metrics["test"]["test_f1_macro"], 0.4)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["config"]["method"], "full-ft")
            self.assertEqual(summary["model_selection"], {})

    def test_writes_failure_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = {"trial_id": "trial001"}

            failure_path = write_failure_file(
                output_dir,
                config=config,
                error=RuntimeError("cuda out of memory"),
                runtime_metrics={"gpu_type": "T4"},
            )

            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["config"]["trial_id"], "trial001")
            self.assertEqual(failure["error"]["type"], "RuntimeError")
            self.assertIn("cuda out of memory", failure["error"]["message"])

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


if __name__ == "__main__":
    unittest.main()
