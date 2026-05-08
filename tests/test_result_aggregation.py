import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.aggregate_results import (
    aggregate_records,
    build_aggregate_report,
    discover_summary_files,
    flatten_summary_record,
    write_aggregate_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def completed_summary(
    *,
    trial_id: str,
    seed: int,
    eval_f1: float,
    test_f1: float | None = None,
) -> dict:
    return {
        "status": "completed",
        "config": {
            "method": "full-ft",
            "search_stage": "final" if test_f1 is not None else "tuning",
            "trial_id": trial_id,
            "config_hash": "abc123",
            "seed": seed,
            "hpo_seed": 42,
            "output_dir": f"outputs/{trial_id}",
        },
        "metrics": {
            "eval": {"eval_f1_macro": eval_f1, "eval_loss": 0.9},
            "test": {"test_f1_macro": test_f1} if test_f1 is not None else None,
        },
        "runtime": {"training_time_sec": 10.0 + seed, "gpu_type": "T4"},
        "model_selection": {
            "best_model_checkpoint": f"outputs/{trial_id}/checkpoint-1",
        },
    }


class ResultAggregationTests(unittest.TestCase):
    def test_discovers_completed_and_failed_summary_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(root / "run1" / "result_summary.json", completed_summary(
                trial_id="run1",
                seed=42,
                eval_f1=0.5,
            ))
            write_json(
                root / "run2" / "failure_summary.json",
                {
                    "status": "failed",
                    "config": {"method": "full-ft", "trial_id": "run2"},
                    "runtime": {"failure_phase": "setup"},
                    "error": {"type": "RuntimeError", "message": "boom"},
                },
            )

            files = discover_summary_files([root])

            self.assertEqual(
                {path.name for path in files},
                {"result_summary.json", "failure_summary.json"},
            )

    def test_flatten_summary_record_extracts_tracking_fields_and_metrics(self):
        payload = completed_summary(
            trial_id="seed42",
            seed=42,
            eval_f1=0.61,
            test_f1=0.58,
        )

        record = flatten_summary_record(payload, Path("outputs/seed42/result_summary.json"))

        self.assertEqual(record["method"], "full-ft")
        self.assertEqual(record["search_stage"], "final")
        self.assertEqual(record["trial_id"], "seed42")
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["eval_f1_macro"], 0.61)
        self.assertEqual(record["test_f1_macro"], 0.58)
        self.assertEqual(record["best_model_checkpoint"], "outputs/seed42/checkpoint-1")
        self.assertEqual(record["summary_path"], "outputs/seed42/result_summary.json")

    def test_aggregate_records_computes_mean_std_and_failure_counts(self):
        records = [
            flatten_summary_record(completed_summary(
                trial_id="seed42",
                seed=42,
                eval_f1=0.60,
                test_f1=0.55,
            ), Path("seed42/result_summary.json")),
            flatten_summary_record(completed_summary(
                trial_id="seed43",
                seed=43,
                eval_f1=0.70,
                test_f1=0.65,
            ), Path("seed43/result_summary.json")),
            flatten_summary_record(
                {
                    "status": "failed",
                    "config": {
                        "method": "full-ft",
                        "search_stage": "final",
                        "trial_id": "seed44",
                        "seed": 44,
                    },
                    "runtime": {},
                    "error": {"type": "RuntimeError", "message": "oom"},
                },
                Path("seed44/failure_summary.json"),
            ),
        ]

        groups = aggregate_records(
            records,
            group_by=["method", "search_stage"],
            metrics=["eval_f1_macro", "test_f1_macro"],
        )

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["runs"], 3)
        self.assertEqual(group["completed"], 2)
        self.assertEqual(group["failed"], 1)
        self.assertAlmostEqual(group["metrics"]["eval_f1_macro"]["mean"], 0.65)
        self.assertAlmostEqual(group["metrics"]["eval_f1_macro"]["std"], 0.0707106781)
        self.assertAlmostEqual(group["metrics"]["test_f1_macro"]["mean"], 0.60)

    def test_build_and_write_aggregate_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(root / "seed42" / "result_summary.json", completed_summary(
                trial_id="seed42",
                seed=42,
                eval_f1=0.60,
                test_f1=0.55,
            ))
            write_json(root / "seed43" / "result_summary.json", completed_summary(
                trial_id="seed43",
                seed=43,
                eval_f1=0.70,
                test_f1=0.65,
            ))

            report = build_aggregate_report(
                [root],
                group_by=["method", "search_stage"],
                metrics=["eval_f1_macro", "test_f1_macro"],
            )
            output_path = write_aggregate_report(
                root / "aggregate_summary.json",
                report,
            )

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["total_runs"], 2)
            self.assertEqual(saved["groups"][0]["completed"], 2)
            self.assertEqual(saved["groups"][0]["metrics"]["test_f1_macro"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
