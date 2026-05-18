import json
import os
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
    trainable_params: int | None = None,
    total_params: int | None = None,
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
            "hpo_trial_cap": 6,
            "hpo_time_cap_gpu_hours": 2.0,
            "output_dir": f"outputs/{trial_id}",
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
        "metrics": {
            "eval": {"eval_f1_macro": eval_f1, "eval_loss": 0.9},
            "test": {"test_f1_macro": test_f1} if test_f1 is not None else None,
        },
        "runtime": {"training_time_sec": 10.0 + seed, "gpu_type": "T4"},
        "model_selection": {
            "best_model_checkpoint": f"outputs/{trial_id}/checkpoint-1",
            "metric_for_best_model": "eval_f1_macro",
            "best_metric_key": "eval_f1_macro",
            "best_metric": eval_f1,
            "best_epoch": 2.0,
            "best_step": 100,
        },
        "artifacts": {
            "predictions": {
                "eval": f"outputs/{trial_id}/eval_predictions.json",
                "test": f"outputs/{trial_id}/test_predictions.json",
            }
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

    def test_discovery_keeps_only_newest_summary_per_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "run1" / "result_summary.json"
            failure_path = root / "run1" / "failure_summary.json"
            write_json(result_path, completed_summary(
                trial_id="run1",
                seed=42,
                eval_f1=0.5,
            ))
            write_json(
                failure_path,
                {
                    "status": "failed",
                    "config": {"method": "full-ft", "trial_id": "run1"},
                    "runtime": {},
                    "error": {"type": "RuntimeError", "message": "old failure"},
                },
            )
            os.utime(failure_path, (100, 100))
            os.utime(result_path, (200, 200))

            files = discover_summary_files([root])

            self.assertEqual(files, [result_path])

    def test_flatten_summary_record_extracts_tracking_fields_and_metrics(self):
        payload = completed_summary(
            trial_id="seed42",
            seed=42,
            eval_f1=0.61,
            test_f1=0.58,
            trainable_params=10,
            total_params=100,
        )

        record = flatten_summary_record(payload, Path("outputs/seed42/result_summary.json"))

        self.assertEqual(record["method"], "full-ft")
        self.assertEqual(record["search_stage"], "final")
        self.assertEqual(record["trial_id"], "seed42")
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["hpo_trial_cap"], 6)
        self.assertEqual(record["hpo_time_cap_gpu_hours"], 2.0)
        self.assertEqual(record["eval_f1_macro"], 0.61)
        self.assertEqual(record["test_f1_macro"], 0.58)
        self.assertEqual(record["trainable_params"], 10)
        self.assertEqual(record["total_params"], 100)
        self.assertEqual(record["trainable_pct"], 10.0)
        self.assertEqual(record["best_model_checkpoint"], "outputs/seed42/checkpoint-1")
        self.assertEqual(record["metric_for_best_model"], "eval_f1_macro")
        self.assertEqual(record["best_metric_key"], "eval_f1_macro")
        self.assertEqual(record["best_metric"], 0.61)
        self.assertEqual(record["best_epoch"], 2.0)
        self.assertEqual(record["best_step"], 100)
        self.assertEqual(
            record["eval_predictions_path"],
            "outputs/seed42/eval_predictions.json",
        )
        self.assertEqual(
            record["test_predictions_path"],
            "outputs/seed42/test_predictions.json",
        )
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
                    "error": {"type": "RuntimeError", "message": "CUDA out of memory"},
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
        self.assertEqual(group["failed_oom"], 1)
        self.assertAlmostEqual(group["metrics"]["eval_f1_macro"]["mean"], 0.65)
        self.assertAlmostEqual(group["metrics"]["eval_f1_macro"]["std"], 0.0707106781)
        self.assertAlmostEqual(group["metrics"]["test_f1_macro"]["mean"], 0.60)

    def test_aggregate_report_sums_hpo_time_and_summarizes_best_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(root / "trial1" / "result_summary.json", completed_summary(
                trial_id="trial1",
                seed=42,
                eval_f1=0.60,
            ))
            write_json(root / "trial2" / "result_summary.json", completed_summary(
                trial_id="trial2",
                seed=43,
                eval_f1=0.70,
            ))
            write_json(
                root / "trial3" / "failure_summary.json",
                {
                    "status": "failed",
                    "config": {
                        "method": "full-ft",
                        "search_stage": "tuning",
                        "trial_id": "trial3",
                        "seed": 42,
                    },
                    "runtime": {"training_time_sec": 5.0},
                    "error": {"type": "RuntimeError", "message": "boom"},
                },
            )

            report = build_aggregate_report([root], group_by=["method", "search_stage"])

            self.assertEqual(report["hpo_total_training_time_sec"], 110.0)
            self.assertAlmostEqual(
                report["hpo_total_training_time_hours"],
                110.0 / 3600,
            )
            group = report["groups"][0]
            self.assertEqual(group["total_training_time_sec"], 110.0)
            self.assertEqual(group["metrics"]["best_epoch"]["count"], 2)
            self.assertEqual(group["metrics"]["best_epoch"]["mean"], 2.0)
            self.assertEqual(group["metrics"]["best_epoch"]["min"], 2.0)
            self.assertEqual(group["metrics"]["best_epoch"]["max"], 2.0)

    def test_aggregate_records_summarizes_parameter_efficiency_fields(self):
        records = [
            flatten_summary_record(completed_summary(
                trial_id="seed42",
                seed=42,
                eval_f1=0.60,
                trainable_params=10,
                total_params=100,
            ), Path("seed42/result_summary.json")),
            flatten_summary_record(completed_summary(
                trial_id="seed43",
                seed=43,
                eval_f1=0.70,
                trainable_params=20,
                total_params=100,
            ), Path("seed43/result_summary.json")),
        ]

        groups = aggregate_records(
            records,
            group_by=["method"],
            metrics=["trainable_params", "total_params", "trainable_pct"],
        )

        self.assertEqual(groups[0]["metrics"]["trainable_params"]["mean"], 15.0)
        self.assertEqual(groups[0]["metrics"]["total_params"]["mean"], 100.0)
        self.assertEqual(groups[0]["metrics"]["trainable_pct"]["mean"], 15.0)

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
