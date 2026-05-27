import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.aggregate_results import parse_args as parse_aggregate_cli_args
from src.experiments.aggregate_results import (
    aggregate_records,
    build_aggregate_report,
    discover_summary_files,
    flatten_summary_record,
    write_aggregate_report,
    write_pareto_csvs,
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
    config_hash: str | None = "abc123",
    search_method: str | None = "random_search",
    search_space_name: str | None = "full_ft",
    hpo_seed: int | None = 42,
    trainable_params: int | None = None,
    total_params: int | None = None,
) -> dict:
    return {
        "status": "completed",
        "config": {
            "method": "full-ft",
            "search_stage": "final" if test_f1 is not None else "tuning",
            "trial_id": trial_id,
            "config_hash": config_hash,
            "search_method": search_method,
            "search_space_name": search_space_name,
            "seed": seed,
            "hpo_seed": hpo_seed,
            "hpo_trial_cap": 6,
            "hpo_time_cap_gpu_hours": 2.0,
            "output_dir": f"outputs/{trial_id}",
            "trainable_params": trainable_params,
            "total_params": total_params,
            "hyperparameters": {
                "learning_rate": 2e-5,
                "warmup_ratio": 0.06,
                "batch_size": 8,
            },
        },
        "metrics": {
            "eval": {
                "eval_f1_macro": eval_f1,
                "eval_accuracy": 0.62,
                "eval_precision_macro": 0.63,
                "eval_recall_macro": 0.64,
                "eval_loss": 0.9,
            },
            "test": {
                "test_f1_macro": test_f1,
                "test_accuracy": 0.59,
                "test_precision_macro": 0.60,
                "test_recall_macro": 0.61,
                "test_f1_hatespeech": 0.50,
                "test_f1_offensive": 0.65,
                "test_f1_normal": 0.62,
            } if test_f1 is not None else None,
        },
        "runtime": {
            "training_time_sec": 10.0 + seed,
            "training_time_hours": (10.0 + seed) / 3600,
            "gpu_hours": (10.0 + seed) / 3600,
            "eval_runtime": 1.25,
            "peak_memory_reserved_mb": 1234.0,
            "gpu_type": "T4",
        },
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
    def test_aggregate_cli_defaults_group_by_config_hash(self):
        with patch.object(sys, "argv", ["aggregate_results.py", "outputs/final"]):
            args = parse_aggregate_cli_args()

        self.assertEqual(args.group_by, ["method", "search_stage", "config_hash"])

    def test_aggregate_cli_accepts_prediction_analysis_options(self):
        with patch.object(
            sys,
            "argv",
            [
                "aggregate_results.py",
                "outputs/final",
                "--write_prediction_analysis",
                "--prediction_analysis_dir",
                "outputs/diagnostics",
                "--max_error_examples",
                "12",
            ],
        ):
            args = parse_aggregate_cli_args()

        self.assertTrue(args.write_prediction_analysis)
        self.assertEqual(args.prediction_analysis_dir, "outputs/diagnostics")
        self.assertEqual(args.max_error_examples, 12)

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
        self.assertEqual(record["search_method"], "random_search")
        self.assertEqual(record["search_space_name"], "full_ft")
        self.assertEqual(record["final_config_id"], "abc123")
        self.assertEqual(record["eval_f1_macro"], 0.61)
        self.assertEqual(record["test_f1_macro"], 0.58)
        self.assertEqual(record["test_f1_hatespeech"], 0.50)
        self.assertEqual(record["trainable_params"], 10)
        self.assertEqual(record["total_params"], 100)
        self.assertEqual(record["trainable_pct"], 10.0)
        self.assertEqual(record["best_model_checkpoint"], "outputs/seed42/checkpoint-1")
        self.assertEqual(record["metric_for_best_model"], "eval_f1_macro")
        self.assertEqual(record["best_metric_key"], "eval_f1_macro")
        self.assertEqual(record["best_metric"], 0.61)
        self.assertEqual(record["best_val_macro_f1"], 0.61)
        self.assertEqual(record["eval_runtime"], 1.25)
        self.assertEqual(record["best_epoch"], 2.0)
        self.assertEqual(record["best_step"], 100)
        self.assertIn('"learning_rate": 2e-05', record["selected_hyperparams_json"])
        self.assertEqual(
            record["eval_predictions_path"],
            "outputs/seed42/eval_predictions.json",
        )
        self.assertEqual(
            record["test_predictions_path"],
            "outputs/seed42/test_predictions.json",
        )
        self.assertEqual(record["summary_path"], "outputs/seed42/result_summary.json")

    def test_flatten_summary_record_handles_tfidf_summary(self):
        payload = {
            "status": "completed",
            "config": {
                "method": "tfidf-logreg",
                "search_stage": "final",
                "trial_id": "tfidf_seed42",
                "config_hash": "tfidf123",
                "search_method": "random_search",
                "search_space_name": "tfidf_logreg",
                "seed": 42,
                "hpo_trial_cap": 12,
                "hpo_time_cap_gpu_hours": 0.5,
                "model_name": "tfidf-logreg",
                "output_dir": "outputs/tfidf_seed42",
                "trainable_params": 150000,
                "total_params": 150000,
                "hyperparameters": {"ngram_range": [1, 2], "C": 1.0},
            },
            "metrics": {
                "eval": {"eval_f1_macro": 0.62, "eval_accuracy": 0.64},
                "test": {"test_f1_macro": 0.60, "test_accuracy": 0.61},
            },
            "runtime": {"training_time_sec": 3.5, "gpu_type": "cpu"},
            "model_selection": {
                "metric_for_best_model": "eval_f1_macro",
                "best_metric_key": "eval_f1_macro",
                "best_metric": 0.62,
                "best_epoch": None,
                "best_model_checkpoint": "model.joblib",
            },
            "artifacts": {
                "predictions": {
                    "eval": "outputs/tfidf_seed42/eval_predictions.json",
                    "test": "outputs/tfidf_seed42/test_predictions.json",
                }
            },
        }

        record = flatten_summary_record(
            payload,
            Path("outputs/tfidf_seed42/result_summary.json"),
        )

        self.assertEqual(record["method"], "tfidf-logreg")
        self.assertEqual(record["eval_f1_macro"], 0.62)
        self.assertEqual(record["test_f1_macro"], 0.60)
        self.assertEqual(record["training_time_sec"], 3.5)
        self.assertEqual(record["trainable_pct"], 100.0)
        self.assertEqual(record["search_space_name"], "tfidf_logreg")
        self.assertEqual(record["best_model_checkpoint"], "model.joblib")
        self.assertEqual(
            record["test_predictions_path"],
            "outputs/tfidf_seed42/test_predictions.json",
        )

    def test_flatten_summary_record_preserves_two_stage_metrics(self):
        payload = completed_summary(
            trial_id="lpft_seed42",
            seed=42,
            eval_f1=0.61,
            test_f1=0.58,
        )
        payload["config"]["method"] = "lp-ft"
        payload["metrics"]["stage1"] = {
            "stage1_eval_f1_macro": 0.57,
            "stage1_eval_loss": 0.8,
        }
        payload["model_selection"].update(
            {
                "stage1_best_metric": 0.57,
                "stage1_best_epoch": 3,
                "stage1_best_step": 300,
                "stage1_best_model_checkpoint": "outputs/lpft/stage1/checkpoint-300",
                "stage2_best_metric": 0.61,
                "stage2_best_epoch": 2,
                "stage2_best_step": 200,
            }
        )

        record = flatten_summary_record(
            payload,
            Path("outputs/lpft_seed42/result_summary.json"),
        )

        self.assertEqual(record["stage1_eval_f1_macro"], 0.57)
        self.assertEqual(record["stage1_eval_loss"], 0.8)
        self.assertEqual(record["stage1_best_metric"], 0.57)
        self.assertEqual(record["stage1_best_epoch"], 3)
        self.assertEqual(record["stage1_best_step"], 300)
        self.assertEqual(
            record["stage1_best_model_checkpoint"],
            "outputs/lpft/stage1/checkpoint-300",
        )
        self.assertEqual(record["stage2_best_metric"], 0.61)
        self.assertEqual(record["stage2_best_epoch"], 2)
        self.assertEqual(record["stage2_best_step"], 200)

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

            self.assertEqual(report["hpo_total_training_time_sec"], 105.0)
            self.assertAlmostEqual(
                report["hpo_total_training_time_hours"],
                105.0 / 3600,
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

    def test_write_pareto_csvs_exports_hpo_final_and_method_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trial1_payload = completed_summary(
                trial_id="trial1",
                seed=42,
                eval_f1=0.60,
                trainable_params=100,
                total_params=1000,
            )
            trial1_payload["metrics"]["stage1"] = {
                "stage1_eval_f1_macro": 0.55,
                "stage1_eval_loss": 0.9,
            }
            trial1_payload["model_selection"]["stage1_best_epoch"] = 2
            write_json(root / "trial1" / "result_summary.json", trial1_payload)
            write_json(
                root / "trial2" / "failure_summary.json",
                {
                    "status": "failed",
                    "config": {
                        "method": "full-ft",
                        "search_stage": "tuning",
                        "trial_id": "trial2",
                        "config_hash": "failed123",
                        "search_method": "random_search",
                        "search_space_name": "full_ft",
                        "seed": 42,
                        "hpo_seed": 42,
                        "hpo_trial_cap": 6,
                        "hpo_time_cap_gpu_hours": 2.0,
                        "hyperparameters": {"learning_rate": 1e-5},
                    },
                    "runtime": {"training_time_sec": 3.0, "gpu_type": "T4"},
                    "error": {"type": "RuntimeError", "message": "CUDA out of memory"},
                },
            )
            seed42_payload = completed_summary(
                trial_id="seed42",
                seed=42,
                eval_f1=0.62,
                test_f1=0.58,
                trainable_params=100,
                total_params=1000,
            )
            seed42_payload["metrics"]["stage1"] = {
                "stage1_eval_f1_macro": 0.56,
                "stage1_eval_loss": 0.85,
            }
            seed42_payload["model_selection"].update(
                {
                    "stage1_best_metric": 0.56,
                    "stage1_best_epoch": 3,
                    "stage1_best_step": 300,
                    "stage2_best_metric": 0.62,
                    "stage2_best_epoch": 2,
                    "stage2_best_step": 200,
                }
            )
            write_json(root / "seed42" / "result_summary.json", seed42_payload)
            write_json(root / "seed43" / "result_summary.json", completed_summary(
                trial_id="seed43",
                seed=43,
                eval_f1=0.64,
                test_f1=0.62,
                trainable_params=100,
                total_params=1000,
            ))
            write_json(
                root / "seed44" / "failure_summary.json",
                {
                    "status": "failed",
                    "config": {
                        "method": "full-ft",
                        "search_stage": "final",
                        "trial_id": "seed44",
                        "config_hash": "abc123",
                        "search_method": "random_search",
                        "search_space_name": "full_ft",
                        "seed": 44,
                        "hpo_seed": 42,
                        "hpo_trial_cap": 6,
                        "hpo_time_cap_gpu_hours": 2.0,
                        "hyperparameters": {"learning_rate": 2e-5},
                    },
                    "runtime": {"training_time_sec": 4.0, "gpu_type": "T4"},
                    "error": {"type": "RuntimeError", "message": "boom"},
                },
            )

            report = build_aggregate_report([root], group_by=["method", "search_stage"])
            outputs = write_pareto_csvs(root / "pareto", report)

            self.assertEqual(
                {path.name for path in outputs},
                {"hpo_runs.csv", "final_runs.csv", "method_summary.csv"},
            )
            with (root / "pareto" / "hpo_runs.csv").open(newline="", encoding="utf-8") as handle:
                hpo_rows = list(csv.DictReader(handle))
            self.assertEqual(len(hpo_rows), 2)
            self.assertEqual(hpo_rows[0]["search_method"], "random_search")
            self.assertEqual(hpo_rows[0]["search_space"], "full_ft")
            self.assertEqual(hpo_rows[0]["eval_time_s"], "1.25")
            self.assertEqual(hpo_rows[0]["stage1_eval_f1_macro"], "0.55")
            self.assertEqual(hpo_rows[0]["stage1_best_epoch"], "2")
            self.assertEqual(hpo_rows[1]["status"], "failed")
            self.assertEqual(hpo_rows[1]["failed_oom"], "True")

            with (root / "pareto" / "final_runs.csv").open(newline="", encoding="utf-8") as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual(len(final_rows), 3)
            self.assertEqual(final_rows[0]["method"], "full-ft")
            self.assertEqual(final_rows[0]["seed"], "42")
            self.assertEqual(final_rows[0]["status"], "completed")
            self.assertEqual(final_rows[0]["test_macro_f1"], "0.58")
            self.assertEqual(final_rows[0]["peak_gpu_memory_mb"], "1234.0")
            self.assertEqual(final_rows[0]["stage1_eval_f1_macro"], "0.56")
            self.assertEqual(final_rows[0]["stage1_best_epoch"], "3")
            self.assertEqual(final_rows[0]["stage2_best_epoch"], "2")
            self.assertIn("learning_rate", final_rows[0]["selected_hyperparams_json"])
            self.assertEqual(final_rows[2]["status"], "failed")
            self.assertEqual(final_rows[2]["error_type"], "RuntimeError")

            with (root / "pareto" / "method_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 1)
            row = summary_rows[0]
            self.assertEqual(row["method"], "full-ft")
            self.assertEqual(row["completed_hpo_trials"], "1")
            self.assertEqual(row["failed_hpo_trials"], "1")
            self.assertEqual(row["failed_oom_trials"], "1")
            self.assertEqual(row["hpo_trial_cap"], "6")
            self.assertEqual(row["hpo_time_cap_gpu_hours"], "2.0")
            self.assertEqual(row["hpo_gpu_type"], "T4")
            self.assertEqual(row["final_gpu_type"], "T4")
            self.assertEqual(row["best_val_macro_f1"], "0.6")
            self.assertIn("trial1", row["selected_hpo_trial_id"])
            self.assertEqual(row["test_macro_f1_mean"], "0.6")
            self.assertGreater(float(row["final_train_time_mean_s"]), 0.0)
            self.assertEqual(row["trainable_params"], "100")
            self.assertEqual(row["total_params"], "1000")
            self.assertEqual(row["final_seed_count"], "3")
            self.assertEqual(row["completed_final_seeds"], "2")
            self.assertEqual(row["failed_final_seeds"], "1")
            self.assertIn(row["pareto_status"], {"pareto_optimal", "dominated"})

    def test_method_summary_matches_best_val_to_selected_config_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(root / "trial_cfg_a" / "result_summary.json", completed_summary(
                trial_id="trial_cfg_a",
                seed=42,
                eval_f1=0.70,
                config_hash="cfgA",
                trainable_params=100,
                total_params=1000,
            ))
            write_json(root / "trial_cfg_b" / "result_summary.json", completed_summary(
                trial_id="trial_cfg_b",
                seed=42,
                eval_f1=0.40,
                config_hash="cfgB",
                trainable_params=100,
                total_params=1000,
            ))
            write_json(root / "final_cfg_a" / "result_summary.json", completed_summary(
                trial_id="final_cfg_a",
                seed=42,
                eval_f1=0.68,
                test_f1=0.66,
                config_hash="cfgA",
                trainable_params=100,
                total_params=1000,
            ))
            write_json(root / "final_cfg_b" / "result_summary.json", completed_summary(
                trial_id="final_cfg_b",
                seed=42,
                eval_f1=0.38,
                test_f1=0.36,
                config_hash="cfgB",
                trainable_params=100,
                total_params=1000,
            ))

            report = build_aggregate_report([root], group_by=["method", "config_hash"])
            write_pareto_csvs(root / "pareto", report)

            with (root / "pareto" / "method_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = {
                    row["final_config_id"]: row
                    for row in csv.DictReader(handle)
                }

            self.assertEqual(rows["cfgA"]["best_val_macro_f1"], "0.7")
            self.assertEqual(rows["cfgB"]["best_val_macro_f1"], "0.4")
            self.assertIn("trial_cfg_a", rows["cfgA"]["selected_hpo_trial_id"])
            self.assertIn("trial_cfg_b", rows["cfgB"]["selected_hpo_trial_id"])
            self.assertEqual(rows["cfgA"]["completed_hpo_trials"], "2")
            self.assertEqual(rows["cfgB"]["completed_hpo_trials"], "2")
            self.assertEqual(rows["cfgB"]["pareto_status"], "dominated")
            self.assertEqual(rows["cfgB"]["dominated_by"], "full-ft:cfgA")

    def test_method_summary_leaves_best_val_blank_when_final_config_has_no_hpo_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(root / "trial_cfg_a" / "result_summary.json", completed_summary(
                trial_id="trial_cfg_a",
                seed=42,
                eval_f1=0.70,
                config_hash="cfgA",
                trainable_params=100,
                total_params=1000,
            ))
            write_json(root / "final_cfg_missing" / "result_summary.json", completed_summary(
                trial_id="final_cfg_missing",
                seed=42,
                eval_f1=0.55,
                test_f1=0.53,
                config_hash="cfgMissing",
                trainable_params=100,
                total_params=1000,
            ))

            report = build_aggregate_report([root], group_by=["method", "config_hash"])
            write_pareto_csvs(root / "pareto", report)

            with (root / "pareto" / "method_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = {
                    row["final_config_id"]: row
                    for row in csv.DictReader(handle)
                }

            self.assertEqual(rows["cfgMissing"]["best_val_macro_f1"], "")
            self.assertEqual(rows["cfgMissing"]["selected_hpo_trial_id"], "")
            self.assertEqual(rows["cfgMissing"]["completed_hpo_trials"], "1")

    def test_method_summary_filters_hpo_budget_by_seed_and_random_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(root / "trial_seed42" / "result_summary.json", completed_summary(
                trial_id="trial_seed42",
                seed=42,
                eval_f1=0.70,
                config_hash="cfgA",
                hpo_seed=42,
            ))
            write_json(root / "trial_seed99" / "result_summary.json", completed_summary(
                trial_id="trial_seed99",
                seed=42,
                eval_f1=0.90,
                config_hash="cfgA",
                hpo_seed=99,
            ))
            write_json(root / "trial_catalog" / "result_summary.json", completed_summary(
                trial_id="trial_catalog",
                seed=42,
                eval_f1=0.95,
                config_hash="cfgA",
                search_method="catalog_run",
                hpo_seed=42,
            ))
            write_json(root / "final_cfg_a" / "result_summary.json", completed_summary(
                trial_id="final_cfg_a",
                seed=42,
                eval_f1=0.68,
                test_f1=0.66,
                config_hash="cfgA",
                hpo_seed=42,
            ))

            report = build_aggregate_report([root], group_by=["method", "config_hash"])
            write_pareto_csvs(root / "pareto", report)

            with (root / "pareto" / "hpo_runs.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                hpo_rows = list(csv.DictReader(handle))
            with (root / "pareto" / "method_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                {row["trial_id"] for row in hpo_rows},
                {"trial_seed42", "trial_seed99"},
            )
            self.assertEqual(rows[0]["completed_hpo_trials"], "1")
            self.assertEqual(rows[0]["actual_hpo_time_s"], "52.0")
            self.assertEqual(rows[0]["hpo_seed"], "42")
            self.assertEqual(rows[0]["best_val_macro_f1"], "0.7")
            self.assertIn("trial_seed42", rows[0]["selected_hpo_trial_id"])

    def test_tfidf_search_space_alias_links_hpo_to_final_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hpo_payload = completed_summary(
                trial_id="tfidf_trial",
                seed=42,
                eval_f1=0.70,
                config_hash="tfidfCfg",
                search_space_name="tfidf_lr",
            )
            hpo_payload["config"]["method"] = "tfidf-logreg"
            final_payload = completed_summary(
                trial_id="tfidf_final",
                seed=42,
                eval_f1=0.68,
                test_f1=0.66,
                config_hash="tfidfCfg",
                search_space_name="tfidf_logreg",
            )
            final_payload["config"]["method"] = "tfidf-logreg"
            write_json(root / "tfidf_trial" / "result_summary.json", hpo_payload)
            write_json(root / "tfidf_final" / "result_summary.json", final_payload)

            report = build_aggregate_report([root], group_by=["method", "config_hash"])
            write_pareto_csvs(root / "pareto", report)

            with (root / "pareto" / "method_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["search_space"], "tfidf_logreg")
            self.assertEqual(rows[0]["best_val_macro_f1"], "0.7")
            self.assertIn("tfidf_trial", rows[0]["selected_hpo_trial_id"])

    def test_method_summary_marks_failed_only_final_config_as_insufficient_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "failed_final" / "failure_summary.json",
                {
                    "status": "failed",
                    "config": {
                        "method": "full-ft",
                        "search_stage": "final",
                        "trial_id": "failed_final",
                        "config_hash": "cfgFailed",
                        "seed": 42,
                        "hyperparameters": {"learning_rate": 2e-5},
                    },
                    "runtime": {"training_time_sec": 4.0, "gpu_type": "T4"},
                    "error": {"type": "RuntimeError", "message": "boom"},
                },
            )

            report = build_aggregate_report([root], group_by=["method", "config_hash"])
            write_pareto_csvs(root / "pareto", report)

            with (root / "pareto" / "method_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["final_config_id"], "cfgFailed")
            self.assertEqual(rows[0]["test_macro_f1_mean"], "")
            self.assertEqual(rows[0]["completed_final_seeds"], "0")
            self.assertEqual(rows[0]["failed_final_seeds"], "1")
            self.assertEqual(rows[0]["pareto_status"], "insufficient_data")
            self.assertEqual(rows[0]["dominated_by"], "")

    def test_missing_final_config_hash_does_not_merge_or_enter_pareto(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(root / "final_a" / "result_summary.json", completed_summary(
                trial_id="final_a",
                seed=42,
                eval_f1=0.61,
                test_f1=0.58,
                config_hash=None,
            ))
            write_json(root / "final_b" / "result_summary.json", completed_summary(
                trial_id="final_b",
                seed=42,
                eval_f1=0.72,
                test_f1=0.69,
                config_hash=None,
            ))

            report = build_aggregate_report([root], group_by=["method", "config_hash"])
            write_pareto_csvs(root / "pareto", report)

            with (root / "pareto" / "method_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["missing_config_hash"] == "True" for row in rows))
            self.assertTrue(
                all(
                    row["final_config_id"].startswith("missing_config_hash:")
                    for row in rows
                )
            )
            self.assertEqual(
                {row["pareto_status"] for row in rows},
                {"insufficient_data"},
            )


if __name__ == "__main__":
    unittest.main()
