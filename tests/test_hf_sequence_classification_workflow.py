import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from src.methods.hf_sequence_classification import (
    HfClassificationRun,
    HfLibraries,
    build_runtime_metrics,
    initialize_hf_run,
    save_final_predictions,
    write_success_outputs,
)
from src.methods.transformer_data import TokenizedSplit
from src.utils.wandb_config import WandbSettings


class FakeTrainer:
    def __init__(self):
        self.predicted_datasets = []

    def predict(self, dataset, metric_key_prefix=None):
        self.predicted_datasets.append((dataset, metric_key_prefix))
        labels = [record["labels"] for record in dataset]
        return argparse.Namespace(
            predictions=[[0.8, 0.1, 0.1] for _ in labels],
            label_ids=labels,
        )


def build_context(output_dir, *, search_stage="final", run_test=True):
    eval_split = TokenizedSplit(
        dataset=[{"labels": 0}],
        records=[{"id": "eval-1", "text": "eval", "label": 0, "label_name": "hatespeech"}],
        raw_size=1,
        preprocessed_size=1,
        dropped_no_majority_count=0,
    )
    test_split = TokenizedSplit(
        dataset=[{"labels": 1}],
        records=[{"id": "test-1", "text": "test", "label": 1, "label_name": "normal"}],
        raw_size=1,
        preprocessed_size=1,
        dropped_no_majority_count=0,
    )
    return HfClassificationRun(
        args=argparse.Namespace(
            output_dir=str(output_dir),
            search_stage=search_stage,
            run_test=run_test,
        ),
        precision_policy={"mixed_precision": "none", "fp16": False, "bf16": False},
        gpu_type="cpu",
        libraries=HfLibraries(
            training_args_cls=object,
            early_stopping_callback_cls=object,
        ),
        model=object(),
        tokenizer=object(),
        data_collator=object(),
        trainer_cls=object,
        class_weights=None,
        train_split="train",
        eval_split="validation",
        test_split="test",
        train_split_data=eval_split,
        eval_split_data=eval_split,
        test_split_data=test_split if run_test else None,
        id2label={0: "hatespeech", 1: "normal", 2: "offensive"},
        label2id={"hatespeech": 0, "normal": 1, "offensive": 2},
        num_labels=3,
    )


class HfSequenceClassificationWorkflowTests(unittest.TestCase):
    def test_initialize_hf_run_clears_managed_outputs_when_overwrite_enabled(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stale_result = output_dir / "result_summary.json"
            stale_result.write_text("{}", encoding="utf-8")
            note = output_dir / "notes.txt"
            note.write_text("keep", encoding="utf-8")
            args = argparse.Namespace(
                output_dir=str(output_dir),
                overwrite_output_dir=True,
                mixed_precision="none",
                fp16=False,
            )

            setup = initialize_hf_run(
                args,
                build_setup_failure_config_fn=lambda args, **kwargs: {
                    "output_dir": args.output_dir,
                    "gpu_type": kwargs["gpu_type"],
                },
                resolve_wandb_settings_fn=lambda _: WandbSettings(enabled=False),
            )

            self.assertFalse(stale_result.exists())
            self.assertTrue(note.exists())
            self.assertEqual(setup.experiment_config["output_dir"], str(output_dir))

    def test_save_final_predictions_writes_eval_and_test_files(self):
        with TemporaryDirectory() as tmp:
            context = build_context(Path(tmp), search_stage="final", run_test=True)
            trainer = FakeTrainer()

            paths = save_final_predictions(context, trainer)

            self.assertEqual(set(paths), {"eval", "test"})
            self.assertTrue(paths["eval"].is_file())
            self.assertTrue(paths["test"].is_file())
            self.assertEqual(
                trainer.predicted_datasets,
                [
                    (context.eval_dataset, "eval_predictions"),
                    (context.test_dataset, "test_predictions"),
                ],
            )
            test_payload = json.loads(paths["test"].read_text(encoding="utf-8"))
            self.assertEqual(test_payload["predictions"][0]["id"], "test-1")

    def test_save_final_predictions_skips_non_final_runs(self):
        with TemporaryDirectory() as tmp:
            context = build_context(Path(tmp), search_stage="tuning", run_test=False)
            trainer = FakeTrainer()

            paths = save_final_predictions(context, trainer)

            self.assertEqual(paths, {})
            self.assertEqual(trainer.predicted_datasets, [])

    def test_write_success_outputs_logs_wandb_and_writes_result_files(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            args = argparse.Namespace(output_dir=str(output_dir))
            wandb_run = Mock()
            runtime_metrics = {"training_time_sec": 1.25, "status": "completed"}
            model_selection = {"best_metric": 0.5}

            paths = write_success_outputs(
                args,
                config={"method": "unit-test"},
                eval_metrics={"eval_f1_macro": 0.5},
                test_metrics=None,
                runtime_metrics=runtime_metrics,
                model_selection=model_selection,
                prediction_paths={},
                wandb_run=wandb_run,
                extra_metrics={"stage1": {"stage1_eval_f1_macro": 0.4}},
            )

            self.assertTrue(paths["summary"].is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            metrics_payload = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics_payload["stage1"]["stage1_eval_f1_macro"], 0.4)
            wandb_run.log.assert_any_call(runtime_metrics)
            wandb_run.log.assert_any_call({"model_selection": model_selection})

    def test_runtime_metrics_records_shared_switches_and_memory(self):
        args = argparse.Namespace(gradient_checkpointing=True)
        with patch("src.methods.hf_sequence_classification.get_peak_memory_mb", return_value=10):
            with patch(
                "src.methods.hf_sequence_classification.get_peak_memory_reserved_mb",
                return_value=12,
            ):
                metrics = build_runtime_metrics(
                    args,
                    training_time_sec=3.0,
                    gpu_type="cpu",
                    precision_policy={"mixed_precision": "none"},
                    status="completed",
                    extra={"stage1_training_time_sec": 1.0},
                )

        self.assertEqual(metrics["training_time_sec"], 3.0)
        self.assertEqual(metrics["peak_memory_mb"], 10)
        self.assertEqual(metrics["peak_memory_reserved_mb"], 12)
        self.assertTrue(metrics["gradient_checkpointing"])
        self.assertEqual(metrics["stage1_training_time_sec"], 1.0)


if __name__ == "__main__":
    unittest.main()
