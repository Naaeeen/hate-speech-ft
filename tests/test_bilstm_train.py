import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.experiments.results import write_json
from src.methods.bilstm import args as bilstm_args
from src.methods.bilstm import config as bilstm_config
from src.methods.bilstm import train as bilstm_train


class BiLSTMTrainEntryTests(unittest.TestCase):
    def test_train_entry_accepts_shared_pipeline_arguments(self):
        completed = subprocess.run(
            [sys.executable, "src/methods/bilstm/train.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--config_hash",
            "--hpo_seed",
            "--hpo_trial_cap",
            "--hpo_time_cap_gpu_hours",
            "--overwrite_output_dir",
            "--max_test_samples",
            "--class_weighting",
            "--early_stopping_patience",
            "--use_wandb",
            "--wandb_entity",
            "--wandb_project",
            "--wandb_tags",
            "--wandb_mode",
            "--wandb_log_model",
        ):
            self.assertIn(option, completed.stdout)

    def test_rejects_unsupported_wandb_model_upload(self):
        args = bilstm_args.parse_args(["--wandb_log_model", "end"])

        with self.assertRaisesRegex(ValueError, "local model artifacts only"):
            bilstm_args.validate_bilstm_args(args)

    def test_runtime_metrics_count_gpu_hours_only_when_training_on_cuda(self):
        cpu_runtime = bilstm_config.build_runtime_metrics(
            training_time_sec=60.0,
            device="cpu",
            gpu_type="NVIDIA A100-SXM4-80GB",
            peak_memory_mb=None,
            peak_memory_reserved_mb=None,
            status="completed",
        )
        cuda_runtime = bilstm_config.build_runtime_metrics(
            training_time_sec=60.0,
            device="cuda",
            gpu_type="NVIDIA A100-SXM4-80GB",
            peak_memory_mb=100.0,
            peak_memory_reserved_mb=200.0,
            status="completed",
        )

        self.assertEqual(cpu_runtime["gpu_type"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(cpu_runtime["device"], "cpu")
        self.assertIsNone(cpu_runtime["gpu_hours"])
        self.assertAlmostEqual(cuda_runtime["gpu_hours"], 60.0 / 3600)

    def test_final_main_writes_standard_artifacts_with_fake_training_stack(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            training_module = types.ModuleType("src.methods.bilstm.training")
            tokenizer_module = types.ModuleType("src.methods.bilstm.tokenizer")

            class FakeTokenizer:
                vocab_size = 100

                @classmethod
                def create(cls, *, max_length):
                    instance = cls()
                    instance.max_length = max_length
                    return instance

                def to_dict(self):
                    return {
                        "tokenizer_name": "fake-tokenizer",
                        "max_length": self.max_length,
                        "vocab_size": self.vocab_size,
                    }

            tokenizer_module.StandardBiLSTMTokenizer = FakeTokenizer

            def fake_save_final_model(output_dir, **_kwargs):
                model_path = Path(output_dir) / "model.pt"
                model_path.write_text("fake model", encoding="utf-8")
                (Path(output_dir) / "tokenizer").mkdir(exist_ok=True)
                return model_path

            def fake_save_prediction_file(path, predictions):
                return write_json(
                    path,
                    {"count": len(predictions), "predictions": predictions},
                )

            training_module.set_seed = lambda _seed: None
            training_module.resolve_device = lambda _device: "cpu"
            training_module.resolve_class_weights = lambda *_args, **_kwargs: None
            training_module.save_final_model = fake_save_final_model
            training_module.save_prediction_file = fake_save_prediction_file
            training_module.run_training = lambda **_kwargs: {
                "model": object(),
                "eval_metrics": {"eval_f1_macro": 0.5, "eval_accuracy": 0.5},
                "test_metrics": {"test_f1_macro": 0.4, "test_accuracy": 0.4},
                "eval_predictions": [{"id": "eval-1", "predicted_label": 1}],
                "test_predictions": [{"id": "test-1", "predicted_label": 2}],
                "runtime": {
                    "training_time_sec": 1.25,
                    "peak_memory_mb": None,
                    "peak_memory_reserved_mb": None,
                    "final_model_source": "checkpoint-epoch1",
                },
                "model_selection": {
                    "best_metric": 0.5,
                    "best_epoch": 1,
                    "best_step": 1,
                    "best_checkpoint": "checkpoint-epoch1",
                },
                "parameters": {
                    "trainable_params": 123,
                    "total_params": 123,
                },
                "history": [{"epoch": 1, "eval_f1_macro": 0.5}],
            }
            fake_split = types.SimpleNamespace(
                records=[{"id": "sample-1", "text": "sample", "label": 1}],
                raw_size=1,
                preprocessed_size=1,
                dropped_no_majority_count=0,
            )
            argv = [
                "train.py",
                "--search_stage",
                "final",
                "--trial_id",
                "bilstm_final_fake",
                "--output_dir",
                str(output_dir),
                "--run_test",
            ]

            with (
                patch.dict(
                    sys.modules,
                    {
                        "src.methods.bilstm.training": training_module,
                        "src.methods.bilstm.tokenizer": tokenizer_module,
                    },
                ),
                patch.object(sys, "argv", argv),
                patch.object(bilstm_train, "get_gpu_type", return_value="cpu"),
                patch.object(bilstm_train, "get_peak_memory_mb", return_value=None),
                patch.object(
                    bilstm_train,
                    "get_peak_memory_reserved_mb",
                    return_value=None,
                ),
                patch.object(
                    bilstm_train,
                    "load_dataset_library",
                    return_value=lambda _name: {"train": [], "validation": [], "test": []},
                ),
                patch.object(
                    bilstm_train,
                    "resolve_bilstm_split_names",
                    return_value=("train", "validation", "test"),
                ),
                patch.object(
                    bilstm_train,
                    "build_bilstm_data_splits",
                    return_value=(fake_split, fake_split, fake_split),
                ),
                patch.object(bilstm_train, "print_split_summary"),
            ):
                bilstm_train.main()

            self.assertTrue((output_dir / "model.pt").is_file())
            self.assertTrue((output_dir / "eval_predictions.json").is_file())
            self.assertTrue((output_dir / "test_predictions.json").is_file())
            summary = json.loads(
                (output_dir / "result_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["metrics"]["eval"]["eval_f1_macro"], 0.5)
            self.assertEqual(summary["metrics"]["test"]["test_f1_macro"], 0.4)
            self.assertEqual(
                summary["artifacts"]["predictions"]["test"],
                (output_dir / "test_predictions.json").as_posix(),
            )
            self.assertEqual(summary["config"]["trainable_params"], 123)
            self.assertIn("git_commit", summary["config"])
            self.assertIn("split_accounting_policy", summary["config"])


if __name__ == "__main__":
    unittest.main()
