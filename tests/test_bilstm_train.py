import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.results import write_json
from src.methods.bilstm import config as bilstm_config
from src.methods.bilstm import train as bilstm_train
from src.methods.bilstm.manual_config import CONFIG as BILSTM_CONFIG


class BiLSTMTrainEntryTests(unittest.TestCase):
    def test_runtime_metrics_count_gpu_hours_only_when_training_on_cuda(self):
        cpu_runtime = bilstm_config.build_runtime_metrics(
            training_time_sec=60.0,
            device="cpu",
            gpu_type="NVIDIA A100-SXM4-80GB",
            peak_memory_mb=None,
            peak_memory_reserved_mb=None,
        )
        cuda_runtime = bilstm_config.build_runtime_metrics(
            training_time_sec=60.0,
            device="cuda",
            gpu_type="NVIDIA A100-SXM4-80GB",
            peak_memory_mb=100.0,
            peak_memory_reserved_mb=200.0,
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
            created_tokenizers = []

            class FakeTokenizer:
                vocab_size = 100

                @classmethod
                def create(cls, *, train_records, max_length, min_freq, max_vocab_size):
                    instance = cls()
                    instance.train_records = train_records
                    instance.max_length = max_length
                    instance.min_freq = min_freq
                    instance.max_vocab_size = max_vocab_size
                    created_tokenizers.append(instance)
                    return instance

                def to_dict(self):
                    return {
                        "tokenizer_name": "fake-tokenizer",
                        "max_length": self.max_length,
                        "min_freq": self.min_freq,
                        "max_vocab_size": self.max_vocab_size,
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
            }
            train_split = types.SimpleNamespace(
                records=[{"id": "train-1", "text": "train sample", "label": 1}],
                raw_size=1,
                preprocessed_size=1,
                dropped_no_majority_count=0,
            )
            eval_split = types.SimpleNamespace(
                records=[{"id": "eval-1", "text": "eval sample", "label": 1}],
                raw_size=1,
                preprocessed_size=1,
                dropped_no_majority_count=0,
            )
            test_split = types.SimpleNamespace(
                records=[{"id": "test-1", "text": "test sample", "label": 1}],
                raw_size=1,
                preprocessed_size=1,
                dropped_no_majority_count=0,
            )
            fake_wandb_run = object()
            run_config = {
                **BILSTM_CONFIG,
                "run_name": "bilstm_final_fake",
                "output_dir": str(output_dir),
                "run_test": True,
                "use_wandb": True,
                "wandb_mode": "disabled",
            }

            with (
                patch.dict(
                    sys.modules,
                    {
                        "src.methods.bilstm.training": training_module,
                        "src.methods.bilstm.tokenizer": tokenizer_module,
                    },
                ),
                patch.object(bilstm_train, "MANUAL_CONFIG", run_config),
                patch.object(bilstm_train, "get_gpu_type", return_value="cpu"),
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
                    return_value=(train_split, eval_split, test_split),
                ),
                patch.object(bilstm_train, "print_split_summary"),
                patch.object(
                    bilstm_train,
                    "init_wandb_run",
                    return_value=fake_wandb_run,
                ) as init_wandb_run,
                patch.object(bilstm_train, "log_wandb") as log_wandb,
                patch.object(bilstm_train, "finish_wandb_run") as finish_wandb_run,
            ):
                bilstm_train.main()

            self.assertEqual(len(created_tokenizers), 1)
            self.assertIs(created_tokenizers[0].train_records, train_split.records)
            self.assertEqual(created_tokenizers[0].max_length, 128)
            self.assertEqual(created_tokenizers[0].min_freq, 2)
            self.assertEqual(created_tokenizers[0].max_vocab_size, 30000)
            self.assertTrue((output_dir / "model.pt").is_file())
            self.assertTrue((output_dir / "eval_predictions.json").is_file())
            self.assertTrue((output_dir / "test_predictions.json").is_file())
            summary = json.loads(
                (output_dir / "result_summary.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("status", summary)
            self.assertEqual(summary["metrics"]["eval"]["eval_f1_macro"], 0.5)
            self.assertEqual(summary["metrics"]["test"]["test_f1_macro"], 0.4)
            self.assertEqual(
                summary["artifacts"]["predictions"]["test"],
                (output_dir / "test_predictions.json").as_posix(),
            )
            self.assertEqual(summary["config"]["trainable_params"], 123)
            self.assertIn("git_commit", summary["config"])
            self.assertIn("split_accounting_policy", summary["config"])
            self.assertEqual(
                summary["config"]["tokenizer_policy"]["tokenizer_name"],
                "fake-tokenizer",
            )
            self.assertEqual(
                summary["config"]["hyperparameters"]["tokenizer_min_freq"],
                2,
            )
            self.assertEqual(
                summary["config"]["hyperparameters"]["max_vocab_size"],
                30000,
            )
            wandb_config = init_wandb_run.call_args.kwargs["config"]
            self.assertEqual(wandb_config["tokenizer_name"], "bilstm-word")
            self.assertEqual(
                wandb_config["tokenizer_policy"]["tokenizer_name"],
                "fake-tokenizer",
            )
            self.assertEqual(wandb_config["vocab_size"], 100)
            self.assertEqual(
                wandb_config["hyperparameters"]["tokenizer_min_freq"],
                2,
            )
            self.assertEqual(
                wandb_config["hyperparameters"]["max_vocab_size"],
                30000,
            )
            log_wandb.assert_called_once()
            self.assertIs(log_wandb.call_args.args[0], fake_wandb_run)
            payloads = log_wandb.call_args.args[1:]
            self.assertIn({"eval_f1_macro": 0.5, "eval_accuracy": 0.5}, payloads)
            self.assertIn({"test_f1_macro": 0.4, "test_accuracy": 0.4}, payloads)
            runtime_payload = payloads[2]
            self.assertEqual(runtime_payload["training_time_sec"], 1.25)
            self.assertEqual(runtime_payload["device"], "cpu")
            model_selection_payload = payloads[3]
            self.assertEqual(
                model_selection_payload["model_selection"]["best_metric"],
                0.5,
            )
            self.assertEqual(
                model_selection_payload["model_selection/best_metric"],
                0.5,
            )
            self.assertEqual(
                model_selection_payload["model_selection/best_epoch"],
                1,
            )
            finish_wandb_run.assert_called_once_with(fake_wandb_run)

if __name__ == "__main__":
    unittest.main()
