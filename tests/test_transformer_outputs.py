import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.methods import transformer_outputs
from src.methods.transformer_setup import start_hf_run, validate_checkpoint_policy


class TransformerOutputsTests(unittest.TestCase):
    def test_checkpoint_policy_rejects_best_model_without_matching_save_and_eval(self):
        args = types.SimpleNamespace(
            early_stopping_patience=0,
            early_stopping_threshold=0.0,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="steps",
            eval_steps=None,
            logging_steps=20,
            save_steps=500,
        )

        with self.assertRaisesRegex(ValueError, "save_strategy"):
            validate_checkpoint_policy(args)

    def test_checkpoint_policy_rejects_early_stopping_without_best_model_loading(self):
        args = types.SimpleNamespace(
            early_stopping_patience=2,
            early_stopping_threshold=0.0,
            load_best_model_at_end=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            eval_steps=None,
            logging_steps=20,
            save_steps=500,
        )

        with self.assertRaisesRegex(ValueError, "Early stopping"):
            validate_checkpoint_policy(args)

    def test_checkpoint_policy_accepts_step_based_best_model_setup(self):
        args = types.SimpleNamespace(
            early_stopping_patience=2,
            early_stopping_threshold=0.0,
            load_best_model_at_end=True,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            logging_steps=20,
            save_steps=500,
        )

        validate_checkpoint_policy(args)

    def test_start_hf_run_uses_single_mixed_precision_config_field(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "new-run"
            args = types.SimpleNamespace(
                output_dir=str(output_dir),
                overwrite_output_dir=False,
                data_fraction=None,
                max_train_samples=None,
                max_eval_samples=None,
                max_test_samples=None,
                early_stopping_patience=0,
                early_stopping_threshold=0.0,
                load_best_model_at_end=False,
                eval_strategy="epoch",
                save_strategy="epoch",
                eval_steps=None,
                logging_steps=20,
                save_steps=500,
                mixed_precision="bf16",
                use_wandb=False,
                wandb_project="hate-speech-ft",
                wandb_entity=None,
                wandb_mode="online",
                run_name="unit-test",
            )

            setup = start_hf_run(args)

            self.assertEqual(
                setup.precision_policy,
                {"mixed_precision": "bf16", "fp16": False, "bf16": True},
            )
            self.assertTrue(output_dir.exists())

    def test_write_success_outputs_logs_two_stage_extra_metrics_once(self):
        with TemporaryDirectory() as tmp, patch.object(
            transformer_outputs, "log_wandb"
        ) as log_wandb:
            transformer_outputs.write_success_outputs(
                types.SimpleNamespace(output_dir=tmp),
                config={"method": "lp-ft"},
                eval_metrics={"eval_f1_macro": 0.7},
                test_metrics=None,
                runtime_metrics={"training_time_sec": 1.0},
                model_selection={"best_metric": 0.7, "best_epoch": 2},
                prediction_paths={},
                wandb_run=object(),
                extra_metrics={"stage1": {"stage1_eval_f1_macro": 0.6}},
            )

        payloads = log_wandb.call_args.args[1:]
        self.assertIn({"stage1/eval/f1_macro": 0.6}, payloads)
        self.assertIn(
            {
                "model_selection": {"best_metric": 0.7, "best_epoch": 2},
                "model_selection/best_metric": 0.7,
                "model_selection/best_epoch": 2,
            },
            payloads,
        )

    def test_write_success_outputs_can_keep_extra_metrics_local_only(self):
        with TemporaryDirectory() as tmp, patch.object(
            transformer_outputs, "log_wandb"
        ) as log_wandb:
            transformer_outputs.write_success_outputs(
                types.SimpleNamespace(output_dir=tmp),
                config={"method": "lp-ft"},
                eval_metrics={"eval_f1_macro": 0.7},
                test_metrics=None,
                runtime_metrics={"training_time_sec": 1.0},
                model_selection={"best_metric": 0.7, "best_epoch": 2},
                prediction_paths={},
                wandb_run=object(),
                extra_metrics={"stage1": {"stage1_eval_f1_macro": 0.6}},
                log_extra_metrics_to_wandb=False,
            )

        payloads = log_wandb.call_args.args[1:]
        self.assertNotIn({"stage1/eval/f1_macro": 0.6}, payloads)

    def test_save_final_model_records_file_level_transformer_artifacts(self):
        class FakeTrainer:
            def save_model(self, output_dir):
                Path(output_dir, "config.json").write_text("{}", encoding="utf-8")
                Path(output_dir, "model.safetensors").write_text("", encoding="utf-8")

        class FakeTokenizer:
            def save_pretrained(self, output_dir):
                Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")

        with TemporaryDirectory() as tmp:
            artifacts = transformer_outputs.save_final_model(
                FakeTrainer(),
                FakeTokenizer(),
                output_dir=tmp,
                no_save_final_model=False,
                model_source="unit-test",
            )

        self.assertEqual(
            set(artifacts),
            {"config.json", "model.safetensors", "tokenizer.json"},
        )


if __name__ == "__main__":
    unittest.main()
