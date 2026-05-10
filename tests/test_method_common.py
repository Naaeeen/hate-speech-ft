import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.methods.common import (
    add_common_method_arguments,
    build_checkpoint_policy,
    build_common_experiment_config,
    build_global_switches,
    build_training_policy,
    find_existing_run_artifacts,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)


class MethodCommonTests(unittest.TestCase):
    def parse_args(self, *args):
        parser = add_common_method_arguments(
            argparse.ArgumentParser(),
            defaults={
                "method": "lora",
                "trial_id": "lora_manual",
                "output_dir": "outputs/lora_manual",
                "load_best_model_at_end": True,
            },
        )
        return parser.parse_args(list(args))

    def test_common_arguments_expose_shared_contract_with_overridable_defaults(self):
        args = self.parse_args()

        self.assertEqual(args.method, "lora")
        self.assertEqual(args.trial_id, "lora_manual")
        self.assertEqual(args.output_dir, "outputs/lora_manual")
        self.assertEqual(args.dataset_name, "Hate-speech-CNERG/hatexplain")
        self.assertEqual(args.mixed_precision, "none")
        self.assertFalse(args.gradient_checkpointing)
        self.assertEqual(args.class_weighting, "none")
        self.assertTrue(args.load_best_model_at_end)

    def test_common_arguments_parse_policy_switches(self):
        args = self.parse_args(
            "--mixed_precision",
            "bf16",
            "--gradient_checkpointing",
            "--class_weighting",
            "balanced",
            "--use_wandb",
            "--wandb_entity",
            "hate-speech-ft",
        )

        self.assertEqual(args.mixed_precision, "bf16")
        self.assertTrue(args.gradient_checkpointing)
        self.assertEqual(args.class_weighting, "balanced")
        self.assertTrue(args.use_wandb)
        self.assertEqual(args.wandb_entity, "hate-speech-ft")

    def test_common_config_builders_merge_method_specific_fields(self):
        args = self.parse_args()
        args.model_name = "distilbert-base-uncased"
        args.tokenizer_name = "distilbert-base-uncased"

        config = build_common_experiment_config(
            args,
            hyperparameters={"learning_rate": 3e-4, "lora_r": 8},
            extra={"trainable_params": 1000, "total_params": 10000},
        )

        self.assertEqual(config["method"], "lora")
        self.assertEqual(config["model_name"], "distilbert-base-uncased")
        self.assertEqual(config["tokenizer_name"], "distilbert-base-uncased")
        self.assertEqual(config["hyperparameters"]["learning_rate"], 3e-4)
        self.assertEqual(config["hyperparameters"]["lora_r"], 8)
        self.assertEqual(config["checkpoint_policy"]["final_model_source"], "best_checkpoint")
        self.assertTrue(config["checkpoint_policy"]["load_best_model_at_end"])
        self.assertEqual(config["trainable_params"], 1000)
        self.assertEqual(config["total_params"], 10000)

    def test_policy_builders_are_method_agnostic(self):
        args = self.parse_args("--class_weighting", "balanced")

        self.assertEqual(
            build_global_switches(args),
            {
                "mixed_precision": "none",
                "gradient_checkpointing": False,
                "class_weighting": "balanced",
                "weighted_ce": True,
                "early_stopping": True,
            },
        )
        self.assertEqual(
            build_training_policy(args, class_weights=[1.0, 2.0, 3.0]),
            {
                "optim": "adamw_torch",
                "lr_scheduler_type": "linear",
                "weight_decay": 0.01,
                "warmup_ratio": 0.06,
                "max_grad_norm": 1.0,
                "mixed_precision": "none",
                "gradient_checkpointing": False,
                "class_weighting": "balanced",
                "class_weights": [1.0, 2.0, 3.0],
            },
        )
        self.assertEqual(build_checkpoint_policy(args)["metric_for_best_model"], "eval_f1_macro")

    def test_output_dir_guard_rejects_existing_run_artifacts(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "result_summary.json").write_text("{}", encoding="utf-8")
            (output_dir / "checkpoint-1").mkdir()

            artifacts = find_existing_run_artifacts(output_dir)

            self.assertIn(output_dir / "result_summary.json", artifacts)
            self.assertIn(output_dir / "checkpoint-1", artifacts)
            with self.assertRaisesRegex(ValueError, "already contains run artifacts"):
                validate_output_dir_for_run(output_dir, overwrite=False)
            validate_output_dir_for_run(output_dir, overwrite=True)

    def test_test_evaluation_policy_allows_only_final_stage(self):
        with self.assertRaises(ValueError):
            validate_test_evaluation_policy(search_stage="tuning", run_test=True)

        validate_test_evaluation_policy(search_stage="final", run_test=True)
        validate_test_evaluation_policy(search_stage="smoke", run_test=False)


if __name__ == "__main__":
    unittest.main()
