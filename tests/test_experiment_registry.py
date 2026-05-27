import unittest
import sys
import re
from pathlib import Path

from src.experiments.registry import (
    build_experiment_command,
    load_experiment_registry,
    parse_override_pairs,
)


class ExperimentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.registry = load_experiment_registry(
            self.repo_root / "configs" / "experiments.json"
        )

    def test_registry_lists_ready_and_planned_methods(self):
        ready_ids = {spec.experiment_id for spec in self.registry.ready_experiments()}
        all_ids = {spec.experiment_id for spec in self.registry.experiments}

        self.assertIn("distilbert_full_smoke", ready_ids)
        self.assertIn("distilbert_full_tuning", ready_ids)
        self.assertIn("tfidf_logreg_tuning", ready_ids)
        self.assertIn("frozen_distilbert_tuning", ready_ids)
        self.assertIn("distilbert_lora_tuning", ready_ids)
        self.assertIn("distilbert_efficient_head_tuning", ready_ids)

    def test_build_ready_experiment_command_includes_common_and_wandb_args(self):
        spec = self.registry.get("distilbert_full_smoke")

        command = build_experiment_command(
            spec,
            repo_root=self.repo_root,
            use_wandb=True,
            wandb_entity="hate-speech-team",
            wandb_project="hate-speech-ft",
        )

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "src/methods/distilbert_full/train.py")
        self.assertIn("--method", command)
        self.assertIn("full-ft", command)
        self.assertIn("--search_stage", command)
        self.assertIn("smoke", command)
        self.assertIn("--max_train_samples", command)
        self.assertIn("256", command)
        self.assertIn("--use_wandb", command)
        self.assertIn("--wandb_entity", command)
        self.assertIn("hate-speech-team", command)
        self.assertIn("--wandb_project", command)
        self.assertIn("hate-speech-ft", command)
        self.assertIn("--wandb_tags", command)
        self.assertIn("distilbert,full-ft,smoke", command)

    def test_wandb_defaults_include_config_hash_and_merge_custom_tags(self):
        spec = self.registry.get("distilbert_full_tuning")

        command = build_experiment_command(
            spec,
            repo_root=self.repo_root,
            overrides={"config_hash": "abc123def456"},
            use_wandb=True,
            wandb_project="hate-speech-ft",
            wandb_tags="custom,distilbert",
        )

        group_index = command.index("--wandb_group")
        tags_index = command.index("--wandb_tags")
        self.assertEqual(command[group_index + 1], "full-ft-tuning-abc123def456")
        self.assertEqual(command[tags_index + 1], "distilbert,full-ft,tuning,custom")

    def test_parse_override_pairs_supports_basic_types(self):
        overrides = parse_override_pairs(
            [
                "learning_rate=3e-5",
                "max_train_samples=128",
                "use_scheduler=true",
                "notes=manual test",
            ]
        )

        self.assertEqual(overrides["learning_rate"], 3e-5)
        self.assertEqual(overrides["max_train_samples"], 128)
        self.assertIs(overrides["use_scheduler"], True)
        self.assertEqual(overrides["notes"], "manual test")

    def test_parse_override_pairs_rejects_removed_determinism_switch(self):
        with self.assertRaisesRegex(ValueError, "full_determinism"):
            parse_override_pairs(["full_determinism=true"])

    def test_build_command_applies_overrides(self):
        spec = self.registry.get("distilbert_full_smoke")

        command = build_experiment_command(
            spec,
            repo_root=self.repo_root,
            overrides={"learning_rate": 3e-5, "max_train_samples": 128},
            use_wandb=False,
        )

        lr_index = command.index("--learning_rate")
        sample_index = command.index("--max_train_samples")
        self.assertEqual(command[lr_index + 1], "3e-05")
        self.assertEqual(command[sample_index + 1], "128")
        self.assertNotIn("--use_wandb", command)

    def test_registry_applies_safe_command_defaults(self):
        spec = self.registry.get("frozen_distilbert_tuning")

        self.assertEqual(spec.args["dataset_name"], "Hate-speech-CNERG/hatexplain")
        self.assertEqual(spec.args["class_weighting"], "none")
        self.assertEqual(spec.args["mixed_precision"], "none")
        self.assertIn("final_seeds", spec.defaults)
        self.assertNotIn("final_seeds", spec.args)
        self.assertIn("class_weighting", spec.command_defaults)
        self.assertNotIn("wandb_project", spec.command_defaults)

        tfidf_spec = self.registry.get("tfidf_logreg_tuning")
        self.assertNotIn("mixed_precision", tfidf_spec.args)
        self.assertNotIn("gradient_checkpointing", tfidf_spec.args)

        lora_spec = self.registry.get("distilbert_lora_tuning")
        self.assertEqual(lora_spec.args["mixed_precision"], "none")
        self.assertEqual(lora_spec.args["optim"], "adamw_torch")

    def test_final_experiment_enables_test_evaluation(self):
        spec = self.registry.get("distilbert_full_final_seed42")
        command = build_experiment_command(spec, repo_root=self.repo_root)

        self.assertIn("--run_test", command)

    def test_direct_final_experiment_gets_config_hash(self):
        spec = self.registry.get("distilbert_full_final_seed42")
        command = build_experiment_command(spec, repo_root=self.repo_root)

        hash_index = command.index("--config_hash")
        self.assertRegex(command[hash_index + 1], r"^[0-9a-f]{12}$")

    def test_missing_planned_script_is_not_runnable(self):
        spec = self.registry.get("partial_distilbert_template")

        with self.assertRaises(FileNotFoundError):
            build_experiment_command(spec, repo_root=self.repo_root)


if __name__ == "__main__":
    unittest.main()
