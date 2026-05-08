import os
import unittest
from unittest.mock import patch

from src.utils.wandb_config import (
    WandbSettings,
    apply_wandb_environment,
    build_wandb_run_name,
    parse_wandb_tags,
)


class WandbConfigTests(unittest.TestCase):
    def test_parse_wandb_tags_deduplicates_and_ignores_empty_values(self):
        self.assertEqual(
            parse_wandb_tags(" smoke,distilbert,, smoke , colab "),
            ("smoke", "distilbert", "colab"),
        )

    def test_build_wandb_run_name_is_stable_and_informative(self):
        self.assertEqual(
            build_wandb_run_name(
                method="full-ft",
                model_name="distilbert-base-uncased",
                seed=42,
                max_train_samples=64,
                num_train_epochs=1.0,
                learning_rate=2e-5,
            ),
            "full-ft_distilbert-base-uncased_seed42_train64_ep1_lr2e-05",
        )

    def test_wandb_settings_report_to_switches_with_enabled_flag(self):
        self.assertEqual(WandbSettings(enabled=True).report_to, "wandb")
        self.assertEqual(WandbSettings(enabled=False).report_to, "none")

    def test_apply_wandb_environment_sets_only_wandb_variables(self):
        settings = WandbSettings(
            enabled=True,
            project="hate-speech-ft",
            entity="hate-speech-team",
            mode="offline",
            run_name="run-name",
            group="distilbert",
            tags=("smoke", "colab"),
            log_model="false",
        )

        with patch.dict(os.environ, {}, clear=True):
            applied = apply_wandb_environment(settings)

            self.assertEqual(os.environ["WANDB_PROJECT"], "hate-speech-ft")
            self.assertEqual(os.environ["WANDB_ENTITY"], "hate-speech-team")
            self.assertEqual(os.environ["WANDB_MODE"], "offline")
            self.assertEqual(os.environ["WANDB_NAME"], "run-name")
            self.assertEqual(os.environ["WANDB_RUN_GROUP"], "distilbert")
            self.assertEqual(os.environ["WANDB_TAGS"], "smoke,colab")
            self.assertEqual(os.environ["WANDB_LOG_MODEL"], "false")
            self.assertEqual(applied["WANDB_PROJECT"], "hate-speech-ft")


if __name__ == "__main__":
    unittest.main()
