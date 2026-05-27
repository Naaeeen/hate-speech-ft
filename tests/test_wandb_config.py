import os
import unittest
from unittest.mock import patch

from src.utils.wandb_config import (
    WandbSettings,
    apply_wandb_environment,
    build_wandb_run_name,
    define_wandb_metric_best_effort,
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

    def test_build_wandb_run_name_can_include_trial_id(self):
        self.assertEqual(
            build_wandb_run_name(
                method="full-ft",
                model_name="distilbert-base-uncased",
                seed=42,
                max_train_samples=64,
                num_train_epochs=1.0,
                learning_rate=2e-5,
                trial_id="distilbert/full smoke",
            ),
            (
                "distilbert-full-smoke_"
                "full-ft_distilbert-base-uncased_seed42_train64_ep1_lr2e-05"
            ),
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

    def test_define_wandb_metric_best_effort_sets_step_metric(self):
        class FakeRun:
            def __init__(self):
                self.calls = []

            def define_metric(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        run = FakeRun()

        define_wandb_metric_best_effort(
            run,
            "train_loss",
            step_metric="global_step",
        )

        self.assertEqual(
            run.calls,
            [(("train_loss",), {"step_metric": "global_step"})],
        )

    def test_log_wandb_best_effort_skips_empty_payloads(self):
        class FakeRun:
            def __init__(self):
                self.calls = []

            def log(self, payload):
                self.calls.append(payload)

        from src.utils.wandb_config import log_wandb_best_effort

        run = FakeRun()
        log_wandb_best_effort(run, {}, {"eval/f1_macro": 0.5})

        self.assertEqual(run.calls, [{"eval/f1_macro": 0.5}])


if __name__ == "__main__":
    unittest.main()
