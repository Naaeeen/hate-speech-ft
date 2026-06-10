import os
import unittest
from unittest.mock import patch

from src.utils.wandb_config import (
    WandbSettings,
    apply_wandb_environment,
    finish_wandb_run,
    log_wandb,
    prefixed_wandb_scalars,
)


class WandbConfigTests(unittest.TestCase):
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
        )

        with patch.dict(os.environ, {}, clear=True):
            applied = apply_wandb_environment(settings)

            self.assertEqual(os.environ["WANDB_PROJECT"], "hate-speech-ft")
            self.assertEqual(os.environ["WANDB_ENTITY"], "hate-speech-team")
            self.assertEqual(os.environ["WANDB_MODE"], "offline")
            self.assertEqual(os.environ["WANDB_NAME"], "run-name")
            self.assertEqual(applied["WANDB_PROJECT"], "hate-speech-ft")

    def test_log_wandb_skips_empty_payloads(self):
        class FakeRun:
            def __init__(self):
                self.calls = []

            def log(self, payload):
                self.calls.append(payload)

        run = FakeRun()
        log_wandb(run, {}, {"eval/f1_macro": 0.5})

        self.assertEqual(run.calls, [{"eval/f1_macro": 0.5}])

    def test_prefixed_wandb_scalars_skips_nested_and_none_values(self):
        payload = prefixed_wandb_scalars(
            "model_selection",
            {"best_epoch": 3, "best_metric": 0.7, "extra": {"nested": True}, "none": None},
        )

        self.assertEqual(
            payload,
            {"model_selection/best_epoch": 3, "model_selection/best_metric": 0.7},
        )

    def test_wandb_runtime_errors_fail_the_run(self):
        class BadRun:
            def log(self, payload):
                raise RuntimeError("log unavailable")

            def finish(self):
                raise RuntimeError("finish unavailable")

        run = BadRun()

        with self.assertRaisesRegex(RuntimeError, "log unavailable"):
            log_wandb(run, {"metric": 1.0})

        with self.assertRaisesRegex(RuntimeError, "finish unavailable"):
            finish_wandb_run(run)


if __name__ == "__main__":
    unittest.main()
