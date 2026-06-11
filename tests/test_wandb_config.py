import os
import unittest
from unittest.mock import patch

from src.utils.wandb_config import (
    WandbSettings,
    apply_wandb_environment,
    build_wandb_settings_from_args,
    define_stage_wandb_metrics,
    define_training_wandb_metrics,
    define_wandb_metric,
    finish_wandb_run,
    init_wandb_run,
    log_wandb_trainer_history,
    log_wandb,
    log_wandb_history,
    namespaced_wandb_metrics,
    prefixed_wandb_scalars,
)


class WandbConfigTests(unittest.TestCase):
    def test_wandb_settings_report_to_switches_with_enabled_flag(self):
        self.assertEqual(WandbSettings(enabled=True).report_to, "wandb")
        self.assertEqual(WandbSettings(enabled=False).report_to, "none")
        self.assertEqual(WandbSettings(enabled=True, mode="disabled").report_to, "none")

    def test_disabled_mode_turns_wandb_off_even_when_use_wandb_is_true(self):
        args = type(
            "Args",
            (),
            {
                "use_wandb": True,
                "wandb_project": "hate-speech-ft",
                "wandb_entity": "team",
                "wandb_mode": "disabled",
                "run_name": "disabled-run",
            },
        )()

        settings = build_wandb_settings_from_args(args)

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.report_to, "none")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(apply_wandb_environment(settings), {})
            self.assertIsNone(init_wandb_run(settings, config={"method": "unit-test"}))
            self.assertNotIn("WANDB_MODE", os.environ)

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

    def test_log_wandb_history_logs_rows_in_order(self):
        class FakeRun:
            def __init__(self):
                self.calls = []

            def log(self, payload):
                self.calls.append(payload)

        run = FakeRun()
        log_wandb_history(
            run,
            [
                {"epoch": 1, "global_step": 10, "train_loss": 0.9},
                {},
                {"epoch": 2, "global_step": 20, "train_loss": 0.7},
            ],
        )

        self.assertEqual(
            run.calls,
            [
                {"train/epoch": 1, "train/global_step": 10, "train/loss": 0.9},
                {"train/epoch": 2, "train/global_step": 20, "train/loss": 0.7},
            ],
        )

    def test_define_wandb_metric_uses_step_metric_when_present(self):
        class FakeRun:
            def __init__(self):
                self.defined = []

            def define_metric(self, name, **kwargs):
                self.defined.append((name, kwargs))

        run = FakeRun()
        define_wandb_metric(run, "global_step")
        define_wandb_metric(run, "train_loss", step_metric="global_step")

        self.assertEqual(
            run.defined,
            [
                ("global_step", {}),
                ("train_loss", {"step_metric": "global_step"}),
            ],
        )

    def test_metric_axis_helpers_define_top_level_test_metrics_only(self):
        class FakeRun:
            def __init__(self):
                self.defined = []

            def define_metric(self, name, **kwargs):
                self.defined.append((name, kwargs))

        run = FakeRun()
        define_training_wandb_metrics(run)
        define_stage_wandb_metrics(run, "stage2")

        self.assertIn(("test/*", {"step_metric": "train/global_step"}), run.defined)
        self.assertNotIn(("stage2/test/*", {"step_metric": "stage2/global_step"}), run.defined)

    def test_namespaced_wandb_metrics_adds_common_eval_test_keys(self):
        payload = namespaced_wandb_metrics(
            {
                "eval_f1_macro": 0.7,
                "test_accuracy": 0.6,
                "training_time_sec": 12.0,
            }
        )

        self.assertEqual(
            payload,
            {
                "eval/f1_macro": 0.7,
                "test/accuracy": 0.6,
                "training_time_sec": 12.0,
            },
        )

    def test_log_wandb_trainer_history_adds_stage_namespace(self):
        class FakeState:
            global_step = 20
            epoch = 2.0
            log_history = [
                {"loss": 0.9, "learning_rate": 1e-5, "step": 10, "epoch": 1.0},
                {"eval_f1_macro": 0.6, "eval_accuracy": 0.7, "step": 20, "epoch": 2.0},
            ]

        class FakeTrainer:
            state = FakeState()

        class FakeRun:
            def __init__(self):
                self.calls = []

            def log(self, payload):
                self.calls.append(payload)

        run = FakeRun()
        log_wandb_trainer_history(
            run,
            FakeTrainer(),
            stage="stage1",
            extra_metrics={"stage1_eval_f1_macro": 0.65},
        )

        self.assertIn(
            {
                "stage1/train/loss": 0.9,
                "stage1/train/learning_rate": 1e-05,
                "stage1/global_step": 10,
                "stage1/epoch": 1.0,
            },
            run.calls,
        )
        self.assertIn(
            {
                "stage1/eval/f1_macro": 0.6,
                "stage1/eval/accuracy": 0.7,
                "stage1/global_step": 20,
                "stage1/epoch": 2.0,
            },
            run.calls,
        )
        self.assertIn(
            {
                "stage1/eval/f1_macro": 0.65,
                "stage1/global_step": 20,
                "stage1/epoch": 2.0,
            },
            run.calls,
        )

    def test_parent_run_can_receive_both_stage_histories(self):
        class FakeState:
            def __init__(self, offset):
                self.global_step = offset + 20
                self.epoch = 2.0
                self.log_history = [
                    {
                        "loss": 0.8 + offset,
                        "learning_rate": 1e-5,
                        "step": offset + 10,
                        "epoch": 1.0,
                    },
                    {
                        "eval_f1_macro": 0.6 + offset,
                        "step": offset + 20,
                        "epoch": 2.0,
                    },
                ]

        class FakeTrainer:
            def __init__(self, offset):
                self.state = FakeState(offset)

        class FakeRun:
            def __init__(self):
                self.calls = []

            def log(self, payload):
                self.calls.append(payload)

        run = FakeRun()
        log_wandb_trainer_history(
            run,
            FakeTrainer(0),
            stage="stage1",
            extra_metrics={"stage1_eval_f1_macro": 0.61},
        )
        log_wandb_trainer_history(
            run,
            FakeTrainer(100),
            stage="stage2",
            extra_metrics={"eval_f1_macro": 0.72},
        )

        merged_keys = {key for payload in run.calls for key in payload}
        self.assertIn("stage1/train/loss", merged_keys)
        self.assertIn("stage1/eval/f1_macro", merged_keys)
        self.assertIn("stage2/train/loss", merged_keys)
        self.assertIn("stage2/eval/f1_macro", merged_keys)
        self.assertNotIn("stage2/test/f1_macro", merged_keys)

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
