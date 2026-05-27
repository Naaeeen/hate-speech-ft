import unittest

from src.methods import staged_wandb
from src.utils.wandb_config import WandbSettings


class FakeRun:
    def __init__(self):
        self.defined_metrics = []
        self.logged_payloads = []

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, payload):
        self.logged_payloads.append(dict(payload))


class FakeTrainer:
    def __init__(self):
        self.state = type(
            "FakeState",
            (),
            {
                "log_history": [
                    {
                        "loss": 0.7,
                        "learning_rate": 1e-4,
                        "grad_norm": 1.2,
                        "epoch": 1.0,
                        "step": 100,
                    },
                    {
                        "eval_loss": 0.6,
                        "eval_f1_macro": 0.5,
                        "eval_accuracy": 0.55,
                        "epoch": 1.0,
                        "step": 100,
                    },
                    {
                        "stage1_eval_loss": 0.58,
                        "stage1_eval_f1_macro": 0.52,
                        "epoch": 1.0,
                        "step": 100,
                    },
                    {
                        "train_runtime": 10.0,
                        "train_loss": 0.65,
                        "epoch": 1.0,
                        "step": 100,
                    },
                ]
            },
        )()


class StagedWandbTests(unittest.TestCase):
    def test_disable_hf_wandb_reporting_preserves_run_metadata(self):
        settings = WandbSettings(
            enabled=True,
            project="hate-speech-ft",
            entity="team",
            mode="online",
            run_name="run",
            group="group",
            tags=("a", "b"),
            log_model="false",
        )

        disabled = staged_wandb.disable_hf_wandb_reporting(settings)

        self.assertFalse(disabled.enabled)
        self.assertEqual(disabled.project, settings.project)
        self.assertEqual(disabled.entity, settings.entity)
        self.assertEqual(disabled.run_name, settings.run_name)
        self.assertEqual(disabled.group, settings.group)
        self.assertEqual(disabled.tags, settings.tags)
        self.assertEqual(disabled.log_model, settings.log_model)
        self.assertEqual(disabled.report_to, "none")

    def test_log_stage_trainer_history_uses_stage_specific_metric_namespace(self):
        run = FakeRun()

        staged_wandb.log_stage_trainer_history(
            run,
            FakeTrainer(),
            stage="stage1",
        )

        self.assertIn(
            (("stage1/train/*",), {"step_metric": "stage1/global_step"}),
            run.defined_metrics,
        )
        self.assertEqual(
            run.logged_payloads[0],
            {
                "stage1/global_step": 100,
                "stage1/epoch": 1.0,
                "stage1/train/loss": 0.7,
                "stage1/train/learning_rate": 1e-4,
                "stage1/train/grad_norm": 1.2,
            },
        )
        self.assertEqual(
            run.logged_payloads[1],
            {
                "stage1/global_step": 100,
                "stage1/epoch": 1.0,
                "stage1/eval/loss": 0.6,
                "stage1/eval/f1_macro": 0.5,
                "stage1/eval/accuracy": 0.55,
            },
        )
        self.assertEqual(
            run.logged_payloads[2],
            {
                "stage1/global_step": 100,
                "stage1/epoch": 1.0,
                "stage1/eval/loss": 0.58,
                "stage1/eval/f1_macro": 0.52,
            },
        )
        self.assertEqual(
            run.logged_payloads[3],
            {
                "stage1/global_step": 100,
                "stage1/epoch": 1.0,
                "stage1/train/runtime": 10.0,
                "stage1/train/loss_average": 0.65,
            },
        )

    def test_log_stage_trainer_history_logs_extra_metrics_once(self):
        class SparseTrainer:
            state = type(
                "FakeState",
                (),
                {
                    "global_step": 200,
                    "epoch": 2.0,
                    "log_history": [{"loss": 0.4, "step": 200, "epoch": 2.0}],
                },
            )()

        run = FakeRun()

        staged_wandb.log_stage_trainer_history(
            run,
            SparseTrainer(),
            stage="stage1",
            extra_metrics={"stage1_eval_f1_macro": 0.61},
        )

        self.assertEqual(
            run.logged_payloads[-1],
            {
                "stage1/global_step": 200,
                "stage1/epoch": 2.0,
                "stage1/eval/f1_macro": 0.61,
            },
        )


if __name__ == "__main__":
    unittest.main()
