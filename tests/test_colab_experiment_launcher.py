import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.colab.experiment_launcher import ExperimentLauncher
from src.experiments.hpo import load_hpo_config
from src.experiments.registry import REPO_ROOT
from src.experiments.registry import load_experiment_registry


class ValueBox:
    def __init__(self, value):
        self.value = value


class ColabExperimentLauncherTests(unittest.TestCase):
    def test_run_uses_repo_root_as_working_directory(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.get_config = lambda: {"suggest_trials": 0}
        launcher.build_command = lambda: ["python", "src/run_experiment.py", "--list"]

        with (
            patch("src.colab.experiment_launcher.subprocess.run") as run,
            patch("builtins.print"),
        ):
            launcher.run()

        run.assert_called_once_with(
            ["python", "src/run_experiment.py", "--list"],
            check=True,
            cwd=REPO_ROOT,
        )

    def test_build_trial_commands_adds_unique_hpo_outputs(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_tuning",
            "overrides": {"mixed_precision": "bf16"},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "overwrite_output_dir": True,
            "suggest_trials": 2,
            "search_space": "full_ft",
            "hpo_seed": 42,
            "trial_output_root": "outputs/hpo",
        }

        commands = launcher.build_trial_commands()

        self.assertEqual(len(commands), 2)
        self.assertIn("--trial_id", commands[0])
        first_command = " ".join(commands[0])
        second_command = " ".join(commands[1])
        self.assertIn("distilbert_full_tuning__full_ft__hpo42__trial001", first_command)
        self.assertIn(
            "outputs/hpo/distilbert_full_tuning__full_ft__hpo42__trial002",
            second_command,
        )
        self.assertRegex(first_command, r"--trial_id [^ ]+__[0-9a-f]{12}")
        self.assertRegex(first_command, r"--output_dir [^ ]+__[0-9a-f]{12}")
        self.assertIn("--optim", commands[0])
        self.assertIn("adamw_torch", commands[0])
        self.assertIn("--hpo_time_cap_gpu_hours", commands[0])
        self.assertIn("2", commands[0])
        self.assertIn("--mixed_precision", commands[0])
        self.assertIn("bf16", commands[0])
        self.assertIn("--overwrite_output_dir", commands[0])

    def test_build_seed_run_commands_uses_final_seed_policy(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_tuning",
            "overrides": {"learning_rate": 2e-5},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "overwrite_output_dir": False,
            "suggest_trials": 0,
            "seed_run_stage": "final",
            "seed_output_root": "outputs/final",
        }

        commands = launcher.build_seed_run_commands()

        self.assertEqual(len(commands), 3)
        self.assertIn("--search_stage", commands[0])
        self.assertIn("final", commands[0])
        self.assertIn("--run_test", commands[0])
        self.assertIn("--hpo_trial_cap", commands[0])
        self.assertIn("3", commands[0])
        self.assertIn("--hpo_time_cap_gpu_hours", commands[0])
        self.assertIn("2", commands[0])
        self.assertIn("--seed", commands[1])
        self.assertIn("43", commands[1])

    def test_bilstm_launcher_rejects_wandb_model_upload(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.get_config = lambda: {
            "experiment": "bilstm_smoke",
            "overrides": {},
            "use_wandb": True,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "end",
            "overwrite_output_dir": False,
            "suggest_trials": 0,
            "seed_run_stage": "none",
        }

        with self.assertRaisesRegex(ValueError, "model artifacts locally only"):
            launcher.build_command()

    def test_tfidf_launcher_rejects_wandb_model_upload(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.get_config = lambda: {
            "experiment": "tfidf_logreg_smoke",
            "overrides": {},
            "use_wandb": True,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "end",
            "overwrite_output_dir": False,
            "suggest_trials": 0,
            "seed_run_stage": "none",
        }

        with self.assertRaisesRegex(ValueError, "model artifacts locally only"):
            launcher.build_command()

    def test_launcher_direct_final_rejects_sample_policy_override(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_final_seed42",
            "overrides": {"data_fraction": 0.5},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "overwrite_output_dir": False,
            "suggest_trials": 0,
            "search_space": "full_ft",
            "seed_run_stage": "none",
        }

        with self.assertRaisesRegex(ValueError, "data_fraction"):
            launcher.build_command()

    def test_trial_commands_reject_smoke_base(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_smoke",
            "overrides": {},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "suggest_trials": 1,
            "search_space": "full_ft",
            "hpo_seed": 42,
            "trial_output_root": "outputs/hpo",
        }

        with self.assertRaises(ValueError):
            launcher.build_trial_commands()

    def test_trial_commands_reject_quick_base(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_quick",
            "overrides": {},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "suggest_trials": 1,
            "search_space": "full_ft",
            "hpo_seed": 42,
            "trial_output_root": "outputs/hpo",
        }

        with self.assertRaisesRegex(ValueError, "tuning experiment"):
            launcher.build_trial_commands()

    def test_trial_commands_reject_fp16_alias_override(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_tuning",
            "overrides": {"fp16": True},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "overwrite_output_dir": False,
            "suggest_trials": 1,
            "search_space": "full_ft",
            "hpo_seed": 42,
            "trial_output_root": "outputs/hpo",
        }

        with self.assertRaisesRegex(ValueError, "mixed_precision=fp16"):
            launcher.build_trial_commands()

    def test_lp_ft_seed_commands_reject_batch_size_alias_override(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_lp_ft_tuning",
            "overrides": {
                "stage1_head_learning_rate": 0.0001,
                "stage1_epochs": 5,
                "stage2_learning_rate": 2e-5,
                "stage2_epochs": 2,
                "batch_size": 16,
            },
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "overwrite_output_dir": False,
            "suggest_trials": 0,
            "seed_run_stage": "final",
            "seed_output_root": "outputs/final",
        }

        with self.assertRaisesRegex(ValueError, "per_device_train_batch_size"):
            launcher.build_seed_run_commands()

    def test_seed_run_commands_reject_smoke_base(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_smoke",
            "overrides": {"learning_rate": 2e-5},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "overwrite_output_dir": False,
            "suggest_trials": 0,
            "seed_run_stage": "final",
            "seed_output_root": "outputs/final",
        }

        with self.assertRaises(ValueError):
            launcher.build_seed_run_commands()

    def test_seed_run_output_root_defaults_to_stage_when_blank(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.registry = load_experiment_registry()
        launcher.search_config = load_hpo_config()
        launcher.get_config = lambda: {
            "experiment": "distilbert_full_tuning",
            "overrides": {"learning_rate": 2e-5},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "overwrite_output_dir": False,
            "suggest_trials": 0,
            "seed_run_stage": "confirm",
            "seed_output_root": "",
        }

        commands = launcher.build_seed_run_commands()

        self.assertEqual(len(commands), 3)
        self.assertIn("outputs/confirm", " ".join(commands[0]))

    def test_run_dispatches_to_trial_commands_when_trials_are_requested(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.get_config = lambda: {"suggest_trials": 2, "seed_run_stage": "none"}
        launcher.run_trial_commands = lambda: ["trial-result"]

        self.assertEqual(launcher.run(), ["trial-result"])

    def test_run_dispatches_to_seed_runs_when_requested(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.get_config = lambda: {"suggest_trials": 0, "seed_run_stage": "final"}
        launcher.run_seed_run_commands = lambda: ["seed-result"]

        self.assertEqual(launcher.run(), ["seed-result"])

    def test_run_rejects_trials_and_seed_runs_together(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.get_config = lambda: {"suggest_trials": 1, "seed_run_stage": "confirm"}

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            launcher.run()

    def test_preview_rejects_trials_and_seed_runs_together(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.get_config = lambda: {"suggest_trials": 1, "seed_run_stage": "final"}

        with self.assertRaisesRegex(ValueError, "Set Trials to 0"):
            launcher.preview_command()

    def test_build_aggregate_command_uses_widget_settings(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.trial_output_root = ValueBox("outputs/hpo")
        launcher.aggregate_input = ValueBox("outputs/hpo")
        launcher.aggregate_output = ValueBox("outputs/hpo/aggregate_summary.json")
        launcher.aggregate_group_by = ValueBox("method search_stage config_hash")
        launcher.aggregate_metrics = ValueBox("eval_f1_macro,training_time_sec")
        launcher.write_pareto_csvs = ValueBox(True)
        launcher.pareto_csv_dir = ValueBox("outputs/pareto")
        launcher.write_prediction_analysis = ValueBox(True)
        launcher.prediction_analysis_dir = ValueBox("outputs/diagnostics")
        launcher.max_error_examples = ValueBox(12)

        command = launcher.build_aggregate_command()

        self.assertIn("src/aggregate_results.py", command)
        self.assertIn("outputs/hpo", command)
        self.assertIn("--group_by", command)
        self.assertIn("config_hash", command)
        self.assertEqual(command.count("--metric"), 2)
        self.assertIn("--write_pareto_csvs", command)
        self.assertIn("--csv_dir", command)
        self.assertIn("outputs/pareto", command)
        self.assertIn("--write_prediction_analysis", command)
        self.assertIn("--prediction_analysis_dir", command)
        self.assertIn("outputs/diagnostics", command)
        self.assertIn("--max_error_examples", command)
        self.assertIn("12", command)

    def test_aggregate_defaults_follow_trial_output_root(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.trial_output_root = ValueBox("outputs/custom_hpo")
        launcher.seed_run_stage = ValueBox("none")
        launcher.seed_output_root = ValueBox("")
        launcher.aggregate_input = ValueBox("")
        launcher.aggregate_output = ValueBox("")
        launcher.aggregate_group_by = ValueBox("method search_stage config_hash")
        launcher.aggregate_metrics = ValueBox("eval_f1_macro,training_time_sec")
        launcher.write_pareto_csvs = ValueBox(True)
        launcher.pareto_csv_dir = ValueBox("")
        launcher.write_prediction_analysis = ValueBox(False)
        launcher.prediction_analysis_dir = ValueBox("")
        launcher.max_error_examples = ValueBox(50)

        config = launcher.get_aggregate_config()

        self.assertEqual(config["input"], "outputs/custom_hpo")
        self.assertEqual(
            config["output"],
            "outputs/custom_hpo/aggregate_summary.json",
        )
        self.assertTrue(config["write_pareto_csvs"])
        self.assertFalse(config["write_prediction_analysis"])

    def test_aggregate_defaults_follow_active_seed_output_root(self):
        launcher = object.__new__(ExperimentLauncher)
        launcher.trial_output_root = ValueBox("outputs/hpo")
        launcher.seed_run_stage = ValueBox("final")
        launcher.seed_output_root = ValueBox("outputs/final")
        launcher.aggregate_input = ValueBox("")
        launcher.aggregate_output = ValueBox("")
        launcher.aggregate_group_by = ValueBox("method search_stage config_hash")
        launcher.aggregate_metrics = ValueBox("eval_f1_macro,test_f1_macro")
        launcher.write_pareto_csvs = ValueBox(False)
        launcher.pareto_csv_dir = ValueBox("")
        launcher.write_prediction_analysis = ValueBox(True)
        launcher.prediction_analysis_dir = ValueBox("")
        launcher.max_error_examples = ValueBox(20)

        config = launcher.get_aggregate_config()

        self.assertEqual(config["input"], "outputs/final")
        self.assertEqual(config["output"], "outputs/final/aggregate_summary.json")
        self.assertFalse(config["write_pareto_csvs"])
        self.assertTrue(config["write_prediction_analysis"])
        self.assertEqual(config["prediction_analysis_dir"], None)
        self.assertEqual(config["max_error_examples"], 20)

    def test_aggregate_results_writes_report_from_widget_settings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "trial001"
            run_dir.mkdir()
            (run_dir / "result_summary.json").write_text(
                """{
  "status": "completed",
  "config": {
    "method": "full-ft",
    "search_stage": "tuning",
    "config_hash": "abc",
    "output_dir": "trial001"
  },
  "metrics": {"eval": {"eval_f1_macro": 0.7}},
  "runtime": {"training_time_sec": 12.0}
}
""",
                encoding="utf-8",
            )
            launcher = object.__new__(ExperimentLauncher)
            launcher.trial_output_root = ValueBox(str(root))
            launcher.aggregate_input = ValueBox(str(root))
            launcher.aggregate_output = ValueBox(str(root / "aggregate_summary.json"))
            launcher.aggregate_group_by = ValueBox("method search_stage config_hash")
            launcher.aggregate_metrics = ValueBox("eval_f1_macro,training_time_sec")
            launcher.write_pareto_csvs = ValueBox(True)
            launcher.pareto_csv_dir = ValueBox(str(root / "pareto"))
            launcher.write_prediction_analysis = ValueBox(True)
            launcher.prediction_analysis_dir = ValueBox(str(root / "diagnostics"))
            launcher.max_error_examples = ValueBox(5)

            with patch("builtins.print"):
                report = launcher.aggregate_results()

            self.assertEqual(report["total_runs"], 1)
            self.assertTrue((root / "aggregate_summary.json").is_file())
            self.assertTrue((root / "pareto" / "hpo_runs.csv").is_file())
            self.assertTrue((root / "diagnostics" / "prediction_analysis.json").is_file())


if __name__ == "__main__":
    unittest.main()
