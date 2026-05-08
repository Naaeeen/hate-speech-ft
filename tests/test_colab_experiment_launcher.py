import unittest
from unittest.mock import patch

from src.colab.experiment_launcher import ExperimentLauncher
from src.experiments.hpo import load_hpo_config
from src.experiments.registry import REPO_ROOT
from src.experiments.registry import load_experiment_registry


class ColabExperimentLauncherTests(unittest.TestCase):
    def test_run_uses_repo_root_as_working_directory(self):
        launcher = object.__new__(ExperimentLauncher)
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
            "experiment": "distilbert_full_smoke",
            "overrides": {},
            "use_wandb": False,
            "wandb_entity": "",
            "wandb_project": "hate-speech-ft",
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_mode": "online",
            "wandb_log_model": "false",
            "suggest_trials": 2,
            "search_space": "full_ft",
            "hpo_seed": 42,
            "trial_output_root": "outputs/hpo",
        }

        commands = launcher.build_trial_commands()

        self.assertEqual(len(commands), 2)
        self.assertIn("--trial_id", commands[0])
        self.assertIn("distilbert_full_smoke__full_ft__trial001", commands[0])
        self.assertIn("outputs/hpo/distilbert_full_smoke__full_ft__trial002", commands[1])


if __name__ == "__main__":
    unittest.main()
