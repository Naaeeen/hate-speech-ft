import unittest
from unittest.mock import patch

from src.colab.experiment_launcher import ExperimentLauncher
from src.experiments.registry import REPO_ROOT


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


if __name__ == "__main__":
    unittest.main()
