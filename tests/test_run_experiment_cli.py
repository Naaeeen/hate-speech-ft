import subprocess
import sys
import unittest


class RunExperimentCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "src/run_experiment.py", *args],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_hpo_set_overrides_global_switches(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--set",
            "mixed_precision=bf16",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--mixed_precision bf16", completed.stdout)
        self.assertNotIn("--mixed_precision none", completed.stdout)

    def test_hpo_rejects_output_dir_override_with_clear_error(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--set",
            "output_dir=outputs/manual",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("HPO trial identity fields", completed.stderr)

    def test_allow_smoke_hpo_marks_smoke_stage(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_smoke",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--allow_smoke_hpo",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_stage smoke", completed.stdout)
        self.assertNotIn("--search_stage tuning", completed.stdout)


if __name__ == "__main__":
    unittest.main()
