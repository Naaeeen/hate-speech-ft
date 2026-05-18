import re
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

    def test_validate_protocol_cli_passes(self):
        completed = self.run_cli("--validate_protocol")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Protocol validation: PASS", completed.stdout)

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
        self.assertIn("--hpo_time_cap_gpu_hours 2", completed.stdout)

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
        self.assertIn("managed by the launcher", completed.stderr)

    def test_hpo_rejects_config_hash_override_with_clear_error(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--set",
            "config_hash=manual",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("config_hash", completed.stderr)

    def test_hpo_overwrite_output_dir_does_not_change_config_hash(self):
        base = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--python",
            "python",
        )
        overwrite = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--overwrite_output_dir",
            "--python",
            "python",
        )

        self.assertEqual(base.returncode, 0, base.stderr)
        self.assertEqual(overwrite.returncode, 0, overwrite.stderr)
        self.assertIn("--overwrite_output_dir", overwrite.stdout)
        base_hash = re.search(r"--config_hash ([0-9a-f]+)", base.stdout)
        overwrite_hash = re.search(r"--config_hash ([0-9a-f]+)", overwrite.stdout)
        self.assertIsNotNone(base_hash)
        self.assertIsNotNone(overwrite_hash)
        self.assertEqual(base_hash.group(1), overwrite_hash.group(1))

    def test_suggest_final_seed_runs_uses_final_seeds_and_test_policy(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "learning_rate=2e-5",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("--search_stage final"), 3)
        self.assertEqual(completed.stdout.count("--run_test"), 3)
        self.assertIn("--seed 42", completed.stdout)
        self.assertIn("--seed 43", completed.stdout)
        self.assertIn("--seed 44", completed.stdout)
        hashes = re.findall(r"--config_hash ([0-9a-f]+)", completed.stdout)
        self.assertEqual(len(hashes), 3)
        self.assertEqual(len(set(hashes)), 1)

    def test_suggest_final_seed_runs_use_effective_wandb_stage_tags(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "learning_rate=2e-5",
            "--use_wandb",
            "--wandb_project",
            "hate-speech-ft",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("--wandb_tags distilbert,full-ft,final"), 3)
        self.assertNotIn("--wandb_tags distilbert,full-ft,tuning", completed.stdout)
        self.assertEqual(completed.stdout.count("--wandb_group full-ft-final"), 3)

    def test_direct_final_experiment_rejects_disabling_test_evaluation(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_final_seed42",
            "--dry_run",
            "--set",
            "run_test=false",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Final-stage experiments must enable --run_test", completed.stderr)

    def test_seed_run_hash_matches_hpo_hash_for_same_fixed_config(self):
        hpo = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--python",
            "python",
        )
        final = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "learning_rate=2e-5",
            "--python",
            "python",
        )

        self.assertEqual(hpo.returncode, 0, hpo.stderr)
        self.assertEqual(final.returncode, 0, final.stderr)
        hpo_hash = re.search(r"--config_hash ([0-9a-f]+)", hpo.stdout)
        final_hash = re.search(r"--config_hash ([0-9a-f]+)", final.stdout)
        self.assertIsNotNone(hpo_hash)
        self.assertIsNotNone(final_hash)
        self.assertEqual(hpo_hash.group(1), final_hash.group(1))

    def test_suggest_seed_runs_rejects_seed_override(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "seed=777",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("seed", completed.stderr)

    def test_suggest_seed_runs_rejects_data_fraction_override(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "data_fraction=0.2",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("data_fraction", completed.stderr)

    def test_suggest_seed_runs_rejects_smoke_and_final_bases(self):
        smoke = self.run_cli(
            "--experiment",
            "distilbert_full_smoke",
            "--suggest_seed_runs",
            "final",
            "--set",
            "learning_rate=2e-5",
            "--python",
            "python",
        )
        final = self.run_cli(
            "--experiment",
            "distilbert_full_final_seed42",
            "--suggest_seed_runs",
            "confirm",
            "--set",
            "learning_rate=2e-5",
            "--python",
            "python",
        )

        self.assertEqual(smoke.returncode, 2)
        self.assertEqual(final.returncode, 2)
        self.assertIn("tuning experiment", smoke.stderr)
        self.assertIn("tuning experiment", final.stderr)

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
