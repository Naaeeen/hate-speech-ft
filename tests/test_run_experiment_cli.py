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
        self.assertIn("--hpo_time_cap_gpu_hours 1", completed.stdout)
        self.assertIn("--search_method random_search", completed.stdout)
        self.assertIn("--search_space_name full_ft", completed.stdout)

    def test_direct_tuning_run_records_search_provenance_without_user_override(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--dry_run",
            "--search_space",
            "full_ft",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_method catalog_run", completed.stdout)
        self.assertIn("--search_space_name full_ft", completed.stdout)

    def test_direct_run_rejects_user_search_provenance_override(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--dry_run",
            "--set",
            "search_space_name=other",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("search_space_name", completed.stderr)

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
        self.assertEqual(completed.stdout.count("--hpo_trial_cap 4"), 3)
        self.assertEqual(completed.stdout.count("--hpo_time_cap_gpu_hours 1"), 3)
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

    def test_direct_run_rejects_stage_and_test_policy_overrides(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_smoke",
            "--dry_run",
            "--set",
            "search_stage=final",
            "--set",
            "run_test=true",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("search_stage", completed.stderr)
        self.assertIn("run_test", completed.stderr)

    def test_direct_final_rejects_sample_policy_overrides(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_final_seed42",
            "--dry_run",
            "--set",
            "data_fraction=0.5",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("data_fraction", completed.stderr)

    def test_direct_final_output_dir_isolated_by_config_hash(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_final_seed42",
            "--dry_run",
            "--set",
            "learning_rate=3e-5",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        config_hash = re.search(r"--config_hash ([0-9a-f]+)", completed.stdout)
        output_dir = re.search(r"--output_dir ([^ ]+)", completed.stdout)
        trial_id = re.search(r"--trial_id ([^ ]+)", completed.stdout)
        self.assertIsNotNone(config_hash)
        self.assertIsNotNone(output_dir)
        self.assertIsNotNone(trial_id)
        self.assertIn(config_hash.group(1), output_dir.group(1))
        self.assertIn(config_hash.group(1), trial_id.group(1))

    def test_direct_tuning_output_dir_and_wandb_group_are_isolated_by_config_hash(self):
        completed = self.run_cli(
            "--experiment",
            "frozen_distilbert_tuning",
            "--dry_run",
            "--set",
            "head_learning_rate=0.0003",
            "--use_wandb",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        config_hash = re.search(r"--config_hash ([0-9a-f]+)", completed.stdout)
        output_dir = re.search(r"--output_dir ([^ ]+)", completed.stdout)
        trial_id = re.search(r"--trial_id ([^ ]+)", completed.stdout)
        self.assertIsNotNone(config_hash)
        self.assertIsNotNone(output_dir)
        self.assertIsNotNone(trial_id)
        self.assertIn(config_hash.group(1), output_dir.group(1))
        self.assertIn(config_hash.group(1), trial_id.group(1))
        self.assertIn(
            f"--wandb_group frozen-backbone-tuning-{config_hash.group(1)}",
            completed.stdout,
        )

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
        learning_rate = re.search(r"--learning_rate ([0-9.eE+-]+)", hpo.stdout)
        self.assertIsNotNone(learning_rate)

        final = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            f"learning_rate={learning_rate.group(1)}",
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

    def test_suggest_trials_rejects_quick_base(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_quick",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("tuning experiment", completed.stderr)

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

    def test_hpo_rejects_fp16_alias_override(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "full_ft",
            "--set",
            "fp16=true",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("mixed_precision=fp16", completed.stderr)

    def test_direct_final_rejects_fp16_alias_override(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_full_final_seed42",
            "--dry_run",
            "--set",
            "fp16=true",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("mixed_precision=fp16", completed.stderr)

    def test_lp_ft_seed_runs_reject_batch_size_alias_override(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_lp_ft_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "stage1_head_learning_rate=0.0001",
            "--set",
            "stage1_epochs=5",
            "--set",
            "stage2_learning_rate=2e-5",
            "--set",
            "stage2_epochs=2",
            "--set",
            "batch_size=16",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("per_device_train_batch_size", completed.stderr)

    def test_bilstm_seed_runs_allow_method_batch_size_override(self):
        completed = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "hidden_size=128",
            "--set",
            "dropout=0.3",
            "--set",
            "learning_rate=0.001",
            "--set",
            "batch_size=64",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("--batch_size 64"), 3)

    def test_lp_ft_smoke_preview_uses_two_stage_method_script(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_lp_ft_smoke",
            "--dry_run",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("src/methods/distilbert_lp_ft/train.py", completed.stdout)
        self.assertIn("--method lp-ft", completed.stdout)
        self.assertIn("--stage1_head_learning_rate 0.0001", completed.stdout)
        self.assertIn("--stage2_learning_rate 2e-05", completed.stdout)
        self.assertIn("--max_train_samples 256", completed.stdout)

    def test_lp_ft_hpo_uses_tuning_base_and_search_space(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_lp_ft_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "lp_ft",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_stage tuning", completed.stdout)
        self.assertIn("--hpo_trial_cap 9", completed.stdout)
        self.assertIn("--stage1_head_learning_rate", completed.stdout)
        self.assertIn("--stage2_learning_rate", completed.stdout)
        self.assertIn("distilbert_lp_ft_tuning__lp_ft__hpo42__trial001", completed.stdout)

    def test_tfidf_smoke_preview_uses_method_script(self):
        completed = self.run_cli(
            "--experiment",
            "tfidf_logreg_smoke",
            "--dry_run",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("src/methods/tfidf_logreg/train.py", completed.stdout)
        self.assertIn("--method tfidf-logreg", completed.stdout)
        self.assertIn("--search_stage smoke", completed.stdout)
        self.assertIn("--max_train_samples 512", completed.stdout)

    def test_tfidf_hpo_uses_tuning_base_and_parseable_ngram_range(self):
        completed = self.run_cli(
            "--experiment",
            "tfidf_logreg_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "tfidf_logreg",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_stage tuning", completed.stdout)
        self.assertIn("--hpo_trial_cap 24", completed.stdout)
        self.assertRegex(completed.stdout, r"--ngram_range \[[0-9],[0-9]\]")
        self.assertIn("tfidf_logreg_tuning__tfidf_logreg__hpo42__trial001", completed.stdout)

    def test_tfidf_final_seed_hash_matches_hpo_hash_for_same_fixed_config(self):
        hpo = self.run_cli(
            "--experiment",
            "tfidf_logreg_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "tfidf_logreg",
            "--python",
            "python",
        )
        self.assertEqual(hpo.returncode, 0, hpo.stderr)
        ngram = re.search(r"--ngram_range (\[[0-9],[0-9]\])", hpo.stdout)
        min_df = re.search(r"--min_df ([0-9]+)", hpo.stdout)
        max_df = re.search(r"--max_df ([0-9.eE+-]+)", hpo.stdout)
        max_features = re.search(r"--max_features ([0-9]+)", hpo.stdout)
        c_value = re.search(r"--C ([0-9.eE+-]+)", hpo.stdout)
        hpo_hash = re.search(r"--config_hash ([0-9a-f]+)", hpo.stdout)
        self.assertIsNotNone(ngram)
        self.assertIsNotNone(min_df)
        self.assertIsNotNone(max_df)
        self.assertIsNotNone(max_features)
        self.assertIsNotNone(c_value)
        self.assertIsNotNone(hpo_hash)
        sublinear_tf = "--sublinear_tf" in hpo.stdout

        final = self.run_cli(
            "--experiment",
            "tfidf_logreg_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            f"ngram_range={ngram.group(1)}",
            "--set",
            f"min_df={min_df.group(1)}",
            "--set",
            f"max_df={max_df.group(1)}",
            "--set",
            f"max_features={max_features.group(1)}",
            "--set",
            f"sublinear_tf={str(sublinear_tf).lower()}",
            "--set",
            f"C={c_value.group(1)}",
            "--python",
            "python",
        )

        self.assertEqual(final.returncode, 0, final.stderr)
        final_hash = re.search(r"--config_hash ([0-9a-f]+)", final.stdout)
        self.assertIsNotNone(final_hash)
        self.assertEqual(hpo_hash.group(1), final_hash.group(1))
        self.assertIn(final_hash.group(1), final.stdout)

    def test_tfidf_direct_final_hash_matches_seed_generated_final_hash(self):
        direct = self.run_cli(
            "--experiment",
            "tfidf_logreg_final_seed42",
            "--dry_run",
            "--python",
            "python",
        )
        generated = self.run_cli(
            "--experiment",
            "tfidf_logreg_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "ngram_range=[1,2]",
            "--set",
            "min_df=2",
            "--set",
            "max_df=1.0",
            "--set",
            "C=1.0",
            "--set",
            "max_features=50000",
            "--set",
            "sublinear_tf=true",
            "--python",
            "python",
        )

        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        direct_hash = re.search(r"--config_hash ([0-9a-f]+)", direct.stdout)
        generated_hash = re.search(r"--config_hash ([0-9a-f]+)", generated.stdout)
        self.assertIsNotNone(direct_hash)
        self.assertIsNotNone(generated_hash)
        self.assertEqual(direct_hash.group(1), generated_hash.group(1))

    def test_bilstm_final_seed_outputs_are_isolated_by_selected_config(self):
        first = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "hidden_size=128",
            "--set",
            "dropout=0.3",
            "--set",
            "learning_rate=0.001",
            "--python",
            "python",
        )
        second = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            "hidden_size=256",
            "--set",
            "dropout=0.3",
            "--set",
            "learning_rate=0.001",
            "--python",
            "python",
        )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        first_output = re.search(r"--output_dir ([^ ]+)", first.stdout)
        second_output = re.search(r"--output_dir ([^ ]+)", second.stdout)
        first_hash = re.search(r"--config_hash ([0-9a-f]+)", first.stdout)
        second_hash = re.search(r"--config_hash ([0-9a-f]+)", second.stdout)
        self.assertIsNotNone(first_output)
        self.assertIsNotNone(second_output)
        self.assertIsNotNone(first_hash)
        self.assertIsNotNone(second_hash)
        self.assertNotEqual(first_hash.group(1), second_hash.group(1))
        self.assertNotEqual(first_output.group(1), second_output.group(1))
        self.assertIn(first_hash.group(1), first_output.group(1))
        self.assertIn(second_hash.group(1), second_output.group(1))

    def test_bilstm_confirm_seed_outputs_are_isolated_by_selected_config(self):
        first = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_seed_runs",
            "confirm",
            "--set",
            "hidden_size=128",
            "--set",
            "dropout=0.3",
            "--set",
            "learning_rate=0.001",
            "--python",
            "python",
        )
        second = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_seed_runs",
            "confirm",
            "--set",
            "hidden_size=256",
            "--set",
            "dropout=0.3",
            "--set",
            "learning_rate=0.001",
            "--python",
            "python",
        )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout.count("--search_stage confirm"), 3)
        self.assertNotIn("--run_test", first.stdout)
        first_output = re.search(r"--output_dir ([^ ]+)", first.stdout)
        second_output = re.search(r"--output_dir ([^ ]+)", second.stdout)
        first_hash = re.search(r"--config_hash ([0-9a-f]+)", first.stdout)
        second_hash = re.search(r"--config_hash ([0-9a-f]+)", second.stdout)
        self.assertIsNotNone(first_output)
        self.assertIsNotNone(second_output)
        self.assertIsNotNone(first_hash)
        self.assertIsNotNone(second_hash)
        self.assertNotEqual(first_hash.group(1), second_hash.group(1))
        self.assertNotEqual(first_output.group(1), second_output.group(1))
        self.assertIn(first_hash.group(1), first_output.group(1))
        self.assertIn(second_hash.group(1), second_output.group(1))

    def test_bilstm_smoke_preview_uses_method_script(self):
        completed = self.run_cli(
            "--experiment",
            "bilstm_smoke",
            "--dry_run",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("src/methods/bilstm/train.py", completed.stdout)
        self.assertIn("--method bilstm", completed.stdout)
        self.assertIn("--search_stage smoke", completed.stdout)
        self.assertIn("--max_train_samples 512", completed.stdout)
        self.assertIn("--max_eval_samples 256", completed.stdout)
        self.assertIn("--weight_decay 0.01", completed.stdout)
        self.assertIn("--warmup_ratio 0.06", completed.stdout)
        self.assertIn("--max_grad_norm 1", completed.stdout)
        self.assertIn("--optim adamw_torch", completed.stdout)
        self.assertIn("--lr_scheduler_type linear", completed.stdout)

    def test_bilstm_hpo_uses_tuning_base_and_search_space(self):
        completed = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "bilstm",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_stage tuning", completed.stdout)
        self.assertIn("--hpo_trial_cap 20", completed.stdout)
        self.assertIn("--hidden_size", completed.stdout)
        self.assertIn("--dropout", completed.stdout)
        self.assertIn("--learning_rate", completed.stdout)
        self.assertIn("bilstm_tuning__bilstm__hpo42__trial001", completed.stdout)

    def test_frozen_distilbert_smoke_preview_uses_method_script(self):
        completed = self.run_cli(
            "--experiment",
            "frozen_distilbert_smoke",
            "--dry_run",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("src/methods/frozen_distilbert/train.py", completed.stdout)
        self.assertIn("--method frozen-backbone", completed.stdout)
        self.assertIn("--search_stage smoke", completed.stdout)
        self.assertIn("--head_learning_rate 0.0003", completed.stdout)
        self.assertIn("--per_device_train_batch_size 8", completed.stdout)
        self.assertIn("--max_train_samples 256", completed.stdout)

    def test_frozen_distilbert_hpo_uses_tuning_base_and_search_space(self):
        completed = self.run_cli(
            "--experiment",
            "frozen_distilbert_tuning",
            "--suggest_trials",
            "2",
            "--search_space",
            "frozen_backbone",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_stage tuning", completed.stdout)
        self.assertIn("--hpo_trial_cap 4", completed.stdout)
        self.assertIn("--head_learning_rate", completed.stdout)
        self.assertIn("--num_train_epochs", completed.stdout)
        self.assertIn(
            "frozen_distilbert_tuning__frozen_backbone__hpo42__trial001",
            completed.stdout,
        )

    def test_lora_smoke_preview_uses_peft_method_script(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_lora_smoke",
            "--dry_run",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("src/methods/distilbert_lora/train.py", completed.stdout)
        self.assertIn("--method lora", completed.stdout)
        self.assertIn("--target_modules", completed.stdout)
        self.assertIn("q_lin", completed.stdout)
        self.assertIn("v_lin", completed.stdout)
        self.assertIn("--lora_r 8", completed.stdout)
        self.assertIn("--max_train_samples 256", completed.stdout)

    def test_lora_hpo_uses_tuning_base_and_search_space(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_lora_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "lora",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_stage tuning", completed.stdout)
        self.assertIn("--hpo_trial_cap 18", completed.stdout)
        self.assertIn("--target_modules", completed.stdout)
        self.assertIn("--lora_alpha", completed.stdout)
        self.assertIn("distilbert_lora_tuning__lora__hpo42__trial001", completed.stdout)

    def test_lora_direct_final_hash_matches_seed_generated_final_hash(self):
        direct = self.run_cli(
            "--experiment",
            "distilbert_lora_final_seed42",
            "--dry_run",
            "--python",
            "python",
        )
        generated = self.run_cli(
            "--experiment",
            "distilbert_lora_tuning",
            "--suggest_seed_runs",
            "final",
            "--python",
            "python",
        )

        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        self.assertEqual(generated.stdout.count("--search_stage final"), 3)
        self.assertEqual(generated.stdout.count("--run_test"), 3)
        direct_hash = re.search(r"--config_hash ([0-9a-f]+)", direct.stdout)
        generated_hash = re.search(r"--config_hash ([0-9a-f]+)", generated.stdout)
        self.assertIsNotNone(direct_hash)
        self.assertIsNotNone(generated_hash)
        self.assertEqual(direct_hash.group(1), generated_hash.group(1))

    def test_efficient_head_smoke_preview_uses_two_stage_peft_script(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_efficient_head_smoke",
            "--dry_run",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("src/methods/distilbert_efficient_head/train.py", completed.stdout)
        self.assertIn("--method efficient-head-ft", completed.stdout)
        self.assertIn("--stage1_target_modules", completed.stdout)
        self.assertIn("q_lin", completed.stdout)
        self.assertIn("v_lin", completed.stdout)
        self.assertIn("--stage2_learning_rate 2e-05", completed.stdout)
        self.assertIn("--max_train_samples 256", completed.stdout)

    def test_efficient_head_hpo_uses_tuning_base_and_search_space(self):
        completed = self.run_cli(
            "--experiment",
            "distilbert_efficient_head_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "efficient_head_ft",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--search_stage tuning", completed.stdout)
        self.assertIn("--hpo_trial_cap 10", completed.stdout)
        self.assertIn("--stage1_lora_r", completed.stdout)
        self.assertIn("--stage1_lora_alpha", completed.stdout)
        self.assertIn("--stage2_learning_rate", completed.stdout)
        self.assertIn(
            "distilbert_efficient_head_tuning__efficient_head_ft__hpo42__trial001",
            completed.stdout,
        )

    def test_efficient_head_direct_final_hash_matches_seed_generated_final_hash(self):
        direct = self.run_cli(
            "--experiment",
            "distilbert_efficient_head_final_seed42",
            "--dry_run",
            "--python",
            "python",
        )
        generated = self.run_cli(
            "--experiment",
            "distilbert_efficient_head_tuning",
            "--suggest_seed_runs",
            "final",
            "--python",
            "python",
        )

        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        self.assertEqual(generated.stdout.count("--search_stage final"), 3)
        self.assertEqual(generated.stdout.count("--run_test"), 3)
        direct_hash = re.search(r"--config_hash ([0-9a-f]+)", direct.stdout)
        generated_hash = re.search(r"--config_hash ([0-9a-f]+)", generated.stdout)
        self.assertIsNotNone(direct_hash)
        self.assertIsNotNone(generated_hash)
        self.assertEqual(direct_hash.group(1), generated_hash.group(1))

    def test_frozen_distilbert_final_seed_hash_matches_hpo_hash(self):
        hpo = self.run_cli(
            "--experiment",
            "frozen_distilbert_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "frozen_backbone",
            "--python",
            "python",
        )
        head_learning_rate = re.search(r"--head_learning_rate ([0-9.eE+-]+)", hpo.stdout)
        self.assertIsNotNone(head_learning_rate)

        final = self.run_cli(
            "--experiment",
            "frozen_distilbert_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            f"head_learning_rate={head_learning_rate.group(1)}",
            "--python",
            "python",
        )

        self.assertEqual(hpo.returncode, 0, hpo.stderr)
        self.assertEqual(final.returncode, 0, final.stderr)
        self.assertEqual(final.stdout.count("--search_stage final"), 3)
        self.assertEqual(final.stdout.count("--run_test"), 3)
        hpo_hash = re.search(r"--config_hash ([0-9a-f]+)", hpo.stdout)
        final_hash = re.search(r"--config_hash ([0-9a-f]+)", final.stdout)
        self.assertIsNotNone(hpo_hash)
        self.assertIsNotNone(final_hash)
        self.assertEqual(hpo_hash.group(1), final_hash.group(1))

    def test_bilstm_hpo_output_paths_are_isolated_by_hpo_seed(self):
        first = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "bilstm",
            "--hpo_seed",
            "42",
            "--python",
            "python",
        )
        second = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "bilstm",
            "--hpo_seed",
            "43",
            "--python",
            "python",
        )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("__hpo42__trial001", first.stdout)
        self.assertIn("__hpo43__trial001", second.stdout)
        first_output = re.search(r"--output_dir ([^ ]+)", first.stdout)
        second_output = re.search(r"--output_dir ([^ ]+)", second.stdout)
        self.assertIsNotNone(first_output)
        self.assertIsNotNone(second_output)
        self.assertNotEqual(first_output.group(1), second_output.group(1))

    def test_bilstm_rejects_wandb_model_upload_before_method_launch(self):
        completed = self.run_cli(
            "--experiment",
            "bilstm_smoke",
            "--dry_run",
            "--use_wandb",
            "--wandb_log_model",
            "end",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("saves model artifacts locally only", completed.stderr)

    def test_tfidf_rejects_wandb_model_upload_before_method_launch(self):
        completed = self.run_cli(
            "--experiment",
            "tfidf_logreg_smoke",
            "--dry_run",
            "--use_wandb",
            "--wandb_log_model",
            "end",
            "--python",
            "python",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("saves model artifacts locally only", completed.stderr)

    def test_bilstm_final_seed_hash_matches_hpo_hash_for_same_fixed_config(self):
        hpo = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_trials",
            "1",
            "--search_space",
            "bilstm",
            "--python",
            "python",
        )
        self.assertEqual(hpo.returncode, 0, hpo.stderr)
        hidden_size = re.search(r"--hidden_size ([0-9]+)", hpo.stdout)
        dropout = re.search(r"--dropout ([0-9.eE+-]+)", hpo.stdout)
        learning_rate = re.search(r"--learning_rate ([0-9.eE+-]+)", hpo.stdout)
        hpo_hash = re.search(r"--config_hash ([0-9a-f]+)", hpo.stdout)
        self.assertIsNotNone(hidden_size)
        self.assertIsNotNone(dropout)
        self.assertIsNotNone(learning_rate)
        self.assertIsNotNone(hpo_hash)

        final = self.run_cli(
            "--experiment",
            "bilstm_tuning",
            "--suggest_seed_runs",
            "final",
            "--set",
            f"hidden_size={hidden_size.group(1)}",
            "--set",
            f"dropout={dropout.group(1)}",
            "--set",
            f"learning_rate={learning_rate.group(1)}",
            "--python",
            "python",
        )

        self.assertEqual(final.returncode, 0, final.stderr)
        self.assertEqual(final.stdout.count("--search_stage final"), 3)
        self.assertEqual(final.stdout.count("--run_test"), 3)
        final_hash = re.search(r"--config_hash ([0-9a-f]+)", final.stdout)
        self.assertIsNotNone(final_hash)
        self.assertEqual(hpo_hash.group(1), final_hash.group(1))


if __name__ == "__main__":
    unittest.main()
