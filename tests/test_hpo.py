import unittest

from src.experiments.hpo import (
    build_config_hash,
    build_seed_run_overrides,
    build_trial_overrides,
    default_search_space_name,
    enumerate_search_space,
    get_time_cap_gpu_hours,
    get_trial_cap,
    get_search_space,
    load_hpo_config,
    merge_trial_overrides,
    sample_search_space,
    shared_fixed_command_overrides,
)


class HpoTests(unittest.TestCase):
    def test_default_search_space_name_matches_method_ids(self):
        self.assertEqual(default_search_space_name("full-ft"), "full_ft")
        self.assertEqual(default_search_space_name("tfidf-logreg"), "tfidf_logreg")

    def test_sample_search_space_is_deterministic_and_applies_lora_rule(self):
        space = {
            "learning_rate": [1e-5, 2e-5],
            "lora_r": [4, 8],
            "lora_alpha_rule": "alpha = 2 * r",
        }

        first = sample_search_space(space, seed=42, trial_index=0)
        second = sample_search_space(space, seed=42, trial_index=0)

        self.assertEqual(first, second)
        self.assertEqual(first["lora_alpha"], 2 * first["lora_r"])

    def test_sample_search_space_applies_stage1_lora_rule(self):
        space = {
            "stage1_learning_rate": [1e-4],
            "stage1_lora_r": [4, 8],
            "stage1_lora_alpha_rule": "alpha = 2 * r",
        }

        trial = sample_search_space(space, seed=42, trial_index=0)

        self.assertEqual(trial["stage1_lora_alpha"], 2 * trial["stage1_lora_r"])

    def test_build_trial_overrides_adds_tracking_fields(self):
        trials = build_trial_overrides(
            base_experiment_id="distilbert_full_tuning",
            method="full-ft",
            search_space={"learning_rate": [1e-5, 2e-5]},
            n_trials=2,
            hpo_seed=123,
            output_root="outputs/hpo",
            fixed_overrides={"optim": "adamw_torch"},
        )

        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0]["search_stage"], "tuning")
        self.assertEqual(trials[0]["hpo_seed"], 123)
        self.assertEqual(trials[0]["optim"], "adamw_torch")
        self.assertIn("config_hash", trials[0])
        self.assertNotEqual(trials[0]["learning_rate"], trials[1]["learning_rate"])
        self.assertIn("trial001", trials[0]["trial_id"])
        self.assertTrue(trials[0]["output_dir"].endswith(trials[0]["trial_id"]))

    def test_build_trial_overrides_enforces_caps(self):
        with self.assertRaises(ValueError):
            build_trial_overrides(
                base_experiment_id="distilbert_full_tuning",
                method="full-ft",
                search_space={"learning_rate": [1e-5, 2e-5]},
                n_trials=3,
                hpo_seed=123,
                output_root="outputs/hpo",
                trial_cap=2,
            )

        trials = build_trial_overrides(
            base_experiment_id="distilbert_full_tuning",
            method="full-ft",
            search_space={"learning_rate": [1e-5, 2e-5]},
            n_trials=3,
            hpo_seed=123,
            output_root="outputs/hpo",
            trial_cap=2,
            time_cap_gpu_hours=1.5,
            allow_over_cap=True,
        )
        self.assertEqual(len(trials), 3)
        self.assertEqual(trials[0]["hpo_trial_cap"], 2)
        self.assertEqual(trials[0]["hpo_time_cap_gpu_hours"], 1.5)

    def test_build_seed_run_overrides_forces_full_data_without_sample_caps(self):
        runs = build_seed_run_overrides(
            base_experiment_id="distilbert_full_tuning",
            method="full-ft",
            seeds=[42],
            output_root="outputs/final",
            search_stage="final",
        )

        self.assertEqual(runs[0]["data_fraction"], 1.0)
        self.assertIsNone(runs[0]["max_train_samples"])
        self.assertIsNone(runs[0]["max_eval_samples"])
        self.assertIsNone(runs[0]["max_test_samples"])
        self.assertIs(runs[0]["run_test"], True)

    def test_enumerate_search_space_expands_combinations(self):
        combinations = enumerate_search_space(
            {
                "learning_rate": [1e-5, 2e-5],
                "lora_r": [4],
                "lora_alpha_rule": "alpha = 2 * r",
            }
        )

        self.assertEqual(len(combinations), 2)
        self.assertEqual({item["lora_alpha"] for item in combinations}, {8})

    def test_enumerate_search_space_applies_stage1_lora_rule(self):
        combinations = enumerate_search_space(
            {
                "stage1_learning_rate": [1e-4, 2e-4],
                "stage1_lora_r": [4],
                "stage1_lora_alpha_rule": "alpha = 2 * r",
            }
        )

        self.assertEqual(len(combinations), 2)
        self.assertEqual({item["stage1_lora_alpha"] for item in combinations}, {8})

    def test_build_config_hash_is_stable(self):
        first = build_config_hash({"b": 2, "a": 1})
        second = build_config_hash({"a": 1, "b": 2})

        self.assertEqual(first, second)

    def test_config_hash_payload_excludes_run_identity_seed(self):
        from src.experiments.hpo import build_config_hash_payload

        payload = build_config_hash_payload(
            {
                "learning_rate": 2e-5,
                "seed": 42,
                "run_test": True,
                "trial_id": "trial001",
                "output_dir": "outputs/trial001",
            }
        )

        self.assertEqual(payload, {"learning_rate": 2e-5})

    def test_merge_trial_overrides_lets_user_override_global_switches_and_rehashes(self):
        trial = {
            "learning_rate": 2e-5,
            "mixed_precision": "none",
            "gradient_checkpointing": False,
            "search_stage": "tuning",
            "trial_id": "trial001",
            "hpo_seed": 42,
            "output_dir": "outputs/hpo/trial001",
            "config_hash": "oldhash",
        }

        merged = merge_trial_overrides(
            base_args={"model_name": "distilbert-base-uncased"},
            user_overrides={"mixed_precision": "bf16", "gradient_checkpointing": True},
            trial_overrides=trial,
        )

        self.assertEqual(merged["mixed_precision"], "bf16")
        self.assertIs(merged["gradient_checkpointing"], True)
        self.assertNotEqual(merged["config_hash"], "oldhash")
        self.assertEqual(merged["trial_id"], "trial001")

    def test_merge_trial_overrides_rejects_identity_overrides(self):
        with self.assertRaises(ValueError):
            merge_trial_overrides(
                base_args={},
                user_overrides={"output_dir": "outputs/manual"},
                trial_overrides={"trial_id": "trial001"},
            )
        with self.assertRaises(ValueError):
            merge_trial_overrides(
                base_args={},
                user_overrides={"config_hash": "manual"},
                trial_overrides={"trial_id": "trial001"},
            )
        with self.assertRaises(ValueError):
            merge_trial_overrides(
                base_args={},
                user_overrides={"seed": 777},
                trial_overrides={"trial_id": "trial001"},
            )
        with self.assertRaises(ValueError):
            merge_trial_overrides(
                base_args={},
                user_overrides={"hpo_time_cap_gpu_hours": 9.0},
                trial_overrides={"trial_id": "trial001"},
            )
        with self.assertRaises(ValueError):
            merge_trial_overrides(
                base_args={},
                user_overrides={"data_fraction": 0.2},
                trial_overrides={"trial_id": "trial001"},
                protected_user_override_keys={
                    "data_fraction",
                },
            )

    def test_overwrite_output_dir_does_not_change_config_hash(self):
        trial = {
            "learning_rate": 2e-5,
            "search_stage": "tuning",
            "trial_id": "trial001",
            "hpo_seed": 42,
            "output_dir": "outputs/hpo/trial001",
        }

        without_overwrite = merge_trial_overrides(
            base_args={"model_name": "distilbert-base-uncased"},
            user_overrides={},
            trial_overrides=trial,
        )
        with_overwrite = merge_trial_overrides(
            base_args={"model_name": "distilbert-base-uncased"},
            user_overrides={"overwrite_output_dir": True},
            trial_overrides=trial,
        )

        self.assertEqual(
            without_overwrite["config_hash"],
            with_overwrite["config_hash"],
        )
        self.assertIs(with_overwrite["overwrite_output_dir"], True)

    def test_get_search_space_reports_available_names(self):
        config = {"search_spaces": {"full_ft": {"learning_rate": [2e-5]}}}

        self.assertEqual(get_search_space(config, "full_ft"), {"learning_rate": [2e-5]})
        with self.assertRaises(KeyError):
            get_search_space(config, "missing")

    def test_project_search_space_config_loads(self):
        config = load_hpo_config()

        self.assertIn("shared_fixed", config)
        self.assertIn("trial_caps", config)
        self.assertIn("full_ft", config["search_spaces"])
        self.assertEqual(config["shared_fixed"]["class_weighting"], "none")
        self.assertEqual(config["shared_fixed"]["optim"], "adamw_torch")
        self.assertEqual(get_trial_cap(config, "full_ft"), 6)
        self.assertEqual(get_time_cap_gpu_hours(config, "full_ft"), 2.0)
        self.assertEqual(
            shared_fixed_command_overrides(config)["mixed_precision"],
            "none",
        )


if __name__ == "__main__":
    unittest.main()
