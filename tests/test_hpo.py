import unittest

from src.experiments.hpo import (
    build_config_hash,
    build_trial_overrides,
    default_search_space_name,
    enumerate_search_space,
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
            allow_over_cap=True,
        )
        self.assertEqual(len(trials), 3)

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

    def test_build_config_hash_is_stable(self):
        first = build_config_hash({"b": 2, "a": 1})
        second = build_config_hash({"a": 1, "b": 2})

        self.assertEqual(first, second)

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
        self.assertEqual(
            shared_fixed_command_overrides(config)["mixed_precision"],
            "none",
        )


if __name__ == "__main__":
    unittest.main()
