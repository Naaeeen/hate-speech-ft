import unittest

from src.experiments.hpo import (
    build_config_hash,
    build_seed_run_overrides,
    build_trial_overrides,
    default_search_space_name,
    enumerate_search_space,
    get_config_hash_keys,
    get_time_cap_gpu_hours,
    get_trial_cap,
    get_search_space,
    load_hpo_config,
    merge_trial_overrides,
    sample_search_space,
    shared_fixed_command_overrides,
    validate_hpo_base_stage,
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
        self.assertIn("hpo123", trials[0]["trial_id"])
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
            search_space={"learning_rate": [1e-5, 2e-5, 3e-5]},
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

    def test_build_trial_overrides_rejects_duplicate_config_requests(self):
        with self.assertRaisesRegex(ValueError, "unique configuration"):
            build_trial_overrides(
                base_experiment_id="distilbert_full_tuning",
                method="full-ft",
                search_space={"learning_rate": [1e-5, 2e-5]},
                n_trials=3,
                hpo_seed=123,
                output_root="outputs/hpo",
            )

    def test_validate_hpo_base_stage_rejects_quick_and_final(self):
        validate_hpo_base_stage("tuning")
        validate_hpo_base_stage("smoke", allow_smoke_hpo=True)

        with self.assertRaisesRegex(ValueError, "tuning experiment"):
            validate_hpo_base_stage("quick")
        with self.assertRaisesRegex(ValueError, "tuning experiment"):
            validate_hpo_base_stage("final")
        with self.assertRaisesRegex(ValueError, "tuning experiment"):
            validate_hpo_base_stage("smoke")

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

    def test_config_hash_payload_canonicalizes_ngram_range(self):
        from src.experiments.hpo import build_config_hash_payload

        self.assertEqual(
            build_config_hash_payload({"ngram_range": "1,2"}),
            {"ngram_range": [1, 2]},
        )
        self.assertEqual(
            build_config_hash_payload({"ngram_range": "[1,2]"}),
            {"ngram_range": [1, 2]},
        )
        self.assertEqual(
            build_config_hash_payload({"C": 10}),
            {"C": 10.0},
        )

    def test_config_hash_payload_can_use_method_effective_keys(self):
        from src.experiments.hpo import build_config_hash_payload

        payload = build_config_hash_payload(
            {
                "ngram_range": "1,2",
                "C": 1,
                "min_df": 2,
                "optim": "adamw_torch",
                "warmup_ratio": 0.06,
            },
            hash_keys=["ngram_range", "C", "min_df"],
        )

        self.assertEqual(
            payload,
            {"ngram_range": [1, 2], "C": 1.0, "min_df": 2},
        )

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
        self.assertTrue(merged["trial_id"].startswith("trial001__"))
        self.assertTrue(merged["output_dir"].endswith(merged["trial_id"]))
        self.assertIn(merged["config_hash"], merged["trial_id"])

    def test_merge_trial_overrides_isolates_seed_run_output_by_config_hash(self):
        first = merge_trial_overrides(
            base_args={"model_name": "distilbert-base-uncased"},
            user_overrides={"learning_rate": 1e-5},
            trial_overrides={
                "search_stage": "final",
                "trial_id": "bilstm_tuning__bilstm__final_seed42",
                "seed": 42,
                "run_test": True,
                "output_dir": "outputs/final/bilstm_tuning__bilstm__final_seed42",
            },
        )
        second = merge_trial_overrides(
            base_args={"model_name": "distilbert-base-uncased"},
            user_overrides={"learning_rate": 2e-5},
            trial_overrides={
                "search_stage": "final",
                "trial_id": "bilstm_tuning__bilstm__final_seed42",
                "seed": 42,
                "run_test": True,
                "output_dir": "outputs/final/bilstm_tuning__bilstm__final_seed42",
            },
        )

        self.assertNotEqual(first["config_hash"], second["config_hash"])
        self.assertNotEqual(first["trial_id"], second["trial_id"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertIn(first["config_hash"], first["output_dir"])
        self.assertIn(second["config_hash"], second["output_dir"])

    def test_merge_trial_overrides_hash_ignores_irrelevant_shared_defaults(self):
        trial = {
            "ngram_range": [1, 2],
            "C": 1.0,
            "min_df": 2,
            "search_stage": "tuning",
            "trial_id": "trial001",
            "output_dir": "outputs/hpo/trial001",
        }
        hash_keys = ["ngram_range", "C", "min_df", "max_features"]

        first = merge_trial_overrides(
            base_args={"max_features": 50000},
            user_overrides={},
            trial_overrides={**trial, "optim": "adamw_torch"},
            hash_keys=hash_keys,
        )
        second = merge_trial_overrides(
            base_args={"max_features": 50000},
            user_overrides={},
            trial_overrides={**trial, "optim": "sgd", "warmup_ratio": 0.5},
            hash_keys=hash_keys,
        )

        self.assertEqual(first["config_hash"], second["config_hash"])

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

    def test_merge_trial_overrides_rejects_hash_unsafe_alias_overrides(self):
        with self.assertRaisesRegex(ValueError, "mixed_precision=fp16"):
            merge_trial_overrides(
                base_args={"method": "full-ft"},
                user_overrides={"fp16": True},
                trial_overrides={"trial_id": "trial001"},
                hash_keys=["method", "mixed_precision"],
            )

        with self.assertRaisesRegex(ValueError, "per_device_train_batch_size"):
            merge_trial_overrides(
                base_args={"method": "lp-ft"},
                user_overrides={"batch_size": 16},
                trial_overrides={"trial_id": "trial001"},
                hash_keys=[
                    "method",
                    "per_device_train_batch_size",
                    "per_device_eval_batch_size",
                ],
            )

        merged = merge_trial_overrides(
            base_args={"method": "bilstm"},
            user_overrides={"batch_size": 64},
            trial_overrides={"trial_id": "trial001"},
            hash_keys=["method", "batch_size"],
        )
        self.assertEqual(merged["batch_size"], 64)

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
        self.assertIn("config_hash_keys", config)
        self.assertIn("full_ft", config["search_spaces"])
        self.assertEqual(config["shared_fixed"]["class_weighting"], "none")
        self.assertEqual(config["shared_fixed"]["optim"], "adamw_torch")
        self.assertEqual(get_trial_cap(config, "full_ft"), 3)
        self.assertEqual(get_time_cap_gpu_hours(config, "full_ft"), 2.0)
        self.assertIn("learning_rate", get_config_hash_keys(config, "full_ft"))
        self.assertIn("ngram_range", get_config_hash_keys(config, "tfidf_logreg"))
        self.assertEqual(
            shared_fixed_command_overrides(config)["mixed_precision"],
            "none",
        )

    def test_lora_final_defaults_are_hpo_reachable(self):
        from src.experiments.registry import load_experiment_registry

        hpo_config = load_hpo_config()
        final_args = load_experiment_registry().get("distilbert_lora_final_seed42").args
        keys = [
            "target_modules",
            "modules_to_save",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "learning_rate",
        ]

        self.assertIn(
            {key: final_args[key] for key in keys},
            [{key: combo[key] for key in keys} for combo in enumerate_search_space(
                hpo_config["search_spaces"]["lora"]
            )],
        )

    def test_efficient_head_final_defaults_are_hpo_reachable(self):
        from src.experiments.registry import load_experiment_registry

        hpo_config = load_hpo_config()
        final_args = load_experiment_registry().get(
            "distilbert_efficient_head_final_seed42"
        ).args
        keys = [
            "stage1_target_modules",
            "stage1_modules_to_save",
            "stage1_learning_rate",
            "stage1_epochs",
            "stage1_lora_r",
            "stage1_lora_alpha",
            "stage1_lora_dropout",
            "stage2_learning_rate",
            "stage2_epochs",
        ]

        self.assertIn(
            {key: final_args[key] for key in keys},
            [{key: combo[key] for key in keys} for combo in enumerate_search_space(
                hpo_config["search_spaces"]["efficient_head_ft"]
            )],
        )


if __name__ == "__main__":
    unittest.main()
