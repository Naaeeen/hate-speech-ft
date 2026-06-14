from __future__ import annotations

import json
import unittest

from src.hpo_random_search import (
    SEARCH_SPACES,
    TRIAL_CAPS,
    build_hpo_trial_report,
    sample_hpo_trials,
)


EXPECTED_TRIAL_CAPS = {
    "tfidf-logreg": 24,
    "bilstm": 20,
    "frozen-backbone": 4,
    "full-ft": 4,
    "lora": 18,
    "lp-ft": 9,
    "efficient-head-ft": 10,
}

EXPECTED_SEARCH_SPACE_KEYS = {
    "tfidf-logreg": {
        "ngram_range",
        "min_df",
        "max_df",
        "max_features",
        "sublinear_tf",
        "C",
    },
    "bilstm": {"embedding_size", "hidden_size", "dropout", "learning_rate"},
    "frozen-backbone": {"head_learning_rate"},
    "full-ft": {"learning_rate"},
    "lora": {"target_modules", "lora_r", "learning_rate"},
    "lp-ft": {"stage1_head_learning_rate", "stage2_learning_rate"},
    "efficient-head-ft": {
        "stage1_lora_r",
        "stage1_learning_rate",
        "stage2_learning_rate",
    },
}


class HpoRandomSearchTests(unittest.TestCase):
    def test_trial_caps_match_manual_hpo_budget(self):
        self.assertEqual(TRIAL_CAPS, EXPECTED_TRIAL_CAPS)

    def test_search_space_method_keys_match_manual_configs(self):
        self.assertEqual(set(SEARCH_SPACES), set(EXPECTED_TRIAL_CAPS))
        self.assertEqual(
            {method: set(space) for method, space in SEARCH_SPACES.items()},
            EXPECTED_SEARCH_SPACE_KEYS,
        )

    def test_default_report_has_unique_trials_at_each_cap(self):
        report = build_hpo_trial_report()

        for method, cap in EXPECTED_TRIAL_CAPS.items():
            trials = report[method]
            self.assertEqual(len(trials), cap)
            serialized_updates = [
                json.dumps(trial["manual_config_updates"], sort_keys=True)
                for trial in trials
            ]
            self.assertEqual(len(serialized_updates), len(set(serialized_updates)))

    def test_sampling_is_deterministic(self):
        first = sample_hpo_trials(
            methods=["bilstm"],
            seed=123,
            trial_caps={"bilstm": 5},
        )
        second = sample_hpo_trials(
            methods=["bilstm"],
            seed=123,
            trial_caps={"bilstm": 5},
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["bilstm"]), 5)

    def test_full_ft_trial_cap_covers_all_learning_rates(self):
        trials = sample_hpo_trials(
            methods=["full-ft"],
            seed=42,
            trial_caps={"full-ft": 4},
        )["full-ft"]

        rates = {config["learning_rate"] for config in trials}
        self.assertEqual(rates, {0.00001, 0.00002, 0.00003, 0.00005})
        self.assertTrue(all(set(config) == {"learning_rate"} for config in trials))

    def test_lora_alpha_follows_sampled_rank(self):
        trials = sample_hpo_trials(
            methods=["lora"],
            seed=42,
            trial_caps={"lora": 6},
        )["lora"]

        for config in trials:
            self.assertEqual(config["lora_alpha"], config["lora_r"])
            self.assertNotIn("output_dir", config)

    def test_over_cap_request_fails_instead_of_returning_duplicates(self):
        with self.assertRaises(ValueError):
            sample_hpo_trials(
                methods=["full-ft"],
                seed=42,
                trial_caps={"full-ft": 5},
            )

    def test_report_only_includes_copyable_config_updates(self):
        report = build_hpo_trial_report(
            methods=["efficient-head-ft"],
            seed=42,
            trial_caps={"efficient-head-ft": 2},
        )
        sampled = sample_hpo_trials(
            methods=["efficient-head-ft"],
            seed=42,
            trial_caps={"efficient-head-ft": 2},
        )["efficient-head-ft"]

        self.assertEqual(len(report["efficient-head-ft"]), 2)
        for index, trial in enumerate(report["efficient-head-ft"], start=1):
            self.assertEqual(set(trial), {"trial_number", "manual_config_updates"})
            self.assertEqual(trial["trial_number"], index)
            self.assertEqual(trial["manual_config_updates"], sampled[index - 1])

    def test_report_does_not_include_reference_result_payloads(self):
        report = build_hpo_trial_report(
            methods=["full-ft"],
            seed=42,
            trial_caps={"full-ft": 1},
        )

        serialized = json.dumps(report).lower()
        self.assertNotIn("hist" "orical", serialized)
        self.assertNotIn("sampled_hparams_json", serialized)


if __name__ == "__main__":
    unittest.main()
