from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from src.hpo_random_search import build_hpo_trial_report, sample_hpo_trials


SAMPLED_KEYS = {
    "bilstm": {"embedding_size", "hidden_size", "dropout", "learning_rate"},
    "efficient-head-ft": {
        "stage1_lora_r",
        "stage1_lora_alpha",
        "stage1_learning_rate",
        "stage2_learning_rate",
    },
    "frozen-backbone": {"head_learning_rate"},
    "full-ft": {"learning_rate"},
    "lora": {"target_modules", "lora_r", "lora_alpha", "learning_rate"},
    "lp-ft": {"stage1_head_learning_rate", "stage2_learning_rate"},
    "tfidf-logreg": {
        "ngram_range",
        "min_df",
        "max_df",
        "max_features",
        "sublinear_tf",
        "C",
    },
}


def _historical_hpo_trials(method: str) -> list[dict]:
    path = Path("results/all/hpo_runs.csv")
    if not path.exists():
        raise unittest.SkipTest("historical HPO CSV is not checked out")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == method]
    keys = SAMPLED_KEYS[method]
    trials = []
    for row in rows:
        payload = json.loads(row["sampled_hparams_json"])
        trial = {key: value for key, value in payload.items() if key in keys}
        if method == "efficient-head-ft":
            stage1_lora = payload["stage1_lora"]
            trial["stage1_lora_r"] = stage1_lora["lora_r"]
            trial["stage1_lora_alpha"] = stage1_lora["lora_alpha"]
        trials.append(trial)
    return trials


def _historical_hpo_payloads(method: str) -> list[dict]:
    path = Path("results/all/hpo_runs.csv")
    if not path.exists():
        raise unittest.SkipTest("historical HPO CSV is not checked out")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            json.loads(row["sampled_hparams_json"])
            for row in csv.DictReader(handle)
            if row["method"] == method
        ]


class HpoRandomSearchTests(unittest.TestCase):
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

    def test_sampling_order_matches_historical_hpo_rows(self):
        trials = sample_hpo_trials(seed=42)

        for method, sampled_trials in trials.items():
            with self.subTest(method=method):
                self.assertEqual(sampled_trials, _historical_hpo_trials(method))

    def test_report_includes_exact_historical_hpo_payload_shape(self):
        report = build_hpo_trial_report(seed=42)

        for method, method_report in report.items():
            with self.subTest(method=method):
                self.assertEqual(
                    [
                        trial["historical_sampled_hparams_json"]
                        for trial in method_report
                    ],
                    _historical_hpo_payloads(method),
                )
                self.assertEqual(
                    [trial["manual_config_updates"] for trial in method_report],
                    sample_hpo_trials(methods=[method], seed=42)[method],
                )


if __name__ == "__main__":
    unittest.main()
