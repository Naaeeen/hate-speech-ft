"""Print-only HPO suggestion helper for manual experiments.

This file prints random-search suggestions only. It does not train models,
create output directories, or aggregate results. Copy one printed trial into
the matching `manual_config.py` when you want to try it.
"""

from __future__ import annotations

import json
import random
from itertools import product
from typing import Any


# Edit these values, then run this file. It only prints sampled hyperparameters.
HPO_SEED = 42
METHODS = [
    "bilstm",
    "efficient-head-ft",
    "frozen-backbone",
    "full-ft",
    "lora",
    "lp-ft",
    "tfidf-logreg",
]

TRIAL_CAPS = {
    "tfidf-logreg": 24,
    "bilstm": 20,
    "frozen-backbone": 4,
    "full-ft": 4,
    "lora": 18,
    "lp-ft": 9,
    "efficient-head-ft": 10,
}

SEARCH_SPACES = {
    "tfidf-logreg": {
        "ngram_range": [[1, 1], [1, 2], [1, 3]],
        "min_df": [1, 2, 5],
        "max_df": [0.9, 1.0],
        "max_features": [20000, 50000, 100000],
        "sublinear_tf": [False, True],
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "bilstm": {
        "embedding_size": [100, 200],
        "hidden_size": [128, 256],
        "dropout": [0.1, 0.3, 0.5],
        "learning_rate": [0.0003, 0.001, 0.003],
    },
    "frozen-backbone": {"head_learning_rate": [0.0001, 0.0003, 0.001, 0.003]},
    "full-ft": {"learning_rate": [0.00001, 0.00002, 0.00003, 0.00005]},
    "lora": {
        "target_modules": [["q_lin", "v_lin"], ["q_lin", "k_lin", "v_lin", "out_lin"]],
        "lora_r": [4, 8, 16],
        "learning_rate": [0.00005, 0.0001, 0.0002, 0.0003],
    },
    "lp-ft": {
        "stage1_head_learning_rate": [0.0001, 0.0003, 0.001],
        "stage2_learning_rate": [0.00001, 0.00002, 0.00003],
    },
    "efficient-head-ft": {
        "stage1_lora_r": [4, 8],
        "stage1_learning_rate": [0.0001, 0.0002, 0.0003],
        "stage2_learning_rate": [0.00001, 0.00002, 0.00003],
    },
}


def sample_hpo_trials(
    *,
    methods: list[str] | None = None,
    seed: int = HPO_SEED,
    trial_caps: dict[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Sample the copyable config updates for each method."""

    selected_methods = methods or METHODS
    caps = trial_caps or TRIAL_CAPS
    return {
        method: _sample_method_trials(method, seed=seed, trial_cap=caps[method])
        for method in selected_methods
    }


def build_hpo_trial_report(
    *,
    methods: list[str] | None = None,
    seed: int = HPO_SEED,
    trial_caps: dict[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build the JSON object printed for Colab/manual runs."""

    return {
        method: [
            {
                "trial_number": index,
                "manual_config_updates": sampled,
            }
            for index, sampled in enumerate(sampled_trials, start=1)
        ]
        for method, sampled_trials in sample_hpo_trials(
            methods=methods,
            seed=seed,
            trial_caps=trial_caps,
        ).items()
    }


def print_hpo_trials() -> None:
    """Print the copyable HPO suggestions and stop."""

    print(json.dumps(build_hpo_trial_report(), indent=2, sort_keys=True))


def _sample_method_trials(
    method: str,
    *,
    seed: int,
    trial_cap: int,
) -> list[dict[str, Any]]:
    if method not in SEARCH_SPACES:
        raise KeyError(f"Unknown HPO method: {method}")
    if trial_cap < 1:
        raise ValueError("trial_cap must be >= 1")

    candidates = _all_method_candidates(method)
    if trial_cap > len(candidates):
        raise ValueError(
            f"trial_cap={trial_cap} exceeds the {len(candidates)} unique "
            f"candidate(s) in the {method} search space."
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:trial_cap]


def _all_method_candidates(method: str) -> list[dict[str, Any]]:
    """Enumerate the search grid before the seeded shuffle."""

    space = SEARCH_SPACES[method]
    keys = list(space)
    candidates = []
    for values in product(*(space[key] for key in keys)):
        sampled = dict(zip(keys, values, strict=True))
        if method == "lora":
            sampled["lora_alpha"] = sampled["lora_r"]
        if method == "efficient-head-ft":
            sampled["stage1_lora_alpha"] = sampled["stage1_lora_r"]
        candidates.append(sampled)
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate candidate dictionaries while preserving grid order."""

    unique = []
    seen = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


if __name__ == "__main__":
    print_hpo_trials()
