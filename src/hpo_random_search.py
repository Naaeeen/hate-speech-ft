"""Print-only HPO suggestion helper for manual experiments.

This is not an experiment launcher. It never trains models, never creates
output directories, and never aggregates results. It only reproduces the
historical random-search order so we can manually copy a trial into the right
`manual_config.py`. The full `historical_sampled_hparams_json` field is printed
so old `results/all/hpo_runs.csv` rows can still be checked exactly.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
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


TRANSFORMER_HPO_DEFAULTS = {
    "batch_size": 16,
    "eval_batch_size": 32,
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_test_samples": None,
    "data_fraction": 1.0,
    "max_length": 128,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "steps",
    "logging_steps": 20,
    "eval_steps": None,
    "save_steps": 500,
    "save_total_limit": 1,
    "overwrite_output_dir": False,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1_macro",
    "greater_is_better": True,
    "save_final_model": True,
    "mixed_precision": "none",
    "fp16": False,
    "bf16": False,
    "gradient_checkpointing": False,
    "class_weighting": "none",
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.001,
}

HISTORICAL_HPO_DEFAULTS = {
    "tfidf-logreg": {
        "seed": 42,
        "data_fraction": 1.0,
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_test_samples": None,
        "class_weighting": "none",
    },
    "bilstm": {
        "max_length": 128,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "class_weighting": "none",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 20,
        "eval_steps": None,
        "save_steps": 500,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1_macro",
        "save_final_model": True,
        "mixed_precision": "none",
        "gradient_checkpointing": False,
        "tokenizer_min_freq": 2,
        "max_vocab_size": 30000,
        "num_layers": 1,
        "batch_size": 64,
        "eval_batch_size": 128,
        "epochs": 10,
        "device": "auto",
    },
    "frozen-backbone": {
        **TRANSFORMER_HPO_DEFAULTS,
        "num_train_epochs": 8.0,
    },
    "full-ft": {
        **TRANSFORMER_HPO_DEFAULTS,
        "epochs": 4.0,
    },
    "lora": {
        **TRANSFORMER_HPO_DEFAULTS,
        "epochs": 4.0,
        "peft_type": "lora",
        "modules_to_save": ["pre_classifier", "classifier"],
        "lora_dropout": 0.0,
    },
    "lp-ft": {
        **TRANSFORMER_HPO_DEFAULTS,
        "stage1_epochs": 5.0,
        "stage2_epochs": 3.0,
        "total_epochs": 8.0,
    },
    "efficient-head-ft": {
        **TRANSFORMER_HPO_DEFAULTS,
        "stage1_epochs": 5.0,
        "stage1_lora": {
            "peft_type": "lora",
            "target_modules": ["q_lin", "v_lin"],
            "modules_to_save": ["pre_classifier", "classifier"],
            "lora_dropout": 0.0,
        },
        "stage2_epochs": 3.0,
        "total_epochs": 8.0,
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
    """Build the JSON object printed for teammates in Colab."""

    return {
        method: [
            {
                "trial_number": index,
                "manual_config_updates": sampled,
                "historical_sampled_hparams_json": _historical_hpo_payload(
                    method,
                    sampled,
                ),
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


def _historical_hpo_payload(method: str, sampled: dict[str, Any]) -> dict[str, Any]:
    """Expand copyable updates into the old sampled_hparams_json shape."""

    payload = deepcopy(HISTORICAL_HPO_DEFAULTS[method])
    if method == "efficient-head-ft":
        payload["stage1_learning_rate"] = sampled["stage1_learning_rate"]
        payload["stage2_learning_rate"] = sampled["stage2_learning_rate"]
        payload["stage1_lora"]["lora_r"] = sampled["stage1_lora_r"]
        payload["stage1_lora"]["lora_alpha"] = sampled["stage1_lora_alpha"]
        return payload

    payload.update(sampled)
    return payload


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
    random.Random(seed).shuffle(candidates)
    return candidates[: min(trial_cap, len(candidates))]


def _all_method_candidates(method: str) -> list[dict[str, Any]]:
    """Enumerate the search grid before the historical seeded shuffle."""

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
    return candidates


if __name__ == "__main__":
    print_hpo_trials()
