from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.methods.bilstm import config as bilstm_config
from src.methods.bilstm.config import validate_bilstm_config
from src.methods.bilstm.manual_config import CONFIG as BILSTM_CONFIG
from src.methods.distilbert_efficient_head import config as efficient_head_config
from src.methods.distilbert_efficient_head.manual_config import (
    CONFIG as EFFICIENT_HEAD_CONFIG,
)
from src.methods.distilbert_full import config as full_config
from src.methods.distilbert_full.manual_config import CONFIG as FULL_CONFIG
from src.methods.distilbert_lora import config as lora_config
from src.methods.distilbert_lora.manual_config import CONFIG as LORA_CONFIG
from src.methods.distilbert_lp_ft import config as lp_ft_config
from src.methods.distilbert_lp_ft.manual_config import CONFIG as LP_FT_CONFIG
from src.methods.frozen_distilbert import config as frozen_config
from src.methods.frozen_distilbert.manual_config import CONFIG as FROZEN_CONFIG
from src.methods.tfidf_logreg import config as tfidf_config
from src.methods.tfidf_logreg.manual_config import CONFIG as TFIDF_CONFIG
from src.methods.tfidf_logreg.training import (
    parse_ngram_range,
    validate_classical_args,
)
from src.methods.transformer_setup import validate_checkpoint_policy
from src.methods.transformer_trainer import resolve_precision_policy


COMMON_RUN_KWARGS = {
    "train_split": "train",
    "eval_split": "validation",
    "train_size": 1,
    "eval_size": 1,
    "full_train_size": 1,
    "full_eval_size": 1,
    "raw_train_size": 1,
    "raw_eval_size": 1,
    "dropped_no_majority_train": 0,
    "dropped_no_majority_eval": 0,
    "test_size": 1,
    "full_test_size": 1,
    "raw_test_size": 1,
    "dropped_no_majority_test": 0,
    "gpu_type": "T4",
    "class_weights": None,
}

METHOD_CONFIG_MODULES = {
    "bilstm": (BILSTM_CONFIG, bilstm_config),
    "efficient-head-ft": (EFFICIENT_HEAD_CONFIG, efficient_head_config),
    "frozen-backbone": (FROZEN_CONFIG, frozen_config),
    "full-ft": (FULL_CONFIG, full_config),
    "lora": (LORA_CONFIG, lora_config),
    "lp-ft": (LP_FT_CONFIG, lp_ft_config),
    "tfidf-logreg": (TFIDF_CONFIG, tfidf_config),
}

RUN_NAME_PREFIXES = {
    "bilstm": "bilstm",
    "efficient-head-ft": "distilbert_efficient_head",
    "frozen-backbone": "frozen_distilbert",
    "full-ft": "distilbert_full",
    "lora": "distilbert_lora",
    "lp-ft": "distilbert_lp_ft",
    "tfidf-logreg": "tfidf_logreg",
}


def _precision_policy(args):
    mixed_precision = args.mixed_precision
    return {
        "mixed_precision": mixed_precision,
        "fp16": mixed_precision == "fp16",
        "bf16": mixed_precision == "bf16",
    }


def _normalized(value):
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    return value


def _manual_config_for_seed(method: str, seed: int) -> dict:
    base_config, _module = METHOD_CONFIG_MODULES[method]
    run_prefix = RUN_NAME_PREFIXES[method]
    config = dict(base_config)
    config["seed"] = seed
    config["run_name"] = f"{run_prefix}_final_seed{seed}"
    config["output_dir"] = f"outputs/{run_prefix}_final_seed{seed}"
    return config


def _current_hyperparameters(method: str, args, module) -> dict:
    if method == "bilstm":
        return module.build_experiment_config(args)["hyperparameters"]
    if method == "tfidf-logreg":
        return module.build_experiment_config(
            args,
            ngram_range=tuple(args.ngram_range),
        )["hyperparameters"]
    if hasattr(module, "build_hyperparameters"):
        return module.build_hyperparameters(args, _precision_policy(args))
    return module.build_experiment_config(
        args,
        **COMMON_RUN_KWARGS,
        trainable_params=1,
        total_params=1,
        precision_policy=_precision_policy(args),
    )["hyperparameters"]


class MethodManualConfigParseTests(unittest.TestCase):
    def test_all_manual_configs_can_be_used_directly(self):
        cases = [
            ("tfidf-logreg", TFIDF_CONFIG),
            ("bilstm", BILSTM_CONFIG),
            ("full-ft", FULL_CONFIG),
            ("frozen-backbone", FROZEN_CONFIG),
            ("lora", LORA_CONFIG),
            ("lp-ft", LP_FT_CONFIG),
            ("efficient-head-ft", EFFICIENT_HEAD_CONFIG),
        ]

        for method, config in cases:
            with self.subTest(method=method):
                args = SimpleNamespace(**config)
                self.assertEqual(args.method, method)
                self.assertEqual(args.seed, 42)
                self.assertTrue(args.run_test)
                self.assertEqual(args.wandb_project, "hate-speech-ft")

    def test_manual_configs_have_required_runtime_fields(self):
        tfidf_args = SimpleNamespace(**TFIDF_CONFIG)
        validate_classical_args(tfidf_args, parse_ngram_range(tfidf_args.ngram_range))

        validate_bilstm_config(SimpleNamespace(**BILSTM_CONFIG))

        for config in (
            FULL_CONFIG,
            FROZEN_CONFIG,
            LORA_CONFIG,
            LP_FT_CONFIG,
            EFFICIENT_HEAD_CONFIG,
        ):
            with self.subTest(method=config["method"]):
                args = SimpleNamespace(**config)
                validate_checkpoint_policy(args)
                self.assertEqual(
                    resolve_precision_policy(args),
                    {"mixed_precision": "none", "fp16": False, "bf16": False},
                )
                self.assertEqual(args.per_device_train_batch_size, 16)
                self.assertEqual(args.per_device_eval_batch_size, 32)

    def test_transformer_mains_delegate_manual_config_directly_to_runner(self):
        cases = [
            (
                "src.methods.distilbert_full.train",
                "run_single_stage_transformer",
                "full-ft",
                {"learning_rate_attr": "learning_rate"},
            ),
            (
                "src.methods.frozen_distilbert.train",
                "run_single_stage_transformer",
                "frozen-backbone",
                {
                    "learning_rate_attr": "head_learning_rate",
                    "prepare_context": "prepare_frozen_context",
                },
            ),
            (
                "src.methods.distilbert_lora.train",
                "run_single_stage_transformer",
                "lora",
                {
                    "learning_rate_attr": "learning_rate",
                    "prepare_context": "apply_lora_to_context",
                },
            ),
            (
                "src.methods.distilbert_lp_ft.train",
                "run_two_stage_transformer",
                "lp-ft",
                {
                    "build_stage_plan": "build_stage_plan",
                    "stage1_learning_rate_attr": "stage1_head_learning_rate",
                    "stage2_learning_rate_attr": "stage2_learning_rate",
                },
            ),
            (
                "src.methods.distilbert_efficient_head.train",
                "run_two_stage_transformer",
                "efficient-head-ft",
                {
                    "build_stage_plan": "build_stage_plan",
                    "stage1_learning_rate_attr": "stage1_learning_rate",
                    "stage2_learning_rate_attr": "stage2_learning_rate",
                },
            ),
        ]

        for module_name, runner_name, method, expected_kwargs in cases:
            with self.subTest(method=method):
                module = importlib.import_module(module_name)
                with patch.object(module, runner_name) as runner:
                    module.main()

                runner.assert_called_once()
                args = runner.call_args.args[0]
                kwargs = runner.call_args.kwargs
                self.assertEqual(args.method, method)
                self.assertIs(kwargs["build_experiment_config"], module.build_experiment_config)
                self.assertNotIn("resolve_wandb_settings", kwargs)
                for key, expected in expected_kwargs.items():
                    actual = kwargs[key]
                    if isinstance(expected, str) and hasattr(module, expected):
                        expected = getattr(module, expected)
                    self.assertEqual(actual, expected)

    def test_manual_final_seed_configs_match_historical_selected_hparams(self):
        result_path = Path("results/all/final_runs (1).csv")
        if not result_path.exists():
            self.skipTest("historical final-runs CSV is not checked out")
        with result_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows:
            method = row["method"]
            seed = int(row["seed"])
            with self.subTest(method=method, seed=seed):
                _base_config, module = METHOD_CONFIG_MODULES[method]
                args = SimpleNamespace(**_manual_config_for_seed(method, seed))
                actual = _current_hyperparameters(method, args, module)
                expected = json.loads(row["selected_hyperparams_json"])
                mismatches = [
                    key
                    for key, value in expected.items()
                    if _normalized(actual.get(key, "<missing>")) != _normalized(value)
                ]
                self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
