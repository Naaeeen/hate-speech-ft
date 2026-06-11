from __future__ import annotations

import importlib
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


MANUAL_CONFIG_FILES = {
    "tfidf-logreg": "src/methods/tfidf_logreg/manual_config.py",
    "bilstm": "src/methods/bilstm/manual_config.py",
    "full-ft": "src/methods/distilbert_full/manual_config.py",
    "frozen-backbone": "src/methods/frozen_distilbert/manual_config.py",
    "lora": "src/methods/distilbert_lora/manual_config.py",
    "lp-ft": "src/methods/distilbert_lp_ft/manual_config.py",
    "efficient-head-ft": "src/methods/distilbert_efficient_head/manual_config.py",
}


def _precision_policy(args):
    mixed_precision = args.mixed_precision
    return {
        "mixed_precision": mixed_precision,
        "fp16": mixed_precision == "fp16",
        "bf16": mixed_precision == "bf16",
    }


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

    def test_manual_config_files_explain_confusing_fields(self):
        common_phrases = (
            "Human-readable run label",
            "Folder where this run writes metrics.json",
            "True means also evaluate the test split",
            "W&B project/entity choose the dashboard",
            "Debug caps",
        )

        for method, filename in MANUAL_CONFIG_FILES.items():
            with self.subTest(method=method):
                text = Path(filename).read_text(encoding="utf-8")
                for phrase in common_phrases:
                    self.assertIn(phrase, text)

        for method in (
            "full-ft",
            "frozen-backbone",
            "lora",
            "lp-ft",
            "efficient-head-ft",
        ):
            with self.subTest(method=method):
                text = Path(MANUAL_CONFIG_FILES[method]).read_text(encoding="utf-8")
                self.assertIn("Hugging Face checkpoint", text)
                self.assertIn("Batch size is per device/GPU", text)

        bilstm_text = Path(MANUAL_CONFIG_FILES["bilstm"]).read_text(encoding="utf-8")
        self.assertIn("Build the word vocab from train split only", bilstm_text)

        tfidf_text = Path(MANUAL_CONFIG_FILES["tfidf-logreg"]).read_text(
            encoding="utf-8"
        )
        self.assertIn("Logistic Regression regularization strength", tfidf_text)

    def test_bilstm_manual_config_matches_new_final_reference(self):
        expected = {
            "batch_size": 64,
            "class_weighting": "none",
            "device": "auto",
            "dropout": 0.5,
            "embedding_size": 100,
            "epochs": 10,
            "eval_batch_size": 128,
            "gradient_checkpointing": False,
            "hidden_size": 256,
            "learning_rate": 0.001,
            "load_best_model_at_end": True,
            "lr_scheduler_type": "linear",
            "max_grad_norm": 1.0,
            "max_length": 128,
            "max_vocab_size": 30000,
            "metric_for_best_model": "eval_f1_macro",
            "mixed_precision": "none",
            "num_layers": 1,
            "optim": "adamw_torch",
            "save_final_model": True,
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "tokenizer_min_freq": 2,
            "warmup_ratio": 0.06,
            "weight_decay": 0.01,
        }

        args = SimpleNamespace(**BILSTM_CONFIG)
        config = bilstm_config.build_experiment_config(args)

        self.assertEqual(config["tokenizer_name"], "bilstm-word")
        self.assertEqual(config["hyperparameters"], expected)

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

if __name__ == "__main__":
    unittest.main()
