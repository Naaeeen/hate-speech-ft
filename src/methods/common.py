from __future__ import annotations

import argparse
from typing import Any

from src.utils.wandb_config import VALID_WANDB_LOG_MODEL_VALUES, VALID_WANDB_MODES


def _option_exists(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def _add_argument_if_missing(
    parser: argparse.ArgumentParser,
    *flags: str,
    **kwargs: Any,
) -> None:
    if not any(_option_exists(parser, flag) for flag in flags):
        parser.add_argument(*flags, **kwargs)


def add_common_method_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_method: str = "template",
    default_search_stage: str = "smoke",
    default_trial_id: str = "manual_template",
) -> argparse.ArgumentParser:
    """Add the shared experiment contract every method script should support."""

    _add_argument_if_missing(parser, "--method", type=str, default=default_method)
    _add_argument_if_missing(
        parser,
        "--search_stage",
        type=str,
        default=default_search_stage,
    )
    _add_argument_if_missing(parser, "--trial_id", type=str, default=default_trial_id)
    _add_argument_if_missing(parser, "--config_hash", type=str, default=None)
    _add_argument_if_missing(parser, "--hpo_seed", type=int, default=None)
    _add_argument_if_missing(
        parser,
        "--dataset_name",
        type=str,
        default="Hate-speech-CNERG/hatexplain",
    )
    _add_argument_if_missing(parser, "--seed", type=int, default=42)
    _add_argument_if_missing(parser, "--data_fraction_seed", type=int, default=42)
    _add_argument_if_missing(parser, "--data_fraction", type=float, default=None)
    _add_argument_if_missing(parser, "--max_train_samples", type=int, default=None)
    _add_argument_if_missing(parser, "--max_eval_samples", type=int, default=None)
    _add_argument_if_missing(parser, "--max_test_samples", type=int, default=None)
    _add_argument_if_missing(
        parser,
        "--output_dir",
        type=str,
        default="outputs/method_template",
    )
    _add_argument_if_missing(parser, "--overwrite_output_dir", action="store_true")
    _add_argument_if_missing(parser, "--run_test", action="store_true")
    _add_argument_if_missing(parser, "--max_length", type=int, default=128)
    _add_argument_if_missing(parser, "--weight_decay", type=float, default=0.01)
    _add_argument_if_missing(parser, "--warmup_ratio", type=float, default=0.06)
    _add_argument_if_missing(parser, "--max_grad_norm", type=float, default=1.0)
    _add_argument_if_missing(parser, "--optim", type=str, default="adamw_torch")
    _add_argument_if_missing(parser, "--lr_scheduler_type", type=str, default="linear")
    _add_argument_if_missing(
        parser,
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    _add_argument_if_missing(
        parser,
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    _add_argument_if_missing(
        parser,
        "--logging_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
    )
    _add_argument_if_missing(parser, "--logging_steps", type=int, default=20)
    _add_argument_if_missing(parser, "--eval_steps", type=int, default=None)
    _add_argument_if_missing(parser, "--save_steps", type=int, default=500)
    _add_argument_if_missing(parser, "--save_total_limit", type=int, default=2)
    _add_argument_if_missing(parser, "--load_best_model_at_end", action="store_true")
    _add_argument_if_missing(
        parser,
        "--metric_for_best_model",
        type=str,
        default="eval_f1_macro",
    )
    _add_argument_if_missing(parser, "--no_save_final_model", action="store_true")
    _add_argument_if_missing(
        parser,
        "--mixed_precision",
        type=str,
        choices=["none", "fp16", "bf16"],
        default="none",
    )
    _add_argument_if_missing(parser, "--gradient_checkpointing", action="store_true")
    _add_argument_if_missing(
        parser,
        "--class_weighting",
        type=str,
        choices=["none", "balanced"],
        default="none",
    )
    _add_argument_if_missing(parser, "--early_stopping_patience", type=int, default=2)
    _add_argument_if_missing(
        parser,
        "--early_stopping_threshold",
        type=float,
        default=0.001,
    )
    _add_argument_if_missing(parser, "--use_wandb", action="store_true")
    _add_argument_if_missing(
        parser,
        "--wandb_project",
        type=str,
        default="hate-speech-ft",
    )
    _add_argument_if_missing(parser, "--wandb_entity", type=str, default=None)
    _add_argument_if_missing(parser, "--wandb_run_name", type=str, default=None)
    _add_argument_if_missing(parser, "--wandb_group", type=str, default=None)
    _add_argument_if_missing(parser, "--wandb_tags", type=str, default=None)
    _add_argument_if_missing(
        parser,
        "--wandb_mode",
        type=str,
        choices=sorted(VALID_WANDB_MODES),
        default="online",
    )
    _add_argument_if_missing(
        parser,
        "--wandb_log_model",
        type=str,
        choices=sorted(VALID_WANDB_LOG_MODEL_VALUES),
        default="false",
    )
    return parser
