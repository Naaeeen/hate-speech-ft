from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Mapping

from src.utils.wandb_config import VALID_WANDB_LOG_MODEL_VALUES, VALID_WANDB_MODES


RUN_ARTIFACT_NAMES = {
    "result_summary.json",
    "failure_summary.json",
    "metrics.json",
    "runtime.json",
    "resolved_config.json",
    "eval_predictions.json",
    "test_predictions.json",
    "trainer_state.json",
    "config.json",
    "training_args.bin",
    "model.pt",
    "finalmodel.pt",
    "model.safetensors",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "model.joblib",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "added_tokens.json",
    "tokenizer",
    "stage1_linear_probe",
    "stage1_lora_head",
    "stage2_full_ft",
    "lp_checkpoints",
    "ft_checkpoints",
}
CHECKPOINT_PREFIX = "checkpoint-"
COMMON_DEFAULTS = {
    "method": "method-template",
    "search_stage": "smoke",
    "trial_id": "method_template_manual",
    "config_hash": None,
    "search_method": None,
    "search_space_name": None,
    "hpo_seed": None,
    "hpo_trial_cap": None,
    "hpo_time_cap_gpu_hours": None,
    "dataset_name": "Hate-speech-CNERG/hatexplain",
    "seed": 42,
    "data_fraction_seed": 42,
    "data_fraction": None,
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_test_samples": None,
    "output_dir": "outputs/method_template",
    "overwrite_output_dir": False,
    "run_test": False,
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
    "save_total_limit": 2,
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_f1_macro",
    "no_save_final_model": False,
    "mixed_precision": "none",
    "gradient_checkpointing": False,
    "class_weighting": "none",
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.001,
    "wandb_project": "hate-speech-ft",
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_group": None,
    "wandb_tags": None,
    "wandb_mode": "online",
    "wandb_log_model": "false",
    "use_wandb": False,
}


def _with_defaults(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {**COMMON_DEFAULTS, **dict(overrides or {})}


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
    defaults: Mapping[str, Any] | None = None,
) -> argparse.ArgumentParser:
    """Add shared experiment arguments without imposing method-specific logic."""

    values = _with_defaults(defaults)
    _add_argument_if_missing(parser, "--method", type=str, default=values["method"])
    _add_argument_if_missing(
        parser,
        "--search_stage",
        type=str,
        default=values["search_stage"],
    )
    _add_argument_if_missing(parser, "--trial_id", type=str, default=values["trial_id"])
    _add_argument_if_missing(parser, "--config_hash", type=str, default=values["config_hash"])
    _add_argument_if_missing(
        parser,
        "--search_method",
        type=str,
        default=values["search_method"],
    )
    _add_argument_if_missing(
        parser,
        "--search_space_name",
        type=str,
        default=values["search_space_name"],
    )
    _add_argument_if_missing(parser, "--hpo_seed", type=int, default=values["hpo_seed"])
    _add_argument_if_missing(
        parser,
        "--hpo_trial_cap",
        type=int,
        default=values["hpo_trial_cap"],
    )
    _add_argument_if_missing(
        parser,
        "--hpo_time_cap_gpu_hours",
        type=float,
        default=values["hpo_time_cap_gpu_hours"],
    )
    _add_argument_if_missing(
        parser,
        "--dataset_name",
        type=str,
        default=values["dataset_name"],
    )
    _add_argument_if_missing(parser, "--seed", type=int, default=values["seed"])
    _add_argument_if_missing(
        parser,
        "--data_fraction_seed",
        type=int,
        default=values["data_fraction_seed"],
    )
    _add_argument_if_missing(
        parser,
        "--data_fraction",
        type=float,
        default=values["data_fraction"],
    )
    _add_argument_if_missing(
        parser,
        "--max_train_samples",
        type=int,
        default=values["max_train_samples"],
    )
    _add_argument_if_missing(
        parser,
        "--max_eval_samples",
        type=int,
        default=values["max_eval_samples"],
    )
    _add_argument_if_missing(
        parser,
        "--max_test_samples",
        type=int,
        default=values["max_test_samples"],
    )
    _add_argument_if_missing(parser, "--output_dir", type=str, default=values["output_dir"])
    _add_argument_if_missing(
        parser,
        "--overwrite_output_dir",
        action="store_true",
        default=values["overwrite_output_dir"],
    )
    _add_argument_if_missing(
        parser,
        "--run_test",
        action="store_true",
        default=values["run_test"],
    )
    _add_argument_if_missing(parser, "--max_length", type=int, default=values["max_length"])
    _add_argument_if_missing(
        parser,
        "--weight_decay",
        type=float,
        default=values["weight_decay"],
    )
    _add_argument_if_missing(
        parser,
        "--warmup_ratio",
        type=float,
        default=values["warmup_ratio"],
    )
    _add_argument_if_missing(
        parser,
        "--max_grad_norm",
        type=float,
        default=values["max_grad_norm"],
    )
    _add_argument_if_missing(parser, "--optim", type=str, default=values["optim"])
    _add_argument_if_missing(
        parser,
        "--lr_scheduler_type",
        type=str,
        default=values["lr_scheduler_type"],
    )
    _add_argument_if_missing(
        parser,
        "--eval_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default=values["eval_strategy"],
    )
    _add_argument_if_missing(
        parser,
        "--save_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default=values["save_strategy"],
    )
    _add_argument_if_missing(
        parser,
        "--logging_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default=values["logging_strategy"],
    )
    _add_argument_if_missing(
        parser,
        "--logging_steps",
        type=int,
        default=values["logging_steps"],
    )
    _add_argument_if_missing(parser, "--eval_steps", type=int, default=values["eval_steps"])
    _add_argument_if_missing(parser, "--save_steps", type=int, default=values["save_steps"])
    _add_argument_if_missing(
        parser,
        "--save_total_limit",
        type=int,
        default=values["save_total_limit"],
    )
    _add_argument_if_missing(
        parser,
        "--load_best_model_at_end",
        action="store_true",
        default=values["load_best_model_at_end"],
    )
    _add_argument_if_missing(
        parser,
        "--metric_for_best_model",
        type=str,
        default=values["metric_for_best_model"],
    )
    _add_argument_if_missing(
        parser,
        "--no_save_final_model",
        action="store_true",
        default=values["no_save_final_model"],
    )
    _add_argument_if_missing(
        parser,
        "--mixed_precision",
        type=str,
        choices=["none", "fp16", "bf16"],
        default=values["mixed_precision"],
    )
    _add_argument_if_missing(
        parser,
        "--gradient_checkpointing",
        action="store_true",
        default=values["gradient_checkpointing"],
    )
    _add_argument_if_missing(
        parser,
        "--class_weighting",
        type=str,
        choices=["none", "balanced"],
        default=values["class_weighting"],
    )
    _add_argument_if_missing(
        parser,
        "--early_stopping_patience",
        type=int,
        default=values["early_stopping_patience"],
    )
    _add_argument_if_missing(
        parser,
        "--early_stopping_threshold",
        type=float,
        default=values["early_stopping_threshold"],
    )
    _add_argument_if_missing(
        parser,
        "--use_wandb",
        action="store_true",
        default=values["use_wandb"],
    )
    _add_argument_if_missing(
        parser,
        "--wandb_project",
        type=str,
        default=values["wandb_project"],
    )
    _add_argument_if_missing(
        parser,
        "--wandb_entity",
        type=str,
        default=values["wandb_entity"],
    )
    _add_argument_if_missing(
        parser,
        "--wandb_run_name",
        type=str,
        default=values["wandb_run_name"],
    )
    _add_argument_if_missing(
        parser,
        "--wandb_group",
        type=str,
        default=values["wandb_group"],
    )
    _add_argument_if_missing(parser, "--wandb_tags", type=str, default=values["wandb_tags"])
    _add_argument_if_missing(
        parser,
        "--wandb_mode",
        type=str,
        choices=sorted(VALID_WANDB_MODES),
        default=values["wandb_mode"],
    )
    _add_argument_if_missing(
        parser,
        "--wandb_log_model",
        type=str,
        choices=sorted(VALID_WANDB_LOG_MODEL_VALUES),
        default=values["wandb_log_model"],
    )
    return parser


def build_global_switches(args: argparse.Namespace) -> dict[str, Any]:
    class_weighting = getattr(args, "class_weighting", "none")
    return {
        "mixed_precision": getattr(args, "mixed_precision", "none"),
        "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
        "class_weighting": class_weighting,
        "weighted_ce": class_weighting == "balanced",
        "early_stopping": int(getattr(args, "early_stopping_patience", 0)) > 0,
    }


def build_training_policy(
    args: argparse.Namespace,
    *,
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "optim": getattr(args, "optim", None),
        "lr_scheduler_type": getattr(args, "lr_scheduler_type", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "warmup_ratio": getattr(args, "warmup_ratio", None),
        "max_grad_norm": getattr(args, "max_grad_norm", None),
        "mixed_precision": getattr(args, "mixed_precision", "none"),
        "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
        "class_weighting": getattr(args, "class_weighting", "none"),
        "class_weights": class_weights,
    }


def build_checkpoint_policy(args: argparse.Namespace) -> dict[str, Any]:
    load_best = bool(getattr(args, "load_best_model_at_end", False))
    return {
        "eval_strategy": getattr(args, "eval_strategy", None),
        "save_strategy": getattr(args, "save_strategy", None),
        "logging_strategy": getattr(args, "logging_strategy", None),
        "logging_steps": getattr(args, "logging_steps", None),
        "eval_steps": getattr(args, "eval_steps", None),
        "save_steps": getattr(args, "save_steps", None),
        "save_total_limit": getattr(args, "save_total_limit", None),
        "load_best_model_at_end": load_best,
        "metric_for_best_model": getattr(args, "metric_for_best_model", None),
        "save_final_model": not bool(getattr(args, "no_save_final_model", False)),
        "wandb_log_model": getattr(args, "wandb_log_model", "false"),
        "overwrite_output_dir": bool(getattr(args, "overwrite_output_dir", False)),
        "final_model_source": "best_checkpoint" if load_best else "last_state",
    }


def build_common_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_length": getattr(args, "max_length", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "warmup_ratio": getattr(args, "warmup_ratio", None),
        "max_grad_norm": getattr(args, "max_grad_norm", None),
        "optim": getattr(args, "optim", None),
        "lr_scheduler_type": getattr(args, "lr_scheduler_type", None),
        "mixed_precision": getattr(args, "mixed_precision", "none"),
        "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
        "class_weighting": getattr(args, "class_weighting", "none"),
        "eval_strategy": getattr(args, "eval_strategy", None),
        "save_strategy": getattr(args, "save_strategy", None),
        "logging_strategy": getattr(args, "logging_strategy", None),
        "logging_steps": getattr(args, "logging_steps", None),
        "eval_steps": getattr(args, "eval_steps", None),
        "save_steps": getattr(args, "save_steps", None),
        "save_total_limit": getattr(args, "save_total_limit", None),
        "load_best_model_at_end": bool(getattr(args, "load_best_model_at_end", False)),
        "metric_for_best_model": getattr(args, "metric_for_best_model", None),
        "save_final_model": not bool(getattr(args, "no_save_final_model", False)),
    }


def build_common_experiment_config(
    args: argparse.Namespace,
    *,
    model_name: str | None = None,
    tokenizer_name: str | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    class_weights: list[float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_model_name = (
        model_name if model_name is not None else getattr(args, "model_name", None)
    )
    resolved_tokenizer_name = (
        tokenizer_name
        if tokenizer_name is not None
        else getattr(args, "tokenizer_name", None)
    )
    config = {
        "method": getattr(args, "method", None),
        "search_stage": getattr(args, "search_stage", None),
        "trial_id": getattr(args, "trial_id", None),
        "config_hash": getattr(args, "config_hash", None),
        "search_method": getattr(args, "search_method", None),
        "search_space_name": getattr(args, "search_space_name", None),
        "hpo_seed": getattr(args, "hpo_seed", None),
        "hpo_trial_cap": getattr(args, "hpo_trial_cap", None),
        "hpo_time_cap_gpu_hours": getattr(args, "hpo_time_cap_gpu_hours", None),
        "dataset": getattr(args, "dataset_name", None),
        "model_name": resolved_model_name,
        "tokenizer_name": resolved_tokenizer_name,
        "seed": getattr(args, "seed", None),
        "data_fraction_seed": getattr(args, "data_fraction_seed", None),
        "data_fraction": getattr(args, "data_fraction", None),
        "max_train_samples": getattr(args, "max_train_samples", None),
        "max_eval_samples": getattr(args, "max_eval_samples", None),
        "max_test_samples": getattr(args, "max_test_samples", None),
        "run_test": bool(getattr(args, "run_test", False)),
        "output_dir": getattr(args, "output_dir", None),
        "global_switches": build_global_switches(args),
        "training_policy": build_training_policy(args, class_weights=class_weights),
        "checkpoint_policy": build_checkpoint_policy(args),
        "hyperparameters": {
            **build_common_hyperparameters(args),
            **dict(hyperparameters or {}),
        },
    }
    config.update(dict(extra or {}))
    return config


def find_existing_run_artifacts(output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    artifacts = []
    for child in output_path.iterdir():
        if child.name in RUN_ARTIFACT_NAMES or child.name.startswith(CHECKPOINT_PREFIX):
            artifacts.append(child)
    return artifacts


def clear_existing_run_artifacts(output_dir: str | Path) -> list[Path]:
    removed = []
    for artifact in find_existing_run_artifacts(output_dir):
        if artifact.is_dir():
            shutil.rmtree(artifact)
        else:
            artifact.unlink()
        removed.append(artifact)
    return removed


def validate_output_dir_for_run(output_dir: str | Path, *, overwrite: bool) -> None:
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_path}")

    artifacts = find_existing_run_artifacts(output_path)
    if artifacts and not overwrite:
        preview = ", ".join(path.name for path in artifacts[:5])
        raise ValueError(
            f"Output directory '{output_dir}' already contains run artifacts "
            f"({preview}). Use a unique --output_dir for a new experiment run, "
            "or pass --overwrite_output_dir only when intentionally replacing artifacts."
        )


def validate_sample_selection_args(args: argparse.Namespace) -> None:
    data_fraction = getattr(args, "data_fraction", None)
    if data_fraction is not None and not 0 < data_fraction <= 1:
        raise ValueError("--data_fraction must be in the interval (0, 1].")

    for option_name in ("max_train_samples", "max_eval_samples", "max_test_samples"):
        value = getattr(args, option_name, None)
        if value is not None and value < 1:
            raise ValueError(f"--{option_name} must be >= 1 when provided.")


def validate_test_evaluation_policy(*, search_stage: str, run_test: bool) -> None:
    if run_test and search_stage != "final":
        raise ValueError(
            "--run_test is only allowed for final-stage experiments. "
            f"Received search_stage={search_stage!r}."
        )
    if search_stage == "final" and not run_test:
        raise ValueError("Final-stage experiments must enable --run_test.")
