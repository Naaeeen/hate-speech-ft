"""Shared config-building pieces for Transformer methods.

The method packages pass their special hyperparameters in, and this file adds
the common dataset, checkpoint, precision, split-accounting, and output fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.run_metadata import get_git_commit_hash


SPLIT_ACCOUNTING_POLICY = (
    "raw_*_size is the loaded Hugging Face split size before local "
    "post-load preprocessing; dropped_no_majority_* counts only "
    "strict-majority drops performed by this code after dataset load. "
    "The HateXplain dataset builder may exclude undecided corpus posts "
    "before these splits are exposed."
)


def common_transformer_hyperparameters(
    args: Any,
    precision_policy: dict[str, Any],
) -> dict[str, Any]:
    """Collect the Transformer knobs that should appear in aggregate tables.

    The manual config uses modern HF names like `per_device_train_batch_size`,
    while the saved result JSON keeps shorter table names like `batch_size`.
    This keeps manual table rebuilding simple.
    """

    return {
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_test_samples": args.max_test_samples,
        "data_fraction": args.data_fraction,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "batch_size": args.per_device_train_batch_size,
        "eval_batch_size": args.per_device_eval_batch_size,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "logging_strategy": args.logging_strategy,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "overwrite_output_dir": args.overwrite_output_dir,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": not args.lower_is_better,
        "save_final_model": not args.no_save_final_model,
        "mixed_precision": precision_policy["mixed_precision"],
        "fp16": precision_policy["fp16"],
        "bf16": precision_policy["bf16"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "class_weighting": args.class_weighting,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
    }


def build_transformer_experiment_config(
    args: Any,
    *,
    repo_root: Path,
    hyperparameters: dict[str, Any],
    method_fields: dict[str, Any] | None = None,
    parameter_fields: dict[str, Any] | None = None,
    split_accounting_policy: str = SPLIT_ACCOUNTING_POLICY,
    **run_info: Any,
) -> dict[str, Any]:
    """Build the readable `resolved_config.json` payload for Transformer runs.

    Method packages pass in their special fields, while this shared function
    adds the dataset policy, split accounting, seed, model/tokenizer names,
    hyperparameters, and parameter/runtime context. It is intentionally just a
    metadata builder, not an experiment registry.
    """

    train_size = run_info["train_size"]
    full_train_size = run_info["full_train_size"]
    effective_train_fraction = train_size / full_train_size if full_train_size else None
    size_fields = {
        "train_size": train_size,
        "eval_size": run_info["eval_size"],
        "raw_train_size": run_info.get("raw_train_size"),
        "raw_eval_size": run_info.get("raw_eval_size"),
        "full_train_size": full_train_size,
        "full_eval_size": run_info["full_eval_size"],
        "dropped_no_majority_train": run_info.get("dropped_no_majority_train"),
        "dropped_no_majority_eval": run_info.get("dropped_no_majority_eval"),
        "test_size": run_info.get("test_size"),
        "full_test_size": run_info.get("full_test_size"),
        "raw_test_size": run_info.get("raw_test_size"),
        "dropped_no_majority_test": run_info.get("dropped_no_majority_test"),
    }
    config = {
        "method": args.method,
        "run_name": args.run_name,
        "dataset": args.dataset_name,
        "train_split": run_info["train_split"],
        "eval_split": run_info["eval_split"],
        "test_split": args.test_split_name,
        "preprocessing_policy": "join_post_tokens_strict_majority",
        "label_policy": "strict_majority_drop_no_majority",
        "split_accounting_policy": split_accounting_policy,
        "selection_metric": "f1_macro",
        "test_policy": "enabled_by_run_test",
        "model_name": args.model_name,
        "tokenizer_name": args.model_name,
        "git_commit": get_git_commit_hash(repo_root),
        "seed": args.seed,
        "data_fraction_seed": args.data_fraction_seed,
        "data_fraction": args.data_fraction,
        "effective_train_fraction": effective_train_fraction,
        "run_test": args.run_test,
        "hyperparameters": hyperparameters,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_test_samples": args.max_test_samples,
        "max_length": args.max_length,
        **(method_fields or {}),
        **size_fields,
        **(parameter_fields or {}),
        "class_weights": run_info.get("class_weights"),
        "gpu_type": run_info.get("gpu_type"),
        "output_dir": args.output_dir,
    }
    return config
