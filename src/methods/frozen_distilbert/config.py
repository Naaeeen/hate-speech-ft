"""Config metadata for frozen-backbone DistilBERT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.methods.transformer_config import (
    build_transformer_experiment_config,
    common_transformer_hyperparameters,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def build_hyperparameters(
    args: Any,
    precision_policy: dict[str, Any],
) -> dict[str, Any]:
    """Return the frozen-head training hyperparameters."""

    return {
        **common_transformer_hyperparameters(args, precision_policy),
        "head_learning_rate": args.head_learning_rate,
        "num_train_epochs": args.num_train_epochs,
    }

def build_experiment_config(
    args: Any,
    *,
    train_split: str,
    eval_split: str,
    train_size: int,
    eval_size: int,
    full_train_size: int,
    full_eval_size: int,
    raw_train_size: int | None,
    raw_eval_size: int | None,
    dropped_no_majority_train: int | None,
    dropped_no_majority_eval: int | None,
    test_size: int | None,
    full_test_size: int | None,
    raw_test_size: int | None,
    dropped_no_majority_test: int | None,
    trainable_params: int,
    total_params: int,
    gpu_type: str | None,
    class_weights: list[float] | None,
    precision_policy: dict[str, Any],
) -> dict[str, Any]:
    """Build resolved config metadata for one frozen-backbone run."""

    return build_transformer_experiment_config(
        args,
        repo_root=REPO_ROOT,
        hyperparameters=build_hyperparameters(args, precision_policy),
        method_fields={
            "head_learning_rate": args.head_learning_rate,
            "learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "epochs": args.num_train_epochs,
        },
        parameter_fields={
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
        train_split=train_split,
        eval_split=eval_split,
        train_size=train_size,
        eval_size=eval_size,
        full_train_size=full_train_size,
        full_eval_size=full_eval_size,
        raw_train_size=raw_train_size,
        raw_eval_size=raw_eval_size,
        dropped_no_majority_train=dropped_no_majority_train,
        dropped_no_majority_eval=dropped_no_majority_eval,
        test_size=test_size,
        full_test_size=full_test_size,
        raw_test_size=raw_test_size,
        dropped_no_majority_test=dropped_no_majority_test,
        gpu_type=gpu_type,
        class_weights=class_weights,
        precision_policy=precision_policy,
    )
