"""Config metadata for Efficient-Head FT.

The main special case is the nested `stage1_lora` hyperparameter block. Saved
results keep that nested shape, while the manual config stays easy to edit with
flat fields like `stage1_lora_r`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.methods.transformer_config import (
    build_transformer_experiment_config,
    common_transformer_hyperparameters,
)
from src.methods.peft_utils import parse_module_names


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_DIR_NAME = "stage1_lora_head"
STAGE2_DIR_NAME = "stage2_full_ft"


def build_stage1_lora_policy(args: Any) -> dict[str, Any]:
    """Build the nested stage-1 LoRA policy used in saved result tables."""

    return {
        "peft_type": "lora",
        "target_modules": parse_module_names(args.stage1_target_modules),
        "modules_to_save": parse_module_names(args.stage1_modules_to_save),
        "lora_r": args.stage1_lora_r,
        "lora_alpha": args.stage1_lora_alpha,
        "lora_dropout": args.stage1_lora_dropout,
    }


def build_hyperparameters(
    args: Any,
    precision_policy: dict[str, Any],
) -> dict[str, Any]:
    """Return the Efficient-Head hyperparameters with nested stage-1 LoRA."""

    return {
        **common_transformer_hyperparameters(args, precision_policy),
        "stage1_learning_rate": args.stage1_learning_rate,
        "stage1_epochs": args.stage1_epochs,
        "stage1_lora": build_stage1_lora_policy(args),
        "stage2_learning_rate": args.stage2_learning_rate,
        "stage2_epochs": args.stage2_epochs,
        "total_epochs": args.stage1_epochs + args.stage2_epochs,
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
    stage1_trainable_params: int,
    stage2_trainable_params: int,
    total_params: int,
    gpu_type: str | None,
    class_weights: list[float] | None,
    precision_policy: dict[str, Any],
) -> dict[str, Any]:
    """Build resolved config metadata for one Efficient-Head run."""

    stage1_lora = build_stage1_lora_policy(args)
    return build_transformer_experiment_config(
        args,
        repo_root=REPO_ROOT,
        hyperparameters=build_hyperparameters(args, precision_policy),
        method_fields={
            "stage1_learning_rate": args.stage1_learning_rate,
            "stage1_epochs": args.stage1_epochs,
            "stage1_lora": stage1_lora,
            "stage2_learning_rate": args.stage2_learning_rate,
            "stage2_epochs": args.stage2_epochs,
            "batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
        },
        parameter_fields={
            "stage1_trainable_params": stage1_trainable_params,
            "stage2_trainable_params": stage2_trainable_params,
            "trainable_params": stage2_trainable_params,
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
