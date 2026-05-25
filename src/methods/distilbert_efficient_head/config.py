from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.methods.hf_common import get_git_commit_hash
from src.methods.peft_utils import parse_module_names


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_DIR_NAME = "stage1_lora_head"
STAGE2_DIR_NAME = "stage2_full_ft"


def build_stage1_lora_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "peft_type": "lora",
        "target_modules": parse_module_names(args.stage1_target_modules),
        "modules_to_save": parse_module_names(args.stage1_modules_to_save),
        "lora_r": args.stage1_lora_r,
        "lora_alpha": args.stage1_lora_alpha,
        "lora_dropout": args.stage1_lora_dropout,
    }


def build_hyperparameters(
    args: argparse.Namespace,
    precision_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_test_samples": args.max_test_samples,
        "data_fraction": args.data_fraction,
        "max_length": args.max_length,
        "stage1_learning_rate": args.stage1_learning_rate,
        "stage1_epochs": args.stage1_epochs,
        "stage1_lora": build_stage1_lora_policy(args),
        "stage2_learning_rate": args.stage2_learning_rate,
        "stage2_epochs": args.stage2_epochs,
        "total_epochs": args.stage1_epochs + args.stage2_epochs,
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


def build_setup_failure_config(
    args: argparse.Namespace,
    *,
    precision_policy: dict[str, Any],
    gpu_type: str | None,
) -> dict[str, Any]:
    return {
        "method": args.method,
        "search_stage": args.search_stage,
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "hpo_seed": args.hpo_seed,
        "hpo_trial_cap": args.hpo_trial_cap,
        "hpo_time_cap_gpu_hours": args.hpo_time_cap_gpu_hours,
        "dataset": args.dataset_name,
        "model_name": args.model_name,
        "git_commit": get_git_commit_hash(REPO_ROOT),
        "output_dir": args.output_dir,
        "seed": args.seed,
        "setup_complete": False,
        "global_switches": {
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "weighted_ce": args.class_weighting != "none",
            "early_stopping": args.early_stopping_patience > 0,
        },
        "training_policy": {
            "method_sequence": [
                "lora_train_classification_head",
                "fresh_backbone_full_finetune_with_transferred_head",
            ],
            "stage1": build_stage1_lora_policy(args),
            "stage2": {"trainable_scope": "all_model_parameters"},
        },
        "hyperparameters": build_hyperparameters(args, precision_policy),
        "runtime_context": {"gpu_type": gpu_type},
    }


def build_experiment_config(
    args: argparse.Namespace,
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
    effective_train_fraction = train_size / full_train_size if full_train_size else None
    stage1_lora = build_stage1_lora_policy(args)
    return {
        "method": args.method,
        "search_stage": args.search_stage,
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "hpo_seed": args.hpo_seed,
        "hpo_trial_cap": args.hpo_trial_cap,
        "hpo_time_cap_gpu_hours": args.hpo_time_cap_gpu_hours,
        "dataset": args.dataset_name,
        "train_split": train_split,
        "eval_split": eval_split,
        "test_split": args.test_split_name,
        "preprocessing_policy": "join_post_tokens_strict_majority",
        "label_policy": "strict_majority_drop_no_majority",
        "split_accounting_policy": (
            "raw_*_size is the loaded Hugging Face split size before local "
            "post-load preprocessing; dropped_no_majority_* counts strict-majority "
            "drops performed by this code after dataset load."
        ),
        "selection_metric": "f1_macro",
        "test_policy": "final_only",
        "model_name": args.model_name,
        "tokenizer_name": args.model_name,
        "git_commit": get_git_commit_hash(REPO_ROOT),
        "seed": args.seed,
        "data_fraction_seed": args.data_fraction_seed,
        "data_fraction": args.data_fraction,
        "effective_train_fraction": effective_train_fraction,
        "run_test": args.run_test,
        "global_switches": {
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "weighted_ce": args.class_weighting != "none",
            "early_stopping": args.early_stopping_patience > 0,
        },
        "training_policy": {
            "method_sequence": [
                "stage1_lora_train_head",
                "stage2_fresh_backbone_full_finetune",
            ],
            "head_transfer_policy": (
                "merge stage-1 LoRA model only to extract classification-head "
                "weights; discard stage-1 backbone/adapters before stage 2"
            ),
            "optim": args.optim,
            "lr_scheduler_type": args.lr_scheduler_type,
            "max_grad_norm": args.max_grad_norm,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "class_weights": class_weights,
            "stage1": {
                "trainable_scope": "lora_adapters_and_classification_head",
                "learning_rate": args.stage1_learning_rate,
                "epochs": args.stage1_epochs,
                "trainable_params": stage1_trainable_params,
                "lora": stage1_lora,
            },
            "stage2": {
                "trainable_scope": "all_model_parameters",
                "learning_rate": args.stage2_learning_rate,
                "epochs": args.stage2_epochs,
                "trainable_params": stage2_trainable_params,
            },
        },
        "hyperparameters": build_hyperparameters(args, precision_policy),
        "checkpoint_policy": {
            "eval_strategy": args.eval_strategy,
            "save_strategy": args.save_strategy,
            "logging_strategy": args.logging_strategy,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": not args.lower_is_better,
            "save_final_model": not args.no_save_final_model,
            "final_model_source": (
                "best_stage2_checkpoint"
                if args.load_best_model_at_end
                else "last_stage2_training_state"
            ),
            "wandb_log_model": args.wandb_log_model,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "overwrite_output_dir": args.overwrite_output_dir,
            "stage1_checkpoint_dir": str(Path(args.output_dir) / STAGE1_DIR_NAME),
            "stage2_checkpoint_dir": str(Path(args.output_dir) / STAGE2_DIR_NAME),
        },
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_test_samples": args.max_test_samples,
        "max_length": args.max_length,
        "stage1_learning_rate": args.stage1_learning_rate,
        "stage1_epochs": args.stage1_epochs,
        "stage1_lora": stage1_lora,
        "stage2_learning_rate": args.stage2_learning_rate,
        "stage2_epochs": args.stage2_epochs,
        "batch_size": args.per_device_train_batch_size,
        "eval_batch_size": args.per_device_eval_batch_size,
        "train_size": train_size,
        "eval_size": eval_size,
        "raw_train_size": raw_train_size,
        "raw_eval_size": raw_eval_size,
        "full_train_size": full_train_size,
        "full_eval_size": full_eval_size,
        "dropped_no_majority_train": dropped_no_majority_train,
        "dropped_no_majority_eval": dropped_no_majority_eval,
        "test_size": test_size,
        "full_test_size": full_test_size,
        "raw_test_size": raw_test_size,
        "dropped_no_majority_test": dropped_no_majority_test,
        "stage1_trainable_params": stage1_trainable_params,
        "stage2_trainable_params": stage2_trainable_params,
        "trainable_params": stage2_trainable_params,
        "total_params": total_params,
        "gpu_type": gpu_type,
        "output_dir": args.output_dir,
    }
