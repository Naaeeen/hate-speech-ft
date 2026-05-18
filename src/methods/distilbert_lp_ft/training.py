from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.methods.hf_common import build_training_arguments
from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    parse_wandb_tags,
)


STAGE1_DIR_NAME = "stage1_linear_probe"
STAGE2_DIR_NAME = "stage2_full_ft"


def resolve_train_batch_size(args: argparse.Namespace) -> int:
    return int(args.batch_size or args.per_device_train_batch_size)


def resolve_eval_batch_size(args: argparse.Namespace) -> int:
    return int(args.batch_size or args.per_device_eval_batch_size)


def is_classification_head_parameter(name: str) -> bool:
    return any(part in {"pre_classifier", "classifier", "score"} for part in name.split("."))


def set_linear_probe_trainability(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_classification_head_parameter(name)


def set_full_finetune_trainability(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def build_callbacks(early_stopping_callback_cls, args: argparse.Namespace) -> list[Any]:
    if args.early_stopping_patience <= 0:
        return []
    return [
        early_stopping_callback_cls(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    ]


def resolve_wandb_settings(args: argparse.Namespace) -> WandbSettings:
    run_name = args.wandb_run_name or build_wandb_run_name(
        method=args.method,
        model_name=args.model_name,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        num_train_epochs=args.stage1_epochs + args.stage2_epochs,
        learning_rate=args.stage2_learning_rate,
        trial_id=args.trial_id,
    )
    return WandbSettings(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=run_name,
        group=args.wandb_group,
        tags=parse_wandb_tags(args.wandb_tags),
        log_model=args.wandb_log_model,
    )


def build_stage_training_arguments(
    training_args_cls,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    learning_rate: float,
    num_train_epochs: float,
    precision_policy: dict[str, Any],
    wandb_settings: WandbSettings,
):
    return build_training_arguments(
        training_args_cls,
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=resolve_train_batch_size(args),
        per_device_eval_batch_size=resolve_eval_batch_size(args),
        num_train_epochs=num_train_epochs,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        overwrite_output_dir=args.overwrite_output_dir,
        report_to=wandb_settings.report_to,
        run_name=wandb_settings.run_name if wandb_settings.enabled else None,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=not args.lower_is_better,
        fp16=precision_policy["fp16"],
        bf16=precision_policy["bf16"],
        gradient_checkpointing=args.gradient_checkpointing,
    )
