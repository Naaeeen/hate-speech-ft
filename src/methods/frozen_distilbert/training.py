from __future__ import annotations

import argparse

from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    parse_wandb_tags,
)


def is_classification_head_parameter(name: str) -> bool:
    return any(part in {"pre_classifier", "classifier", "score"} for part in name.split("."))


def set_frozen_backbone_trainability(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_classification_head_parameter(name)


def resolve_wandb_settings(args: argparse.Namespace) -> WandbSettings:
    run_name = args.wandb_run_name or build_wandb_run_name(
        method=args.method,
        model_name=args.model_name,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.head_learning_rate,
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
