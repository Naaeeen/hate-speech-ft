from __future__ import annotations

import argparse

from src.methods.peft_utils import (
    apply_lora_to_model,
    replace_context_model,
    validate_modules_to_save_cover_classification_head,
)
from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    parse_wandb_tags,
)


def apply_lora_to_context(context, args: argparse.Namespace):
    validate_modules_to_save_cover_classification_head(
        context.model,
        args.modules_to_save,
    )
    return replace_context_model(context, apply_lora_to_model(context.model, args))


def resolve_wandb_settings(args: argparse.Namespace) -> WandbSettings:
    run_name = args.wandb_run_name or build_wandb_run_name(
        method=args.method,
        model_name=args.model_name,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
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
