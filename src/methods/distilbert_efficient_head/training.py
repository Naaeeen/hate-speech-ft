from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.methods.distilbert_efficient_head.config import STAGE1_DIR_NAME, STAGE2_DIR_NAME
from src.methods.hf_sequence_classification import (
    build_early_stopping_callbacks,
    build_hf_training_arguments_from_args,
)
from src.methods.peft_utils import (
    apply_lora_to_model,
    extract_classification_head_state_dict,
    load_classification_head_state_dict,
    replace_context_model,
    validate_modules_to_save_cover_classification_head,
)
from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    parse_wandb_tags,
)


def apply_stage1_lora_to_context(context, args: argparse.Namespace):
    validate_modules_to_save_cover_classification_head(
        context.model,
        args.stage1_modules_to_save,
    )
    return replace_context_model(
        context,
        apply_lora_to_model(context.model, args, prefix="stage1_"),
    )


def set_full_finetune_trainability(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def build_stage2_context(stage1_model, context, args: argparse.Namespace):
    from transformers import AutoModelForSequenceClassification

    head_state = extract_classification_head_state_dict(stage1_model)
    stage2_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=context.num_labels,
        id2label=context.id2label,
        label2id=context.label2id,
    )
    if args.gradient_checkpointing:
        stage2_model.gradient_checkpointing_enable()
    load_classification_head_state_dict(stage2_model, head_state)
    set_full_finetune_trainability(stage2_model)
    return replace_context_model(context, stage2_model)


def build_callbacks(early_stopping_callback_cls, args: argparse.Namespace) -> list[Any]:
    return build_early_stopping_callbacks(early_stopping_callback_cls, args)


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
    return build_hf_training_arguments_from_args(
        training_args_cls,
        args=args,
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        precision_policy=precision_policy,
        wandb_settings=wandb_settings,
    )


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
