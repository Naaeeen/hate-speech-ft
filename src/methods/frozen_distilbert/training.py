from __future__ import annotations

import argparse
from collections.abc import Iterable

from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    parse_wandb_tags,
)


def is_classification_head_parameter(name: str) -> bool:
    return any(part in {"pre_classifier", "classifier", "score"} for part in name.split("."))


def _iter_backbone_modules(model) -> Iterable:
    """Yield backbone modules that should stay in eval mode while the head trains."""

    seen = set()
    for module in (
        getattr(model, "base_model", None),
        getattr(model, getattr(model, "base_model_prefix", ""), None),
    ):
        if module is None or module is model:
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield module


def _set_backbone_eval_mode(model) -> None:
    for backbone_module in _iter_backbone_modules(model):
        backbone_module.eval()


def keep_frozen_backbone_in_eval_mode(model) -> None:
    """Keep frozen backbone dropout behavior stable during head-only training."""

    if getattr(model, "_frozen_backbone_train_patch_applied", False):
        _set_backbone_eval_mode(model)
        return

    original_train = model.train

    def train_with_frozen_backbone(mode: bool = True):
        result = original_train(mode)
        _set_backbone_eval_mode(model)
        return result

    model.train = train_with_frozen_backbone
    model._frozen_backbone_train_patch_applied = True
    _set_backbone_eval_mode(model)


def set_frozen_backbone_trainability(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_classification_head_parameter(name)
    keep_frozen_backbone_in_eval_mode(model)


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
