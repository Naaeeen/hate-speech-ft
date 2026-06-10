"""Config metadata for the DistilBERT LoRA method."""

from __future__ import annotations

from typing import Any

from src.methods.distilbert_full.config import (
    build_experiment_config as build_base_experiment_config,
)
from src.methods.peft_utils import parse_module_names


def build_lora_policy(args) -> dict[str, Any]:
    """Collect the LoRA adapter fields from `manual_config.py`."""

    return {
        "peft_type": "lora",
        "target_modules": parse_module_names(args.target_modules),
        "modules_to_save": parse_module_names(args.modules_to_save),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }


def _merge_lora_fields(config: dict[str, Any], args) -> dict[str, Any]:
    lora_policy = build_lora_policy(args)
    config["hyperparameters"] = {
        **config.get("hyperparameters", {}),
        **lora_policy,
    }
    config.update(lora_policy)
    return config


def build_experiment_config(args, **kwargs):
    """Build full-FT metadata, then add LoRA-specific adapter fields."""

    return _merge_lora_fields(build_base_experiment_config(args, **kwargs), args)
