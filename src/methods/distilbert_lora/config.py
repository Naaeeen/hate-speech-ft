from __future__ import annotations

from typing import Any

from src.methods.distilbert_full.config import (
    build_experiment_config as build_base_experiment_config,
)
from src.methods.distilbert_full.config import (
    build_setup_failure_config as build_base_setup_failure_config,
)
from src.methods.peft_utils import parse_module_names


def build_lora_policy(args) -> dict[str, Any]:
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
    config["training_policy"] = {
        **config.get("training_policy", {}),
        "trainable_scope": "lora_adapters_and_classification_head",
        "peft": lora_policy,
    }
    config["hyperparameters"] = {
        **config.get("hyperparameters", {}),
        **lora_policy,
    }
    config.update(lora_policy)
    return config


def build_experiment_config(args, **kwargs):
    return _merge_lora_fields(build_base_experiment_config(args, **kwargs), args)


def build_setup_failure_config(args, **kwargs) -> dict[str, Any]:
    return _merge_lora_fields(build_base_setup_failure_config(args, **kwargs), args)
