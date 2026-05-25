from __future__ import annotations

import json
from dataclasses import replace
from typing import Any


CLASSIFICATION_HEAD_PARTS = {"pre_classifier", "classifier", "score"}


def parse_module_names(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        raise ValueError("Module list must not be empty.")
    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            raw_values = json.loads(text)
        else:
            raw_values = text.split(",")
    else:
        raise TypeError(f"Module list must be a string or list, got {type(value)!r}.")

    names = [str(item).strip() for item in raw_values if str(item).strip()]
    if not names:
        raise ValueError("Module list must not be empty.")
    return names


def is_classification_head_name(name: str) -> bool:
    return any(part in CLASSIFICATION_HEAD_PARTS for part in name.split("."))


def classification_head_modules(model: Any) -> set[str]:
    modules = set()
    for name in model.state_dict():
        for part in name.split("."):
            if part in CLASSIFICATION_HEAD_PARTS:
                modules.add(part)
    return modules


def validate_modules_to_save_cover_classification_head(
    model: Any,
    modules_to_save: str | list[str] | tuple[str, ...] | None,
) -> None:
    expected_modules = classification_head_modules(model)
    if not expected_modules:
        raise ValueError("Target model has no classification-head parameters.")
    saved_modules = set(parse_module_names(modules_to_save))
    missing_modules = sorted(expected_modules - saved_modules)
    if missing_modules:
        raise ValueError(
            "stage1_modules_to_save must include every classification-head module "
            "needed for stage-2 transfer. Missing: " + ", ".join(missing_modules)
        )


def _prefixed_arg(args: Any, prefix: str, name: str) -> Any:
    return getattr(args, f"{prefix}{name}" if prefix else name)


def build_lora_config_from_args(args: Any, *, prefix: str = ""):
    from peft import LoraConfig, TaskType

    modules_to_save = parse_module_names(_prefixed_arg(args, prefix, "modules_to_save"))
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(_prefixed_arg(args, prefix, "lora_r")),
        lora_alpha=int(_prefixed_arg(args, prefix, "lora_alpha")),
        lora_dropout=float(_prefixed_arg(args, prefix, "lora_dropout")),
        target_modules=parse_module_names(_prefixed_arg(args, prefix, "target_modules")),
        modules_to_save=modules_to_save or None,
        bias="none",
    )


def apply_lora_to_model(model: Any, args: Any, *, prefix: str = ""):
    from peft import get_peft_model

    return get_peft_model(model, build_lora_config_from_args(args, prefix=prefix))


def replace_context_model(context: Any, model: Any) -> Any:
    try:
        return replace(context, model=model)
    except TypeError:
        context.model = model
        return context


def clone_state_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().clone()
    if hasattr(value, "clone"):
        return value.clone()
    return value


def extract_classification_head_state_dict(model: Any) -> dict[str, Any]:
    source_model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    head_state = {
        name: clone_state_value(value)
        for name, value in source_model.state_dict().items()
        if is_classification_head_name(name)
    }
    if not head_state:
        raise ValueError("No classification-head parameters found to transfer.")
    return head_state


def _unexpected_keys(load_result: Any) -> list[str]:
    if hasattr(load_result, "unexpected_keys"):
        return list(load_result.unexpected_keys)
    if isinstance(load_result, tuple) and len(load_result) >= 2:
        return list(load_result[1])
    return []


def load_classification_head_state_dict(model: Any, head_state: dict[str, Any]) -> None:
    if not head_state:
        raise ValueError("Cannot load an empty classification-head state dict.")
    expected_head_keys = {
        name for name in model.state_dict() if is_classification_head_name(name)
    }
    if not expected_head_keys:
        raise ValueError("Target model has no classification-head parameters.")
    missing_head_keys = sorted(expected_head_keys - set(head_state))
    if missing_head_keys:
        raise ValueError(
            "Missing classification-head keys while transferring head state: "
            + ", ".join(missing_head_keys)
        )
    load_result = model.load_state_dict(head_state, strict=False)
    unexpected = _unexpected_keys(load_result)
    if unexpected:
        raise ValueError(
            "Unexpected keys while loading classification-head state: "
            + ", ".join(unexpected)
        )
