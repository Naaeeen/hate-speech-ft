from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


VALID_WANDB_MODES = ("online", "offline", "disabled")
VALID_WANDB_LOG_MODEL_VALUES = ("false", "end", "checkpoint")


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_wandb_tags(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()

    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items.extend(value.split(","))
    else:
        for item in value:
            raw_items.extend(str(item).split(","))

    tags: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = item.strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tuple(tags)


def normalize_wandb_mode(value: str | None) -> str:
    mode = (value or "online").strip().lower()
    if mode not in VALID_WANDB_MODES:
        valid = ", ".join(VALID_WANDB_MODES)
        raise ValueError(f"Invalid W&B mode '{value}'. Expected one of: {valid}")
    return mode


def normalize_wandb_log_model(value: str | None) -> str:
    log_model = (value or "false").strip().lower()
    if log_model not in VALID_WANDB_LOG_MODEL_VALUES:
        valid = ", ".join(VALID_WANDB_LOG_MODEL_VALUES)
        raise ValueError(
            f"Invalid W&B log model value '{value}'. Expected one of: {valid}"
        )
    return log_model


def slugify_run_part(value: str, *, default: str = "run") -> str:
    text = str(value).strip().replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return text or default


def build_wandb_run_name(
    *,
    method: str,
    model_name: str,
    seed: int,
    max_train_samples: int | None,
    num_train_epochs: float,
    learning_rate: float,
) -> str:
    method_part = slugify_run_part(method, default="method")
    model_part = slugify_run_part(model_name, default="model")
    sample_part = f"train{max_train_samples}" if max_train_samples else "full"
    epoch_part = f"{num_train_epochs:g}"
    lr_part = f"{learning_rate:g}"
    return (
        f"{method_part}_{model_part}_seed{seed}_"
        f"{sample_part}_ep{epoch_part}_lr{lr_part}"
    )


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    mode: str = "online"
    run_name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    log_model: str = "false"

    def __post_init__(self) -> None:
        object.__setattr__(self, "project", _clean_optional_text(self.project))
        object.__setattr__(self, "entity", _clean_optional_text(self.entity))
        object.__setattr__(self, "mode", normalize_wandb_mode(self.mode))
        object.__setattr__(self, "run_name", _clean_optional_text(self.run_name))
        object.__setattr__(self, "group", _clean_optional_text(self.group))
        object.__setattr__(self, "tags", parse_wandb_tags(self.tags))
        object.__setattr__(
            self, "log_model", normalize_wandb_log_model(self.log_model)
        )

    @property
    def report_to(self) -> str:
        return "wandb" if self.enabled else "none"


def apply_wandb_environment(settings: WandbSettings) -> dict[str, str]:
    if not settings.enabled:
        return {}

    updates: dict[str, str] = {
        "WANDB_MODE": settings.mode,
        "WANDB_LOG_MODEL": settings.log_model,
        "WANDB_JOB_TYPE": "train",
    }
    if settings.project:
        updates["WANDB_PROJECT"] = settings.project
    if settings.entity:
        updates["WANDB_ENTITY"] = settings.entity
    if settings.run_name:
        updates["WANDB_NAME"] = settings.run_name
    if settings.group:
        updates["WANDB_RUN_GROUP"] = settings.group
    if settings.tags:
        updates["WANDB_TAGS"] = ",".join(settings.tags)

    for key, value in updates.items():
        os.environ[key] = value
    return updates


def init_wandb_run(
    settings: WandbSettings,
    *,
    config: dict[str, Any],
):
    if not settings.enabled:
        return None

    apply_wandb_environment(settings)

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested, but the 'wandb' package is not installed. "
            "Install it with 'pip install wandb' or disable --use_wandb."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "config": config,
        "job_type": "train",
        "mode": settings.mode,
    }
    if settings.project:
        init_kwargs["project"] = settings.project
    if settings.entity:
        init_kwargs["entity"] = settings.entity
    if settings.run_name:
        init_kwargs["name"] = settings.run_name
    if settings.group:
        init_kwargs["group"] = settings.group
    if settings.tags:
        init_kwargs["tags"] = list(settings.tags)

    return wandb.init(**init_kwargs)


def finish_wandb_run(run) -> None:
    if run is not None:
        run.finish()
