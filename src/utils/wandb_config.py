"""Small W&B helper layer for the manual workflow.

W&B is useful for tracking runs, but it is not the experiment orchestrator
anymore. The manual config decides whether one run logs online/offline/disabled;
this file normalizes that choice, starts one run, logs our final/manual summary
payloads, and finishes it.

For single-stage Transformer methods, Hugging Face Trainer also receives
`report_to="wandb"`, so W&B can show normal trainer step/eval logs in addition
to the final payloads we log ourselves. Two-stage methods disable stage-local
Trainer W&B and log only parent-run final metrics plus the one-shot stage-1
summary payload. If online W&B fails, we let the run fail instead of writing a
special failure-summary file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


VALID_WANDB_MODES = ("online", "offline", "disabled")


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_wandb_mode(value: str | None) -> str:
    """Normalize and validate the manual W&B mode string."""

    mode = (value or "online").strip().lower()
    if mode not in VALID_WANDB_MODES:
        valid = ", ".join(VALID_WANDB_MODES)
        raise ValueError(f"Invalid W&B mode '{value}'. Expected one of: {valid}")
    return mode


@dataclass(frozen=True)
class WandbSettings:
    """Clean W&B settings copied from a method's `manual_config.py`."""

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    mode: str = "online"
    run_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project", _clean_optional_text(self.project))
        object.__setattr__(self, "entity", _clean_optional_text(self.entity))
        object.__setattr__(self, "mode", normalize_wandb_mode(self.mode))
        object.__setattr__(self, "run_name", _clean_optional_text(self.run_name))

    @property
    def report_to(self) -> str:
        """Return the value expected by Hugging Face `TrainingArguments`."""

        return "wandb" if self.enabled else "none"


def build_wandb_settings_from_args(args: Any) -> WandbSettings:
    """Build clean W&B settings from a method's `manual_config.py` namespace."""

    return WandbSettings(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=args.run_name,
    )


def apply_wandb_environment(settings: WandbSettings) -> dict[str, str]:
    """Set the W&B environment variables expected by wandb and HF Trainer."""

    if not settings.enabled:
        return {}

    updates: dict[str, str] = {
        "WANDB_MODE": settings.mode,
        "WANDB_JOB_TYPE": "train",
    }
    if settings.project:
        updates["WANDB_PROJECT"] = settings.project
    if settings.entity:
        updates["WANDB_ENTITY"] = settings.entity
    if settings.run_name:
        updates["WANDB_NAME"] = settings.run_name

    for key, value in updates.items():
        os.environ[key] = value
    return updates


def init_wandb_run(
    settings: WandbSettings,
    *,
    config: dict[str, Any],
):
    """Start one W&B run, or return None when logging is disabled."""

    if not settings.enabled:
        return None

    apply_wandb_environment(settings)
    import wandb

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

    return wandb.init(**init_kwargs)


def log_wandb(run, *payloads: dict[str, Any]) -> None:
    """Log one or more final metric payloads to an existing W&B run."""

    if run is None:
        return
    for payload in payloads:
        if not payload:
            continue
        run.log(payload)


def prefixed_wandb_scalars(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    """Flatten scalar values as `prefix/key` for W&B charts."""

    return {
        f"{prefix}/{key}": value
        for key, value in values.items()
        if value is not None and not isinstance(value, (dict, list, tuple, set))
    }


def finish_wandb_run(run) -> None:
    """Finish a W&B run when logging was enabled."""

    if run is not None:
        run.finish()
