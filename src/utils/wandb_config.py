"""Small W&B helper layer for the manual workflow.

W&B is useful for tracking runs, but it is only a per-run logger here. The
manual config decides whether one run logs online/offline/disabled; this file
normalizes that choice, starts one run, logs final summary payloads, and
finishes it.

For single-stage Transformer methods, Hugging Face Trainer also receives
`report_to="wandb"`, so W&B can show normal trainer step/eval logs in addition
to the final payloads. Two-stage methods disable stage-local
Trainer W&B and relay stage histories into the parent run. Online W&B errors are
left visible so the login or config can be fixed before another run.
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
        mode = normalize_wandb_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        if mode == "disabled":
            object.__setattr__(self, "enabled", False)
        object.__setattr__(self, "run_name", _clean_optional_text(self.run_name))

    @property
    def report_to(self) -> str:
        """Return the value expected by Hugging Face `TrainingArguments`."""

        return "wandb" if self.enabled else "none"


def build_wandb_settings_from_args(args: Any) -> WandbSettings:
    """Build clean W&B settings from a method's `manual_config.py` namespace."""

    mode = normalize_wandb_mode(args.wandb_mode)
    return WandbSettings(
        enabled=bool(args.use_wandb) and mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=mode,
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


def _is_wandb_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _namespaced_key(key: str, namespace: str | None) -> str:
    base = f"{namespace}/" if namespace else ""
    stage_prefix = f"{namespace}_" if namespace else ""
    if namespace and key.startswith(stage_prefix):
        key = key.removeprefix(stage_prefix)
    if key in {"step", "global_step"}:
        return f"{base}global_step" if namespace else "train/global_step"
    if key == "epoch":
        return f"{base}epoch" if namespace else "train/epoch"
    if key in {"loss", "train_loss"}:
        return f"{base}train/loss"
    if key == "learning_rate":
        return f"{base}train/learning_rate"
    if key == "grad_norm":
        return f"{base}train/grad_norm"
    if key.startswith("train_"):
        return f"{base}train/{key.removeprefix('train_')}"
    if key.startswith("eval_"):
        return f"{base}eval/{key.removeprefix('eval_')}"
    if key.startswith("test_"):
        return f"{base}test/{key.removeprefix('test_')}"
    return f"{base}{key}" if namespace else key


def namespaced_wandb_metrics(
    values: dict[str, Any] | None,
    *,
    namespace: str | None = None,
    scalar_only: bool = False,
) -> dict[str, Any]:
    """Copy common flat metric names into W&B-friendly slash namespaces."""

    if not values:
        return {}
    payload = {}
    for key, value in values.items():
        if scalar_only and not _is_wandb_scalar(value):
            continue
        payload[_namespaced_key(str(key), namespace)] = value
    return payload


def log_wandb_history(
    run,
    rows: list[dict[str, Any]],
    *,
    namespace: str | None = None,
) -> None:
    """Log ordered training-history rows to an existing W&B run."""

    if run is None:
        return
    for row in rows:
        payload = namespaced_wandb_metrics(row, namespace=namespace, scalar_only=True)
        if payload:
            run.log(payload)


def define_wandb_metric(
    run,
    name: str,
    *,
    step_metric: str | None = None,
) -> None:
    """Define a W&B metric axis when a run is active."""

    if run is None:
        return
    kwargs = {"step_metric": step_metric} if step_metric else {}
    run.define_metric(name, **kwargs)


def define_training_wandb_metrics(run) -> None:
    """Define the common single-run train/eval W&B chart axes."""

    define_wandb_metric(run, "train/global_step")
    define_wandb_metric(run, "train/loss", step_metric="train/global_step")
    define_wandb_metric(run, "train/epoch", step_metric="train/global_step")
    define_wandb_metric(run, "eval/*", step_metric="train/global_step")
    define_wandb_metric(run, "test/*", step_metric="train/global_step")


def define_stage_wandb_metrics(run, stage: str) -> None:
    """Define stage-specific chart axes for two-stage methods."""

    step_metric = f"{stage}/global_step"
    define_wandb_metric(run, step_metric)
    define_wandb_metric(run, f"{stage}/epoch", step_metric=step_metric)
    define_wandb_metric(run, f"{stage}/train/*", step_metric=step_metric)
    define_wandb_metric(run, f"{stage}/eval/*", step_metric=step_metric)


def log_wandb_trainer_history(
    run,
    trainer,
    *,
    stage: str,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    """Relay one HF Trainer's history into the parent W&B run."""

    if run is None:
        return
    state = getattr(trainer, "state", None)
    rows = list(getattr(state, "log_history", []) or [])
    log_wandb_history(run, rows, namespace=stage)

    if not extra_metrics:
        return
    payload = namespaced_wandb_metrics(
        extra_metrics,
        namespace=stage,
        scalar_only=True,
    )
    global_step = getattr(state, "global_step", None)
    epoch = getattr(state, "epoch", None)
    if global_step is not None:
        payload[f"{stage}/global_step"] = global_step
    if epoch is not None:
        payload[f"{stage}/epoch"] = epoch
    if payload:
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
