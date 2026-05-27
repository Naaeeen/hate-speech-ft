from __future__ import annotations

import sys
from typing import Any

from src.utils.wandb_config import WandbSettings, define_wandb_metric_best_effort


def disable_hf_wandb_reporting(settings: WandbSettings) -> WandbSettings:
    """Keep the user W&B run active while disabling Trainer auto-reporting."""
    return WandbSettings(
        enabled=False,
        project=settings.project,
        entity=settings.entity,
        mode=settings.mode,
        run_name=settings.run_name,
        group=settings.group,
        tags=settings.tags,
        log_model=settings.log_model,
    )


def _log_best_effort(run, payload: dict[str, Any]) -> None:
    if run is None or not payload:
        return
    try:
        run.log(payload)
    except Exception as exc:  # pragma: no cover - defensive around remote logging
        print(f"Warning: W&B stage history logging failed: {exc}", file=sys.stderr)


def _stage_metric_name(key: str, *, stage: str) -> str | None:
    if key == "loss":
        return "train/loss"
    if key == "learning_rate":
        return "train/learning_rate"
    if key == "grad_norm":
        return "train/grad_norm"
    if key == "train_runtime":
        return "train/runtime"
    if key == "train_samples_per_second":
        return "train/samples_per_second"
    if key == "train_steps_per_second":
        return "train/steps_per_second"
    if key == "train_loss":
        return "train/loss_average"
    stage_eval_prefix = f"{stage}_eval_"
    if key.startswith(stage_eval_prefix):
        return f"eval/{key.removeprefix(stage_eval_prefix)}"
    if key.startswith("eval_"):
        return f"eval/{key.removeprefix('eval_')}"
    return None


def _payload_metric_keys(
    payload: dict[str, Any],
    *,
    global_step_metric: str,
    stage: str,
) -> set[str]:
    metadata_keys = {global_step_metric, f"{stage}/epoch"}
    return set(payload).difference(metadata_keys)


def _build_stage_payload(
    record: dict[str, Any],
    *,
    stage: str,
    global_step_metric: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "step" in record:
        payload[global_step_metric] = record["step"]
    if "epoch" in record:
        payload[f"{stage}/epoch"] = record["epoch"]
    for key, value in record.items():
        if key in {"step", "epoch"}:
            continue
        metric_name = _stage_metric_name(key, stage=stage)
        if metric_name is not None:
            payload[f"{stage}/{metric_name}"] = value
    return payload


def log_stage_trainer_history(
    run,
    trainer,
    *,
    stage: str,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    """Log HF Trainer history under stage-specific metric names.

    Two-stage methods reuse HF Trainer global steps in each phase. Letting the
    W&B Trainer callback write both phases to train/loss makes the x-axis
    non-monotonic and produces misleading joined curves. This helper writes the
    same useful history as stage1/train/loss, stage2/train/loss, etc.
    """
    if run is None:
        return

    global_step_metric = f"{stage}/global_step"
    define_wandb_metric_best_effort(run, global_step_metric)
    define_wandb_metric_best_effort(
        run,
        f"{stage}/train/*",
        step_metric=global_step_metric,
    )
    define_wandb_metric_best_effort(
        run,
        f"{stage}/eval/*",
        step_metric=global_step_metric,
    )

    trainer_state = getattr(trainer, "state", None)
    last_step = getattr(trainer_state, "global_step", None)
    last_epoch = getattr(trainer_state, "epoch", None)
    logged_metric_names: set[str] = set()
    log_history = getattr(trainer_state, "log_history", []) or []
    for record in log_history:
        payload = _build_stage_payload(
            record,
            stage=stage,
            global_step_metric=global_step_metric,
        )
        metric_keys = _payload_metric_keys(
            payload,
            global_step_metric=global_step_metric,
            stage=stage,
        )
        last_step = payload.get(global_step_metric, last_step)
        last_epoch = payload.get(f"{stage}/epoch", last_epoch)
        if metric_keys:
            logged_metric_names.update(metric_keys)
            _log_best_effort(run, payload)

    if not extra_metrics:
        return

    extra_record = dict(extra_metrics)
    if last_step is not None:
        extra_record.setdefault("step", last_step)
    if last_epoch is not None:
        extra_record.setdefault("epoch", last_epoch)
    extra_payload = _build_stage_payload(
        extra_record,
        stage=stage,
        global_step_metric=global_step_metric,
    )
    for metric_name in list(
        _payload_metric_keys(
            extra_payload,
            global_step_metric=global_step_metric,
            stage=stage,
        )
    ):
        if metric_name in logged_metric_names:
            extra_payload.pop(metric_name, None)
    if _payload_metric_keys(
        extra_payload,
        global_step_metric=global_step_metric,
        stage=stage,
    ):
        _log_best_effort(run, extra_payload)
