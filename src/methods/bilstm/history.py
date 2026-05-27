from __future__ import annotations

from typing import Any


def append_epoch_history(
    history: list[dict[str, Any]],
    *,
    eval_metrics: dict[str, Any],
    train_loss: float,
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    record = {
        **eval_metrics,
        "train_loss": train_loss,
        "epoch": epoch,
        "global_step": global_step,
    }
    history.append(record)
    return record
