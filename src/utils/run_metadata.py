"""Tiny runtime-metadata helpers.

These helpers stay lightweight. If CUDA, torch, or git is unavailable, they
return `None`, `cpu`, or `unknown` instead of adding setup machinery. That keeps
Colab runs and local commands on the same metadata path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def count_model_parameters(model) -> tuple[int, int]:
    """Return `(trainable_params, total_params)` for any torch-like model."""

    total_params = 0
    trainable_params = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_params += count
        if parameter.requires_grad:
            trainable_params += count
    return trainable_params, total_params


def get_gpu_type() -> str:
    """Return the current CUDA device name, `cpu`, or `unknown`."""

    try:
        import torch
    except ImportError:
        return "unknown"
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def get_peak_memory_mb() -> float | None:
    """Return peak allocated CUDA memory in MB when torch/CUDA is available."""

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def get_peak_memory_reserved_mb() -> float | None:
    """Return peak reserved CUDA memory in MB when torch/CUDA is available."""

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / (1024 * 1024)


def reset_peak_memory_stats() -> None:
    """Reset torch CUDA peak-memory stats if possible."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def synchronize_cuda() -> None:
    """Synchronize CUDA so runtime measurements include queued kernels."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_git_commit_hash(repo_root: Path) -> str | None:
    """Read the current git commit hash for run provenance."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def build_compute_cost_fields(
    training_time_sec: float | None,
    *,
    gpu_type: str | None,
) -> dict[str, float | None]:
    """Return time in hours and GPU-hours for GPU-backed runs."""

    training_time_hours = (
        training_time_sec / 3600 if training_time_sec is not None else None
    )
    has_gpu = bool(gpu_type and gpu_type not in {"cpu", "unknown"})
    return {
        "training_time_hours": training_time_hours,
        "gpu_hours": training_time_hours if has_gpu else None,
    }
