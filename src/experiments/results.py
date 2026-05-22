from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def remove_if_exists(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()


def write_resolved_config(output_dir: str | Path, config: dict[str, Any]) -> Path:
    return write_json(Path(output_dir) / "resolved_config.json", config)


def write_result_files(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    eval_metrics: dict[str, Any],
    runtime_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
    model_selection: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    remove_if_exists(output_path / "failure_summary.json")
    metrics_payload = {
        "eval": eval_metrics,
        "test": test_metrics,
    }
    summary_payload = {
        "status": status,
        "config": config,
        "metrics": metrics_payload,
        "runtime": runtime_metrics,
        "model_selection": model_selection or {},
    }

    return {
        "metrics": write_json(output_path / "metrics.json", metrics_payload),
        "runtime": write_json(output_path / "runtime.json", runtime_metrics),
        "summary": write_json(output_path / "result_summary.json", summary_payload),
    }


def write_failure_file(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    error: BaseException,
    runtime_metrics: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    remove_if_exists(output_path / "result_summary.json")
    remove_if_exists(output_path / "metrics.json")
    remove_if_exists(output_path / "runtime.json")
    return write_json(
        output_path / "failure_summary.json",
        {
            "status": "failed",
            "config": config,
            "runtime": runtime_metrics or {},
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        },
    )
