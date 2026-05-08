from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def write_resolved_config(output_dir: str | Path, config: dict[str, Any]) -> Path:
    return write_json(Path(output_dir) / "resolved_config.json", config)


def write_result_files(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    eval_metrics: dict[str, Any],
    runtime_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    metrics_payload = {
        "eval": eval_metrics,
        "test": test_metrics,
    }
    summary_payload = {
        "config": config,
        "metrics": metrics_payload,
        "runtime": runtime_metrics,
    }

    return {
        "metrics": write_json(output_path / "metrics.json", metrics_payload),
        "runtime": write_json(output_path / "runtime.json", runtime_metrics),
        "summary": write_json(output_path / "result_summary.json", summary_payload),
    }
