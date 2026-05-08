from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.experiments.results import write_json


DEFAULT_METRICS = (
    "eval_f1_macro",
    "eval_accuracy",
    "eval_precision_macro",
    "eval_recall_macro",
    "test_f1_macro",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "training_time_sec",
    "peak_memory_reserved_mb",
)


def discover_summary_files(paths: Iterable[str | Path]) -> list[Path]:
    summary_files: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        candidates = [path] if path.is_file() else []
        if path.is_dir():
            candidates.extend(path.rglob("result_summary.json"))
            candidates.extend(path.rglob("failure_summary.json"))
        for candidate in candidates:
            if candidate.name not in {"result_summary.json", "failure_summary.json"}:
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                summary_files.append(candidate)
    return sorted(summary_files, key=lambda item: item.as_posix())


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_value(payload: dict[str, Any], key: str) -> Any:
    metrics = payload.get("metrics") or {}
    runtime = payload.get("runtime") or {}
    for section in ("eval", "test"):
        section_metrics = metrics.get(section) or {}
        if key in section_metrics:
            return section_metrics[key]
    if key in runtime:
        return runtime[key]
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten_summary_record(
    payload: dict[str, Any],
    summary_path: str | Path,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    config = payload.get("config") or {}
    runtime = payload.get("runtime") or {}
    model_selection = payload.get("model_selection") or {}
    error = payload.get("error") or {}

    record: dict[str, Any] = {
        "summary_path": Path(summary_path).as_posix(),
        "status": payload.get("status", "unknown"),
        "method": config.get("method"),
        "search_stage": config.get("search_stage"),
        "trial_id": config.get("trial_id"),
        "config_hash": config.get("config_hash"),
        "seed": config.get("seed"),
        "hpo_seed": config.get("hpo_seed"),
        "dataset": config.get("dataset"),
        "model_name": config.get("model_name"),
        "output_dir": config.get("output_dir"),
        "gpu_type": runtime.get("gpu_type") or config.get("gpu_type"),
        "failure_phase": runtime.get("failure_phase"),
        "error_type": error.get("type"),
        "error_message": error.get("message"),
        "best_model_checkpoint": model_selection.get("best_model_checkpoint"),
        "best_metric": model_selection.get("best_metric"),
    }
    for metric in metrics:
        value = _metric_value(payload, metric)
        if value is not None:
            record[metric] = value
    return record


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    group_by: Iterable[str],
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    group_keys = list(group_by)
    metric_names = list(metrics)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(record)

    groups = []
    for key, group_records in sorted(grouped.items(), key=lambda item: str(item[0])):
        completed = [record for record in group_records if record.get("status") == "completed"]
        failed = [record for record in group_records if record.get("status") == "failed"]
        metric_summaries = {}
        for metric in metric_names:
            values = [
                value
                for record in completed
                for value in [_as_float(record.get(metric))]
                if value is not None
            ]
            if values:
                metric_summaries[metric] = _metric_summary(values)
        groups.append(
            {
                "group": dict(zip(group_keys, key)),
                "runs": len(group_records),
                "completed": len(completed),
                "failed": len(failed),
                "metrics": metric_summaries,
            }
        )
    return groups


def build_aggregate_report(
    paths: Iterable[str | Path],
    *,
    group_by: Iterable[str],
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    metric_names = list(metrics)
    summary_files = discover_summary_files(paths)
    records = [
        flatten_summary_record(load_summary(path), path, metrics=metric_names)
        for path in summary_files
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [Path(path).as_posix() for path in paths],
        "group_by": list(group_by),
        "metrics": metric_names,
        "total_runs": len(records),
        "completed_runs": sum(1 for record in records if record.get("status") == "completed"),
        "failed_runs": sum(1 for record in records if record.get("status") == "failed"),
        "runs": records,
        "groups": aggregate_records(records, group_by=group_by, metrics=metric_names),
    }


def write_aggregate_report(path: str | Path, report: dict[str, Any]) -> Path:
    return write_json(path, report)
