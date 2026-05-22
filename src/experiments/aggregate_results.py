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
    "trainable_params",
    "total_params",
    "trainable_pct",
)


def discover_summary_files(paths: Iterable[str | Path]) -> list[Path]:
    latest_by_output_dir: dict[Path, Path] = {}
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
                output_dir = resolved.parent
                current = latest_by_output_dir.get(output_dir)
                if current is None or _summary_sort_key(candidate) > _summary_sort_key(current):
                    latest_by_output_dir[output_dir] = candidate
    return sorted(latest_by_output_dir.values(), key=lambda item: item.as_posix())


def _summary_sort_key(path: Path) -> tuple[float, int]:
    status_priority = 1 if path.name == "failure_summary.json" else 0
    return (path.stat().st_mtime, status_priority)


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_value(payload: dict[str, Any], key: str) -> Any:
    metrics = payload.get("metrics") or {}
    runtime = payload.get("runtime") or {}
    config = payload.get("config") or {}
    if key == "trainable_pct":
        trainable = _as_float(config.get("trainable_params"))
        total = _as_float(config.get("total_params"))
        if trainable is None or not total:
            return None
        return 100 * trainable / total
    for section in ("eval", "test"):
        section_metrics = metrics.get(section) or {}
        if key in section_metrics:
            return section_metrics[key]
    if key in runtime:
        return runtime[key]
    if key in config:
        return config[key]
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
    trainable_params = config.get("trainable_params")
    total_params = config.get("total_params")

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
        "is_oom": is_oom_failure(error.get("message"), error.get("type")),
        "best_model_checkpoint": model_selection.get("best_model_checkpoint"),
        "best_metric": model_selection.get("best_metric"),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": _metric_value(payload, "trainable_pct"),
    }
    for metric in metrics:
        value = _metric_value(payload, metric)
        if value is not None:
            record[metric] = value
    return record


def is_oom_failure(message: Any, error_type: Any = None) -> bool:
    text = " ".join(
        str(item).lower()
        for item in (error_type, message)
        if item is not None
    )
    oom_markers = (
        "out of memory",
        "cuda oom",
        "cublas_status_alloc_failed",
        "memoryerror",
        "resourceexhaustederror",
    )
    return any(marker in text for marker in oom_markers)


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
        failed_oom = [record for record in failed if record.get("is_oom")]
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
                "failed_oom": len(failed_oom),
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
        "failed_oom_runs": sum(
            1
            for record in records
            if record.get("status") == "failed" and record.get("is_oom")
        ),
        "runs": records,
        "groups": aggregate_records(records, group_by=group_by, metrics=metric_names),
    }


def write_aggregate_report(path: str | Path, report: dict[str, Any]) -> Path:
    return write_json(path, report)
