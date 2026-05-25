from __future__ import annotations

import csv
import json
import math
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
    "eval_f1_hatespeech",
    "eval_f1_normal",
    "eval_f1_offensive",
    "test_f1_macro",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_hatespeech",
    "test_f1_normal",
    "test_f1_offensive",
    "training_time_sec",
    "training_time_hours",
    "gpu_hours",
    "peak_memory_mb",
    "peak_memory_reserved_mb",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "best_epoch",
)

HPO_RUN_COLUMNS = (
    "method",
    "trial_id",
    "search_stage",
    "hpo_seed",
    "training_seed",
    "search_method",
    "search_space",
    "hpo_trial_cap",
    "hpo_time_cap_gpu_hours",
    "sampled_hparams_json",
    "status",
    "failed_oom",
    "val_macro_f1",
    "val_precision",
    "val_recall",
    "val_accuracy",
    "best_epoch",
    "train_time_s",
    "train_time_hours",
    "gpu_hours",
    "eval_time_s",
    "peak_gpu_memory_mb",
    "gpu_type",
    "trainable_params",
    "total_params",
    "notes",
    "summary_path",
    "error_type",
    "error_message",
)

FINAL_RUN_COLUMNS = (
    "method",
    "final_config_id",
    "missing_config_hash",
    "seed",
    "status",
    "failed_oom",
    "selected_hyperparams_json",
    "test_macro_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
    "test_per_class_f1_hate",
    "test_per_class_f1_offensive",
    "test_per_class_f1_normal",
    "val_macro_f1",
    "best_epoch",
    "final_train_time_s",
    "final_train_time_hours",
    "gpu_hours",
    "peak_gpu_memory_mb",
    "gpu_type",
    "trainable_params",
    "total_params",
    "summary_path",
    "test_predictions_path",
    "model_artifacts",
    "error_type",
    "error_message",
)

SEARCH_SPACE_ALIASES = {
    "tfidf_lr": "tfidf_logreg",
}

METHOD_SUMMARY_COLUMNS = (
    "method",
    "final_config_id",
    "missing_config_hash",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "test_precision_mean",
    "test_precision_std",
    "test_recall_mean",
    "test_recall_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "final_train_time_mean_s",
    "final_train_time_std_s",
    "peak_gpu_memory_mean_mb",
    "peak_gpu_memory_std_mb",
    "trainable_params",
    "total_params",
    "completed_hpo_trials",
    "failed_hpo_trials",
    "failed_oom_trials",
    "actual_hpo_time_s",
    "actual_hpo_gpu_hours",
    "hpo_gpu_type",
    "final_gpu_type",
    "hpo_seed",
    "hpo_trial_cap",
    "hpo_time_cap_gpu_hours",
    "search_method",
    "search_space",
    "selection_metric",
    "best_val_macro_f1",
    "selected_hpo_trial_id",
    "selected_hpo_summary_path",
    "best_epoch_mean",
    "best_epoch_min",
    "best_epoch_max",
    "selected_hyperparams_json",
    "pareto_status",
    "dominated_by",
    "final_seed_count",
    "completed_final_seeds",
    "failed_final_seeds",
    "failed_final_oom_seeds",
    "final_seeds",
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


def _stable_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _canonical_search_space_name(value: Any) -> str | None:
    if value is None or value == "":
        return None
    search_space_name = str(value)
    return SEARCH_SPACE_ALIASES.get(search_space_name, search_space_name)


def _missing_config_hash_id(summary_path: str | Path) -> str:
    return f"missing_config_hash:{Path(summary_path).parent.as_posix()}"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_non_empty(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _is_hpo_search_record(record: dict[str, Any]) -> bool:
    return record.get("search_method") == "random_search"


def _hpo_budget_records(
    records: Iterable[dict[str, Any]],
    stages: set[str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("search_stage") in stages and _is_hpo_search_record(record)
    ]


def flatten_summary_record(
    payload: dict[str, Any],
    summary_path: str | Path,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    config = payload.get("config") or {}
    runtime = payload.get("runtime") or {}
    model_selection = payload.get("model_selection") or {}
    prediction_artifacts = (payload.get("artifacts") or {}).get("predictions") or {}
    model_artifacts = (payload.get("artifacts") or {}).get("model") or {}
    error = payload.get("error") or {}
    hyperparameters = config.get("hyperparameters") or {}
    trainable_params = config.get("trainable_params")
    total_params = config.get("total_params")
    method = config.get("method")
    search_stage = config.get("search_stage")
    search_space_name = _canonical_search_space_name(
        config.get("search_space_name") or config.get("search_space")
    )
    search_method = config.get("search_method")
    config_hash = config.get("config_hash")
    missing_config_hash = search_stage == "final" and not config_hash
    final_config_id = (
        _missing_config_hash_id(summary_path)
        if missing_config_hash
        else config_hash
    )
    best_metric_key = model_selection.get("best_metric_key")
    eval_f1_macro = _metric_value(payload, "eval_f1_macro")
    best_val_macro_f1 = (
        model_selection.get("best_metric")
        if best_metric_key == "eval_f1_macro"
        else eval_f1_macro
    )

    record: dict[str, Any] = {
        "summary_path": Path(summary_path).as_posix(),
        "status": payload.get("status", "unknown"),
        "method": method,
        "search_stage": search_stage,
        "trial_id": config.get("trial_id"),
        "config_hash": config_hash,
        "final_config_id": final_config_id,
        "missing_config_hash": missing_config_hash,
        "search_method": search_method,
        "search_space_name": search_space_name,
        "selected_hyperparams_json": _stable_json(hyperparameters),
        "sampled_hparams_json": _stable_json(hyperparameters),
        "seed": config.get("seed"),
        "hpo_seed": config.get("hpo_seed"),
        "hpo_trial_cap": config.get("hpo_trial_cap"),
        "hpo_time_cap_gpu_hours": config.get("hpo_time_cap_gpu_hours"),
        "git_commit": config.get("git_commit"),
        "dataset": config.get("dataset"),
        "model_name": config.get("model_name"),
        "data_fraction": config.get("data_fraction"),
        "effective_train_fraction": config.get("effective_train_fraction"),
        "train_size": config.get("train_size"),
        "eval_size": config.get("eval_size"),
        "test_size": config.get("test_size"),
        "raw_train_size": config.get("raw_train_size"),
        "raw_eval_size": config.get("raw_eval_size"),
        "raw_test_size": config.get("raw_test_size"),
        "full_train_size": config.get("full_train_size"),
        "full_eval_size": config.get("full_eval_size"),
        "full_test_size": config.get("full_test_size"),
        "dropped_no_majority_train": config.get("dropped_no_majority_train"),
        "dropped_no_majority_eval": config.get("dropped_no_majority_eval"),
        "dropped_no_majority_test": config.get("dropped_no_majority_test"),
        "output_dir": config.get("output_dir"),
        "gpu_type": runtime.get("gpu_type") or config.get("gpu_type"),
        "failure_phase": runtime.get("failure_phase"),
        "error_type": error.get("type"),
        "error_message": error.get("message"),
        "is_oom": is_oom_failure(error.get("message"), error.get("type")),
        "metric_for_best_model": model_selection.get("metric_for_best_model"),
        "best_metric_key": best_metric_key,
        "best_model_checkpoint": model_selection.get("best_model_checkpoint"),
        "best_metric": model_selection.get("best_metric"),
        "selection_metric": (
            config.get("selection_metric")
            or model_selection.get("metric_for_best_model")
        ),
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": model_selection.get("best_epoch"),
        "best_step": model_selection.get("best_step"),
        "eval_predictions_path": prediction_artifacts.get("eval"),
        "test_predictions_path": prediction_artifacts.get("test"),
        "model_artifacts": model_artifacts,
        "training_time_sec": _metric_value(payload, "training_time_sec"),
        "training_time_hours": _metric_value(payload, "training_time_hours"),
        "gpu_hours": _metric_value(payload, "gpu_hours"),
        "eval_runtime": _metric_value(payload, "eval_runtime"),
        "peak_memory_mb": _metric_value(payload, "peak_memory_mb"),
        "peak_memory_reserved_mb": _metric_value(payload, "peak_memory_reserved_mb"),
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


def _sum_metric(records: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        value
        for record in records
        for value in [_as_float(record.get(metric))]
        if value is not None
    ]
    return math.fsum(values) if values else None


def _hours(seconds: float | None) -> float | None:
    return seconds / 3600 if seconds is not None else None


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
        total_training_time_sec = _sum_metric(group_records, "training_time_sec")
        groups.append(
            {
                "group": dict(zip(group_keys, key)),
                "runs": len(group_records),
                "completed": len(completed),
                "failed": len(failed),
                "failed_oom": len(failed_oom),
                "total_training_time_sec": total_training_time_sec,
                "total_training_time_hours": _hours(total_training_time_sec),
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
    hpo_records = _hpo_budget_records(records, {"tuning", "confirm"})
    tuning_hpo_records = _hpo_budget_records(records, {"tuning"})
    confirmation_hpo_records = _hpo_budget_records(records, {"confirm"})
    total_training_time_sec = _sum_metric(records, "training_time_sec")
    hpo_total_training_time_sec = _sum_metric(hpo_records, "training_time_sec")
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
        "total_training_time_sec": total_training_time_sec,
        "total_training_time_hours": _hours(total_training_time_sec),
        "hpo_total_training_time_sec": hpo_total_training_time_sec,
        "hpo_total_training_time_hours": _hours(hpo_total_training_time_sec),
        "tuning_hpo_total_training_time_sec": _sum_metric(
            tuning_hpo_records,
            "training_time_sec",
        ),
        "confirmation_total_training_time_sec": _sum_metric(
            confirmation_hpo_records,
            "training_time_sec",
        ),
        "final_total_training_time_sec": _sum_metric(
            [
                record
                for record in records
                if record.get("search_stage") == "final"
            ],
            "training_time_sec",
        ),
        "runs": records,
        "groups": aggregate_records(records, group_by=group_by, metrics=metric_names),
    }


def write_aggregate_report(path: str | Path, report: dict[str, Any]) -> Path:
    return write_json(path, report)


def _peak_memory_mb(record: dict[str, Any]) -> float | None:
    return _first_present(
        _as_float(record.get("peak_memory_reserved_mb")),
        _as_float(record.get("peak_memory_mb")),
    )


def _numeric_values(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [
        value
        for record in records
        for value in [_as_float(record.get(field))]
        if value is not None
    ]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _unique_join(values: Iterable[Any]) -> str:
    items = sorted({str(value) for value in values if value is not None and value != ""})
    return ",".join(items)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    column_names = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in column_names})
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _stable_json(value)
    return value


def _hpo_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": record.get("method"),
        "trial_id": record.get("trial_id"),
        "search_stage": record.get("search_stage"),
        "hpo_seed": record.get("hpo_seed"),
        "training_seed": record.get("seed"),
        "search_method": record.get("search_method"),
        "search_space": record.get("search_space_name"),
        "hpo_trial_cap": record.get("hpo_trial_cap"),
        "hpo_time_cap_gpu_hours": record.get("hpo_time_cap_gpu_hours"),
        "sampled_hparams_json": record.get("sampled_hparams_json"),
        "status": record.get("status"),
        "failed_oom": bool(record.get("is_oom")),
        "val_macro_f1": _first_present(
            record.get("eval_f1_macro"),
            record.get("best_val_macro_f1"),
        ),
        "val_precision": record.get("eval_precision_macro"),
        "val_recall": record.get("eval_recall_macro"),
        "val_accuracy": record.get("eval_accuracy"),
        "best_epoch": record.get("best_epoch"),
        "train_time_s": record.get("training_time_sec"),
        "train_time_hours": record.get("training_time_hours"),
        "gpu_hours": record.get("gpu_hours"),
        "eval_time_s": record.get("eval_runtime"),
        "peak_gpu_memory_mb": _peak_memory_mb(record),
        "gpu_type": record.get("gpu_type"),
        "trainable_params": record.get("trainable_params"),
        "total_params": record.get("total_params"),
        "notes": record.get("failure_phase"),
        "summary_path": record.get("summary_path"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
    }


def _final_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": record.get("method"),
        "final_config_id": record.get("final_config_id"),
        "missing_config_hash": bool(record.get("missing_config_hash")),
        "seed": record.get("seed"),
        "status": record.get("status"),
        "failed_oom": bool(record.get("is_oom")),
        "selected_hyperparams_json": record.get("selected_hyperparams_json"),
        "test_macro_f1": record.get("test_f1_macro"),
        "test_precision": record.get("test_precision_macro"),
        "test_recall": record.get("test_recall_macro"),
        "test_accuracy": record.get("test_accuracy"),
        "test_per_class_f1_hate": record.get("test_f1_hatespeech"),
        "test_per_class_f1_offensive": record.get("test_f1_offensive"),
        "test_per_class_f1_normal": record.get("test_f1_normal"),
        "val_macro_f1": record.get("eval_f1_macro"),
        "best_epoch": record.get("best_epoch"),
        "final_train_time_s": record.get("training_time_sec"),
        "final_train_time_hours": record.get("training_time_hours"),
        "gpu_hours": record.get("gpu_hours"),
        "peak_gpu_memory_mb": _peak_memory_mb(record),
        "gpu_type": record.get("gpu_type"),
        "trainable_params": record.get("trainable_params"),
        "total_params": record.get("total_params"),
        "summary_path": record.get("summary_path"),
        "test_predictions_path": record.get("test_predictions_path"),
        "model_artifacts": record.get("model_artifacts"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
    }


def _best_hpo_record(hpo_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [record for record in hpo_records if record.get("status") == "completed"]
    scored = [
        (score, record)
        for record in completed
        for score in [
            _as_float(
                _first_present(
                    record.get("eval_f1_macro"),
                    record.get("best_val_macro_f1"),
                )
            )
        ]
        if score is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _selected_hpo_record(
    hpo_records: list[dict[str, Any]],
    final_config_id: Any,
) -> dict[str, Any] | None:
    matching_config_records = [
        record
        for record in hpo_records
        if final_config_id is not None and record.get("config_hash") == final_config_id
    ]
    return _best_hpo_record(matching_config_records)


def _summarize_metric(
    records: list[dict[str, Any]],
    field: str,
) -> tuple[float | None, float | None]:
    values = _numeric_values(records, field)
    return _mean(values), _std(values)


def _method_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_final_records = [
        record
        for record in records
        if record.get("search_stage") == "final"
    ]
    hpo_records = _hpo_budget_records(records, {"tuning"})
    grouped_final: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for record in all_final_records:
        key = (record.get("method"), record.get("final_config_id"))
        grouped_final.setdefault(key, []).append(record)

    rows = []
    sorted_groups = sorted(grouped_final.items(), key=lambda item: str(item[0]))
    for (method, final_config_id), all_group in sorted_groups:
        completed_group = [
            record for record in all_group if record.get("status") == "completed"
        ]
        failed_final = [
            record for record in all_group if record.get("status") == "failed"
        ]
        failed_final_oom = [record for record in failed_final if record.get("is_oom")]
        search_space_name = _first_non_empty(
            record.get("search_space_name") for record in all_group
        )
        final_hpo_seed = _first_non_empty(record.get("hpo_seed") for record in all_group)
        missing_config_hash = any(record.get("missing_config_hash") for record in all_group)
        method_hpo = [
            record
            for record in hpo_records
            if record.get("method") == method
            and (
                search_space_name is None
                or record.get("search_space_name") == search_space_name
            )
            and (
                final_hpo_seed is None
                or record.get("hpo_seed") == final_hpo_seed
            )
        ]
        completed_hpo = [
            record for record in method_hpo if record.get("status") == "completed"
        ]
        failed_hpo = [
            record for record in method_hpo if record.get("status") == "failed"
        ]
        failed_oom = [record for record in failed_hpo if record.get("is_oom")]
        selected_hpo = _selected_hpo_record(method_hpo, final_config_id)
        train_time_mean, train_time_std = _summarize_metric(
            completed_group,
            "training_time_sec",
        )
        test_f1_mean, test_f1_std = _summarize_metric(completed_group, "test_f1_macro")
        test_precision_mean, test_precision_std = _summarize_metric(
            completed_group,
            "test_precision_macro",
        )
        test_recall_mean, test_recall_std = _summarize_metric(
            completed_group,
            "test_recall_macro",
        )
        test_accuracy_mean, test_accuracy_std = _summarize_metric(
            completed_group,
            "test_accuracy",
        )
        peak_values = [
            value
            for record in completed_group
            for value in [_peak_memory_mb(record)]
            if value is not None
        ]
        best_epoch_values = _numeric_values(completed_group, "best_epoch")
        actual_hpo_time_s = _sum_metric(method_hpo, "training_time_sec")
        hpo_gpu_hours = _sum_metric(method_hpo, "gpu_hours")
        row = {
            "method": method,
            "final_config_id": final_config_id,
            "missing_config_hash": missing_config_hash,
            "test_macro_f1_mean": test_f1_mean,
            "test_macro_f1_std": test_f1_std,
            "test_precision_mean": test_precision_mean,
            "test_precision_std": test_precision_std,
            "test_recall_mean": test_recall_mean,
            "test_recall_std": test_recall_std,
            "test_accuracy_mean": test_accuracy_mean,
            "test_accuracy_std": test_accuracy_std,
            "final_train_time_mean_s": train_time_mean,
            "final_train_time_std_s": train_time_std,
            "peak_gpu_memory_mean_mb": _mean(peak_values),
            "peak_gpu_memory_std_mb": _std(peak_values),
            "trainable_params": _first_non_empty(
                record.get("trainable_params") for record in completed_group + all_group
            ),
            "total_params": _first_non_empty(
                record.get("total_params") for record in completed_group + all_group
            ),
            "completed_hpo_trials": len(completed_hpo),
            "failed_hpo_trials": len(failed_hpo),
            "failed_oom_trials": len(failed_oom),
            "actual_hpo_time_s": actual_hpo_time_s,
            "actual_hpo_gpu_hours": hpo_gpu_hours,
            "hpo_gpu_type": _unique_join(record.get("gpu_type") for record in method_hpo),
            "final_gpu_type": _unique_join(record.get("gpu_type") for record in all_group),
            "hpo_seed": _first_non_empty(
                record.get("hpo_seed") for record in method_hpo + all_group
            ),
            "hpo_trial_cap": _first_non_empty(
                record.get("hpo_trial_cap") for record in method_hpo + all_group
            ),
            "hpo_time_cap_gpu_hours": _first_non_empty(
                record.get("hpo_time_cap_gpu_hours") for record in method_hpo + all_group
            ),
            "search_method": _first_non_empty(
                record.get("search_method") for record in method_hpo + all_group
            ),
            "search_space": search_space_name,
            "selection_metric": _first_non_empty(
                record.get("metric_for_best_model") or record.get("selection_metric")
                for record in method_hpo + all_group
            ),
            "best_val_macro_f1": (
                _as_float(
                    _first_present(
                        selected_hpo.get("eval_f1_macro"),
                        selected_hpo.get("best_val_macro_f1"),
                    )
                )
                if selected_hpo is not None
                else None
            ),
            "selected_hpo_trial_id": (
                selected_hpo.get("trial_id") if selected_hpo is not None else None
            ),
            "selected_hpo_summary_path": (
                selected_hpo.get("summary_path") if selected_hpo is not None else None
            ),
            "best_epoch_mean": _mean(best_epoch_values),
            "best_epoch_min": min(best_epoch_values) if best_epoch_values else None,
            "best_epoch_max": max(best_epoch_values) if best_epoch_values else None,
            "selected_hyperparams_json": _first_non_empty(
                record.get("selected_hyperparams_json") for record in completed_group + all_group
            ),
            "final_seed_count": len(all_group),
            "completed_final_seeds": len(completed_group),
            "failed_final_seeds": len(failed_final),
            "failed_final_oom_seeds": len(failed_final_oom),
            "final_seeds": _unique_join(record.get("seed") for record in all_group),
        }
        rows.append(row)
    _annotate_pareto(rows)
    return rows


def _annotate_pareto(rows: list[dict[str, Any]]) -> None:
    for candidate in rows:
        if (
            candidate.get("missing_config_hash")
            or _as_float(candidate.get("test_macro_f1_mean")) is None
        ):
            candidate["pareto_status"] = "insufficient_data"
            candidate["dominated_by"] = ""
            continue
        dominated_by = []
        for competitor in rows:
            if competitor is candidate:
                continue
            if _dominates(competitor, candidate):
                dominated_by.append(_pareto_row_id(competitor))
        candidate["pareto_status"] = "dominated" if dominated_by else "pareto_optimal"
        candidate["dominated_by"] = ",".join(sorted(set(dominated_by)))


def _pareto_row_id(row: dict[str, Any]) -> str:
    method = row.get("method") or "unknown_method"
    config_id = row.get("final_config_id") or "unknown_config"
    return f"{method}:{config_id}"


def _dominates(competitor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if competitor.get("missing_config_hash") or candidate.get("missing_config_hash"):
        return False
    competitor_score = _as_float(competitor.get("test_macro_f1_mean"))
    candidate_score = _as_float(candidate.get("test_macro_f1_mean"))
    if competitor_score is None or candidate_score is None:
        return False
    if competitor_score < candidate_score:
        return False

    cost_keys = (
        "final_train_time_mean_s",
        "peak_gpu_memory_mean_mb",
        "trainable_params",
    )
    comparable_costs = []
    for key in cost_keys:
        competitor_cost = _as_float(competitor.get(key))
        candidate_cost = _as_float(candidate.get(key))
        if competitor_cost is not None and candidate_cost is not None:
            comparable_costs.append((competitor_cost, candidate_cost))
    if not comparable_costs:
        return False
    if any(
        competitor_cost > candidate_cost
        for competitor_cost, candidate_cost in comparable_costs
    ):
        return False
    strictly_better_score = competitor_score > candidate_score
    strictly_better_cost = any(
        competitor_cost < candidate_cost
        for competitor_cost, candidate_cost in comparable_costs
    )
    return strictly_better_score or strictly_better_cost


def write_pareto_csvs(output_dir: str | Path, report: dict[str, Any]) -> list[Path]:
    """Write Pareto-ready HPO, final-run, and method-summary CSV artifacts."""

    output_path = Path(output_dir)
    records = list(report.get("runs") or [])
    hpo_rows = [
        _hpo_row(record)
        for record in sorted(
            records,
            key=lambda item: (
                str(item.get("method")),
                str(item.get("trial_id")),
                str(item.get("summary_path")),
            ),
        )
        if record.get("search_stage") == "tuning" and _is_hpo_search_record(record)
    ]
    final_rows = [
        _final_row(record)
        for record in sorted(
            records,
            key=lambda item: (
                str(item.get("method")),
                str(item.get("final_config_id")),
                str(item.get("seed")),
                str(item.get("summary_path")),
            ),
        )
        if record.get("search_stage") == "final"
    ]
    method_rows = _method_summary_rows(records)
    return [
        _write_csv(output_path / "hpo_runs.csv", hpo_rows, HPO_RUN_COLUMNS),
        _write_csv(output_path / "final_runs.csv", final_rows, FINAL_RUN_COLUMNS),
        _write_csv(
            output_path / "method_summary.csv",
            method_rows,
            METHOD_SUMMARY_COLUMNS,
        ),
    ]
