"""Output helpers for Transformer runs.

The runners call this file after training. It handles the final validation/test
passes, prediction JSONs, local model artifacts, runtime metrics, W&B metric
payloads, and the small console report. Keeping this here makes every
Transformer method produce the same per-run evidence for manual aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.methods.predictions import save_prediction_file
from src.methods.transformer_types import HfClassificationRun
from src.results import (
    MODEL_ARTIFACT_NAMES,
    write_resolved_config,
    write_result_files,
)
from src.utils.run_metadata import (
    build_compute_cost_fields,
    get_peak_memory_mb,
    get_peak_memory_reserved_mb,
)
from src.utils.wandb_config import log_wandb, prefixed_wandb_scalars


def write_config_snapshot(output_dir: str | Path, config: dict[str, Any]):
    """Write `resolved_config.json` and print its path."""

    path = write_resolved_config(output_dir, config)
    print(f"Resolved config: {path}")
    return path


def evaluate_validation_and_optional_test(trainer, context: HfClassificationRun):
    """Run final validation, and test only when `run_test=True`."""

    print("\nRunning final validation evaluation...")
    metrics = trainer.evaluate(metric_key_prefix="eval")
    test_metrics = None
    if context.args.run_test:
        print("\nRunning final test evaluation...")
        test_metrics = trainer.evaluate(
            eval_dataset=context.test_dataset,
            metric_key_prefix="test",
        )
    return metrics, test_metrics


def save_final_model(
    trainer,
    tokenizer,
    *,
    output_dir: str | Path,
    no_save_final_model: bool,
    model_source: str,
) -> dict[str, Path]:
    """Save the final selected model/tokenizer unless disabled."""

    if no_save_final_model:
        print("\nSkipping final model save because no_save_final_model=True.")
        return {}
    print(f"\nSaving final model and tokenizer from {model_source}...")
    output_path = Path(output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    artifacts = {
        name: output_path / name
        for name in MODEL_ARTIFACT_NAMES
        if (output_path / name).exists()
    }
    return artifacts or {"model_dir": output_path}


def save_final_predictions(context: HfClassificationRun, trainer) -> dict[str, Path]:
    """Write eval/test prediction rows when the run is allowed to use test data."""

    if not context.args.run_test:
        return {}
    print("\nSaving evaluation predictions...")
    eval_output = trainer.predict(context.eval_dataset, metric_key_prefix="eval_predictions")
    paths = {
        "eval": save_prediction_file(
            Path(context.args.output_dir) / "eval_predictions.json",
            records=context.eval_split_data.records,
            prediction_output=eval_output,
            id2label=context.id2label,
        )
    }
    if context.test_split_data is None:
        raise ValueError("Cannot save test predictions before loading a test split.")
    print("Saving test predictions...")
    test_output = trainer.predict(context.test_dataset, metric_key_prefix="test_predictions")
    paths["test"] = save_prediction_file(
        Path(context.args.output_dir) / "test_predictions.json",
        records=context.test_split_data.records,
        prediction_output=test_output,
        id2label=context.id2label,
    )
    return paths


def build_runtime_metrics(
    args: Any,
    *,
    training_time_sec: float | None,
    gpu_type: str,
    precision_policy: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build runtime/memory metadata for a Transformer run."""

    metrics = {
        "training_time_sec": training_time_sec,
        **build_compute_cost_fields(training_time_sec, gpu_type=gpu_type),
        "peak_memory_mb": get_peak_memory_mb(),
        "peak_memory_allocated_mb": get_peak_memory_mb(),
        "peak_memory_reserved_mb": get_peak_memory_reserved_mb(),
        "gpu_type": gpu_type,
        "mixed_precision": precision_policy["mixed_precision"],
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if extra:
        metrics.update(extra)
    return metrics


def write_success_outputs(
    args: Any,
    *,
    config: dict[str, Any],
    eval_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    runtime_metrics: dict[str, Any],
    model_selection: dict[str, Any],
    prediction_paths: dict[str, Path],
    wandb_run,
    model_artifact_paths: dict[str, Path] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write local result files and log the same final facts to W&B."""

    result_paths = write_result_files(
        args.output_dir,
        config=config,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        runtime_metrics=runtime_metrics,
        model_selection=model_selection,
        prediction_paths=prediction_paths,
        artifact_paths=model_artifact_paths,
        extra_metrics=extra_metrics,
    )
    log_wandb(
        wandb_run,
        _wandb_metric_payload(eval_metrics),
        _wandb_metric_payload(test_metrics),
        _wandb_extra_metric_payload(extra_metrics),
        runtime_metrics,
        {
            "model_selection": model_selection,
            **prefixed_wandb_scalars("model_selection", model_selection),
        },
    )
    return result_paths


def _wandb_metric_payload(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    return {_wandb_metric_key(key): value for key, value in metrics.items()}


def _wandb_metric_key(key: str) -> str:
    if key.startswith("eval_"):
        return f"eval/{key.removeprefix('eval_')}"
    if key.startswith("test_"):
        return f"test/{key.removeprefix('test_')}"
    return key


def _wandb_extra_metric_payload(extra_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Turn nested stage metrics into readable W&B keys like stage1/eval/f1."""

    if not extra_metrics:
        return {}
    payload = {}
    for group, metrics in extra_metrics.items():
        if isinstance(metrics, dict):
            prefix = f"{group}_"
            payload.update(
                {
                    f"{group}/{_wandb_metric_key(key.removeprefix(prefix))}": value
                    for key, value in metrics.items()
                }
            )
        else:
            payload[group] = metrics
    return payload


def print_run_report(
    *,
    eval_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    runtime_metrics: dict[str, Any],
    model_selection: dict[str, Any],
    result_paths: dict[str, Path],
    prediction_paths: dict[str, Path],
    stage_metrics: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print validation/test metrics, runtime, paths, and stage metrics."""

    if stage_metrics:
        for title, metrics in stage_metrics.items():
            print(f"\n{title}:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
    print("\nFinal validation metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")
    if test_metrics is not None:
        print("\nFinal test metrics:")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")
    print("\nRuntime metrics:")
    for key, value in runtime_metrics.items():
        print(f"{key}: {value}")
    print("\nModel selection:")
    for key, value in model_selection.items():
        print(f"{key}: {value}")
    print("\nResult files:")
    for key, value in result_paths.items():
        print(f"{key}: {value}")
    if prediction_paths:
        print("\nPrediction files:")
        for key, value in prediction_paths.items():
            print(f"{key}: {value}")
