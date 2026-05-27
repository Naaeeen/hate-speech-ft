from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.results import (  # noqa: E402
    write_failure_file,
    write_resolved_config,
    write_result_files,
)
from src.methods.bilstm.args import parse_args, validate_bilstm_args  # noqa: E402
from src.methods.bilstm.config import (  # noqa: E402
    build_experiment_config,
    build_model_selection,
    build_runtime_metrics,
    resolve_wandb_settings,
)
from src.methods.bilstm.data import (  # noqa: E402
    build_bilstm_data_splits,
    load_dataset_library,
    print_split_summary,
    resolve_bilstm_split_names,
)
from src.methods.common import (  # noqa: E402
    clear_existing_run_artifacts,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)
from src.methods.hf_common import (  # noqa: E402
    get_gpu_type,
    get_peak_memory_mb,
    get_peak_memory_reserved_mb,
)
from src.utils.wandb_config import (  # noqa: E402
    define_wandb_metric_best_effort,
    finish_wandb_run,
    init_wandb_run,
    log_wandb_best_effort,
    update_wandb_config_best_effort,
)


def _validate_startup_args(args, output_dir: Path) -> None:
    validate_bilstm_args(args)
    validate_test_evaluation_policy(
        search_stage=args.search_stage,
        run_test=args.run_test,
    )
    validate_output_dir_for_run(
        output_dir,
        overwrite=args.overwrite_output_dir,
    )


def _prepare_output_dir(args, output_dir: Path) -> None:
    if args.overwrite_output_dir:
        removed_artifacts = clear_existing_run_artifacts(output_dir)
        if removed_artifacts:
            preview = ", ".join(path.name for path in removed_artifacts[:8])
            print(f"Cleared existing run artifacts from {output_dir}: {preview}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _define_bilstm_wandb_metrics(wandb_run) -> None:
    define_wandb_metric_best_effort(wandb_run, "global_step")
    define_wandb_metric_best_effort(
        wandb_run,
        "train_loss",
        step_metric="global_step",
    )
    define_wandb_metric_best_effort(
        wandb_run,
        "eval_*",
        step_metric="global_step",
    )


def _with_wandb_global_step(
    metrics: dict[str, Any] | None,
    *,
    global_step: int | None,
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    payload = dict(metrics)
    if global_step is not None:
        payload["global_step"] = global_step
    return payload


def _write_final_prediction_files(
    *,
    output_dir: Path,
    args,
    eval_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
) -> dict[str, Path]:
    if args.search_stage != "final":
        return {}

    from src.methods.bilstm.training import save_prediction_file

    prediction_paths = {
        "eval": save_prediction_file(
            output_dir / "eval_predictions.json",
            eval_predictions,
        )
    }
    if args.run_test:
        prediction_paths["test"] = save_prediction_file(
            output_dir / "test_predictions.json",
            test_predictions,
        )
    return prediction_paths


def _print_result_report(
    *,
    output_dir: Path,
    eval_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    runtime_metrics: dict[str, Any],
    model_selection: dict[str, Any],
    result_paths: dict[str, Path],
    prediction_paths: dict[str, Path],
) -> None:
    print("\nFinal validation metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")
    if test_metrics is not None:
        print("\nFinal test metrics:")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")
    print("\nModel selection:")
    for key, value in model_selection.items():
        print(f"{key}: {value}")
    print("\nRuntime metrics:")
    for key, value in runtime_metrics.items():
        print(f"{key}: {value}")
    print("\nResult files:")
    for key, value in result_paths.items():
        print(f"{key}: {value}")
    if prediction_paths:
        print("\nPrediction files:")
        for key, value in prediction_paths.items():
            print(f"{key}: {value}")
    print(f"\nDone. Saved to: {output_dir}")


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    output_dir = Path(args.output_dir)
    gpu_type = get_gpu_type()
    resolved_device = "unknown"
    wandb_run = None
    run_artifacts_prepared = False
    config = build_experiment_config(
        args,
        gpu_type=gpu_type,
        setup_complete=False,
    )

    try:
        _validate_startup_args(args, output_dir)
    except ValueError as exc:
        print(f"Cannot start run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        wandb_run = init_wandb_run(resolve_wandb_settings(args), config=config)
        _define_bilstm_wandb_metrics(wandb_run)
        from src.methods.bilstm.training import (
            resolve_class_weights,
            resolve_device,
            run_training,
            save_final_model,
            set_seed,
        )

        set_seed(args.seed)
        device = resolve_device(args.device)
        resolved_device = str(device)
        print(f"Using device: {device}")
        if gpu_type:
            print(f"GPU type: {gpu_type}")

        load_dataset = load_dataset_library()
        print(f"Loading dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)
        print("Available splits:", list(dataset.keys()))

        train_split, eval_split, test_split = resolve_bilstm_split_names(dataset, args)
        train_data, eval_data, test_data = build_bilstm_data_splits(
            dataset,
            args,
            train_split=train_split,
            eval_split=eval_split,
            test_split=test_split,
        )
        print_split_summary(
            train_split=train_split,
            eval_split=eval_split,
            test_split=test_split,
            train_data=train_data,
            eval_data=eval_data,
            test_data=test_data,
        )

        from src.methods.bilstm.tokenizer import StandardBiLSTMTokenizer

        tokenizer = StandardBiLSTMTokenizer.create(max_length=args.max_length)
        class_weights = resolve_class_weights(
            train_data.records,
            class_weighting=args.class_weighting,
            num_labels=3,
        )
        if class_weights is not None:
            print(f"Using balanced class weights: {class_weights}")

        result = run_training(
            args=args,
            train_data=train_data,
            eval_data=eval_data,
            test_data=test_data,
            tokenizer=tokenizer,
            device=device,
            class_weights=class_weights,
        )
        parameters = result["parameters"]
        config = build_experiment_config(
            args,
            train_split=train_split,
            eval_split=eval_split,
            test_split=test_split,
            train_data=train_data,
            eval_data=eval_data,
            test_data=test_data,
            tokenizer=tokenizer,
            gpu_type=gpu_type,
            trainable_params=parameters["trainable_params"],
            total_params=parameters["total_params"],
            class_weights=class_weights,
            setup_complete=True,
        )
        _prepare_output_dir(args, output_dir)
        run_artifacts_prepared = True
        write_resolved_config(output_dir, config)
        update_wandb_config_best_effort(wandb_run, config)

        model_path = save_final_model(
            output_dir,
            model=result["model"],
            tokenizer=tokenizer,
            config=config,
            no_save_final_model=args.no_save_final_model,
        )
        if model_path is not None:
            print(f"\nSaved final Bi-LSTM model: {model_path}")
            model_artifact_paths = {
                "model.pt": model_path,
                "tokenizer": output_dir / "tokenizer",
            }
        else:
            print("\nSkipping final model save because --no_save_final_model was set.")
            model_artifact_paths = {}

        prediction_paths = _write_final_prediction_files(
            output_dir=output_dir,
            args=args,
            eval_predictions=result["eval_predictions"],
            test_predictions=result["test_predictions"],
        )
        runtime_metrics = build_runtime_metrics(
            training_time_sec=result["runtime"]["training_time_sec"],
            device=str(device),
            gpu_type=gpu_type,
            peak_memory_mb=result["runtime"]["peak_memory_mb"],
            status="completed",
            peak_memory_reserved_mb=result["runtime"]["peak_memory_reserved_mb"],
            final_model_source=result["runtime"]["final_model_source"],
        )
        model_selection = build_model_selection(
            metric_for_best_model=args.metric_for_best_model,
            best_metric=result["model_selection"]["best_metric"],
            best_epoch=result["model_selection"]["best_epoch"],
            best_step=result["model_selection"]["best_step"],
            best_checkpoint=result["model_selection"]["best_checkpoint"],
        )
        result_paths = write_result_files(
            output_dir=output_dir,
            config=config,
            eval_metrics=result["eval_metrics"],
            test_metrics=result["test_metrics"],
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            prediction_paths=prediction_paths,
            artifact_paths=model_artifact_paths,
            extra_metrics={"history": result["history"]},
            status="completed",
        )
        final_global_step = result["model_selection"].get("best_step")
        eval_wandb_metrics = _with_wandb_global_step(
            result["eval_metrics"],
            global_step=final_global_step,
        )
        test_wandb_metrics = _with_wandb_global_step(
            result["test_metrics"],
            global_step=final_global_step,
        )
        log_wandb_best_effort(
            wandb_run,
            *result["history"],
            eval_wandb_metrics,
            *([test_wandb_metrics] if test_wandb_metrics is not None else []),
            runtime_metrics,
            {"model_selection": model_selection},
        )
        _print_result_report(
            output_dir=output_dir,
            eval_metrics=result["eval_metrics"],
            test_metrics=result["test_metrics"],
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            result_paths=result_paths,
            prediction_paths=prediction_paths,
        )
    except Exception as exc:
        runtime_metrics = build_runtime_metrics(
            training_time_sec=time.perf_counter() - run_start,
            device=resolved_device,
            gpu_type=gpu_type,
            peak_memory_mb=get_peak_memory_mb(),
            status="failed",
            peak_memory_reserved_mb=get_peak_memory_reserved_mb(),
            failure_phase="setup_or_train_eval",
        )
        write_failure_file(
            output_dir,
            config=config,
            error=exc,
            runtime_metrics=runtime_metrics,
            clear_existing_artifacts=run_artifacts_prepared,
        )
        log_wandb_best_effort(
            wandb_run,
            {
                "status": "failed",
                "failure_phase": runtime_metrics["failure_phase"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        print(f"\nFailure summary: {output_dir / 'failure_summary.json'}")
        raise
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
