"""Manual entrypoint for one BiLSTM run.

The script reads `manual_config.py`, builds a train-split word vocabulary,
runs the custom PyTorch loop, writes the shared result files, and logs one W&B
run when enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.results import (  # noqa: E402
    prepare_output_dir_for_run,
    write_resolved_config,
    write_result_files,
)
from src.methods.bilstm.manual_config import CONFIG as MANUAL_CONFIG  # noqa: E402
from src.methods.bilstm.config import (  # noqa: E402
    build_experiment_config,
    build_model_selection,
    build_runtime_metrics,
    validate_bilstm_config,
)
from src.methods.bilstm.data import (  # noqa: E402
    build_bilstm_data_splits,
    load_dataset_library,
    print_split_summary,
    resolve_bilstm_split_names,
)
from src.utils.run_metadata import (  # noqa: E402
    get_gpu_type,
)
from src.utils.wandb_config import (  # noqa: E402
    build_wandb_settings_from_args,
    define_training_wandb_metrics,
    finish_wandb_run,
    init_wandb_run,
    log_wandb,
    log_wandb_history,
    namespaced_wandb_metrics,
    prefixed_wandb_scalars,
)


def _write_final_prediction_files(
    *,
    output_dir: Path,
    args,
    eval_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
) -> dict[str, Path]:
    if not args.run_test:
        return {}

    from src.methods.bilstm.training import save_prediction_file

    prediction_paths = {
        "eval": save_prediction_file(
            output_dir / "eval_predictions.json",
            eval_predictions,
        )
    }
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
    """Run one BiLSTM experiment from `manual_config.py`."""

    args = SimpleNamespace(**MANUAL_CONFIG)
    output_dir = Path(args.output_dir)
    gpu_type = get_gpu_type()

    validate_bilstm_config(args)
    prepare_output_dir_for_run(output_dir, overwrite=args.overwrite_output_dir)

    from src.methods.bilstm.training import (
        resolve_class_weights,
        resolve_device,
        run_training,
        save_final_model,
        set_seed,
    )

    set_seed(args.seed)
    device = resolve_device(args.device)
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

    tokenizer = StandardBiLSTMTokenizer.create(
        train_records=train_data.records,
        max_length=args.max_length,
        min_freq=args.tokenizer_min_freq,
        max_vocab_size=args.max_vocab_size,
    )
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
    )
    write_resolved_config(output_dir, config)

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
        print("\nSkipping final model save because no_save_final_model=True.")
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
    )
    wandb_run = init_wandb_run(build_wandb_settings_from_args(args), config=config)
    define_training_wandb_metrics(wandb_run)
    log_wandb_history(wandb_run, result.get("history", []))
    log_wandb(
        wandb_run,
        namespaced_wandb_metrics(result["eval_metrics"]),
        namespaced_wandb_metrics(result["test_metrics"]),
        prefixed_wandb_scalars("runtime", runtime_metrics),
        {
            "model_selection": model_selection,
            **prefixed_wandb_scalars("model_selection", model_selection),
        },
    )
    finish_wandb_run(wandb_run)
    _print_result_report(
        output_dir=output_dir,
        eval_metrics=result["eval_metrics"],
        test_metrics=result["test_metrics"],
        runtime_metrics=runtime_metrics,
        model_selection=model_selection,
        result_paths=result_paths,
        prediction_paths=prediction_paths,
    )


if __name__ == "__main__":
    main()
