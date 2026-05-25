from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.distilbert_efficient_head.args import parse_args
from src.methods.distilbert_efficient_head.config import (
    STAGE1_DIR_NAME,
    STAGE2_DIR_NAME,
    build_experiment_config,
    build_setup_failure_config,
)
from src.methods.distilbert_efficient_head.training import (
    apply_stage1_lora_to_context,
    build_callbacks,
    build_stage2_context,
    build_stage_training_arguments,
    resolve_wandb_settings,
)
from src.methods.hf_common import build_model_selection_summary, count_model_parameters, synchronize_cuda
from src.methods.hf_sequence_classification import (
    build_hf_trainer,
    build_runtime_metrics,
    evaluate_validation_and_optional_test,
    finish_failed_setup_run,
    finish_failed_train_run,
    initialize_hf_run,
    prepare_hf_classification_run,
    print_run_report,
    reset_peak_memory_stats,
    save_final_model,
    save_final_predictions,
    start_hf_run,
    write_config_snapshot,
    write_success_outputs,
)
from src.utils.wandb_config import finish_wandb_run


def _prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key if key.startswith(f"{prefix}_") else f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def _merge_stage_model_selection(
    stage1_selection: dict[str, Any],
    stage2_selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        **stage2_selection,
        "stage1_best_metric": stage1_selection.get("best_metric"),
        "stage1_best_epoch": stage1_selection.get("best_epoch"),
        "stage1_best_step": stage1_selection.get("best_step"),
        "stage1_best_model_checkpoint": stage1_selection.get("best_model_checkpoint"),
        "stage2_best_metric": stage2_selection.get("best_metric"),
        "stage2_best_epoch": stage2_selection.get("best_epoch"),
        "stage2_best_step": stage2_selection.get("best_step"),
    }


def main() -> None:
    args = parse_args()
    wandb_run = None
    train_start_time = None
    setup = initialize_hf_run(
        args,
        build_setup_failure_config_fn=build_setup_failure_config,
        resolve_wandb_settings_fn=resolve_wandb_settings,
    )
    precision_policy = setup.precision_policy
    experiment_config = setup.experiment_config

    try:
        precision_policy, experiment_config, wandb_run = start_hf_run(
            args,
            setup,
            build_setup_failure_config_fn=build_setup_failure_config,
        )
        base_context = prepare_hf_classification_run(
            args,
            precision_policy=precision_policy,
            gpu_type=setup.gpu_type,
        )
        stage2_trainable_params, total_params = count_model_parameters(base_context.model)
        stage1_context = apply_stage1_lora_to_context(base_context, args)
        stage1_trainable_params, stage1_total_params = count_model_parameters(
            stage1_context.model
        )

        print(
            "Stage 1 LoRA trainable params: "
            f"{stage1_trainable_params:,} / {stage1_total_params:,}"
        )
        print(
            "Stage 2 trainable params: "
            f"{stage2_trainable_params:,} / {total_params:,}"
        )

        experiment_config = build_experiment_config(
            args,
            **base_context.config_kwargs(),
            stage1_trainable_params=stage1_trainable_params,
            stage2_trainable_params=stage2_trainable_params,
            total_params=total_params,
        )
        write_config_snapshot(args.output_dir, experiment_config, wandb_run)

        stage1_args = build_stage_training_arguments(
            stage1_context.libraries.training_args_cls,
            args=args,
            output_dir=Path(args.output_dir) / STAGE1_DIR_NAME,
            learning_rate=args.stage1_learning_rate,
            num_train_epochs=args.stage1_epochs,
            precision_policy=precision_policy,
            wandb_settings=setup.wandb_settings,
        )
        stage2_args = build_stage_training_arguments(
            stage1_context.libraries.training_args_cls,
            args=args,
            output_dir=Path(args.output_dir) / STAGE2_DIR_NAME,
            learning_rate=args.stage2_learning_rate,
            num_train_epochs=args.stage2_epochs,
            precision_policy=precision_policy,
            wandb_settings=setup.wandb_settings,
        )
        stage1_trainer = build_hf_trainer(
            stage1_context,
            stage1_args,
            callbacks=build_callbacks(
                stage1_context.libraries.early_stopping_callback_cls,
                args,
            ),
        )

    except Exception as exc:
        finish_failed_setup_run(
            args,
            config=experiment_config,
            error=exc,
            gpu_type=setup.gpu_type,
            precision_policy=precision_policy,
            wandb_run=wandb_run,
        )
        raise

    try:
        print("\nStarting efficient-head DistilBERT training...")
        reset_peak_memory_stats()
        synchronize_cuda()
        train_start_time = time.perf_counter()

        print("\nStage 1: LoRA parameter-efficient head training...")
        stage1_start_time = time.perf_counter()
        stage1_trainer.train()
        synchronize_cuda()
        stage1_training_time_sec = time.perf_counter() - stage1_start_time
        stage1_eval_metrics = stage1_trainer.evaluate(metric_key_prefix="stage1_eval")
        stage1_model_selection = build_model_selection_summary(
            stage1_trainer,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=not args.lower_is_better,
        )

        print("\nStage 2: full fine-tuning fresh backbone with transferred head...")
        stage2_context = build_stage2_context(stage1_trainer.model, base_context, args)
        stage2_trainer = build_hf_trainer(
            stage2_context,
            stage2_args,
            callbacks=build_callbacks(
                stage2_context.libraries.early_stopping_callback_cls,
                args,
            ),
        )
        stage2_start_time = time.perf_counter()
        stage2_trainer.train()
        synchronize_cuda()
        stage2_training_time_sec = time.perf_counter() - stage2_start_time
        training_time_sec = time.perf_counter() - train_start_time

        metrics, test_metrics = evaluate_validation_and_optional_test(
            stage2_trainer,
            stage2_context,
        )
        model_selection = _merge_stage_model_selection(
            stage1_model_selection,
            build_model_selection_summary(
                stage2_trainer,
                metric_for_best_model=args.metric_for_best_model,
                greater_is_better=not args.lower_is_better,
            ),
        )
        model_artifact_paths = save_final_model(
            stage2_trainer,
            stage2_context.tokenizer,
            output_dir=args.output_dir,
            no_save_final_model=args.no_save_final_model,
            model_source=(
                "best stage-2 checkpoint"
                if args.load_best_model_at_end
                else "last stage-2 training state"
            ),
        )
        prediction_paths = save_final_predictions(stage2_context, stage2_trainer)
        runtime_metrics = build_runtime_metrics(
            args,
            training_time_sec=training_time_sec,
            gpu_type=setup.gpu_type,
            precision_policy=precision_policy,
            status="completed",
            extra={
                "stage1_training_time_sec": stage1_training_time_sec,
                "stage2_training_time_sec": stage2_training_time_sec,
            },
        )
        if wandb_run is not None:
            wandb_run.log(_prefixed_metrics("stage1", stage1_eval_metrics))
        result_paths = write_success_outputs(
            args,
            config=experiment_config,
            eval_metrics=metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            prediction_paths=prediction_paths,
            model_artifact_paths=model_artifact_paths,
            wandb_run=wandb_run,
            extra_metrics={"stage1": stage1_eval_metrics},
        )

        print_run_report(
            eval_metrics=metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            result_paths=result_paths,
            prediction_paths=prediction_paths,
            stage_metrics={"Stage 1 validation metrics": stage1_eval_metrics},
        )
        print(f"\nDone. Saved to: {args.output_dir}")

    except Exception as exc:
        finish_failed_train_run(
            args,
            config=experiment_config,
            error=exc,
            gpu_type=setup.gpu_type,
            precision_policy=precision_policy,
            wandb_run=wandb_run,
            train_start_time=train_start_time,
        )
        raise
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
