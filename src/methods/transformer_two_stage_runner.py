"""Shared runner for the two-stage Transformer methods.

LP-FT and Efficient-Head FT both train in two stages, but they do not mean the
same thing. So this file only owns the common shell: build stage training args,
run stage 1, build the stage-2 context, run stage 2, then save one final set of
artifacts. Stage-local W&B is disabled on purpose, because the manual workflow
wants one readable W&B run per experiment run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.methods.transformer_outputs import (
    build_runtime_metrics,
    evaluate_validation_and_optional_test,
    print_run_report,
    save_final_model,
    save_final_predictions,
    write_config_snapshot,
    write_success_outputs,
)
from src.methods.transformer_setup import prepare_hf_classification_run, start_hf_run
from src.methods.transformer_trainer import (
    build_early_stopping_callbacks,
    build_hf_trainer,
    build_hf_training_arguments_from_args,
    build_model_selection_summary,
)
from src.utils.run_metadata import reset_peak_memory_stats, synchronize_cuda
from src.utils.wandb_config import WandbSettings, finish_wandb_run, init_wandb_run


@dataclass(frozen=True)
class TwoStagePlan:
    """Small bundle that tells the shared runner how this method stages work.

    `stage1_context` is the model/dataset setup used for stage 1. The callable
    builds the stage-2 context from the trained stage-1 model, which lets LP-FT
    continue from the same model while Efficient-Head can rebuild a fresh
    backbone and copy only the head.
    """

    stage1_context: Any
    stage1_trainable_params: int
    stage2_trainable_params: int
    total_params: int
    build_stage2_context: Callable[[Any], Any]


def merge_stage_model_selection(
    stage1_selection: dict[str, Any],
    stage2_selection: dict[str, Any],
) -> dict[str, Any]:
    """Flatten stage best-checkpoint details into the final summary dict."""

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


def run_two_stage_transformer(
    args: Any,
    *,
    build_experiment_config: Callable[..., dict[str, Any]],
    build_stage_plan: Callable[[Any, Any], TwoStagePlan],
    stage1_dir_name: str,
    stage2_dir_name: str,
    stage1_learning_rate_attr: str,
    stage1_epochs_attr: str,
    stage2_learning_rate_attr: str,
    stage2_epochs_attr: str,
    train_message: str,
    stage1_message: str,
    stage2_message: str,
) -> None:
    """Run one complete two-stage manual experiment.

    The final metrics and saved model always come from stage 2. Stage 1 metrics
    are kept as extra evidence, but they are not treated as a separate run.
    """

    setup = start_hf_run(args)
    base_context = prepare_hf_classification_run(
        args,
        precision_policy=setup.precision_policy,
        gpu_type=setup.gpu_type,
    )
    plan = build_stage_plan(base_context, args)
    print(
        "Stage 1 trainable params: "
        f"{plan.stage1_trainable_params:,} / {plan.total_params:,}"
    )
    print(
        "Stage 2 trainable params: "
        f"{plan.stage2_trainable_params:,} / {plan.total_params:,}"
    )

    experiment_config = build_experiment_config(
        args,
        **base_context.config_kwargs(),
        stage1_trainable_params=plan.stage1_trainable_params,
        stage2_trainable_params=plan.stage2_trainable_params,
        total_params=plan.total_params,
    )
    wandb_run = init_wandb_run(setup.wandb_settings, config=experiment_config)
    try:
        write_config_snapshot(args.output_dir, experiment_config)
        # The parent manual run owns W&B. The two internal Trainer objects should
        # not create their own runs, otherwise one seed becomes three runs.
        stage_wandb_settings = WandbSettings(enabled=False)
        stage1_args = build_hf_training_arguments_from_args(
            args=args,
            output_dir=Path(args.output_dir) / stage1_dir_name,
            learning_rate=getattr(args, stage1_learning_rate_attr),
            num_train_epochs=getattr(args, stage1_epochs_attr),
            precision_policy=setup.precision_policy,
            wandb_settings=stage_wandb_settings,
        )
        stage2_args = build_hf_training_arguments_from_args(
            args=args,
            output_dir=Path(args.output_dir) / stage2_dir_name,
            learning_rate=getattr(args, stage2_learning_rate_attr),
            num_train_epochs=getattr(args, stage2_epochs_attr),
            precision_policy=setup.precision_policy,
            wandb_settings=stage_wandb_settings,
        )
        stage1_trainer = build_hf_trainer(
            plan.stage1_context,
            stage1_args,
            callbacks=build_early_stopping_callbacks(args),
        )

        print(f"\n{train_message}...")
        reset_peak_memory_stats()
        synchronize_cuda()
        train_start_time = time.perf_counter()

        print(f"\nStage 1: {stage1_message}...")
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

        print(f"\nStage 2: {stage2_message}...")
        # This is the key method-specific handoff. LP-FT keeps training the same
        # model state; Efficient-Head rebuilds a fresh backbone and copies the
        # trained head. The runner does not assume which version it is.
        stage2_context = plan.build_stage2_context(stage1_trainer.model)
        stage2_trainer = build_hf_trainer(
            stage2_context,
            stage2_args,
            callbacks=build_early_stopping_callbacks(args),
        )
        stage2_start_time = time.perf_counter()
        stage2_trainer.train()
        synchronize_cuda()
        stage2_training_time_sec = time.perf_counter() - stage2_start_time
        training_time_sec = time.perf_counter() - train_start_time

        eval_metrics, test_metrics = evaluate_validation_and_optional_test(
            stage2_trainer,
            stage2_context,
        )
        model_selection = merge_stage_model_selection(
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
            precision_policy=setup.precision_policy,
            extra={
                "stage1_training_time_sec": stage1_training_time_sec,
                "stage2_training_time_sec": stage2_training_time_sec,
            },
        )
        result_paths = write_success_outputs(
            args,
            config=experiment_config,
            eval_metrics=eval_metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            prediction_paths=prediction_paths,
            model_artifact_paths=model_artifact_paths,
            wandb_run=wandb_run,
            extra_metrics={"stage1": stage1_eval_metrics},
        )

        print_run_report(
            eval_metrics=eval_metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            result_paths=result_paths,
            prediction_paths=prediction_paths,
            stage_metrics={"Stage 1 validation metrics": stage1_eval_metrics},
        )
        print(f"\nDone. Saved to: {args.output_dir}")
    finally:
        finish_wandb_run(wandb_run)
