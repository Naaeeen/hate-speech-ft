"""Shared runner for the one-stage Transformer methods.

This is the "boring but important" path used by Full FT, Frozen DistilBERT,
and LoRA. A method file gives this runner its manual config, tells it which
learning-rate field to read, and optionally tweaks the model context before
training. The runner then does the same lifecycle for everybody: setup,
training, validation/test evaluation, local JSON outputs, model saving, and one
W&B run.
"""

from __future__ import annotations

import time
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
from src.methods.transformer_setup import (
    prepare_hf_classification_run,
    start_hf_run,
)
from src.methods.transformer_trainer import (
    build_early_stopping_callbacks,
    build_hf_trainer,
    build_hf_training_arguments_from_args,
    build_model_selection_summary,
)
from src.utils.run_metadata import (
    count_model_parameters,
    reset_peak_memory_stats,
    synchronize_cuda,
)
from src.utils.wandb_config import (
    finish_wandb_run,
    init_wandb_run,
)


PrepareContext = Callable[[Any, Any], Any]
BuildConfig = Callable[..., dict[str, Any]]


def run_single_stage_transformer(
    args: Any,
    *,
    build_experiment_config: BuildConfig,
    learning_rate_attr: str,
    prepare_context: PrepareContext | None = None,
    params_label: str = "Trainable params",
    train_message: str = "Starting training",
) -> None:
    """Run one manual Transformer experiment from start to finish.

    Method packages keep their own identity and hyperparameter config, but this
    helper owns the shared mechanics. That keeps Full FT, Frozen, and LoRA
    comparable while each method still has its own direct entrypoint.
    """

    setup = start_hf_run(args)
    context = prepare_hf_classification_run(
        args,
        precision_policy=setup.precision_policy,
        gpu_type=setup.gpu_type,
    )
    if prepare_context is not None:
        context = prepare_context(context, args) or context

    trainable_params, total_params = count_model_parameters(context.model)
    print(f"{params_label}: {trainable_params:,} / {total_params:,}")

    experiment_config = build_experiment_config(
        args,
        **context.config_kwargs(),
        trainable_params=trainable_params,
        total_params=total_params,
    )
    wandb_run = init_wandb_run(setup.wandb_settings, config=experiment_config)
    try:
        write_config_snapshot(args.output_dir, experiment_config)
        training_args = build_hf_training_arguments_from_args(
            args=args,
            output_dir=args.output_dir,
            learning_rate=getattr(args, learning_rate_attr),
            num_train_epochs=args.num_train_epochs,
            precision_policy=setup.precision_policy,
            wandb_settings=setup.wandb_settings,
        )
        trainer = build_hf_trainer(
            context,
            training_args,
            callbacks=build_early_stopping_callbacks(args),
        )

        print(f"\n{train_message}...")
        reset_peak_memory_stats()
        synchronize_cuda()
        train_start_time = time.perf_counter()
        trainer.train()
        synchronize_cuda()
        training_time_sec = time.perf_counter() - train_start_time

        eval_metrics, test_metrics = evaluate_validation_and_optional_test(
            trainer,
            context,
        )
        model_selection = build_model_selection_summary(
            trainer,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=not args.lower_is_better,
        )
        model_artifact_paths = save_final_model(
            trainer,
            context.tokenizer,
            output_dir=args.output_dir,
            no_save_final_model=args.no_save_final_model,
            model_source=(
                "best checkpoint" if args.load_best_model_at_end else "last training state"
            ),
        )
        prediction_paths = save_final_predictions(context, trainer)
        runtime_metrics = build_runtime_metrics(
            args,
            training_time_sec=training_time_sec,
            gpu_type=setup.gpu_type,
            precision_policy=setup.precision_policy,
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
        )

        print_run_report(
            eval_metrics=eval_metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            result_paths=result_paths,
            prediction_paths=prediction_paths,
        )
        print(f"\nDone. Saved to: {args.output_dir}")
    finally:
        finish_wandb_run(wandb_run)
