from __future__ import annotations

import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.frozen_distilbert.args import parse_args
from src.methods.frozen_distilbert.config import (
    build_experiment_config,
    build_setup_failure_config,
)
from src.methods.frozen_distilbert.training import (
    resolve_wandb_settings,
    set_frozen_backbone_trainability,
)
from src.methods.hf_common import (
    build_model_selection_summary,
    count_model_parameters,
    synchronize_cuda,
)
from src.methods.hf_sequence_classification import (
    build_early_stopping_callbacks,
    build_hf_trainer,
    build_hf_training_arguments_from_args,
    build_runtime_metrics,
    evaluate_validation_and_optional_test,
    finish_failed_setup_run,
    finish_failed_train_run,
    initialize_hf_run,
    prepare_hf_output_dir,
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
        context = prepare_hf_classification_run(
            args,
            precision_policy=precision_policy,
            gpu_type=setup.gpu_type,
        )
        set_frozen_backbone_trainability(context.model)
        trainable_params, total_params = count_model_parameters(context.model)
        print(f"Trainable head params: {trainable_params:,} / {total_params:,}")

        experiment_config = build_experiment_config(
            args,
            **context.config_kwargs(),
            trainable_params=trainable_params,
            total_params=total_params,
        )
        prepare_hf_output_dir(args)
        write_config_snapshot(args.output_dir, experiment_config, wandb_run)

        training_args = build_hf_training_arguments_from_args(
            context.libraries.training_args_cls,
            args=args,
            output_dir=args.output_dir,
            learning_rate=args.head_learning_rate,
            num_train_epochs=args.num_train_epochs,
            precision_policy=precision_policy,
            wandb_settings=setup.wandb_settings,
        )
        trainer = build_hf_trainer(
            context,
            training_args,
            callbacks=build_early_stopping_callbacks(
                context.libraries.early_stopping_callback_cls,
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
        print("\nStarting frozen-backbone DistilBERT training...")
        reset_peak_memory_stats()
        synchronize_cuda()
        train_start_time = time.perf_counter()
        trainer.train()
        synchronize_cuda()
        training_time_sec = time.perf_counter() - train_start_time

        metrics, test_metrics = evaluate_validation_and_optional_test(
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
            precision_policy=precision_policy,
            status="completed",
        )
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
        )

        print_run_report(
            eval_metrics=metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            result_paths=result_paths,
            prediction_paths=prediction_paths,
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
