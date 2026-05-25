import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.distilbert_full.args import parse_args
from src.methods.distilbert_full.config import (
    build_experiment_config,
    build_setup_failure_config,
)
from src.methods.distilbert_full.data import (
    build_fixed_label_maps,
    build_tokenized_dataset,
    build_tokenized_dataset_with_count,
    build_tokenized_dataset_with_stats,
    find_split_name,
    resolve_eval_split_name,
)
from src.methods.predictions import save_prediction_file
from src.methods.common import (
    clear_existing_run_artifacts,
    find_existing_run_artifacts,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
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
from src.methods.hf_common import (
    build_model_selection_summary,
    build_trainer,
    build_training_arguments,
    compute_balanced_class_weights,
    count_model_parameters,
    resolve_class_weights,
    synchronize_cuda,
    validate_checkpoint_policy,
)
from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    finish_wandb_run,
    parse_wandb_tags,
)

def resolve_wandb_settings(args) -> WandbSettings:
    run_name = args.wandb_run_name or build_wandb_run_name(
        method=args.method,
        model_name=args.model_name,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        trial_id=args.trial_id,
    )
    return WandbSettings(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=run_name,
        group=args.wandb_group,
        tags=parse_wandb_tags(args.wandb_tags),
        log_model=args.wandb_log_model,
    )


def main():
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
        trainable_params, total_params = count_model_parameters(context.model)
        print(f"Trainable params: {trainable_params:,} / {total_params:,}")

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
            learning_rate=args.learning_rate,
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
        print("\nStarting training...")
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
