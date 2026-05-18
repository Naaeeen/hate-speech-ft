from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.results import (
    write_failure_file,
    write_resolved_config,
    write_result_files,
)
from src.methods.common import (
    clear_existing_run_artifacts,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)
from src.methods.distilbert_lp_ft.args import parse_args
from src.methods.distilbert_lp_ft.config import (
    build_experiment_config,
    build_setup_failure_config,
)
from src.methods.distilbert_lp_ft.training import (
    STAGE1_DIR_NAME,
    STAGE2_DIR_NAME,
    build_callbacks,
    build_stage_training_arguments,
    resolve_wandb_settings,
    set_full_finetune_trainability,
    set_linear_probe_trainability,
)
from src.methods.hf_common import (
    build_model_selection_summary,
    build_trainer,
    build_weighted_trainer_class,
    compute_metrics_fn,
    count_model_parameters,
    get_gpu_type,
    get_peak_memory_mb,
    get_peak_memory_reserved_mb,
    resolve_class_weights,
    resolve_precision_policy,
    synchronize_cuda,
    validate_checkpoint_policy,
)
from src.methods.predictions import save_prediction_file
from src.methods.transformer_data import (
    build_fixed_label_maps,
    build_tokenized_dataset_with_stats,
    find_split_name,
    resolve_eval_split_name,
)
from src.utils.wandb_config import (
    finish_wandb_run,
    init_wandb_run,
)


def _prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    try:
        validate_output_dir_for_run(args.output_dir, overwrite=args.overwrite_output_dir)
    except ValueError as exc:
        print(f"Cannot start run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.overwrite_output_dir:
        removed_artifacts = clear_existing_run_artifacts(args.output_dir)
        if removed_artifacts:
            preview = ", ".join(path.name for path in removed_artifacts[:8])
            print(f"Cleared existing run artifacts from {args.output_dir}: {preview}")

    os.makedirs(args.output_dir, exist_ok=True)
    wandb_run = None
    gpu_type = get_gpu_type()
    precision_policy = {
        "mixed_precision": args.mixed_precision,
        "fp16": args.fp16,
        "bf16": args.mixed_precision == "bf16",
    }
    experiment_config = build_setup_failure_config(
        args,
        precision_policy=precision_policy,
        gpu_type=gpu_type,
    )
    wandb_settings = resolve_wandb_settings(args)

    try:
        precision_policy = resolve_precision_policy(args)
        experiment_config = build_setup_failure_config(
            args,
            precision_policy=precision_policy,
            gpu_type=gpu_type,
        )
        validate_test_evaluation_policy(
            search_stage=args.search_stage,
            run_test=args.run_test,
        )
        validate_checkpoint_policy(args)
        wandb_run = init_wandb_run(wandb_settings, config=experiment_config)

        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        set_seed(args.seed)

        print(f"Loading dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)
        print("Available splits:", list(dataset.keys()))

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_split = find_split_name(dataset, ["train"])
        eval_split = resolve_eval_split_name(
            dataset,
            test_split_name=args.test_split_name,
        )
        test_split = find_split_name(dataset, [args.test_split_name])
        if train_split is None:
            raise ValueError(
                f"No train split found. Available splits: {list(dataset.keys())}"
            )
        if args.run_test and test_split is None:
            raise ValueError(
                f"No test split named '{args.test_split_name}' found. "
                f"Available splits: {list(dataset.keys())}"
            )
        if args.run_test and eval_split == test_split:
            raise ValueError(
                f"Evaluation split '{eval_split}' and test split '{test_split}' "
                "must be distinct."
            )

        train_split_data = build_tokenized_dataset_with_stats(
            dataset[train_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            data_fraction=args.data_fraction,
            fraction_seed=args.data_fraction_seed,
            max_samples=args.max_train_samples,
        )
        eval_split_data = build_tokenized_dataset_with_stats(
            dataset[eval_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_eval_samples,
        )
        train_dataset = train_split_data.dataset
        eval_dataset = eval_split_data.dataset
        test_dataset = None
        test_split_data = None
        full_test_size = None
        if args.run_test:
            test_split_data = build_tokenized_dataset_with_stats(
                dataset[test_split],
                tokenizer=tokenizer,
                max_length=args.max_length,
                max_samples=args.max_test_samples,
            )
            test_dataset = test_split_data.dataset
            full_test_size = test_split_data.preprocessed_size

        id2label, label2id, num_labels = build_fixed_label_maps()
        print(
            f"Train split: {train_split}, size={len(train_dataset)} "
            f"(preprocessed full={train_split_data.preprocessed_size})"
        )
        print(
            f"Eval split: {eval_split}, size={len(eval_dataset)} "
            f"(preprocessed full={eval_split_data.preprocessed_size})"
        )
        print(
            "Strict-majority dropped: "
            f"train={train_split_data.dropped_no_majority_count}, "
            f"eval={eval_split_data.dropped_no_majority_count}"
        )
        if args.run_test:
            print(
                f"Test split: {test_split}, size={len(test_dataset)} "
                f"(preprocessed full={full_test_size})"
            )
            print(
                "Strict-majority dropped: "
                f"test={test_split_data.dropped_no_majority_count}"
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        set_linear_probe_trainability(model)
        stage1_trainable_params, total_params = count_model_parameters(model)
        set_full_finetune_trainability(model)
        stage2_trainable_params, _ = count_model_parameters(model)
        set_linear_probe_trainability(model)
        print(
            "Stage 1 trainable params: "
            f"{stage1_trainable_params:,} / {total_params:,}"
        )
        print(
            "Stage 2 trainable params: "
            f"{stage2_trainable_params:,} / {total_params:,}"
        )

        class_weights = resolve_class_weights(
            class_weighting=args.class_weighting,
            train_dataset=train_dataset,
            num_labels=num_labels,
        )
        if class_weights is not None:
            print(f"Class weights ({args.class_weighting}): {class_weights}")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        experiment_config = build_experiment_config(
            args,
            train_split=train_split,
            eval_split=eval_split,
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
            full_train_size=train_split_data.preprocessed_size,
            full_eval_size=eval_split_data.preprocessed_size,
            raw_train_size=train_split_data.raw_size,
            raw_eval_size=eval_split_data.raw_size,
            dropped_no_majority_train=train_split_data.dropped_no_majority_count,
            dropped_no_majority_eval=eval_split_data.dropped_no_majority_count,
            test_size=len(test_dataset) if test_dataset is not None else None,
            full_test_size=full_test_size,
            raw_test_size=test_split_data.raw_size if test_split_data is not None else None,
            dropped_no_majority_test=(
                test_split_data.dropped_no_majority_count
                if test_split_data is not None
                else None
            ),
            stage1_trainable_params=stage1_trainable_params,
            stage2_trainable_params=stage2_trainable_params,
            total_params=total_params,
            gpu_type=gpu_type,
            class_weights=class_weights,
            precision_policy=precision_policy,
        )
        resolved_config_path = write_resolved_config(args.output_dir, experiment_config)
        print(f"Resolved config: {resolved_config_path}")
        if wandb_run is not None:
            wandb_run.config.update(experiment_config, allow_val_change=True)

        trainer_cls = (
            build_weighted_trainer_class(Trainer, class_weights)
            if class_weights is not None
            else Trainer
        )

        stage1_args = build_stage_training_arguments(
            TrainingArguments,
            args=args,
            output_dir=Path(args.output_dir) / STAGE1_DIR_NAME,
            learning_rate=args.stage1_head_learning_rate,
            num_train_epochs=args.stage1_epochs,
            precision_policy=precision_policy,
            wandb_settings=wandb_settings,
        )
        stage1_trainer = build_trainer(
            trainer_cls=trainer_cls,
            model=model,
            training_args=stage1_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn(),
            callbacks=build_callbacks(EarlyStoppingCallback, args),
        )

        stage2_args = build_stage_training_arguments(
            TrainingArguments,
            args=args,
            output_dir=Path(args.output_dir) / STAGE2_DIR_NAME,
            learning_rate=args.stage2_learning_rate,
            num_train_epochs=args.stage2_epochs,
            precision_policy=precision_policy,
            wandb_settings=wandb_settings,
        )

    except Exception as exc:
        runtime_metrics = {
            "training_time_sec": None,
            "peak_memory_mb": get_peak_memory_mb(),
            "peak_memory_allocated_mb": get_peak_memory_mb(),
            "peak_memory_reserved_mb": get_peak_memory_reserved_mb(),
            "gpu_type": gpu_type,
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "status": "failed",
            "failure_phase": "setup",
        }
        failure_path = write_failure_file(
            args.output_dir,
            config=experiment_config,
            error=exc,
            runtime_metrics=runtime_metrics,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "status": "failed",
                    "failure_phase": "setup",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        finish_wandb_run(wandb_run)
        print(f"\nFailure summary: {failure_path}")
        raise

    try:
        print("\nStarting LP+FT training...")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

        synchronize_cuda()
        train_start_time = time.perf_counter()

        print("\nStage 1: linear probing classification head...")
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

        print("\nStage 2: full fine-tuning all parameters...")
        set_full_finetune_trainability(model)
        stage2_trainer = build_trainer(
            trainer_cls=trainer_cls,
            model=model,
            training_args=stage2_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn(),
            callbacks=build_callbacks(EarlyStoppingCallback, args),
        )
        stage2_start_time = time.perf_counter()
        stage2_trainer.train()
        synchronize_cuda()
        stage2_training_time_sec = time.perf_counter() - stage2_start_time
        training_time_sec = time.perf_counter() - train_start_time

        print("\nRunning final validation evaluation...")
        metrics = stage2_trainer.evaluate(metric_key_prefix="eval")
        test_metrics = None
        if args.run_test:
            print("\nRunning final test evaluation...")
            test_metrics = stage2_trainer.evaluate(
                eval_dataset=test_dataset,
                metric_key_prefix="test",
            )

        model_selection = build_model_selection_summary(
            stage2_trainer,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=not args.lower_is_better,
        )
        model_selection.update(
            {
                "stage1_best_metric": stage1_model_selection.get("best_metric"),
                "stage1_best_epoch": stage1_model_selection.get("best_epoch"),
                "stage1_best_step": stage1_model_selection.get("best_step"),
                "stage1_best_model_checkpoint": stage1_model_selection.get(
                    "best_model_checkpoint"
                ),
                "stage2_best_metric": model_selection.get("best_metric"),
                "stage2_best_epoch": model_selection.get("best_epoch"),
                "stage2_best_step": model_selection.get("best_step"),
            }
        )

        if args.no_save_final_model:
            print("\nSkipping final model save because --no_save_final_model was set.")
        else:
            model_source = (
                "best stage-2 checkpoint"
                if args.load_best_model_at_end
                else "last stage-2 training state"
            )
            print(f"\nSaving final model and tokenizer from {model_source}...")
            stage2_trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

        prediction_paths = {}
        if args.search_stage == "final":
            print("\nSaving evaluation predictions...")
            eval_prediction_output = stage2_trainer.predict(
                eval_dataset,
                metric_key_prefix="eval_predictions",
            )
            prediction_paths["eval"] = save_prediction_file(
                Path(args.output_dir) / "eval_predictions.json",
                records=eval_split_data.records,
                prediction_output=eval_prediction_output,
                id2label=id2label,
            )
            if args.run_test:
                print("Saving test predictions...")
                test_prediction_output = stage2_trainer.predict(
                    test_dataset,
                    metric_key_prefix="test_predictions",
                )
                prediction_paths["test"] = save_prediction_file(
                    Path(args.output_dir) / "test_predictions.json",
                    records=test_split_data.records,
                    prediction_output=test_prediction_output,
                    id2label=id2label,
                )

        runtime_metrics = {
            "training_time_sec": training_time_sec,
            "stage1_training_time_sec": stage1_training_time_sec,
            "stage2_training_time_sec": stage2_training_time_sec,
            "peak_memory_mb": get_peak_memory_mb(),
            "peak_memory_allocated_mb": get_peak_memory_mb(),
            "peak_memory_reserved_mb": get_peak_memory_reserved_mb(),
            "gpu_type": gpu_type,
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "status": "completed",
        }
        if wandb_run is not None:
            wandb_run.log(runtime_metrics)
            wandb_run.log({"model_selection": model_selection})
            wandb_run.log(_prefixed_metrics("stage1", stage1_eval_metrics))

        result_paths = write_result_files(
            args.output_dir,
            config=experiment_config,
            eval_metrics=metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            prediction_paths=prediction_paths,
        )

        print("\nStage 1 validation metrics:")
        for key, value in stage1_eval_metrics.items():
            print(f"{key}: {value}")
        print("\nFinal validation metrics:")
        for key, value in metrics.items():
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
        print(f"\nDone. Saved to: {args.output_dir}")

    except Exception as exc:
        runtime_metrics = {
            "training_time_sec": (
                time.perf_counter() - train_start_time
                if "train_start_time" in locals()
                else None
            ),
            "peak_memory_mb": get_peak_memory_mb(),
            "peak_memory_allocated_mb": get_peak_memory_mb(),
            "peak_memory_reserved_mb": get_peak_memory_reserved_mb(),
            "gpu_type": gpu_type,
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "status": "failed",
            "failure_phase": "train_eval",
        }
        failure_path = write_failure_file(
            args.output_dir,
            config=experiment_config,
            error=exc,
            runtime_metrics=runtime_metrics,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "status": "failed",
                    "failure_phase": "train_eval",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        print(f"\nFailure summary: {failure_path}")
        raise
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
