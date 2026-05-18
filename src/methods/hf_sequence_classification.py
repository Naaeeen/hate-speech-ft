from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from src.methods.hf_common import (
    build_trainer,
    build_training_arguments,
    build_weighted_trainer_class,
    compute_metrics_fn,
    get_gpu_type,
    get_peak_memory_mb,
    get_peak_memory_reserved_mb,
    resolve_class_weights,
    resolve_precision_policy,
    validate_checkpoint_policy,
)
from src.methods.predictions import save_prediction_file
from src.methods.transformer_data import (
    TokenizedSplit,
    build_fixed_label_maps,
    build_tokenized_dataset_with_stats,
    find_split_name,
    resolve_eval_split_name,
)
from src.utils.wandb_config import WandbSettings, finish_wandb_run, init_wandb_run


@dataclass(frozen=True)
class HfLibraries:
    training_args_cls: Any
    early_stopping_callback_cls: Any


@dataclass(frozen=True)
class HfRunSetup:
    gpu_type: str
    precision_policy: dict[str, Any]
    experiment_config: dict[str, Any]
    wandb_settings: WandbSettings


@dataclass(frozen=True)
class HfClassificationRun:
    args: Any
    precision_policy: dict[str, Any]
    gpu_type: str
    libraries: HfLibraries
    model: Any
    tokenizer: Any
    data_collator: Any
    trainer_cls: Any
    class_weights: list[float] | None
    train_split: str
    eval_split: str
    test_split: str | None
    train_split_data: TokenizedSplit
    eval_split_data: TokenizedSplit
    test_split_data: TokenizedSplit | None
    id2label: dict[int, str]
    label2id: dict[str, int]
    num_labels: int

    @property
    def train_dataset(self) -> list[dict[str, Any]]:
        return self.train_split_data.dataset

    @property
    def eval_dataset(self) -> list[dict[str, Any]]:
        return self.eval_split_data.dataset

    @property
    def test_dataset(self) -> list[dict[str, Any]] | None:
        return self.test_split_data.dataset if self.test_split_data is not None else None

    def config_kwargs(self) -> dict[str, Any]:
        return {
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "train_size": len(self.train_dataset),
            "eval_size": len(self.eval_dataset),
            "full_train_size": self.train_split_data.preprocessed_size,
            "full_eval_size": self.eval_split_data.preprocessed_size,
            "raw_train_size": self.train_split_data.raw_size,
            "raw_eval_size": self.eval_split_data.raw_size,
            "dropped_no_majority_train": (
                self.train_split_data.dropped_no_majority_count
            ),
            "dropped_no_majority_eval": self.eval_split_data.dropped_no_majority_count,
            "test_size": (
                len(self.test_dataset) if self.test_dataset is not None else None
            ),
            "full_test_size": (
                self.test_split_data.preprocessed_size
                if self.test_split_data is not None
                else None
            ),
            "raw_test_size": (
                self.test_split_data.raw_size
                if self.test_split_data is not None
                else None
            ),
            "dropped_no_majority_test": (
                self.test_split_data.dropped_no_majority_count
                if self.test_split_data is not None
                else None
            ),
            "gpu_type": self.gpu_type,
            "class_weights": self.class_weights,
            "precision_policy": self.precision_policy,
        }


def initial_precision_policy(args: Any) -> dict[str, Any]:
    return {
        "mixed_precision": getattr(args, "mixed_precision", "unknown"),
        "fp16": bool(getattr(args, "fp16", False)),
        "bf16": getattr(args, "mixed_precision", None) == "bf16",
    }


def initialize_hf_run(
    args: Any,
    *,
    build_setup_failure_config_fn,
    resolve_wandb_settings_fn,
) -> HfRunSetup:
    try:
        validate_output_dir_for_run(
            args.output_dir,
            overwrite=args.overwrite_output_dir,
        )
    except ValueError as exc:
        print(f"Cannot start run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.overwrite_output_dir:
        removed_artifacts = clear_existing_run_artifacts(args.output_dir)
        if removed_artifacts:
            preview = ", ".join(path.name for path in removed_artifacts[:8])
            print(f"Cleared existing run artifacts from {args.output_dir}: {preview}")

    os.makedirs(args.output_dir, exist_ok=True)
    gpu_type = get_gpu_type()
    precision_policy = initial_precision_policy(args)
    experiment_config = build_setup_failure_config_fn(
        args,
        precision_policy=precision_policy,
        gpu_type=gpu_type,
    )
    return HfRunSetup(
        gpu_type=gpu_type,
        precision_policy=precision_policy,
        experiment_config=experiment_config,
        wandb_settings=resolve_wandb_settings_fn(args),
    )


def start_hf_run(
    args: Any,
    setup: HfRunSetup,
    *,
    build_setup_failure_config_fn,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    precision_policy = resolve_precision_policy(args)
    experiment_config = build_setup_failure_config_fn(
        args,
        precision_policy=precision_policy,
        gpu_type=setup.gpu_type,
    )
    validate_test_evaluation_policy(
        search_stage=args.search_stage,
        run_test=args.run_test,
    )
    validate_checkpoint_policy(args)
    wandb_run = init_wandb_run(setup.wandb_settings, config=experiment_config)
    return precision_policy, experiment_config, wandb_run


def prepare_hf_classification_run(
    args: Any,
    *,
    precision_policy: dict[str, Any],
    gpu_type: str,
) -> HfClassificationRun:
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
        raise ValueError(f"No train split found. Available splits: {list(dataset.keys())}")
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
    test_split_data = None
    if args.run_test:
        test_split_data = build_tokenized_dataset_with_stats(
            dataset[test_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_test_samples,
        )

    id2label, label2id, num_labels = build_fixed_label_maps()
    print_split_summary(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        train_split_data=train_split_data,
        eval_split_data=eval_split_data,
        test_split_data=test_split_data,
    )
    print(f"Using num_labels: {num_labels}")
    print(f"id2label: {id2label}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    class_weights = resolve_class_weights(
        class_weighting=args.class_weighting,
        train_dataset=train_split_data.dataset,
        num_labels=num_labels,
    )
    if class_weights is not None:
        print(f"Class weights ({args.class_weighting}): {class_weights}")

    trainer_cls = (
        build_weighted_trainer_class(Trainer, class_weights)
        if class_weights is not None
        else Trainer
    )
    return HfClassificationRun(
        args=args,
        precision_policy=precision_policy,
        gpu_type=gpu_type,
        libraries=HfLibraries(
            training_args_cls=TrainingArguments,
            early_stopping_callback_cls=EarlyStoppingCallback,
        ),
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        trainer_cls=trainer_cls,
        class_weights=class_weights,
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        train_split_data=train_split_data,
        eval_split_data=eval_split_data,
        test_split_data=test_split_data,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
    )


def print_split_summary(
    *,
    train_split: str,
    eval_split: str,
    test_split: str | None,
    train_split_data: TokenizedSplit,
    eval_split_data: TokenizedSplit,
    test_split_data: TokenizedSplit | None,
) -> None:
    print(
        f"Train split: {train_split}, size={len(train_split_data.dataset)} "
        f"(preprocessed full={train_split_data.preprocessed_size})"
    )
    print(
        f"Eval split: {eval_split}, size={len(eval_split_data.dataset)} "
        f"(preprocessed full={eval_split_data.preprocessed_size})"
    )
    print(
        "Strict-majority dropped: "
        f"train={train_split_data.dropped_no_majority_count}, "
        f"eval={eval_split_data.dropped_no_majority_count}"
    )
    if test_split_data is not None:
        print(
            f"Test split: {test_split}, size={len(test_split_data.dataset)} "
            f"(preprocessed full={test_split_data.preprocessed_size})"
        )
        print(
            "Strict-majority dropped: "
            f"test={test_split_data.dropped_no_majority_count}"
        )


def build_hf_training_arguments_from_args(
    training_args_cls,
    *,
    args: Any,
    output_dir: str | Path,
    learning_rate: float,
    num_train_epochs: float,
    precision_policy: dict[str, Any],
    wandb_settings: WandbSettings,
    per_device_train_batch_size: int | None = None,
    per_device_eval_batch_size: int | None = None,
):
    return build_training_arguments(
        training_args_cls,
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=(
            per_device_train_batch_size
            if per_device_train_batch_size is not None
            else args.per_device_train_batch_size
        ),
        per_device_eval_batch_size=(
            per_device_eval_batch_size
            if per_device_eval_batch_size is not None
            else args.per_device_eval_batch_size
        ),
        num_train_epochs=num_train_epochs,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        overwrite_output_dir=args.overwrite_output_dir,
        report_to=wandb_settings.report_to,
        run_name=wandb_settings.run_name if wandb_settings.enabled else None,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=not args.lower_is_better,
        fp16=precision_policy["fp16"],
        bf16=precision_policy["bf16"],
        gradient_checkpointing=args.gradient_checkpointing,
    )


def build_early_stopping_callbacks(early_stopping_callback_cls, args: Any) -> list[Any]:
    if args.early_stopping_patience <= 0:
        return []
    return [
        early_stopping_callback_cls(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    ]


def build_hf_trainer(
    context: HfClassificationRun,
    training_args,
    *,
    callbacks: list[Any] | None = None,
):
    return build_trainer(
        trainer_cls=context.trainer_cls,
        model=context.model,
        training_args=training_args,
        train_dataset=context.train_dataset,
        eval_dataset=context.eval_dataset,
        tokenizer=context.tokenizer,
        data_collator=context.data_collator,
        compute_metrics=compute_metrics_fn(),
        callbacks=callbacks,
    )


def write_config_snapshot(
    output_dir: str | Path,
    experiment_config: dict[str, Any],
    wandb_run,
) -> Path:
    resolved_config_path = write_resolved_config(output_dir, experiment_config)
    print(f"Resolved config: {resolved_config_path}")
    if wandb_run is not None:
        wandb_run.config.update(experiment_config, allow_val_change=True)
    return resolved_config_path


def reset_peak_memory_stats() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def evaluate_validation_and_optional_test(
    trainer,
    context: HfClassificationRun,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
) -> None:
    if no_save_final_model:
        print("\nSkipping final model save because --no_save_final_model was set.")
        return

    print(f"\nSaving final model and tokenizer from {model_source}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_final_predictions(
    context: HfClassificationRun,
    trainer,
) -> dict[str, Path]:
    prediction_paths = {}
    if context.args.search_stage != "final":
        return prediction_paths

    print("\nSaving evaluation predictions...")
    eval_prediction_output = trainer.predict(
        context.eval_dataset,
        metric_key_prefix="eval_predictions",
    )
    prediction_paths["eval"] = save_prediction_file(
        Path(context.args.output_dir) / "eval_predictions.json",
        records=context.eval_split_data.records,
        prediction_output=eval_prediction_output,
        id2label=context.id2label,
    )
    if context.args.run_test:
        if context.test_split_data is None:
            raise ValueError("Cannot save test predictions before loading a test split.")
        print("Saving test predictions...")
        test_prediction_output = trainer.predict(
            context.test_dataset,
            metric_key_prefix="test_predictions",
        )
        prediction_paths["test"] = save_prediction_file(
            Path(context.args.output_dir) / "test_predictions.json",
            records=context.test_split_data.records,
            prediction_output=test_prediction_output,
            id2label=context.id2label,
        )
    return prediction_paths


def build_runtime_metrics(
    args: Any,
    *,
    training_time_sec: float | None,
    gpu_type: str,
    precision_policy: dict[str, Any],
    status: str,
    failure_phase: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        "training_time_sec": training_time_sec,
        "peak_memory_mb": get_peak_memory_mb(),
        "peak_memory_allocated_mb": get_peak_memory_mb(),
        "peak_memory_reserved_mb": get_peak_memory_reserved_mb(),
        "gpu_type": gpu_type,
        "mixed_precision": precision_policy["mixed_precision"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "status": status,
    }
    if failure_phase is not None:
        metrics["failure_phase"] = failure_phase
    if extra:
        metrics.update(extra)
    return metrics


def elapsed_since(start_time: float | None) -> float | None:
    return time.perf_counter() - start_time if start_time is not None else None


def write_failure_summary(
    args: Any,
    *,
    config: dict[str, Any],
    error: Exception,
    runtime_metrics: dict[str, Any],
    wandb_run,
) -> Path:
    failure_path = write_failure_file(
        args.output_dir,
        config=config,
        error=error,
        runtime_metrics=runtime_metrics,
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "status": "failed",
                "failure_phase": runtime_metrics.get("failure_phase"),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
    print(f"\nFailure summary: {failure_path}")
    return failure_path


def finish_failed_setup_run(
    args: Any,
    *,
    config: dict[str, Any],
    error: Exception,
    gpu_type: str,
    precision_policy: dict[str, Any],
    wandb_run,
) -> None:
    runtime_metrics = build_runtime_metrics(
        args,
        training_time_sec=None,
        gpu_type=gpu_type,
        precision_policy=precision_policy,
        status="failed",
        failure_phase="setup",
    )
    write_failure_summary(
        args,
        config=config,
        error=error,
        runtime_metrics=runtime_metrics,
        wandb_run=wandb_run,
    )
    finish_wandb_run(wandb_run)


def finish_failed_train_run(
    args: Any,
    *,
    config: dict[str, Any],
    error: Exception,
    gpu_type: str,
    precision_policy: dict[str, Any],
    wandb_run,
    train_start_time: float | None,
) -> None:
    runtime_metrics = build_runtime_metrics(
        args,
        training_time_sec=elapsed_since(train_start_time),
        gpu_type=gpu_type,
        precision_policy=precision_policy,
        status="failed",
        failure_phase="train_eval",
    )
    write_failure_summary(
        args,
        config=config,
        error=error,
        runtime_metrics=runtime_metrics,
        wandb_run=wandb_run,
    )


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
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    if wandb_run is not None:
        wandb_run.log(runtime_metrics)
        wandb_run.log({"model_selection": model_selection})

    return write_result_files(
        args.output_dir,
        config=config,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        runtime_metrics=runtime_metrics,
        model_selection=model_selection,
        prediction_paths=prediction_paths,
        extra_metrics=extra_metrics,
    )


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
