import os
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
from src.experiments.results import (
    write_json,
    write_failure_file,
    write_resolved_config,
    write_result_files,
)
from src.methods.common import (
    clear_existing_run_artifacts,
    find_existing_run_artifacts,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)
from src.methods.hf_common import (
    build_model_selection_summary,
    build_trainer,
    build_training_arguments,
    build_weighted_trainer_class,
    compute_balanced_class_weights,
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
from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    finish_wandb_run,
    init_wandb_run,
    parse_wandb_tags,
)


def _as_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def save_prediction_file(
    path: str | Path,
    *,
    records: list[dict],
    prediction_output,
    id2label: dict[int, str],
):
    logits = _as_list(prediction_output.predictions)
    labels = _as_list(prediction_output.label_ids)
    if len(records) != len(logits) or len(records) != len(labels):
        raise ValueError(
            "Prediction output length does not match source records: "
            f"records={len(records)}, logits={len(logits)}, labels={len(labels)}"
        )

    predictions = []
    for record, logit_values, label_id in zip(records, logits, labels):
        logit_list = [float(value) for value in _as_list(logit_values)]
        predicted_label = int(_argmax(logit_list))
        gold_label = int(label_id)
        predictions.append(
            {
                "id": record.get("id"),
                "text": record.get("text"),
                "label": gold_label,
                "label_name": id2label.get(gold_label),
                "predicted_label": predicted_label,
                "predicted_label_name": id2label.get(predicted_label),
                "logits": logit_list,
            }
        )

    return write_json(
        path,
        {
            "count": len(predictions),
            "predictions": predictions,
        },
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
        ds = load_dataset(args.dataset_name)

        print("Available splits:", list(ds.keys()))

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        train_split = find_split_name(ds, ["train"])
        eval_split = resolve_eval_split_name(ds, test_split_name=args.test_split_name)
        test_split = find_split_name(ds, [args.test_split_name])

        if train_split is None:
            raise ValueError(f"No train split found. Available splits: {list(ds.keys())}")
        if args.run_test and test_split is None:
            raise ValueError(
                f"No test split named '{args.test_split_name}' found. "
                f"Available splits: {list(ds.keys())}"
            )
        if args.run_test and eval_split == test_split:
            raise ValueError(
                f"Evaluation split '{eval_split}' and test split '{test_split}' must be distinct."
            )

        train_split_data = build_tokenized_dataset_with_stats(
            ds[train_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            data_fraction=args.data_fraction,
            fraction_seed=args.data_fraction_seed,
            max_samples=args.max_train_samples,
        )
        eval_split_data = build_tokenized_dataset_with_stats(
            ds[eval_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_eval_samples,
        )
        train_dataset = train_split_data.dataset
        eval_dataset = eval_split_data.dataset
        full_train_size = train_split_data.preprocessed_size
        full_eval_size = eval_split_data.preprocessed_size
        test_dataset = None
        test_split_data = None
        full_test_size = None
        if args.run_test:
            test_split_data = build_tokenized_dataset_with_stats(
                ds[test_split],
                tokenizer=tokenizer,
                max_length=args.max_length,
                max_samples=args.max_test_samples,
            )
            test_dataset = test_split_data.dataset
            full_test_size = test_split_data.preprocessed_size

        id2label, label2id, num_labels = build_fixed_label_maps()

        print(
            f"Train split: {train_split}, size={len(train_dataset)} "
            f"(preprocessed full={full_train_size})"
        )
        print(
            f"Eval split: {eval_split}, size={len(eval_dataset)} "
            f"(preprocessed full={full_eval_size})"
        )
        print(
            "Strict-majority dropped: "
            f"train={train_split_data.dropped_no_majority_count}, "
            f"eval={eval_split_data.dropped_no_majority_count}"
        )
        print(f"Using num_labels: {num_labels}")
        print(f"id2label: {id2label}")
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
        trainable_params, total_params = count_model_parameters(model)
        print(f"Trainable params: {trainable_params:,} / {total_params:,}")

        precision_policy = resolve_precision_policy(args)
        class_weights = resolve_class_weights(
            class_weighting=args.class_weighting,
            train_dataset=train_dataset,
            num_labels=num_labels,
        )
        if class_weights is not None:
            print(f"Class weights ({args.class_weighting}): {class_weights}")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        gpu_type = get_gpu_type()
        experiment_config = build_experiment_config(
            args,
            train_split=train_split,
            eval_split=eval_split,
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
            full_train_size=full_train_size,
            full_eval_size=full_eval_size,
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
            trainable_params=trainable_params,
            total_params=total_params,
            gpu_type=gpu_type,
            class_weights=class_weights,
            precision_policy=precision_policy,
        )
        resolved_config_path = write_resolved_config(args.output_dir, experiment_config)
        print(f"Resolved config: {resolved_config_path}")
        if wandb_run is not None:
            wandb_run.config.update(experiment_config, allow_val_change=True)

        training_args = build_training_arguments(
            TrainingArguments,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
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

        callbacks = []
        if args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=args.early_stopping_threshold,
                )
            )
        trainer_cls = (
            build_weighted_trainer_class(Trainer, class_weights)
            if class_weights is not None
            else Trainer
        )
        trainer = build_trainer(
            trainer_cls=trainer_cls,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn(),
            callbacks=callbacks,
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
        print("\nStarting training...")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        synchronize_cuda()
        train_start_time = time.perf_counter()
        trainer.train()
        synchronize_cuda()
        training_time_sec = time.perf_counter() - train_start_time

        print("\nRunning evaluation...")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        test_metrics = None
        if args.run_test:
            print("\nRunning test evaluation...")
            test_metrics = trainer.evaluate(
                eval_dataset=test_dataset,
                metric_key_prefix="test",
            )
        model_selection = build_model_selection_summary(
            trainer,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=not args.lower_is_better,
        )

        if args.no_save_final_model:
            print("\nSkipping final model save because --no_save_final_model was set.")
        else:
            model_source = (
                "best checkpoint" if args.load_best_model_at_end else "last training state"
            )
            print(f"\nSaving final model and tokenizer from {model_source}...")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

        prediction_paths = {}
        if args.search_stage == "final":
            print("\nSaving evaluation predictions...")
            eval_prediction_output = trainer.predict(
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
                test_prediction_output = trainer.predict(
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
        result_paths = write_result_files(
            args.output_dir,
            config=experiment_config,
            eval_metrics=metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            model_selection=model_selection,
            prediction_paths=prediction_paths,
        )

        print("\nFinal evaluation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        if test_metrics is not None:
            print("\nFinal test metrics:")
            for k, v in test_metrics.items():
                print(f"{k}: {v}")
        print("\nRuntime metrics:")
        for k, v in runtime_metrics.items():
            print(f"{k}: {v}")
        print("\nModel selection:")
        for k, v in model_selection.items():
            print(f"{k}: {v}")
        print("\nResult files:")
        for k, v in result_paths.items():
            print(f"{k}: {v}")
        if prediction_paths:
            print("\nPrediction files:")
            for k, v in prediction_paths.items():
                print(f"{k}: {v}")

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
