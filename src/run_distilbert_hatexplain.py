import argparse
import inspect
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.label_policy import LABEL_ID_TO_NAME, LABEL_NAME_TO_ID
from src.data.preprocessing import (
    preprocess_hatexplain_split,
    select_data_fraction,
    tokenize_preprocessed_record,
)
from src.experiments.results import (
    write_failure_file,
    write_resolved_config,
    write_result_files,
)
from src.utils.wandb_config import (
    VALID_WANDB_LOG_MODEL_VALUES,
    VALID_WANDB_MODES,
    WandbSettings,
    build_wandb_run_name,
    finish_wandb_run,
    init_wandb_run,
    parse_wandb_tags,
)


MODEL_NAME = "distilbert-base-uncased"
RUN_ARTIFACT_NAMES = {
    "result_summary.json",
    "failure_summary.json",
    "metrics.json",
    "runtime.json",
    "resolved_config.json",
    "trainer_state.json",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run DistilBERT on HateXplain")

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, default="Hate-speech-CNERG/hatexplain")
    parser.add_argument("--output_dir", type=str, default="./outputs/distilbert_hatexplain")
    parser.add_argument("--method", type=str, default="full-ft")
    parser.add_argument("--search_stage", type=str, default="smoke")
    parser.add_argument("--trial_id", type=str, default=None)
    parser.add_argument("--config_hash", type=str, default=None)
    parser.add_argument("--hpo_seed", type=int, default=None)
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument("--run_test", action="store_true")

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_fraction_seed", type=int, default=42)
    parser.add_argument("--data_fraction", type=float, default=None)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument(
        "--eval_strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
    )
    parser.add_argument(
        "--save_strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
    )
    parser.add_argument(
        "--logging_strategy",
        choices=("no", "steps", "epoch"),
        default="steps",
    )
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help=(
            "Allow writing into an output directory that already contains run "
            "artifacts. By default, existing result/checkpoint/model files are "
            "protected to avoid accidental experiment loss."
        ),
    )
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_f1_macro")
    parser.add_argument("--lower_is_better", action="store_true")
    parser.add_argument("--no_save_final_model", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        choices=("none", "fp16", "bf16"),
        default="none",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--class_weighting",
        choices=("none", "balanced"),
        default="none",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        choices=VALID_WANDB_MODES,
        default="online",
    )
    parser.add_argument(
        "--wandb_log_model",
        choices=VALID_WANDB_LOG_MODEL_VALUES,
        default="false",
    )

    return parser.parse_args()


def find_split_name(ds_dict, candidates):
    for name in candidates:
        if name in ds_dict:
            return name
    return None


def maybe_select_subset(records: list[dict[str, Any]], max_samples: int | None):
    if max_samples is None:
        return records
    return records[: min(max_samples, len(records))]


def build_fixed_label_maps():
    return dict(LABEL_ID_TO_NAME), dict(LABEL_NAME_TO_ID), len(LABEL_ID_TO_NAME)


def count_model_parameters(model) -> tuple[int, int]:
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
    return trainable_params, total_params


def get_gpu_type() -> str:
    try:
        import torch
    except ImportError:
        return "unknown"
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def get_peak_memory_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def get_peak_memory_reserved_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / (1024 * 1024)


def find_existing_run_artifacts(output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        return []

    artifacts = [
        output_path / name
        for name in RUN_ARTIFACT_NAMES
        if (output_path / name).exists()
    ]
    artifacts.extend(path for path in output_path.glob("checkpoint-*") if path.exists())
    return sorted(set(artifacts), key=lambda path: path.as_posix())


def validate_output_dir_for_run(output_dir: str | Path, *, overwrite: bool) -> None:
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_path}")

    artifacts = find_existing_run_artifacts(output_path)
    if artifacts and not overwrite:
        preview = ", ".join(path.name for path in artifacts[:5])
        if len(artifacts) > 5:
            preview += f", +{len(artifacts) - 5} more"
        raise ValueError(
            f"Output directory already contains run artifacts: {output_path} "
            f"({preview}). Use a unique --output_dir for a new experiment run, "
            "or pass --overwrite_output_dir only when intentionally replacing "
            "the previous local results."
        )


def synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_git_commit_hash(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def resolve_precision_policy(args) -> dict[str, Any]:
    mixed_precision = args.mixed_precision
    if args.fp16:
        if mixed_precision not in {"none", "fp16"}:
            raise ValueError("--fp16 cannot be combined with --mixed_precision bf16.")
        mixed_precision = "fp16"
    return {
        "mixed_precision": mixed_precision,
        "fp16": mixed_precision == "fp16",
        "bf16": mixed_precision == "bf16",
    }


def compute_balanced_class_weights(
    dataset: list[dict[str, Any]],
    *,
    num_labels: int,
) -> list[float]:
    counts = Counter(int(record["labels"]) for record in dataset)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot compute class weights for an empty training dataset.")

    weights = []
    for label_id in range(num_labels):
        count = counts.get(label_id, 0)
        if count == 0:
            raise ValueError(
                f"Cannot compute balanced class weights: label {label_id} has no samples."
            )
        weights.append(total / (num_labels * count))
    return weights


def resolve_class_weights(
    *,
    class_weighting: str,
    train_dataset: list[dict[str, Any]],
    num_labels: int,
) -> list[float] | None:
    if class_weighting == "none":
        return None
    if class_weighting == "balanced":
        return compute_balanced_class_weights(train_dataset, num_labels=num_labels)
    raise ValueError(f"Unsupported class weighting mode: {class_weighting}")


def build_weighted_trainer_class(trainer_cls, class_weights: list[float]):
    import torch

    weights = torch.tensor(class_weights, dtype=torch.float)

    class WeightedLossTrainer(trainer_cls):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            outputs = model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(logits.device))
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


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


def build_experiment_config(
    args,
    *,
    train_split: str,
    eval_split: str,
    train_size: int,
    eval_size: int,
    full_train_size: int,
    full_eval_size: int,
    test_size: int | None = None,
    full_test_size: int | None = None,
    trainable_params: int,
    total_params: int,
    gpu_type: str | None = None,
    class_weights: list[float] | None = None,
    precision_policy: dict[str, Any] | None = None,
):
    effective_train_fraction = train_size / full_train_size if full_train_size else None
    precision_policy = precision_policy or resolve_precision_policy(args)
    return {
        "method": args.method,
        "search_stage": args.search_stage,
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "hpo_seed": args.hpo_seed,
        "dataset": args.dataset_name,
        "train_split": train_split,
        "eval_split": eval_split,
        "test_split": args.test_split_name,
        "preprocessing_policy": "join_post_tokens_strict_majority",
        "label_policy": "strict_majority_drop_no_majority",
        "selection_metric": "f1_macro",
        "test_policy": "final_only",
        "model_name": args.model_name,
        "tokenizer_name": args.model_name,
        "git_commit": get_git_commit_hash(),
        "seed": args.seed,
        "data_fraction_seed": args.data_fraction_seed,
        "data_fraction": args.data_fraction,
        "effective_train_fraction": effective_train_fraction,
        "run_test": args.run_test,
        "global_switches": {
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "weighted_ce": args.class_weighting != "none",
            "early_stopping": args.early_stopping_patience > 0,
        },
        "training_policy": {
            "optim": args.optim,
            "lr_scheduler_type": args.lr_scheduler_type,
            "max_grad_norm": args.max_grad_norm,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "class_weights": class_weights,
        },
        "hyperparameters": {
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_test_samples": args.max_test_samples,
            "data_fraction": args.data_fraction,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "optim": args.optim,
            "lr_scheduler_type": args.lr_scheduler_type,
            "batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "epochs": args.num_train_epochs,
            "eval_strategy": args.eval_strategy,
            "save_strategy": args.save_strategy,
            "logging_strategy": args.logging_strategy,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "overwrite_output_dir": args.overwrite_output_dir,
            "load_best_model_at_end": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": not args.lower_is_better,
            "save_final_model": not args.no_save_final_model,
            "mixed_precision": precision_policy["mixed_precision"],
            "fp16": precision_policy["fp16"],
            "bf16": precision_policy["bf16"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
        },
        "checkpoint_policy": {
            "eval_strategy": args.eval_strategy,
            "save_strategy": args.save_strategy,
            "logging_strategy": args.logging_strategy,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": not args.lower_is_better,
            "save_final_model": not args.no_save_final_model,
            "final_model_source": (
                "best_checkpoint" if args.load_best_model_at_end else "last_training_state"
            ),
            "wandb_log_model": args.wandb_log_model,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "overwrite_output_dir": args.overwrite_output_dir,
        },
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_test_samples": args.max_test_samples,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "batch_size": args.per_device_train_batch_size,
        "eval_batch_size": args.per_device_eval_batch_size,
        "epochs": args.num_train_epochs,
        "train_size": train_size,
        "eval_size": eval_size,
        "full_train_size": full_train_size,
        "full_eval_size": full_eval_size,
        "effective_train_fraction": effective_train_fraction,
        "test_size": test_size,
        "full_test_size": full_test_size,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "gpu_type": gpu_type,
        "output_dir": args.output_dir,
    }


def build_setup_failure_config(
    args,
    *,
    precision_policy: dict[str, Any] | None = None,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    precision_policy = precision_policy or {
        "mixed_precision": getattr(args, "mixed_precision", "unknown"),
        "fp16": bool(getattr(args, "fp16", False)),
        "bf16": getattr(args, "mixed_precision", None) == "bf16",
    }
    return {
        "method": args.method,
        "search_stage": args.search_stage,
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "hpo_seed": args.hpo_seed,
        "dataset": args.dataset_name,
        "model_name": args.model_name,
        "git_commit": get_git_commit_hash(),
        "output_dir": args.output_dir,
        "seed": args.seed,
        "data_fraction_seed": args.data_fraction_seed,
        "setup_complete": False,
        "global_switches": {
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "weighted_ce": args.class_weighting != "none",
            "early_stopping": args.early_stopping_patience > 0,
        },
        "training_policy": {
            "optim": args.optim,
            "lr_scheduler_type": args.lr_scheduler_type,
            "max_grad_norm": args.max_grad_norm,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "mixed_precision": precision_policy["mixed_precision"],
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
        },
        "runtime_context": {
            "gpu_type": gpu_type,
        },
        "output_safety": {
            "overwrite_output_dir": getattr(args, "overwrite_output_dir", False),
        },
    }


def build_tokenized_dataset_with_count(
    examples,
    tokenizer,
    max_length: int,
    data_fraction: float | None = None,
    fraction_seed: int = 42,
    max_samples: int | None = None,
):
    records = preprocess_hatexplain_split(examples)
    full_count = len(records)
    if data_fraction is not None:
        records = select_data_fraction(records, data_fraction, seed=fraction_seed)
    records = maybe_select_subset(records, max_samples)
    tokenized = [
        tokenize_preprocessed_record(record, tokenizer, max_length=max_length)
        for record in records
    ]
    return tokenized, full_count


def build_tokenized_dataset(examples, tokenizer, max_length: int, max_samples: int | None = None):
    tokenized, _ = build_tokenized_dataset_with_count(
        examples,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
    )
    return tokenized


def validate_test_evaluation_policy(*, search_stage: str, run_test: bool) -> None:
    if run_test and search_stage != "final":
        raise ValueError(
            "--run_test is allowed only when --search_stage final. "
            "Do not use the test split during smoke, quick, or tuning runs."
        )


def validate_checkpoint_policy(args) -> None:
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be >= 0.")
    if args.early_stopping_threshold < 0:
        raise ValueError("--early_stopping_threshold must be >= 0.")
    if args.fp16 and args.mixed_precision == "bf16":
        raise ValueError("--fp16 cannot be combined with --mixed_precision bf16.")

    if args.early_stopping_patience > 0 and not args.load_best_model_at_end:
        raise ValueError(
            "Early stopping requires --load_best_model_at_end so the final model "
            "matches the monitored validation metric."
        )
    if not args.load_best_model_at_end:
        return

    if args.eval_strategy == "no":
        raise ValueError(
            "Best-model selection or early stopping requires evaluation. "
            "Set --eval_strategy steps or epoch."
        )
    if args.save_strategy == "no":
        raise ValueError(
            "Best-model selection or early stopping requires checkpoint saving. "
            "Set --save_strategy steps or epoch."
        )
    if args.save_strategy != args.eval_strategy:
        raise ValueError(
            "--load_best_model_at_end requires --save_strategy to match "
            "--eval_strategy for Hugging Face Trainer."
        )
    if args.save_strategy == "steps":
        eval_steps = args.eval_steps or args.logging_steps
        if eval_steps <= 0 or args.save_steps <= 0:
            raise ValueError("Step-based best-model selection requires positive steps.")
        if args.save_steps % eval_steps != 0:
            raise ValueError(
                "For step-based best-model selection, --save_steps must be a "
                "multiple of --eval_steps."
            )


def compute_metrics_fn():
    import evaluate
    import numpy as np

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        label_ids = sorted(LABEL_ID_TO_NAME)

        results = {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            "precision_macro": precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall_macro": recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"],
        }
        per_class_f1 = f1_metric.compute(
            predictions=preds,
            references=labels,
            average=None,
            labels=label_ids,
        )["f1"]
        per_class_precision = precision_metric.compute(
            predictions=preds,
            references=labels,
            average=None,
            labels=label_ids,
            zero_division=0,
        )["precision"]
        per_class_recall = recall_metric.compute(
            predictions=preds,
            references=labels,
            average=None,
            labels=label_ids,
            zero_division=0,
        )["recall"]
        for index, label_id in enumerate(label_ids):
            label_name = LABEL_ID_TO_NAME[label_id]
            results[f"f1_{label_name}"] = float(per_class_f1[index])
            results[f"precision_{label_name}"] = float(per_class_precision[index])
            results[f"recall_{label_name}"] = float(per_class_recall[index])
        return results

    return compute_metrics


def build_trainer(
    *,
    trainer_cls,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics,
    callbacks=None,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    return trainer_cls(**trainer_kwargs)


def build_training_arguments(training_args_cls, **kwargs):
    signature = inspect.signature(training_args_cls.__init__)
    parameters = signature.parameters
    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_arbitrary_kwargs:
        return training_args_cls(**kwargs)

    supported_keys = set(parameters) - {"self"}
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_keys
    }
    return training_args_cls(**filtered_kwargs)


def build_model_selection_summary(
    trainer,
    *,
    metric_for_best_model: str,
    greater_is_better: bool,
) -> dict[str, Any]:
    log_history = getattr(getattr(trainer, "state", None), "log_history", []) or []
    metric_keys = [metric_for_best_model]
    if metric_for_best_model.startswith("eval_"):
        metric_keys.append(metric_for_best_model.removeprefix("eval_"))
    else:
        metric_keys.append(f"eval_{metric_for_best_model}")

    candidates = []
    for record in log_history:
        for metric_key in metric_keys:
            if metric_key in record:
                candidates.append((record, metric_key, record[metric_key]))
                break

    best_record = None
    best_metric_key = metric_for_best_model
    best_metric = None
    if candidates:
        best_record, best_metric_key, best_metric = (
            max(candidates, key=lambda item: item[2])
            if greater_is_better
            else min(candidates, key=lambda item: item[2])
        )

    state = getattr(trainer, "state", None)
    return {
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "best_metric_key": best_metric_key,
        "best_metric": (
            getattr(state, "best_metric", None)
            if getattr(state, "best_metric", None) is not None
            else best_metric
        ),
        "best_epoch": best_record.get("epoch") if best_record else None,
        "best_step": best_record.get("step") if best_record else None,
        "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
    }


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
        eval_split = find_split_name(ds, ["validation", "valid", "test"])
        test_split = find_split_name(ds, [args.test_split_name])

        if train_split is None:
            raise ValueError(f"No train split found. Available splits: {list(ds.keys())}")
        if eval_split is None:
            raise ValueError(
                f"No validation/valid/test split found. Available splits: {list(ds.keys())}"
            )
        if args.run_test and test_split is None:
            raise ValueError(
                f"No test split named '{args.test_split_name}' found. "
                f"Available splits: {list(ds.keys())}"
            )

        train_dataset, full_train_size = build_tokenized_dataset_with_count(
            ds[train_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            data_fraction=args.data_fraction,
            fraction_seed=args.data_fraction_seed,
            max_samples=args.max_train_samples,
        )
        eval_dataset, full_eval_size = build_tokenized_dataset_with_count(
            ds[eval_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_eval_samples,
        )
        test_dataset = None
        full_test_size = None
        if args.run_test:
            test_dataset, full_test_size = build_tokenized_dataset_with_count(
                ds[test_split],
                tokenizer=tokenizer,
                max_length=args.max_length,
                max_samples=args.max_test_samples,
            )

        id2label, label2id, num_labels = build_fixed_label_maps()

        print(
            f"Train split: {train_split}, size={len(train_dataset)} "
            f"(preprocessed full={full_train_size})"
        )
        print(
            f"Eval split: {eval_split}, size={len(eval_dataset)} "
            f"(preprocessed full={full_eval_size})"
        )
        print(f"Using num_labels: {num_labels}")
        print(f"id2label: {id2label}")
        if args.run_test:
            print(
                f"Test split: {test_split}, size={len(test_dataset)} "
                f"(preprocessed full={full_test_size})"
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
        wandb_settings = resolve_wandb_settings(args)
        gpu_type = get_gpu_type()
        experiment_config = build_experiment_config(
            args,
            train_split=train_split,
            eval_split=eval_split,
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
            full_train_size=full_train_size,
            full_eval_size=full_eval_size,
            test_size=len(test_dataset) if test_dataset is not None else None,
            full_test_size=full_test_size,
            trainable_params=trainable_params,
            total_params=total_params,
            gpu_type=gpu_type,
            class_weights=class_weights,
            precision_policy=precision_policy,
        )
        resolved_config_path = write_resolved_config(args.output_dir, experiment_config)
        print(f"Resolved config: {resolved_config_path}")
        wandb_run = init_wandb_run(wandb_settings, config=experiment_config)

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
