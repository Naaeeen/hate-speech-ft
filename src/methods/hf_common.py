from __future__ import annotations

import inspect
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


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


def synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_git_commit_hash(repo_root: Path) -> str | None:
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


def build_compute_cost_fields(
    training_time_sec: float | None,
    *,
    gpu_type: str | None,
) -> dict[str, float | None]:
    training_time_hours = (
        training_time_sec / 3600 if training_time_sec is not None else None
    )
    has_gpu = bool(gpu_type and gpu_type not in {"cpu", "unknown"})
    return {
        "training_time_hours": training_time_hours,
        "gpu_hours": training_time_hours if has_gpu else None,
    }


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


def compute_metrics_fn(
    label_id_to_name: Mapping[int, str] | None = None,
):
    import evaluate
    import numpy as np

    if label_id_to_name is None:
        from src.data.label_policy import LABEL_ID_TO_NAME

        label_id_to_name = LABEL_ID_TO_NAME

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        label_ids = sorted(label_id_to_name)

        results = {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1_macro": f1_metric.compute(
                predictions=preds,
                references=labels,
                average="macro",
            )["f1"],
            "precision_macro": precision_metric.compute(
                predictions=preds,
                references=labels,
                average="macro",
                zero_division=0,
            )["precision"],
            "recall_macro": recall_metric.compute(
                predictions=preds,
                references=labels,
                average="macro",
                zero_division=0,
            )["recall"],
        }
        labels_array = np.asarray(labels)
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
            label_name = label_id_to_name[label_id]
            results[f"f1_{label_name}"] = float(per_class_f1[index])
            results[f"precision_{label_name}"] = float(per_class_precision[index])
            results[f"recall_{label_name}"] = float(per_class_recall[index])
            results[f"support_{label_name}"] = int(np.sum(labels_array == label_id))
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
