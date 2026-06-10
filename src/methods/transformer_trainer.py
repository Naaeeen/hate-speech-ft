"""Trainer-building helpers for DistilBERT-style methods.

The simplified repo still uses Hugging Face Trainer for Transformer methods.
This file keeps the slightly fiddly parts in one place: precision flags,
optional class weights, Trainer arguments, early stopping, and the best-metric
summary that we save into `result_summary.json`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from src.methods.transformer_types import HfClassificationRun
from src.utils.wandb_config import WandbSettings


def resolve_precision_policy(args) -> dict[str, Any]:
    """Translate the simple `mixed_precision` string into HF fp16/bf16 flags."""

    mixed_precision = args.mixed_precision
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
    """Compute balanced class weights from a tokenized training dataset."""

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
    """Resolve `class_weighting` into weights or `None`."""

    if class_weighting == "none":
        return None
    if class_weighting == "balanced":
        return compute_balanced_class_weights(train_dataset, num_labels=num_labels)
    raise ValueError(f"Unsupported class weighting mode: {class_weighting}")


def build_weighted_trainer_class(trainer_cls, class_weights: list[float]):
    """Patch Trainer loss with explicit class weights when requested."""

    import torch

    weights = torch.tensor(class_weights, dtype=torch.float)

    class WeightedLossTrainer(trainer_cls):
        """Trainer subclass that swaps in weighted cross-entropy."""

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            """Compute weighted classification loss for one Trainer batch."""

            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            outputs = model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def compute_metrics_fn(label_id_to_name: Mapping[int, str] | None = None):
    """Return the metric callback used by Hugging Face Trainer."""

    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if label_id_to_name is None:
        from src.data.label_policy import LABEL_ID_TO_NAME

        label_id_to_name = LABEL_ID_TO_NAME

    def compute_metrics(eval_pred):
        """Compute macro and per-class metrics from logits/labels."""

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        label_ids = sorted(label_id_to_name)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )
        class_precision, class_recall, class_f1, support = precision_recall_fscore_support(
            labels,
            preds,
            labels=label_ids,
            zero_division=0,
        )
        results = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(macro_f1),
            "precision_macro": float(macro_precision),
            "recall_macro": float(macro_recall),
        }
        for index, label_id in enumerate(label_ids):
            label_name = label_id_to_name[label_id]
            results[f"f1_{label_name}"] = float(class_f1[index])
            results[f"precision_{label_name}"] = float(class_precision[index])
            results[f"recall_{label_name}"] = float(class_recall[index])
            results[f"support_{label_name}"] = int(support[index])
        return results

    return compute_metrics


def build_hf_training_arguments_from_args(
    *,
    args: Any,
    output_dir: str,
    learning_rate: float,
    num_train_epochs: float,
    precision_policy: dict[str, Any],
    wandb_settings: WandbSettings,
):
    """Build HF `TrainingArguments` from the flat manual config."""

    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
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
        report_to=wandb_settings.report_to,
        run_name=wandb_settings.run_name if wandb_settings.enabled else None,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=not args.lower_is_better,
        fp16=precision_policy["fp16"],
        bf16=precision_policy["bf16"],
        gradient_checkpointing=args.gradient_checkpointing,
    )


def build_early_stopping_callbacks(args: Any) -> list[Any]:
    """Return an EarlyStoppingCallback list, or empty when disabled."""

    if args.early_stopping_patience <= 0:
        return []
    from transformers import EarlyStoppingCallback

    return [
        EarlyStoppingCallback(
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
    """Create the Trainer for one already-prepared Transformer context."""

    trainer_kwargs = {
        "model": context.model,
        "args": training_args,
        "train_dataset": context.train_dataset,
        "eval_dataset": context.eval_dataset,
        "processing_class": context.tokenizer,
        "data_collator": context.data_collator,
        "compute_metrics": compute_metrics_fn(),
    }
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    return context.trainer_cls(**trainer_kwargs)


def build_model_selection_summary(
    trainer,
    *,
    metric_for_best_model: str,
    greater_is_better: bool,
) -> dict[str, Any]:
    """Recover the best validation metric/epoch from Trainer state.

    Hugging Face stores part of this on `trainer.state`, and the epoch/step is
    easiest to recover from `log_history`. This helper keeps the saved summary
    stable across all Transformer methods.
    """

    log_history = getattr(getattr(trainer, "state", None), "log_history", []) or []
    metric_keys = [metric_for_best_model]
    metric_keys.append(
        metric_for_best_model.removeprefix("eval_")
        if metric_for_best_model.startswith("eval_")
        else f"eval_{metric_for_best_model}"
    )
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
