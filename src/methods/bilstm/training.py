"""Custom PyTorch training loop for the BiLSTM baseline.

BiLSTM does not use Hugging Face Trainer, so this file owns the whole loop:
seed/device setup, DataLoaders, AdamW, linear warmup/decay, epoch validation,
checkpoint selection by macro-F1, early stopping, optional prediction rows, and
runtime metadata. Prediction rows are useful for final/test runs, but the
entrypoint only writes prediction files when `run_test=True`. It mirrors the
Transformer output contract so the manual aggregation step stays the same.
"""

from __future__ import annotations

import math
import random
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.results import write_json
from src.methods.bilstm.data import BiLSTMSplit
from src.methods.bilstm.dataset import HateXplainBiLSTMDataset
from src.methods.bilstm.model import BiLSTMClassifier
from src.methods.classification_metrics import build_classification_metrics

if TYPE_CHECKING:
    from src.methods.bilstm.tokenizer import StandardBiLSTMTokenizer


LABEL_ID_TO_NAME = {0: "hatespeech", 1: "normal", 2: "offensive"}


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and torch for one manual run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve `auto` to CUDA when available, otherwise CPU."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
    return device


def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset CUDA peak-memory tracking before training starts."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: torch.device) -> float | None:
    """Return peak allocated CUDA memory in MB, or `None` on CPU."""

    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def get_peak_memory_reserved_mb(device: torch.device) -> float | None:
    """Return peak reserved CUDA memory in MB, or `None` on CPU."""

    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved() / (1024 * 1024)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return `(trainable_params, total_params)` for result tables."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def resolve_class_weights(
    records: Sequence[Mapping[str, Any]],
    *,
    class_weighting: str,
    num_labels: int,
) -> list[float] | None:
    """Compute optional balanced class weights for the loss function."""

    if class_weighting == "none":
        return None
    if class_weighting != "balanced":
        raise ValueError(f"Unsupported class weighting mode: {class_weighting}")

    counts = {label_id: 0 for label_id in range(num_labels)}
    for record in records:
        counts[int(record["label"])] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot compute class weights for an empty training split.")

    weights = []
    for label_id in range(num_labels):
        count = counts[label_id]
        if count == 0:
            raise ValueError(
                f"Cannot compute balanced class weights: label {label_id} has no samples."
            )
        weights.append(total / (num_labels * count))
    return weights


def make_dataloader(
    records: list[dict[str, Any]],
    tokenizer: StandardBiLSTMTokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a deterministic DataLoader for one BiLSTM split."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = HateXplainBiLSTMDataset(records, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def build_model(
    args: Any,
    tokenizer: StandardBiLSTMTokenizer,
    *,
    num_labels: int = 3,
) -> BiLSTMClassifier:
    """Create the from-scratch BiLSTM model from manual config values."""

    return BiLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_labels,
        dropout=args.dropout,
        pad_idx=tokenizer.pad_id,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the simple linear warmup/decay scheduler used in old runs."""

    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        """Scale LR for this optimizer step."""

        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        remaining_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - current_step) / float(remaining_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_model(
    model: BiLSTMClassifier,
    dataloader: DataLoader,
    *,
    device: torch.device,
    split_name: str,
    source_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Evaluate a split and optionally build per-example prediction rows."""

    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_probabilities: list[list[float]] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, lengths=lengths)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    metrics = build_classification_metrics(
        all_labels,
        all_predictions,
        prefix=split_name,
    )
    prediction_rows = build_prediction_rows(
        records=source_records or [],
        predicted_labels=all_predictions,
        probabilities=all_probabilities,
    )
    return metrics, prediction_rows


def build_prediction_rows(
    *,
    records: Sequence[Mapping[str, Any]],
    predicted_labels: Sequence[int],
    probabilities: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    """Create JSON-friendly prediction rows aligned to source records."""

    if not records:
        return []
    if len(records) != len(predicted_labels):
        raise ValueError(
            "Prediction output length does not match source records: "
            f"records={len(records)}, predictions={len(predicted_labels)}."
        )
    if len(records) != len(probabilities):
        raise ValueError(
            "Probability output length does not match source records: "
            f"records={len(records)}, probabilities={len(probabilities)}."
        )

    rows = []
    for record, predicted_label, probability_row in zip(
        records,
        predicted_labels,
        probabilities,
    ):
        gold_label = int(record["label"])
        predicted_label_id = int(predicted_label)
        rows.append(
            {
                "id": record.get("id"),
                "text": record.get("text"),
                "label": gold_label,
                "label_name": LABEL_ID_TO_NAME.get(gold_label),
                "predicted_label": predicted_label_id,
                "predicted_label_name": LABEL_ID_TO_NAME.get(predicted_label_id),
                "probabilities": [float(value) for value in probability_row],
            }
        )
    return rows


def save_prediction_file(path: str | Path, predictions: list[dict[str, Any]]) -> Path:
    """Save BiLSTM prediction rows using the shared JSON writer."""

    return write_json(path, {"count": len(predictions), "predictions": predictions})


def save_checkpoint(
    output_dir: str | Path,
    *,
    epoch: int,
    step: int,
    model: BiLSTMClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    metrics: dict[str, Any],
) -> Path:
    """Save one epoch checkpoint with model/optimizer/scheduler state."""

    checkpoint_dir = Path(output_dir) / f"checkpoint-epoch{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        checkpoint_dir / "model.pt",
    )
    write_json(checkpoint_dir / "metrics.json", metrics)
    return checkpoint_dir


def cleanup_checkpoints(
    output_dir: str | Path,
    *,
    save_total_limit: int,
    best_checkpoint: str | Path | None,
) -> None:
    """Keep only the best/recent checkpoints allowed by `save_total_limit`."""

    if save_total_limit <= 0:
        return

    output_path = Path(output_dir)
    checkpoints = sorted(
        output_path.glob("checkpoint-epoch*"),
        key=lambda path: path.stat().st_mtime,
    )
    if len(checkpoints) <= save_total_limit:
        return

    keep: set[Path] = set()
    if best_checkpoint is not None:
        keep.add(Path(best_checkpoint))

    for checkpoint in reversed(checkpoints):
        if len(keep) >= save_total_limit:
            break
        keep.add(checkpoint)

    for checkpoint in checkpoints:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint, ignore_errors=True)


def load_checkpoint_model_state(
    model: BiLSTMClassifier,
    checkpoint_dir: Path,
    *,
    device: torch.device,
) -> None:
    """Load the selected checkpoint back into the current model."""

    checkpoint = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def save_final_model(
    output_dir: str | Path,
    *,
    model: BiLSTMClassifier,
    tokenizer: StandardBiLSTMTokenizer,
    config: dict[str, Any],
    no_save_final_model: bool,
) -> Path | None:
    """Save the final BiLSTM model and tokenizer files for this run."""

    if no_save_final_model:
        return None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config,
            "experiment_config": config,
        },
        model_path,
    )
    tokenizer.save_pretrained(str(output_path / "tokenizer"))
    return model_path


def _metric_value(metrics: Mapping[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"Metric {metric_name!r} not found. Available: {available}")
    return float(metrics[metric_name])


def _train_one_epoch(
    *,
    model: BiLSTMClassifier,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, lengths=lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=loss.item())

    return total_loss / max(1, total_examples)


def run_training(
    *,
    args: Any,
    train_data: BiLSTMSplit,
    eval_data: BiLSTMSplit,
    test_data: BiLSTMSplit | None,
    tokenizer: StandardBiLSTMTokenizer,
    device: torch.device,
    class_weights: list[float] | None,
) -> dict[str, Any]:
    """Train one BiLSTM run and return everything the caller needs to save."""

    if not train_data.records:
        raise ValueError("Training split is empty after preprocessing/subsetting.")
    if not eval_data.records:
        raise ValueError("Evaluation split is empty after preprocessing/subsetting.")

    model = build_model(args, tokenizer).to(device)
    trainable_params, total_params = count_parameters(model)
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    train_loader = make_dataloader(
        train_data.records,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    eval_loader = make_dataloader(
        eval_data.records,
        tokenizer,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, args.epochs * len(train_loader)),
        warmup_ratio=args.warmup_ratio,
    )
    class_weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float, device=device)
        if class_weights is not None
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

    best_metric = -math.inf
    best_epoch: int | None = None
    best_step: int | None = None
    best_checkpoint: Path | None = None
    epochs_without_improvement = 0
    global_step = 0

    reset_peak_memory_stats(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        global_step += len(train_loader)

        eval_metrics, _prediction_rows = evaluate_model(
            model,
            eval_loader,
            device=device,
            split_name="eval",
        )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"eval_f1_macro={eval_metrics['eval_f1_macro']:.4f}"
        )

        checkpoint_dir: Path | None = None
        if args.save_strategy == "epoch":
            checkpoint_dir = save_checkpoint(
                args.output_dir,
                epoch=epoch,
                step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=eval_metrics,
            )

        current_metric = _metric_value(eval_metrics, args.metric_for_best_model)
        # Checkpoint selection is intentionally simple: validation macro-F1 must
        # improve by more than the threshold, otherwise patience starts ticking.
        improved = current_metric > best_metric + args.early_stopping_threshold
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_step = global_step
            best_checkpoint = checkpoint_dir
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if checkpoint_dir is not None:
            # Keep the best checkpoint plus the newest allowed checkpoints, so
            # Colab output folders do not grow forever during manual runs.
            cleanup_checkpoints(
                args.output_dir,
                save_total_limit=args.save_total_limit,
                best_checkpoint=best_checkpoint,
            )

        if args.early_stopping_patience > 0 and (
            epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} non-improving epoch(s)."
            )
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time_sec = time.perf_counter() - train_start
    final_model_source = "last_epoch"
    if args.load_best_model_at_end and best_checkpoint is not None:
        load_checkpoint_model_state(model, best_checkpoint, device=device)
        final_model_source = best_checkpoint.as_posix()

    final_eval_metrics, final_eval_predictions = evaluate_model(
        model,
        eval_loader,
        device=device,
        split_name="eval",
        # We only attach source text/ids to eval predictions when the run is in
        # final/test mode. Validation-only HPO reruns can stay lighter.
        source_records=eval_data.records if args.run_test else None,
    )

    test_metrics = None
    test_predictions: list[dict[str, Any]] = []
    if args.run_test:
        if test_data is None:
            raise ValueError("Cannot run final test evaluation before loading test data.")
        test_loader = make_dataloader(
            test_data.records,
            tokenizer,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
        )
        test_metrics, test_predictions = evaluate_model(
            model,
            test_loader,
            device=device,
            split_name="test",
            source_records=test_data.records,
        )

    return {
        "model": model,
        "eval_metrics": final_eval_metrics,
        "test_metrics": test_metrics,
        "eval_predictions": final_eval_predictions,
        "test_predictions": test_predictions,
        "runtime": {
            "training_time_sec": training_time_sec,
            "peak_memory_mb": get_peak_memory_mb(device),
            "peak_memory_reserved_mb": get_peak_memory_reserved_mb(device),
            "final_model_source": final_model_source,
        },
        "model_selection": {
            "best_metric": best_metric if best_metric != -math.inf else None,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_checkpoint": best_checkpoint.as_posix() if best_checkpoint else None,
        },
        "parameters": {
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
    }
