from __future__ import annotations

from datasets import load_dataset

import random
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .tokenizer import FrozenDistilBertTokenizer, MODEL_NAME
from .dataset import HateXplainFrozenDistilBertDataset
from .model import FrozenDistilBertClassifier
from src.data.preprocessing import preprocess_hatexplain_split


LABEL_ID_TO_NAME = {0: "hatespeech", 1: "normal", 2: "offensive"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_shared_records(dataset_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    try:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    except TypeError:
        dataset = load_dataset(dataset_name)

    train_records = preprocess_hatexplain_split(dataset["train"])
    val_records = preprocess_hatexplain_split(dataset["validation"])
    test_records = preprocess_hatexplain_split(dataset["test"])
    return train_records, val_records, test_records


def maybe_limit_records(records: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None or max_samples <= 0:
        return records
    return records[:max_samples]


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count trainable and total model parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def make_dataloader(
    records: list[dict],
    tokenizer: FrozenDistilBertTokenizer,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = HateXplainFrozenDistilBertDataset(records, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_model(
    model: FrozenDistilBertClassifier,
    dataloader: DataLoader,
    *,
    device: torch.device,
    split_name: str,
) -> dict[str, float]:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        f"{split_name}_accuracy": accuracy_score(all_labels, all_predictions),
        f"{split_name}_f1_macro": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        f"{split_name}_precision_macro": precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        f"{split_name}_recall_macro": recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }


def save_checkpoint(
    output_dir: str | Path,
    *,
    epoch: int,
    model: FrozenDistilBertClassifier,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
) -> Path:
    checkpoint_dir = Path(output_dir) / f"checkpoint-epoch{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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


def save_final_model(
    output_dir: str | Path,
    *,
    model: FrozenDistilBertClassifier,
    tokenizer: FrozenDistilBertTokenizer,
    resolved_config: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "finalmodel.pt")
    write_json(output_path / "resolved_config.json", resolved_config)

    tokenizer.save_pretrained(str(output_path / "tokenizer"))


def write_standard_result_files(
    output_dir: str | Path,
    *,
    resolved_config: dict[str, Any],
    eval_metrics: dict[str, Any],
    runtime_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
) -> None:
    output_path = Path(output_dir)
    metrics_payload = {"eval": eval_metrics, "test": test_metrics}
    summary_payload = {
        "config": resolved_config,
        "metrics": metrics_payload,
        "runtime": runtime_metrics,
    }
    write_json(output_path / "metrics.json", metrics_payload)
    write_json(output_path / "runtime.json", runtime_metrics)
    write_json(output_path / "result_summary.json", summary_payload)


def metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"Metric {metric_name!r} not found. Available: {available}")
    return float(value)


def train_frozen_distilbert(config: dict[str, Any]) -> dict[str, Any]:
    start_time = time.time()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    requested_device = config.get("device", "auto")

    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    train_records, val_records, test_records = load_shared_records(str(config["dataset_name"]))
    train_records = maybe_limit_records(train_records, config.get("max_train_samples"))
    val_records = maybe_limit_records(val_records, config.get("max_eval_samples"))

    tokenizer = FrozenDistilBertTokenizer.create(max_length=int(config["max_length"]))

    train_loader = make_dataloader(
        train_records,
        tokenizer,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
    )
    val_loader = make_dataloader(
        val_records,
        tokenizer,
        batch_size=int(config.get("eval_batch_size", config.get("batch_size", 32))),
        shuffle=False,
    )

    model = FrozenDistilBertClassifier(
        model_name=MODEL_NAME,
        num_classes=int(config.get("num_classes", 3)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    criterion = torch.nn.CrossEntropyLoss()

    resolved_config = {
        **config,
        "device": str(device),
        "label_mapping": LABEL_ID_TO_NAME,
        "tokenizer_policy": "src/methods/frozen_distilbert/tokenizer.py",
        "frozen_backbone": True,
        "checkpoint_policy": {
            "eval_strategy": config.get("eval_strategy", "epoch"),
            "save_strategy": config.get("save_strategy", "epoch"),
            "save_total_limit": config.get("save_total_limit", 2),
            "load_best_model_at_end": config.get("load_best_model_at_end", True),
            "metric_for_best_model": config.get("metric_for_best_model", "eval_f1_macro"),
        },
        **count_parameters(model),
    }
    write_json(output_dir / "resolved_config.json", resolved_config)

    best_metric = -math.inf
    best_checkpoint: Path | None = None
    best_eval_metrics: dict[str, Any] = {}
    epochs = int(config.get("epochs", 5))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            progress.set_postfix(loss=loss.item())

        train_loss = total_loss / max(1, total_examples)
        eval_metrics = evaluate_model(model, val_loader, device=device, split_name="eval")
        eval_metrics["train_loss"] = train_loss
        eval_metrics["epoch"] = epoch

        checkpoint_dir: Path | None = None
        if config.get("save_strategy", "epoch") == "epoch":
            checkpoint_dir = save_checkpoint(
                output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=eval_metrics,
            )

        current_metric = metric_value(eval_metrics, str(config.get("metric_for_best_model", "eval_f1_macro")))
        if current_metric > best_metric:
            best_metric = current_metric
            best_eval_metrics = dict(eval_metrics)
            best_checkpoint = checkpoint_dir

        if checkpoint_dir is not None:
            cleanup_checkpoints(
                output_dir,
                save_total_limit=int(config.get("save_total_limit", 2)),
                best_checkpoint=best_checkpoint,
            )

    final_model_source = "last_epoch"
    if config.get("load_best_model_at_end", True) and best_checkpoint is not None:
        checkpoint = torch.load(best_checkpoint / "model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_model_source = best_checkpoint.as_posix()

    test_metrics = None
    if bool(config.get("run_test", False)):
        test_loader = make_dataloader(
            test_records,
            tokenizer,
            batch_size=int(config.get("eval_batch_size", config.get("batch_size", 32))),
            shuffle=False,
        )
        test_metrics = evaluate_model(model, test_loader, device=device, split_name="test")

    runtime_metrics = {
        "training_time_sec": time.time() - start_time,
        "device": str(device),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
        ),
        "final_model_source": final_model_source
        if config.get("load_best_model_at_end", True)
        else "last_epoch",
    }

    save_final_model(
        output_dir,
        model=model,
        tokenizer=tokenizer,
        resolved_config=resolved_config,
    )
    write_standard_result_files(
        output_dir,
        resolved_config=resolved_config,
        eval_metrics=best_eval_metrics,
        runtime_metrics=runtime_metrics,
        test_metrics=test_metrics,
    )

    return {
        "output_dir": output_dir.as_posix(),
        "eval_metrics": best_eval_metrics,
        "test_metrics": test_metrics,
        "runtime_metrics": runtime_metrics,
    }
