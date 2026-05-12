from __future__ import annotations

import inspect
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# Edit these constants directly for a different run.
METHOD = "full-ft"
TRIAL_ID = "standalone_distilbert_full_ft_tuning_seed42"
SEARCH_STAGE = "tuning"  # smoke, tuning, confirm, or final
SEED = 42
DATASET_NAME = "Hate-speech-CNERG/hatexplain"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft"
OVERWRITE_OUTPUT_DIR = False

MAX_LENGTH = 128
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
OPTIM = "adamw_torch"
LR_SCHEDULER_TYPE = "linear"
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001
MIXED_PRECISION = "none"  # none, fp16, or bf16
GRADIENT_CHECKPOINTING = False

# Set these to small numbers for a smoke run. Use None for full data.
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None

# Keep RUN_TEST=False during smoke/tuning/confirmation.
# Use True only after the final configuration is frozen.
RUN_TEST = False

USE_WANDB = True
WANDB_PROJECT = "hate-speech-ft"
WANDB_ENTITY = None
WANDB_MODE = "online"  # online, offline, or disabled
WANDB_RUN_NAME = f"standalone_distilbert_full_ft_seed{SEED}"

ID_TO_LABEL = {0: "hatespeech", 1: "normal", 2: "offensive"}
LABEL_TO_ID = {value: key for key, value in ID_TO_LABEL.items()}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_metrics() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "peak_mem_allocated_mb": None,
            "peak_mem_reserved_mb": None,
        }
    return {
        "peak_mem_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_mem_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def count_parameters(model) -> dict[str, float | int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    trainable_pct = 100 * trainable_params / total_params if total_params else 0.0
    return {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_pct": float(trainable_pct),
    }


def strict_majority_label(labels: list[int]) -> int | None:
    if not labels:
        return None
    counts = Counter(int(label) for label in labels)
    label, count = counts.most_common(1)[0]
    if count > len(labels) / 2:
        return label
    return None


def validate_stage_policy() -> None:
    valid_stages = {"smoke", "tuning", "confirm", "final"}
    if SEARCH_STAGE not in valid_stages:
        raise ValueError(f"SEARCH_STAGE must be one of {sorted(valid_stages)}")
    if RUN_TEST and SEARCH_STAGE != "final":
        raise ValueError(
            "RUN_TEST=True is only allowed when SEARCH_STAGE='final'. "
            "Use validation only for smoke, tuning, and confirmation runs."
        )
    if SEARCH_STAGE == "final" and not RUN_TEST:
        print(
            "Warning: SEARCH_STAGE='final' but RUN_TEST=False; "
            "no test metrics will be saved."
        )


def make_records(
    raw_split,
    split_name: str,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    processed_examples = 0
    dropped_no_majority = 0
    for row in raw_split:
        processed_examples += 1
        label = strict_majority_label(row["annotators"]["label"])
        if label is None:
            dropped_no_majority += 1
            continue
        records.append(
            {
                "id": str(row.get("id", len(records))),
                "text": " ".join(str(token) for token in row["post_tokens"]),
                "label": int(label),
            }
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    return records, {
        "split": split_name,
        "raw_examples": len(raw_split),
        "processed_examples": processed_examples,
        "kept_examples": len(records),
        "dropped_no_majority": dropped_no_majority,
        "max_samples": max_samples,
        "stopped_at_sample_cap": max_samples is not None and len(records) >= max_samples,
    }


def tokenize_dataset(records: list[dict[str, Any]], tokenizer) -> Dataset:
    dataset = Dataset.from_list(records)

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["id", "text", "label"],
    )


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def build_training_args() -> TrainingArguments:
    if MIXED_PRECISION not in {"none", "fp16", "bf16"}:
        raise ValueError("MIXED_PRECISION must be one of: none, fp16, bf16")

    kwargs = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "optim": OPTIM,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "fp16": MIXED_PRECISION == "fp16",
        "bf16": MIXED_PRECISION == "bf16",
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1_macro",
        "greater_is_better": True,
        "report_to": ["wandb"] if USE_WANDB and WANDB_MODE != "disabled" else [],
        "run_name": WANDB_RUN_NAME,
        "seed": SEED,
        "data_seed": SEED,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters) - {"self"}
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(
        **{key: value for key, value in kwargs.items() if key in supported}
    )


def build_trainer(model, training_args, train_dataset, eval_dataset, tokenizer) -> Trainer:
    callbacks = []
    if EARLY_STOPPING_PATIENCE is not None and EARLY_STOPPING_PATIENCE > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        )

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "callbacks" in trainer_signature.parameters:
        kwargs["callbacks"] = callbacks
    if "processing_class" in trainer_signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_json(title: str, payload: Any) -> None:
    print(f"\n{title}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def ensure_output_dir_is_available() -> None:
    if OVERWRITE_OUTPUT_DIR or not OUTPUT_DIR.exists():
        return
    blocking_paths = [
        OUTPUT_DIR / "metrics.json",
        OUTPUT_DIR / "run_summary.json",
        OUTPUT_DIR / "predictions_validation.json",
        OUTPUT_DIR / "predictions_test.json",
        OUTPUT_DIR / "final_model",
        OUTPUT_DIR / "checkpoints",
    ]
    existing = [path for path in blocking_paths if path.exists()]
    if existing:
        existing_list = "\n".join(f"- {path}" for path in existing)
        raise RuntimeError(
            "OUTPUT_DIR already contains experiment artifacts. Use a new OUTPUT_DIR "
            "or set OVERWRITE_OUTPUT_DIR=True if you intentionally want to replace them.\n"
            f"Existing artifacts:\n{existing_list}"
        )


def model_selection_summary(trainer) -> dict[str, Any]:
    state = trainer.state
    log_history = getattr(state, "log_history", []) or []
    best_metric = getattr(state, "best_metric", None)
    best_epoch = None
    best_step = None
    if best_metric is not None:
        for record in log_history:
            metric_value = record.get("eval_f1_macro")
            if metric_value is not None and abs(metric_value - best_metric) < 1e-12:
                best_epoch = record.get("epoch")
                best_step = record.get("step")
                break
    return {
        "metric_for_best_model": "eval_f1_macro",
        "greater_is_better": True,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
        "global_step": getattr(state, "global_step", None),
    }


def prediction_rows(records: list[dict[str, Any]], logits: np.ndarray) -> list[dict[str, Any]]:
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(probabilities, axis=-1)
    rows = []
    for record, pred, probs in zip(records, predictions, probabilities):
        label = int(record["label"])
        pred = int(pred)
        rows.append(
            {
                "id": record["id"],
                "text": record["text"],
                "label": label,
                "label_name": ID_TO_LABEL[label],
                "prediction": pred,
                "prediction_name": ID_TO_LABEL[pred],
                "probabilities": {
                    ID_TO_LABEL[index]: float(value)
                    for index, value in enumerate(probs)
                },
            }
        )
    return rows


def init_wandb(config: dict[str, Any]):
    if not USE_WANDB or WANDB_MODE == "disabled":
        return None

    import wandb

    if WANDB_MODE == "online" and not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set. Run `wandb login` or set WANDB_MODE='offline'.")

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        mode=WANDB_MODE,
        config=config,
        job_type="train",
    )


def main() -> None:
    script_start_time = time.perf_counter()
    set_all_seeds(SEED)
    validate_stage_policy()
    ensure_output_dir_is_available()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "trial_id": TRIAL_ID,
        "search_stage": SEARCH_STAGE,
        "seed": SEED,
        "dataset_name": DATASET_NAME,
        "model_name": MODEL_NAME,
        "output_dir": str(OUTPUT_DIR),
        "overwrite_output_dir": OVERWRITE_OUTPUT_DIR,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "optim": OPTIM,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
        "mixed_precision": MIXED_PRECISION,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "max_test_samples": MAX_TEST_SAMPLES,
        "run_test": RUN_TEST,
        "label_mapping": ID_TO_LABEL,
        "wandb": {
            "enabled": USE_WANDB,
            "project": WANDB_PROJECT,
            "entity": WANDB_ENTITY,
            "mode": WANDB_MODE,
            "run_name": WANDB_RUN_NAME,
        },
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_type": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    save_json(OUTPUT_DIR / "config_snapshot.json", config)
    print_json("Run config snapshot", config)

    wandb_run = init_wandb(config)
    try:
        print(f"Loading dataset: {DATASET_NAME}")
        raw = load_dataset(DATASET_NAME)

        print("Preparing records")
        train_records, train_audit = make_records(
            raw["train"],
            "train",
            MAX_TRAIN_SAMPLES,
        )
        eval_records, eval_audit = make_records(
            raw["validation"],
            "validation",
            MAX_EVAL_SAMPLES,
        )
        if RUN_TEST:
            test_records, test_audit = make_records(
                raw["test"],
                "test",
                MAX_TEST_SAMPLES,
            )
        else:
            test_records = []
            test_audit = {
                "split": "test",
                "raw_examples": len(raw["test"]),
                "processed_examples": 0,
                "kept_examples": None,
                "dropped_no_majority": None,
                "max_samples": MAX_TEST_SAMPLES,
                "stopped_at_sample_cap": False,
                "skipped_because_run_test_false": True,
            }
        dataset_audit = {
            "train": train_audit,
            "validation": eval_audit,
            "test": test_audit,
        }
        config["dataset_audit"] = dataset_audit
        save_json(OUTPUT_DIR / "config_snapshot.json", config)

        print(f"Train examples: {len(train_records)}")
        print(f"Validation examples: {len(eval_records)}")
        if RUN_TEST:
            print(f"Test examples: {len(test_records)}")
        print_json("Dataset preprocessing audit", dataset_audit)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_dataset = tokenize_dataset(train_records, tokenizer)
        eval_dataset = tokenize_dataset(eval_records, tokenizer)
        test_dataset = tokenize_dataset(test_records, tokenizer) if RUN_TEST else None

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(ID_TO_LABEL),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
        parameter_metrics = count_parameters(model)
        config.update(parameter_metrics)
        save_json(OUTPUT_DIR / "config_snapshot.json", config)
        if wandb_run is not None:
            wandb_run.config.update(parameter_metrics, allow_val_change=True)
        print_json("Model parameter counts", parameter_metrics)

        trainer = build_trainer(
            model,
            build_training_args(),
            train_dataset,
            eval_dataset,
            tokenizer,
        )

        print("Starting training")
        reset_peak_memory()
        synchronize_cuda()
        train_start_time = time.perf_counter()
        train_result = trainer.train()
        synchronize_cuda()
        train_time_sec = time.perf_counter() - train_start_time
        train_memory_metrics = peak_memory_metrics()
        train_metrics = dict(getattr(train_result, "metrics", {}) or {})
        train_metrics["gpu_synchronized_train_time_sec"] = train_time_sec
        train_metrics.update(
            {
                "train_peak_mem_allocated_mb": train_memory_metrics[
                    "peak_mem_allocated_mb"
                ],
                "train_peak_mem_reserved_mb": train_memory_metrics[
                    "peak_mem_reserved_mb"
                ],
            }
        )
        print_json("Training metrics", train_metrics)

        print("Evaluating validation split")
        validation_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix="eval",
        )
        print(json.dumps(validation_metrics, indent=2, sort_keys=True))

        eval_predictions = trainer.predict(eval_dataset)
        save_json(
            OUTPUT_DIR / "predictions_validation.json",
            prediction_rows(eval_records, eval_predictions.predictions),
        )

        test_metrics = None
        if RUN_TEST and test_dataset is not None:
            print("Evaluating test split")
            test_metrics = trainer.evaluate(
                eval_dataset=test_dataset,
                metric_key_prefix="test",
            )
            print(json.dumps(test_metrics, indent=2, sort_keys=True))
            test_predictions = trainer.predict(test_dataset)
            save_json(
                OUTPUT_DIR / "predictions_test.json",
                prediction_rows(test_records, test_predictions.predictions),
            )

        print("Saving final model")
        final_model_dir = OUTPUT_DIR / "final_model"
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

        selection = model_selection_summary(trainer)
        run_memory_metrics = peak_memory_metrics()
        runtime = {
            "status": "completed",
            "gpu_synchronized_train_time_sec": train_time_sec,
            "total_runtime_sec": time.perf_counter() - script_start_time,
            "train_examples": len(train_records),
            "validation_examples": len(eval_records),
            "test_examples": len(test_records) if RUN_TEST else None,
            "gpu_type": config["gpu_type"],
            "peak_mem_allocated_mb": train_memory_metrics["peak_mem_allocated_mb"],
            "peak_mem_reserved_mb": train_memory_metrics["peak_mem_reserved_mb"],
            "run_peak_mem_allocated_mb": run_memory_metrics["peak_mem_allocated_mb"],
            "run_peak_mem_reserved_mb": run_memory_metrics["peak_mem_reserved_mb"],
        }
        metrics = {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "runtime": runtime,
            "model_selection": selection,
            "parameters": parameter_metrics,
            "dataset_audit": dataset_audit,
        }
        save_json(OUTPUT_DIR / "metrics.json", metrics)
        save_json(OUTPUT_DIR / "trainer_log_history.json", trainer.state.log_history)
        save_json(
            OUTPUT_DIR / "run_summary.json",
            {
                "config": config,
                "metrics": metrics,
                "final_model_dir": str(final_model_dir),
            },
        )
        if wandb_run is not None:
            wandb_run.log(runtime)
            wandb_run.log(parameter_metrics)
            wandb_run.log({"model_selection": selection})
            wandb_run.log({"final_eval": validation_metrics})
            if test_metrics is not None:
                wandb_run.log({"final_test": test_metrics})

        print_json("Model selection", selection)
        print_json("Runtime and memory", runtime)
        print_json(
            "Manual record fields",
            {
                "method": METHOD,
                "trial_id": TRIAL_ID,
                "seed": SEED,
                "hparams_json": {
                    "learning_rate": LEARNING_RATE,
                    "train_batch_size": TRAIN_BATCH_SIZE,
                    "eval_batch_size": EVAL_BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "max_length": MAX_LENGTH,
                    "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "optim": OPTIM,
                    "lr_scheduler_type": LR_SCHEDULER_TYPE,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
                    "mixed_precision": MIXED_PRECISION,
                    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
                },
                "best_epoch": selection["best_epoch"],
                "val_macro_f1": validation_metrics.get("eval_f1_macro"),
                "train_time_s": runtime["gpu_synchronized_train_time_sec"],
                "total_runtime_s": runtime["total_runtime_sec"],
                "peak_mem_allocated_mb": runtime["peak_mem_allocated_mb"],
                "peak_mem_reserved_mb": runtime["peak_mem_reserved_mb"],
                "trainable_params": parameter_metrics["trainable_params"],
                "total_params": parameter_metrics["total_params"],
                "gpu_type": runtime["gpu_type"],
                "status": runtime["status"],
            },
        )
        print(f"Done. Outputs saved to: {OUTPUT_DIR}")

    except Exception as exc:
        failure = {
            "status": "failed",
            "method": METHOD,
            "trial_id": TRIAL_ID,
            "search_stage": SEARCH_STAGE,
            "seed": SEED,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "total_runtime_sec": time.perf_counter() - script_start_time,
            "gpu_type": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            **peak_memory_metrics(),
        }
        save_json(OUTPUT_DIR / "failure.json", failure)
        if wandb_run is not None:
            wandb_run.log(failure)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
