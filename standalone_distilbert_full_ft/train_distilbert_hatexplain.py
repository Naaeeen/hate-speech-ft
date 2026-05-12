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
    Trainer,
    TrainingArguments,
)


# Edit these constants directly for a different run.
SEED = 42
DATASET_NAME = "Hate-speech-CNERG/hatexplain"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft"

MAX_LENGTH = 128
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

# Set these to small numbers for a smoke run. Use None for full data.
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None

# Keep RUN_TEST=False during tuning. Use True only for a final-style run.
RUN_TEST = True

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


def strict_majority_label(labels: list[int]) -> int | None:
    counts = Counter(int(label) for label in labels)
    label, count = counts.most_common(1)[0]
    if count > len(labels) / 2:
        return label
    return None


def make_records(raw_split, max_samples: int | None = None) -> list[dict[str, Any]]:
    records = []
    for row in raw_split:
        label = strict_majority_label(row["annotators"]["label"])
        if label is None:
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
    return records


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
    kwargs = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
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
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    set_all_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "dataset_name": DATASET_NAME,
        "model_name": MODEL_NAME,
        "output_dir": str(OUTPUT_DIR),
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
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
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(OUTPUT_DIR / "config_snapshot.json", config)

    wandb_run = init_wandb(config)
    start_time = time.perf_counter()
    try:
        print(f"Loading dataset: {DATASET_NAME}")
        raw = load_dataset(DATASET_NAME)

        print("Preparing records")
        train_records = make_records(raw["train"], MAX_TRAIN_SAMPLES)
        eval_records = make_records(raw["validation"], MAX_EVAL_SAMPLES)
        test_records = make_records(raw["test"], MAX_TEST_SAMPLES) if RUN_TEST else []

        print(f"Train examples: {len(train_records)}")
        print(f"Validation examples: {len(eval_records)}")
        if RUN_TEST:
            print(f"Test examples: {len(test_records)}")

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

        trainer = build_trainer(
            model,
            build_training_args(),
            train_dataset,
            eval_dataset,
            tokenizer,
        )

        print("Starting training")
        trainer.train()

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

        runtime = {
            "status": "completed",
            "training_time_sec": time.perf_counter() - start_time,
            "train_examples": len(train_records),
            "validation_examples": len(eval_records),
            "test_examples": len(test_records) if RUN_TEST else None,
        }
        metrics = {
            "validation": validation_metrics,
            "test": test_metrics,
            "runtime": runtime,
        }
        save_json(OUTPUT_DIR / "metrics.json", metrics)
        if wandb_run is not None:
            wandb_run.log(runtime)
            wandb_run.log({"final_eval": validation_metrics})
            if test_metrics is not None:
                wandb_run.log({"final_test": test_metrics})

        print(f"Done. Outputs saved to: {OUTPUT_DIR}")

    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "runtime_sec": time.perf_counter() - start_time,
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
