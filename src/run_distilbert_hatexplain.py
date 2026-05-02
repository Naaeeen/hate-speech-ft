import argparse
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.label_policy import LABEL_ID_TO_NAME, LABEL_NAME_TO_ID
from src.data.preprocessing import (
    preprocess_hatexplain_split,
    tokenize_preprocessed_record,
)


MODEL_NAME = "distilbert-base-uncased"


def parse_args():
    parser = argparse.ArgumentParser(description="Run DistilBERT on HateXplain")

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, default="Hate-speech-CNERG/hatexplain")
    parser.add_argument("--output_dir", type=str, default="./outputs/distilbert_hatexplain")

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

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


def build_tokenized_dataset(examples, tokenizer, max_length: int, max_samples: int | None = None):
    records = preprocess_hatexplain_split(examples)
    records = maybe_select_subset(records, max_samples)
    return [
        tokenize_preprocessed_record(record, tokenizer, max_length=max_length)
        for record in records
    ]


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

        results = {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            "precision_macro": precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall_macro": recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"],
        }
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
):
    return trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def main():
    args = parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)

    print("Available splits:", list(ds.keys()))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_split = find_split_name(ds, ["train"])
    eval_split = find_split_name(ds, ["validation", "valid", "test"])

    if train_split is None:
        raise ValueError(f"No train split found. Available splits: {list(ds.keys())}")
    if eval_split is None:
        raise ValueError(
            f"No validation/valid/test split found. Available splits: {list(ds.keys())}"
        )

    train_dataset = build_tokenized_dataset(
        ds[train_split],
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples,
    )
    eval_dataset = build_tokenized_dataset(
        ds[eval_split],
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_eval_samples,
    )

    id2label, label2id, num_labels = build_fixed_label_maps()

    print(f"Train split: {train_split}, size={len(train_dataset)}")
    print(f"Eval split: {eval_split}, size={len(eval_dataset)}")
    print(f"Using num_labels: {num_labels}")
    print(f"id2label: {id2label}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        report_to="none",
        load_best_model_at_end=False,
        fp16=False,  # safer default; Colab can handle fp16 later if you want
    )

    trainer = build_trainer(
        trainer_cls=Trainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(),
    )

    print("\nStarting training...")
    trainer.train()

    print("\nRunning evaluation...")
    metrics = trainer.evaluate()

    print("\nFinal evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nSaving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
