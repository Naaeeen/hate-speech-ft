from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.label_policy import LABEL_ID_TO_NAME, LABEL_NAME_TO_ID
from src.data.preprocessing import (
    preprocess_hatexplain_split,
    tokenize_preprocessed_record,
)
from src.experiments.results import write_failure_file, write_resolved_config, write_result_files
from src.methods.common import (
    add_common_method_arguments,
    build_common_experiment_config,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)

DEFAULT_METHOD_ID = "lp-ft"
DEFAULT_METHOD_PACKAGE = "LP-FT"
DEFAULT_DESCRIPTION = "Conduct a 2-step method: step 1 with Linear Probing, and Step 2 with full Fine-Tuning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    add_common_method_arguments(
        parser,
        defaults={
            "method": DEFAULT_METHOD_ID,
            "trial_id": f"{DEFAULT_METHOD_PACKAGE}_manual",
            "output_dir": f"outputs/{DEFAULT_METHOD_PACKAGE}",
        },
    )

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type = int, default = 8)

    parser.add_argument("--stage1_head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--stage1_epochs", type=float, default=2.0)

    parser.add_argument("--stage2_learning_rate", type=float, default=2e-5)
    parser.add_argument("--stage2_epochs", type=float, default=2.0)
    return parser.parse_args()


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
            "precision_macro": precision_metric.compute(predictions=preds, references=labels, average="macro")[
                "precision"],
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


def build_experiment_config(args: argparse.Namespace) -> dict[str, object]:
    return build_common_experiment_config(
        args,
        model_name=args.model_name,
        tokenizer_name=None,
        hyperparameters={
            "stage1_head_learning_rate": args.stage1_head_learning_rate,
            "stage1_epochs": args.stage1_epochs,
            "stage2_learning_rate": args.stage2_learning_rate,
            "stage2_epochs": args.stage2_epochs,
        },
        extra={
            "status": "success",
        },
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    config = build_experiment_config(args)
    output_dir = Path(args.output_dir)

    validate_test_evaluation_policy(
        search_stage=args.search_stage,
        run_test=args.run_test,
    )
    validate_output_dir_for_run(
        output_dir,
        overwrite=args.overwrite_output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(output_dir, config)

    try:
        dataset = load_dataset(args.dataset_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_records = build_tokenized_dataset(dataset["train"], tokenizer, args.max_length, args.max_train_samples)
        validation_records = build_tokenized_dataset(dataset["validation"], tokenizer, args.max_length, args.max_eval_samples)
        test_records = build_tokenized_dataset(dataset["test"], tokenizer, args.max_length, args.max_eval_samples)

        idToName, NameToId, num_labels = build_fixed_label_maps()
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels, id2label=idToName, label2id=NameToId
        )

        # Linear Probing Phase
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        linear_probing_args = TrainingArguments(
            output_dir=str(output_dir / "lp_checkpoints"),
            learning_rate=args.stage1_head_learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.stage1_epochs,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            report_to="none",
            load_best_model_at_end=False,
            fp16=False,
        )

        lp_trainer = build_trainer(
            trainer_cls=Trainer,
            model=model,
            training_args=linear_probing_args,
            train_dataset=train_records,
            eval_dataset=validation_records,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn(),
        )
        lp_trainer.train()

        # Fine-tuning phase
        for param in model.parameters():
            param.requires_grad = True

        fine_tuning_args = TrainingArguments(
            output_dir=str(output_dir / "ft_checkpoints"),
            learning_rate=args.stage2_learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.stage2_epochs,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            report_to="all",
            load_best_model_at_end=False,
            fp16=False,
        )

        final_trainer = Trainer(
            model=model,
            args=fine_tuning_args,
            train_dataset=train_records,
            eval_dataset=validation_records,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn(),
        )
        final_trainer.train()
        val_trainer = final_trainer.evaluate(eval_dataset = validation_records)
        test_trainer = final_trainer.evaluate(eval_dataset = test_records)


        val_metrics = {
            "val_macro_f1": val_trainer["eval_f1_macro"],
            "val_accuracy": val_trainer["eval_accuracy"]  }

        test_metrics = {
            "test_macro_f1": test_trainer["eval_f1_macro"],
            "test_accuracy": test_trainer["eval_accuracy"]  }

        runtime_metrics = {
            "total_duration_seconds": time.perf_counter() - start,
            "status": "completed"
        }

        result_summary = {
            "status": "completed",
            "selection_metric": "val_macro_f1",
        }

        model_selection = {

        }

        write_result_files(
            output_dir=output_dir,
            config=config,
            eval_metrics=val_metrics,
            runtime_metrics=runtime_metrics,
            test_metrics=test_metrics,
            model_selection=model_selection,
            status="completed"
        )


    except Exception as exc:
        config["extra"]["status"] = "failed"
        write_failure_file(
            output_dir,
            config=config,
            error=exc,
            runtime_metrics={
                "status": "failed",
                "failure_phase": "execution",
                "training_time_sec": time.perf_counter() - start,
            },
        )
        raise


if __name__ == "__main__":
    main()