"""Setup helpers shared by the Transformer methods.

Transformer setup validates the manual config, resolves precision and W&B
settings, protects the output directory, loads HateXplain, tokenizes it, builds
the DistilBERT classifier, and optionally prepares class weights. Training
itself happens in the runner files.
"""

from __future__ import annotations

from typing import Any

from src.methods.transformer_data import (
    build_fixed_label_maps,
    build_tokenized_dataset_with_stats,
    find_split_name,
    resolve_eval_split_name,
)
from src.methods.transformer_trainer import (
    build_weighted_trainer_class,
    resolve_class_weights,
    resolve_precision_policy,
)
from src.methods.transformer_types import (
    HfClassificationRun,
    HfRunSetup,
)
from src.results import prepare_output_dir_for_run, validate_sample_selection_args
from src.utils.run_metadata import get_gpu_type
from src.utils.wandb_config import build_wandb_settings_from_args


def start_hf_run(args: Any):
    """Validate run-level settings before loading the expensive model/data."""

    validate_sample_selection_args(args)
    validate_checkpoint_policy(args)
    precision_policy = resolve_precision_policy(args)
    wandb_settings = build_wandb_settings_from_args(args)
    prepare_output_dir_for_run(
        args.output_dir,
        overwrite=getattr(args, "overwrite_output_dir", False),
    )
    gpu_type = get_gpu_type()
    return HfRunSetup(
        gpu_type=gpu_type,
        precision_policy=precision_policy,
        wandb_settings=wandb_settings,
    )


def validate_checkpoint_policy(args: Any) -> None:
    """Check that early stopping and best-model selection make sense together."""

    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if args.early_stopping_threshold < 0:
        raise ValueError("early_stopping_threshold must be >= 0.")

    if args.early_stopping_patience > 0 and not args.load_best_model_at_end:
        raise ValueError(
            "Early stopping requires load_best_model_at_end=True so the final model "
            "matches the monitored validation metric."
        )
    if not args.load_best_model_at_end:
        return

    if args.eval_strategy == "no":
        raise ValueError(
            "Best-model selection or early stopping requires evaluation. "
            "Set eval_strategy to 'steps' or 'epoch'."
        )
    if args.save_strategy == "no":
        raise ValueError(
            "Best-model selection or early stopping requires checkpoint saving. "
            "Set save_strategy to 'steps' or 'epoch'."
        )
    if args.save_strategy != args.eval_strategy:
        raise ValueError(
            "load_best_model_at_end requires save_strategy to match "
            "eval_strategy for Hugging Face Trainer."
        )
    if args.save_strategy == "steps":
        eval_steps = args.eval_steps or args.logging_steps
        if eval_steps <= 0 or args.save_steps <= 0:
            raise ValueError("Step-based best-model selection requires positive steps.")
        if args.save_steps % eval_steps != 0:
            raise ValueError(
                "For step-based best-model selection, save_steps must be a "
                "multiple of eval_steps."
            )


def prepare_hf_classification_run(
    args: Any,
    *,
    precision_policy: dict[str, Any],
    gpu_type: str,
) -> HfClassificationRun:
    """Load and tokenize HateXplain, then return everything Trainer needs."""

    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        set_seed,
    )

    set_seed(args.seed)
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    print("Available splits:", list(dataset.keys()))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_split = find_split_name(dataset, ["train"])
    eval_split = resolve_eval_split_name(dataset, test_split_name=args.test_split_name)
    test_split = find_split_name(dataset, [args.test_split_name])
    if train_split is None:
        raise ValueError(f"No train split found. Available splits: {list(dataset.keys())}")
    if args.run_test and test_split is None:
        raise ValueError(f"No test split named '{args.test_split_name}' found.")
    if args.run_test and eval_split == test_split:
        raise ValueError("Evaluation and test split must be distinct.")

    train_data = build_tokenized_dataset_with_stats(
        dataset[train_split],
        tokenizer=tokenizer,
        max_length=args.max_length,
        data_fraction=args.data_fraction,
        fraction_seed=args.data_fraction_seed,
        max_samples=args.max_train_samples,
    )
    eval_data = build_tokenized_dataset_with_stats(
        dataset[eval_split],
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_eval_samples,
    )
    test_data = None
    if args.run_test:
        # Test data is only loaded when the manual config asks for it. That keeps
        # validation-only checks from touching the test split.
        test_data = build_tokenized_dataset_with_stats(
            dataset[test_split],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_test_samples,
        )
    id2label, label2id, num_labels = build_fixed_label_maps()
    print(
        f"Train split: {train_split}, size={len(train_data.dataset)} "
        f"(preprocessed full={train_data.preprocessed_size})"
    )
    print(
        f"Eval split: {eval_split}, size={len(eval_data.dataset)} "
        f"(preprocessed full={eval_data.preprocessed_size})"
    )
    if test_data is not None:
        print(
            f"Test split: {test_split}, size={len(test_data.dataset)} "
            f"(preprocessed full={test_data.preprocessed_size})"
        )
    print(f"Using num_labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    class_weights = resolve_class_weights(
        class_weighting=args.class_weighting,
        train_dataset=train_data.dataset,
        num_labels=num_labels,
    )
    trainer_cls = (
        build_weighted_trainer_class(Trainer, class_weights)
        if class_weights is not None
        else Trainer
    )
    return HfClassificationRun(
        args=args,
        precision_policy=precision_policy,
        gpu_type=gpu_type,
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        trainer_cls=trainer_cls,
        class_weights=class_weights,
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        train_split_data=train_data,
        eval_split_data=eval_data,
        test_split_data=test_data,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
    )
