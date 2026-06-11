"""Config metadata for the TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.run_metadata import (
    build_compute_cost_fields,
    get_git_commit_hash,
)
from src.methods.tfidf_logreg.data import ClassicalSplit


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_NAME = "tfidf-logreg"


def build_experiment_config(
    args: Any,
    *,
    ngram_range: tuple[int, int],
    train_split: str | None = None,
    eval_split: str | None = None,
    test_split: str | None = None,
    train_data: ClassicalSplit | None = None,
    eval_data: ClassicalSplit | None = None,
    test_data: ClassicalSplit | None = None,
    gpu_type: str | None = None,
    trainable_params: int | None = None,
    total_params: int | None = None,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    """Build the saved config for one TF-IDF + Logistic Regression run.

    The shape is meant for manual table rebuilding: split sizes,
    strict-majority drop counts, seed, model stats, and selected hyperparameters
    all land in one JSON object.
    """

    train_size = len(train_data.records) if train_data is not None else None
    full_train_size = train_data.preprocessed_size if train_data is not None else None
    effective_train_fraction = (
        train_size / full_train_size if train_size is not None and full_train_size else None
    )
    return {
        "method": args.method,
        "run_name": args.run_name,
        "dataset": args.dataset_name,
        "train_split": train_split,
        "eval_split": eval_split,
        "test_split": args.test_split_name,
        "preprocessing_policy": "join_post_tokens_strict_majority",
        "label_policy": "strict_majority_drop_no_majority",
        "split_accounting_policy": (
            "raw_*_size is the loaded Hugging Face split size before local "
            "post-load preprocessing; dropped_no_majority_* counts strict-majority "
            "drops performed after dataset load."
        ),
        "selection_metric": "f1_macro",
        "test_policy": "enabled_by_run_test",
        "model_name": DEFAULT_MODEL_NAME,
        "tokenizer_name": "tfidf",
        "git_commit": get_git_commit_hash(REPO_ROOT),
        "seed": args.seed,
        "data_fraction_seed": args.data_fraction_seed,
        "data_fraction": args.data_fraction,
        "effective_train_fraction": effective_train_fraction,
        "run_test": args.run_test,
        "hyperparameters": {
            "ngram_range": list(ngram_range),
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features,
            "sublinear_tf": args.sublinear_tf,
            "C": args.C,
            "seed": args.seed,
            "data_fraction": args.data_fraction,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_test_samples": args.max_test_samples,
            "class_weighting": args.class_weighting,
        },
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_test_samples": args.max_test_samples,
        "ngram_range": list(ngram_range),
        "min_df": args.min_df,
        "max_df": args.max_df,
        "max_features": args.max_features,
        "sublinear_tf": args.sublinear_tf,
        "C": args.C,
        "train_size": train_size,
        "eval_size": len(eval_data.records) if eval_data is not None else None,
        "test_size": len(test_data.records) if test_data is not None else None,
        "raw_train_size": train_data.raw_size if train_data is not None else None,
        "raw_eval_size": eval_data.raw_size if eval_data is not None else None,
        "raw_test_size": test_data.raw_size if test_data is not None else None,
        "full_train_size": full_train_size,
        "full_eval_size": eval_data.preprocessed_size if eval_data is not None else None,
        "full_test_size": test_data.preprocessed_size if test_data is not None else None,
        "dropped_no_majority_train": (
            train_data.dropped_no_majority_count if train_data is not None else None
        ),
        "dropped_no_majority_eval": (
            eval_data.dropped_no_majority_count if eval_data is not None else None
        ),
        "dropped_no_majority_test": (
            test_data.dropped_no_majority_count if test_data is not None else None
        ),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "vocab_size": vocab_size,
        "gpu_type": gpu_type,
        "output_dir": args.output_dir,
    }


def build_runtime_metrics(
    *,
    training_time_sec: float | None,
    gpu_type: str,
    peak_memory_mb: float | None = None,
    peak_memory_reserved_mb: float | None = None,
) -> dict[str, Any]:
    """Return the runtime block for the CPU sklearn baseline."""

    resolved_peak_memory_mb = peak_memory_mb
    resolved_peak_memory_reserved_mb = peak_memory_reserved_mb
    runtime = {
        "training_time_sec": training_time_sec,
        **build_compute_cost_fields(training_time_sec, gpu_type="cpu"),
        "compute_device": "cpu",
        "peak_memory_mb": resolved_peak_memory_mb,
        "peak_memory_allocated_mb": resolved_peak_memory_mb,
        "peak_memory_reserved_mb": resolved_peak_memory_reserved_mb,
        "gpu_type": gpu_type,
        "mixed_precision": "not_applicable",
        "gradient_checkpointing": False,
    }
    return runtime


def build_model_selection(
    eval_metrics: dict[str, Any],
    *,
    best_model_checkpoint: str | None,
) -> dict[str, Any]:
    """Describe model selection for a method that has no epochs/checkpoints."""

    return {
        "metric_for_best_model": "eval_f1_macro",
        "best_metric_key": "eval_f1_macro",
        "best_metric": eval_metrics.get("eval_f1_macro"),
        "greater_is_better": True,
        "best_epoch": None,
        "best_step": None,
        "best_model_checkpoint": best_model_checkpoint,
    }
