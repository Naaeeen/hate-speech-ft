from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.methods.hf_common import (
    build_compute_cost_fields,
    get_git_commit_hash,
)
from src.methods.tfidf_logreg.data import ClassicalSplit
from src.methods.tfidf_logreg.training import parse_ngram_range
from src.utils.wandb_config import (
    WandbSettings,
    parse_wandb_tags,
    slugify_run_part,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_NAME = "tfidf-logreg"


def build_wandb_run_name(args: argparse.Namespace) -> str:
    ngram_lower, ngram_upper = parse_ngram_range(args.ngram_range)
    sample_part = f"train{args.max_train_samples}" if args.max_train_samples else "full"
    base = (
        f"{slugify_run_part(args.method, default='method')}_"
        f"{slugify_run_part(DEFAULT_MODEL_NAME, default='model')}_"
        f"seed{args.seed}_{sample_part}_"
        f"ngram{ngram_lower}-{ngram_upper}_min_df{args.min_df}_C{args.C:g}"
    )
    if args.trial_id:
        return f"{slugify_run_part(args.trial_id, default='trial')}_{base}"
    return base


def resolve_wandb_settings(args: argparse.Namespace) -> WandbSettings:
    return WandbSettings(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=args.wandb_run_name or build_wandb_run_name(args),
        group=args.wandb_group,
        tags=parse_wandb_tags(args.wandb_tags),
        log_model=args.wandb_log_model,
    )


def build_experiment_config(
    args: argparse.Namespace,
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
    setup_complete: bool = True,
) -> dict[str, Any]:
    train_size = len(train_data.records) if train_data is not None else None
    full_train_size = train_data.preprocessed_size if train_data is not None else None
    effective_train_fraction = (
        train_size / full_train_size if train_size is not None and full_train_size else None
    )
    class_weight = "balanced" if args.class_weighting == "balanced" else None
    return {
        "method": args.method,
        "search_stage": args.search_stage,
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "search_method": getattr(args, "search_method", None),
        "search_space_name": getattr(args, "search_space_name", None),
        "hpo_seed": args.hpo_seed,
        "hpo_trial_cap": getattr(args, "hpo_trial_cap", None),
        "hpo_time_cap_gpu_hours": getattr(args, "hpo_time_cap_gpu_hours", None),
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
        "test_policy": "final_only",
        "model_name": DEFAULT_MODEL_NAME,
        "tokenizer_name": "tfidf",
        "git_commit": get_git_commit_hash(REPO_ROOT),
        "seed": args.seed,
        "data_fraction_seed": args.data_fraction_seed,
        "data_fraction": args.data_fraction,
        "effective_train_fraction": effective_train_fraction,
        "run_test": args.run_test,
        "global_switches": {
            "mixed_precision": "not_applicable",
            "gradient_checkpointing": False,
            "class_weighting": args.class_weighting,
            "weighted_ce": False,
            "early_stopping": False,
        },
        "training_policy": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "solver": "liblinear",
            "max_iter": 1000,
            "class_weighting": args.class_weighting,
            "class_weight": class_weight,
            "random_state": args.seed,
            "mixed_precision": "not_applicable",
            "gradient_checkpointing": False,
        },
        "checkpoint_policy": {
            "save_final_model": not args.no_save_final_model,
            "final_model_source": "final_fit",
            "overwrite_output_dir": args.overwrite_output_dir,
            "wandb_log_model": args.wandb_log_model,
        },
        "hyperparameters": {
            "ngram_range": list(ngram_range),
            "min_df": args.min_df,
            "max_features": args.max_features,
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
        "max_features": args.max_features,
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
        "setup_complete": setup_complete,
    }


def build_runtime_metrics(
    *,
    training_time_sec: float | None,
    gpu_type: str,
    status: str,
    failure_phase: str | None = None,
    peak_memory_mb: float | None = None,
    peak_memory_reserved_mb: float | None = None,
) -> dict[str, Any]:
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
        "status": status,
    }
    if failure_phase is not None:
        runtime["failure_phase"] = failure_phase
    return runtime


def build_model_selection(eval_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_for_best_model": "eval_f1_macro",
        "best_metric_key": "eval_f1_macro",
        "best_metric": eval_metrics.get("eval_f1_macro"),
        "greater_is_better": True,
        "best_epoch": None,
        "best_step": None,
        "best_model_checkpoint": "model.joblib",
    }
