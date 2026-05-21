from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.methods.bilstm.data import BiLSTMSplit
from src.methods.common import build_common_experiment_config
from src.methods.hf_common import build_compute_cost_fields, get_git_commit_hash
from src.utils.wandb_config import (
    WandbSettings,
    build_wandb_run_name,
    parse_wandb_tags,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "bilstm-random-embedding"
TOKENIZER_NAME = "distilbert-base-uncased"

if TYPE_CHECKING:
    from src.methods.bilstm.tokenizer import StandardBiLSTMTokenizer


def resolve_wandb_settings(args: argparse.Namespace) -> WandbSettings:
    run_name = args.wandb_run_name or build_wandb_run_name(
        method=args.method,
        model_name=MODEL_NAME,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        trial_id=args.trial_id,
    )
    return WandbSettings(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=run_name,
        group=args.wandb_group,
        tags=parse_wandb_tags(args.wandb_tags),
        log_model=args.wandb_log_model,
    )


def _split_sizes(split: BiLSTMSplit | None) -> dict[str, Any]:
    if split is None:
        return {
            "size": None,
            "raw_size": None,
            "preprocessed_size": None,
            "dropped_no_majority_count": None,
        }
    return {
        "size": len(split.records),
        "raw_size": split.raw_size,
        "preprocessed_size": split.preprocessed_size,
        "dropped_no_majority_count": split.dropped_no_majority_count,
    }


def build_experiment_config(
    args: argparse.Namespace,
    *,
    train_split: str | None = None,
    eval_split: str | None = None,
    test_split: str | None = None,
    train_data: BiLSTMSplit | None = None,
    eval_data: BiLSTMSplit | None = None,
    test_data: BiLSTMSplit | None = None,
    tokenizer: StandardBiLSTMTokenizer | None = None,
    gpu_type: str | None = None,
    trainable_params: int | None = None,
    total_params: int | None = None,
    class_weights: list[float] | None = None,
    setup_complete: bool = True,
) -> dict[str, Any]:
    train = _split_sizes(train_data)
    eval_ = _split_sizes(eval_data)
    test = _split_sizes(test_data)
    effective_train_fraction = (
        train["size"] / train["preprocessed_size"]
        if train["size"] is not None and train["preprocessed_size"]
        else None
    )
    hyperparameters = {
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "device": args.device,
    }
    config = build_common_experiment_config(
        args,
        model_name=MODEL_NAME,
        tokenizer_name=TOKENIZER_NAME,
        hyperparameters=hyperparameters,
        class_weights=class_weights,
        extra={
            "train_split": train_split,
            "eval_split": eval_split,
            "test_split": test_split or args.test_split_name,
            "preprocessing_policy": "join_post_tokens_strict_majority",
            "label_policy": "strict_majority_drop_no_majority",
            "split_accounting_policy": (
                "raw_*_size is the loaded Hugging Face split size before local "
                "post-load preprocessing; dropped_no_majority_* counts "
                "strict-majority drops performed after dataset load."
            ),
            "selection_metric": "f1_macro",
            "test_policy": "final_only",
            "run_test": args.run_test,
            "git_commit": get_git_commit_hash(REPO_ROOT),
            "train_size": train["size"],
            "eval_size": eval_["size"],
            "test_size": test["size"],
            "raw_train_size": train["raw_size"],
            "raw_eval_size": eval_["raw_size"],
            "raw_test_size": test["raw_size"],
            "full_train_size": train["preprocessed_size"],
            "full_eval_size": eval_["preprocessed_size"],
            "full_test_size": test["preprocessed_size"],
            "dropped_no_majority_train": train["dropped_no_majority_count"],
            "dropped_no_majority_eval": eval_["dropped_no_majority_count"],
            "dropped_no_majority_test": test["dropped_no_majority_count"],
            "effective_train_fraction": effective_train_fraction,
            "tokenizer_policy": (
                tokenizer.to_dict()
                if tokenizer is not None
                else {"tokenizer_name": TOKENIZER_NAME}
            ),
            "gpu_type": gpu_type,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "vocab_size": tokenizer.vocab_size if tokenizer is not None else None,
            "setup_complete": setup_complete,
        },
    )
    config["training_policy"] = {
        **config["training_policy"],
        "model_class": "src.methods.bilstm.model.BiLSTMClassifier",
        "optimizer": "torch.optim.AdamW",
        "scheduler": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
    }
    return config


def build_runtime_metrics(
    *,
    training_time_sec: float | None,
    device: str,
    gpu_type: str | None,
    peak_memory_mb: float | None,
    status: str,
    peak_memory_reserved_mb: float | None = None,
    final_model_source: str | None = None,
    failure_phase: str | None = None,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "training_time_sec": training_time_sec,
        **build_compute_cost_fields(training_time_sec, gpu_type=gpu_type),
        "device": device,
        "gpu_type": gpu_type,
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_allocated_mb": peak_memory_mb,
        "peak_memory_reserved_mb": peak_memory_reserved_mb,
        "mixed_precision": "none",
        "gradient_checkpointing": False,
        "status": status,
    }
    if final_model_source is not None:
        runtime["final_model_source"] = final_model_source
    if failure_phase is not None:
        runtime["failure_phase"] = failure_phase
    return runtime


def build_model_selection(
    *,
    metric_for_best_model: str,
    best_metric: float | None,
    best_epoch: int | None,
    best_step: int | None,
    best_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "metric_for_best_model": metric_for_best_model,
        "best_metric_key": metric_for_best_model,
        "best_metric": best_metric,
        "greater_is_better": True,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_model_checkpoint": best_checkpoint,
    }
