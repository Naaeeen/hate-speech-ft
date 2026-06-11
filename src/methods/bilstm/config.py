"""Config metadata helpers for the BiLSTM baseline.

This file turns the editable `manual_config.py` values plus split/model stats
into `resolved_config.json`. The shape is intentionally close to the other
methods so manual final tables can be rebuilt later.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.methods.bilstm.data import BiLSTMSplit
from src.utils.run_metadata import build_compute_cost_fields, get_git_commit_hash


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "bilstm-random-embedding"
TOKENIZER_NAME = "bilstm-word"

if TYPE_CHECKING:
    from src.methods.bilstm.tokenizer import StandardBiLSTMTokenizer


def validate_bilstm_config(args: Any) -> None:
    """Fail early for manual BiLSTM settings that would make the run nonsense."""

    positive_int_options = (
        "max_length",
        "embedding_size",
        "hidden_size",
        "num_layers",
        "batch_size",
        "eval_batch_size",
        "epochs",
        "tokenizer_min_freq",
    )
    for option_name in positive_int_options:
        if int(getattr(args, option_name)) < 1:
            raise ValueError(f"{option_name} must be >= 1.")

    for option_name in ("max_train_samples", "max_eval_samples", "max_test_samples"):
        value = getattr(args, option_name)
        if value is not None and value < 1:
            raise ValueError(f"{option_name} must be >= 1 when provided.")

    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in the interval [0, 1).")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0.")
    if args.warmup_ratio < 0:
        raise ValueError("warmup_ratio must be >= 0.")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be >= 0.")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if args.early_stopping_threshold < 0:
        raise ValueError("early_stopping_threshold must be >= 0.")
    if args.max_vocab_size < 2:
        raise ValueError("max_vocab_size must be >= 2 for pad and unk tokens.")
    if args.data_fraction is not None and not 0 < args.data_fraction <= 1:
        raise ValueError("data_fraction must be in the interval (0, 1].")
    if args.eval_strategy != "epoch":
        raise ValueError("Bi-LSTM currently supports only eval_strategy='epoch'.")
    if args.save_strategy not in {"no", "epoch"}:
        raise ValueError("Bi-LSTM supports save_strategy='no' or 'epoch'.")
    if args.load_best_model_at_end and args.save_strategy == "no":
        raise ValueError("load_best_model_at_end requires save_strategy='epoch'.")
    if args.metric_for_best_model != "eval_f1_macro":
        raise ValueError("Bi-LSTM currently selects checkpoints by eval_f1_macro.")


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
    args: Any,
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
) -> dict[str, Any]:
    """Build the saved config for one BiLSTM run.

    BiLSTM is custom PyTorch rather than HF Trainer, but the output config still
    records the same research evidence: split accounting, tokenizer policy,
    hyperparameters, parameter counts, class weights, and runtime context.
    """

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
        "tokenizer_min_freq": args.tokenizer_min_freq,
        "max_vocab_size": args.max_vocab_size,
        "device": args.device,
    }
    return {
        "method": args.method,
        "run_name": args.run_name,
        "dataset": args.dataset_name,
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "seed": args.seed,
        "data_fraction_seed": args.data_fraction_seed,
        "data_fraction": args.data_fraction,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_test_samples": args.max_test_samples,
        "run_test": args.run_test,
        "output_dir": args.output_dir,
        "hyperparameters": {
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "optim": "adamw_torch",
            "lr_scheduler_type": "linear",
            "class_weighting": args.class_weighting,
            "eval_strategy": args.eval_strategy,
            "save_strategy": args.save_strategy,
            "logging_strategy": args.logging_strategy,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
            "save_final_model": not args.no_save_final_model,
            "mixed_precision": "none",
            "gradient_checkpointing": False,
            **hyperparameters,
        },
        "train_split": train_split,
        "eval_split": eval_split,
        "test_split": test_split or args.test_split_name,
        "preprocessing_policy": "join_post_tokens_strict_majority",
        "label_policy": "strict_majority_drop_no_majority",
        "split_accounting_policy": (
            "raw_*_size is the loaded Hugging Face split size before local "
            "post-load preprocessing; dropped_no_majority_* counts strict-majority "
            "drops performed after dataset load."
        ),
        "selection_metric": "f1_macro",
        "test_policy": "enabled_by_run_test",
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
        "class_weights": class_weights,
        "vocab_size": tokenizer.vocab_size if tokenizer is not None else None,
    }


def build_runtime_metrics(
    *,
    training_time_sec: float | None,
    device: str,
    gpu_type: str | None,
    peak_memory_mb: float | None,
    peak_memory_reserved_mb: float | None = None,
    final_model_source: str | None = None,
) -> dict[str, Any]:
    """Create the BiLSTM runtime block with CPU/GPU memory fields."""

    cost_gpu_type = gpu_type if str(device).startswith("cuda") else "cpu"
    runtime: dict[str, Any] = {
        "training_time_sec": training_time_sec,
        **build_compute_cost_fields(training_time_sec, gpu_type=cost_gpu_type),
        "device": device,
        "gpu_type": gpu_type,
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_allocated_mb": peak_memory_mb,
        "peak_memory_reserved_mb": peak_memory_reserved_mb,
        "mixed_precision": "none",
        "gradient_checkpointing": False,
    }
    if final_model_source is not None:
        runtime["final_model_source"] = final_model_source
    return runtime


def build_model_selection(
    *,
    metric_for_best_model: str,
    best_metric: float | None,
    best_epoch: int | None,
    best_step: int | None,
    best_checkpoint: str | None,
) -> dict[str, Any]:
    """Record which epoch checkpoint won validation macro-F1."""

    return {
        "metric_for_best_model": metric_for_best_model,
        "best_metric_key": metric_for_best_model,
        "best_metric": best_metric,
        "greater_is_better": True,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_model_checkpoint": best_checkpoint,
    }
