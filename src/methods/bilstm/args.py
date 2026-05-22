from __future__ import annotations

import argparse

from src.methods.common import add_common_method_arguments


SUPPORTED_OPTIMIZERS = {"adamw_torch"}
SUPPORTED_LR_SCHEDULERS = {"linear"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Bi-LSTM HateXplain baseline."
    )
    add_common_method_arguments(
        parser,
        defaults={
            "method": "bilstm",
            "trial_id": "bilstm_manual",
            "output_dir": "outputs/bilstm_manual",
            "load_best_model_at_end": True,
        },
    )
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args(argv)


def validate_bilstm_args(args: argparse.Namespace) -> None:
    positive_int_options = (
        "max_length",
        "embedding_size",
        "hidden_size",
        "num_layers",
        "batch_size",
        "eval_batch_size",
        "epochs",
    )
    for option_name in positive_int_options:
        if int(getattr(args, option_name)) < 1:
            raise ValueError(f"--{option_name} must be >= 1.")

    for option_name in ("max_train_samples", "max_eval_samples", "max_test_samples"):
        value = getattr(args, option_name)
        if value is not None and value < 1:
            raise ValueError(f"--{option_name} must be >= 1 when provided.")

    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in the interval [0, 1).")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be >= 0.")
    if args.warmup_ratio < 0:
        raise ValueError("--warmup_ratio must be >= 0.")
    if args.max_grad_norm < 0:
        raise ValueError("--max_grad_norm must be >= 0.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be >= 0.")
    if args.early_stopping_threshold < 0:
        raise ValueError("--early_stopping_threshold must be >= 0.")
    if args.data_fraction is not None and not 0 < args.data_fraction <= 1:
        raise ValueError("--data_fraction must be in the interval (0, 1].")
    if args.optim not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise ValueError(f"Bi-LSTM supports --optim values: {supported}.")
    if args.lr_scheduler_type not in SUPPORTED_LR_SCHEDULERS:
        supported = ", ".join(sorted(SUPPORTED_LR_SCHEDULERS))
        raise ValueError(f"Bi-LSTM supports --lr_scheduler_type values: {supported}.")
    if args.mixed_precision != "none":
        raise ValueError("Bi-LSTM currently supports only --mixed_precision none.")
    if args.gradient_checkpointing:
        raise ValueError("Bi-LSTM does not support --gradient_checkpointing.")
    if args.eval_strategy != "epoch":
        raise ValueError("Bi-LSTM currently supports only --eval_strategy epoch.")
    if args.save_strategy not in {"no", "epoch"}:
        raise ValueError("Bi-LSTM supports --save_strategy no or epoch.")
    if args.load_best_model_at_end and args.save_strategy == "no":
        raise ValueError("--load_best_model_at_end requires --save_strategy epoch.")
    if args.metric_for_best_model != "eval_f1_macro":
        raise ValueError("Bi-LSTM currently selects checkpoints by eval_f1_macro.")
    if args.wandb_log_model != "false":
        raise ValueError(
            "Bi-LSTM currently records local model artifacts only. "
            "Use --wandb_log_model false until W&B artifact upload is implemented "
            "for this method."
        )
