from __future__ import annotations

import argparse

from src.methods.common import add_common_method_arguments


DEFAULT_METHOD_ID = "lp-ft"
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DESCRIPTION = "Run DistilBERT linear probing followed by full fine-tuning."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    add_common_method_arguments(
        parser,
        defaults={
            "method": DEFAULT_METHOD_ID,
            "trial_id": "distilbert_lp_ft_manual",
            "output_dir": "outputs/distilbert_lp_ft_manual",
            "load_best_model_at_end": True,
        },
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Legacy alias for both per-device train/eval batch sizes. Prefer "
            "--per_device_train_batch_size and --per_device_eval_batch_size."
        ),
    )
    parser.add_argument("--stage1_head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--stage1_epochs", type=float, default=2.0)
    parser.add_argument("--stage2_learning_rate", type=float, default=2e-5)
    parser.add_argument("--stage2_epochs", type=float, default=2.0)
    parser.add_argument("--lower_is_better", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()
