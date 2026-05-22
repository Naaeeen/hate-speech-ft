from __future__ import annotations

import argparse

from src.methods.common import add_common_method_arguments


DEFAULT_METHOD_ID = "frozen-backbone"
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DESCRIPTION = "Run DistilBERT with a frozen backbone and trainable classifier head."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    add_common_method_arguments(
        parser,
        defaults={
            "method": DEFAULT_METHOD_ID,
            "trial_id": "frozen_distilbert_manual",
            "output_dir": "outputs/frozen_distilbert_manual",
            "load_best_model_at_end": True,
        },
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--lower_is_better", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args(argv)
