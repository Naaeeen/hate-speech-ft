from __future__ import annotations

import argparse

from src.methods.common import add_common_method_arguments


DEFAULT_METHOD_ID = "lora"
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DESCRIPTION = "Run DistilBERT with LoRA parameter-efficient fine-tuning."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    add_common_method_arguments(
        parser,
        defaults={
            "method": DEFAULT_METHOD_ID,
            "trial_id": "distilbert_lora_manual",
            "output_dir": "outputs/distilbert_lora_manual",
            "load_best_model_at_end": True,
        },
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--target_modules", type=str, default="q_lin,k_lin,v_lin")
    parser.add_argument("--modules_to_save", type=str, default="pre_classifier,classifier")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lower_is_better", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args(argv)
