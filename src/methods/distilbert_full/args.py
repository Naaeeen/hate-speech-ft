from __future__ import annotations

import argparse

from src.utils.wandb_config import VALID_WANDB_LOG_MODEL_VALUES, VALID_WANDB_MODES


MODEL_NAME = "distilbert-base-uncased"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run DistilBERT on HateXplain")

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, default="Hate-speech-CNERG/hatexplain")
    parser.add_argument("--output_dir", type=str, default="./outputs/distilbert_hatexplain")
    parser.add_argument("--method", type=str, default="full-ft")
    parser.add_argument("--search_stage", type=str, default="smoke")
    parser.add_argument("--trial_id", type=str, default=None)
    parser.add_argument("--config_hash", type=str, default=None)
    parser.add_argument("--hpo_seed", type=int, default=None)
    parser.add_argument("--hpo_trial_cap", type=int, default=None)
    parser.add_argument("--hpo_time_cap_gpu_hours", type=float, default=None)
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument("--run_test", action="store_true")

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_fraction_seed", type=int, default=42)
    parser.add_argument("--data_fraction", type=float, default=None)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument(
        "--eval_strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
    )
    parser.add_argument(
        "--save_strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
    )
    parser.add_argument(
        "--logging_strategy",
        choices=("no", "steps", "epoch"),
        default="steps",
    )
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help=(
            "Allow writing into an output directory that already contains run "
            "artifacts. By default, existing result/checkpoint/model files are "
            "protected to avoid accidental experiment loss."
        ),
    )
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_f1_macro")
    parser.add_argument("--lower_is_better", action="store_true")
    parser.add_argument("--no_save_final_model", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        choices=("none", "fp16", "bf16"),
        default="none",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--class_weighting",
        choices=("none", "balanced"),
        default="none",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        choices=VALID_WANDB_MODES,
        default="online",
    )
    parser.add_argument(
        "--wandb_log_model",
        choices=VALID_WANDB_LOG_MODEL_VALUES,
        default="false",
    )

    return parser.parse_args(argv)
