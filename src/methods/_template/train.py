from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.results import write_failure_file, write_resolved_config


DEFAULT_METHOD_ID = "method-template"
DEFAULT_METHOD_PACKAGE = "method_template"
DEFAULT_DESCRIPTION = "Copyable method implementation template."


def add_shared_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD_ID)
    parser.add_argument("--search_stage", type=str, default="smoke")
    parser.add_argument("--trial_id", type=str, default=f"{DEFAULT_METHOD_PACKAGE}_manual")
    parser.add_argument("--config_hash", type=str, default=None)
    parser.add_argument("--hpo_seed", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="Hate-speech-CNERG/hatexplain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_fraction_seed", type=int, default=42)
    parser.add_argument("--data_fraction", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=f"outputs/{DEFAULT_METHOD_PACKAGE}")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--logging_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_f1_macro")
    parser.add_argument("--no_save_final_model", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="none", choices=["none", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--class_weighting", type=str, default="none", choices=["none", "balanced"])
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hate-speech-ft")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_model", type=str, default="false", choices=["false", "end", "checkpoint"])
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    add_shared_arguments(parser)

    # Replace or extend these with method-specific parameters.
    parser.add_argument("--model_name", type=str, default="replace-me")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1)
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": args.method,
        "search_stage": args.search_stage,
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "hpo_seed": args.hpo_seed,
        "dataset": args.dataset_name,
        "model_name": args.model_name,
        "seed": args.seed,
        "data_fraction": args.data_fraction,
        "output_dir": args.output_dir,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
        },
        "global_switches": {
            "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "class_weighting": args.class_weighting,
            "early_stopping": args.early_stopping_patience > 0,
        },
        "checkpoint_policy": {
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
            "wandb_log_model": args.wandb_log_model,
        },
        "status": "template_not_implemented",
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    config = build_experiment_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(output_dir, config)

    try:
        raise NotImplementedError(
            "This is a copyable template. Replace this block with method-specific "
            "training, evaluation, W&B logging, and write_result_files()."
        )
    except Exception as exc:
        write_failure_file(
            output_dir,
            config=config,
            error=exc,
            runtime_metrics={
                "status": "failed",
                "failure_phase": "template",
                "training_time_sec": time.perf_counter() - start,
            },
        )
        raise


if __name__ == "__main__":
    main()
