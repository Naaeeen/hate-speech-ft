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
from src.methods.common import add_common_method_arguments


_METHOD_ID_PLACEHOLDER = "{{METHOD_ID}}"
_METHOD_PACKAGE_PLACEHOLDER = "{{METHOD_PACKAGE}}"
_DESCRIPTION_PLACEHOLDER = "{{DESCRIPTION}}"

DEFAULT_METHOD_ID = (
    "template" if _METHOD_ID_PLACEHOLDER.startswith("{{") else _METHOD_ID_PLACEHOLDER
)
DEFAULT_METHOD_PACKAGE = (
    "method_template"
    if _METHOD_PACKAGE_PLACEHOLDER.startswith("{{")
    else _METHOD_PACKAGE_PLACEHOLDER
)
DEFAULT_DESCRIPTION = (
    "Method scaffold template."
    if _DESCRIPTION_PLACEHOLDER.startswith("{{")
    else _DESCRIPTION_PLACEHOLDER
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    add_common_method_arguments(
        parser,
        default_method=DEFAULT_METHOD_ID,
        default_search_stage="smoke",
        default_trial_id=f"{DEFAULT_METHOD_PACKAGE}_manual",
    )

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
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "optim": args.optim,
            "lr_scheduler_type": args.lr_scheduler_type,
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
            "This scaffold is intentionally incomplete. Implement data loading, "
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
