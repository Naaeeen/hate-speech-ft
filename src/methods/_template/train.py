from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.results import write_failure_file, write_resolved_config
from src.methods.common import (
    add_common_method_arguments,
    build_common_experiment_config,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)


DEFAULT_METHOD_ID = "method-template"
DEFAULT_METHOD_PACKAGE = "method_template"
DEFAULT_DESCRIPTION = "Copyable method implementation template."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    add_common_method_arguments(
        parser,
        defaults={
            "method": DEFAULT_METHOD_ID,
            "trial_id": f"{DEFAULT_METHOD_PACKAGE}_manual",
            "output_dir": f"outputs/{DEFAULT_METHOD_PACKAGE}",
        },
    )

    # Replace or extend these with method-specific parameters.
    parser.add_argument("--model_name", type=str, default="replace-me")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1)
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> dict[str, object]:
    return build_common_experiment_config(
        args,
        model_name=args.model_name,
        tokenizer_name=None,
        hyperparameters={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        extra={
            "status": "template_not_implemented",
        },
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    config = build_experiment_config(args)
    output_dir = Path(args.output_dir)
    validate_test_evaluation_policy(
        search_stage=args.search_stage,
        run_test=args.run_test,
    )
    validate_output_dir_for_run(
        output_dir,
        overwrite=args.overwrite_output_dir,
    )
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
