from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.frozen_distilbert.training import train_frozen_distilbert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-backbone DistilBERT HateXplain baseline."
    )

    # Metadata passed by src/run_experiment.py
    parser.add_argument("--method", type=str)
    parser.add_argument("--search_stage", type=str)
    parser.add_argument("--trial_id", type=str)

    # Shared/default arguments
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--selection_metric", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--data_fraction", type=float)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_test", action=argparse.BooleanOptionalAction)

    # Frozen DistilBERT arguments
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--epochs", type=int)

    # Checkpoint arguments
    parser.add_argument("--eval_strategy", type=str)
    parser.add_argument("--save_strategy", type=str)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--metric_for_best_model", type=str)
    parser.add_argument(
    "--device",
    type=str,
    choices=["auto", "cpu", "cuda"],
    )   

    return parser.parse_args()


def namespace_to_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None
    }


def main() -> int:
    args = parse_args()
    config = namespace_to_config(args)

    print("Starting frozen-backbone DistilBERT training...", flush=True)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)

    result = train_frozen_distilbert(config)

    print("Finished frozen-backbone DistilBERT training.", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
