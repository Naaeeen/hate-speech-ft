from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.hpo import (
    DEFAULT_SEARCH_SPACE_PATH,
    build_trial_overrides,
    get_search_space,
    get_seed_policy,
    get_trial_cap,
    load_hpo_config,
    shared_fixed_command_overrides,
)
from src.methods.frozen_distilbert.training import train_frozen_distilbert


WORKFLOW_ARG_KEYS = {
    "hpo_config",
    "search_space_name",
    "n_trials",
    "hpo_seed",
    "trial_output_root",
    "confirm_top_k",
    "confirm_output_root",
    "final_output_root",
    "confirm_seeds",
    "final_seeds",
    "final_run_test",
    "allow_over_cap",
    "dry_run_workflow",
    "workflow_summary_name",
}

HYPERPARAMETER_KEYS = {
    "learning_rate",
    "epochs",
    "dropout",
    "batch_size",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, sort_keys=True)
        file.write("\n")


def parse_int_list(raw_values: list[int] | None) -> list[int] | None:
    if raw_values is None or len(raw_values) == 0:
        return None
    return [int(value) for value in raw_values]


def get_stage_seeds(
    *,
    hpo_config: dict[str, Any],
    stage: str,
    explicit_seeds: list[int] | None,
    fallback: list[int],
) -> list[int]:
    if explicit_seeds:
        return explicit_seeds
    try:
        return get_seed_policy(hpo_config, stage)
    except ValueError:
        return fallback


def extract_eval_metrics(result: dict[str, Any]) -> dict[str, Any]:
    if "eval_metrics" in result and isinstance(result["eval_metrics"], dict):
        return result["eval_metrics"]
    if "metrics" in result and isinstance(result["metrics"], dict):
        metrics = result["metrics"]
        if "eval" in metrics and isinstance(metrics["eval"], dict):
            return metrics["eval"]
        if "best_eval" in metrics and isinstance(metrics["best_eval"], dict):
            return metrics["best_eval"]
    if "best_eval" in result and isinstance(result["best_eval"], dict):
        return result["best_eval"]
    raise KeyError("Could not find eval metrics in train_frozen_distilbert() result.")


def get_metric(result: dict[str, Any], metric_name: str) -> float:
    eval_metrics = extract_eval_metrics(result)
    if metric_name not in eval_metrics:
        available = ", ".join(sorted(eval_metrics))
        raise KeyError(f"Metric {metric_name!r} not found. Available metrics: {available}")
    return float(eval_metrics[metric_name])


def normalize_frozen_distilbert_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Keep the HPO search-space names, but also provide the names expected by
    train_frozen_distilbert().

    search_spaces.json uses:
      - head_learning_rate
      - num_train_epochs

    training.py reads:
      - learning_rate
      - epochs
    """
    normalized = dict(config)

    if normalized.get("head_learning_rate") is not None:
        normalized["learning_rate"] = float(normalized["head_learning_rate"])

    if normalized.get("num_train_epochs") is not None:
        normalized["epochs"] = int(normalized["num_train_epochs"])

    if normalized.get("learning_rate") is not None:
        normalized["learning_rate"] = float(normalized["learning_rate"])

    if normalized.get("epochs") is not None:
        normalized["epochs"] = int(normalized["epochs"])

    if normalized.get("weight_decay") is not None:
        normalized["weight_decay"] = float(normalized["weight_decay"])

    return normalized


def select_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in sorted(HYPERPARAMETER_KEYS) if key in config}


def make_stage_trial_id(prefix: str, config_rank: int, seed: int, stage: str) -> str:
    return f"{prefix}__{stage}_config{config_rank:02d}_seed{seed}"


def run_one_config(config: dict[str, Any], *, metric_name: str) -> dict[str, Any]:
    config = normalize_frozen_distilbert_config(config)

    print("\nRunning config:", flush=True)
    print(json.dumps(json_safe(config), indent=2, sort_keys=True), flush=True)

    result = train_frozen_distilbert(config)
    metric = get_metric(result, metric_name)

    return {
        "config": config,
        "result": result,
        "metric_name": metric_name,
        "metric_value": metric,
        "eval_metrics": extract_eval_metrics(result),
    }


def summarize_runs(runs: list[dict[str, Any]], *, metric_name: str) -> dict[str, Any]:
    values = [float(run["metric_value"]) for run in runs]
    return {
        "metric_name": metric_name,
        "values": values,
        "mean": statistics.mean(values) if values else None,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "num_runs": len(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full frozen DistilBERT HPO workflow: tune, confirm, and final seed runs."
    )

    # Workflow arguments.
    parser.add_argument("--hpo_config", type=str, default=str(DEFAULT_SEARCH_SPACE_PATH))
    parser.add_argument("--search_space_name", type=str, default="frozen_backbone")
    parser.add_argument("--n_trials", type=int, default=6)
    parser.add_argument("--hpo_seed", type=int, default=2026)
    parser.add_argument("--trial_output_root", type=str, default="outputs/frozen_distilbert_hpo")
    parser.add_argument("--confirm_top_k", type=int, default=2)
    parser.add_argument("--confirm_output_root", type=str, default=None)
    parser.add_argument("--final_output_root", type=str, default=None)
    parser.add_argument("--confirm_seeds", type=int, nargs="*", default=None)
    parser.add_argument("--final_seeds", type=int, nargs="*", default=None)
    parser.add_argument("--final_run_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_over_cap", action="store_true")
    parser.add_argument("--dry_run_workflow", action="store_true")
    parser.add_argument("--workflow_summary_name", type=str, default="hpo_workflow_summary.json")

    # Metadata commonly passed by run_experiment.py.
    parser.add_argument("--method", type=str)
    parser.add_argument("--search_stage", type=str)
    parser.add_argument("--trial_id", type=str)
    parser.add_argument("--config_hash", type=str)
    parser.add_argument("--overwrite_output_dir", action=argparse.BooleanOptionalAction)

    # Shared dataset/training args.
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--selection_metric", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--data_fraction", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--run_test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    # Frozen DistilBERT/training hyperparameters.
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--head_learning_rate", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--num_classes", type=int, default=3)

    # Extra protocol args that may be shared across methods.
    parser.add_argument("--class_weighting", type=str)
    parser.add_argument("--early_stopping_patience", type=int)
    parser.add_argument("--early_stopping_threshold", type=float)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--mixed_precision", type=str)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction)

    # Checkpoint/selection args.
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_f1_macro")

    return parser.parse_args()


def base_training_config(args: argparse.Namespace) -> dict[str, Any]:
    raw = vars(args)
    return {
        key: value
        for key, value in raw.items()
        if key not in WORKFLOW_ARG_KEYS and value is not None
    }


def main() -> int:
    args = parse_args()
    if args.n_trials <= 0:
        raise SystemExit("--n_trials must be positive.")
    if args.confirm_top_k <= 0:
        raise SystemExit("--confirm_top_k must be positive.")

    workflow_start = time.time()
    hpo_config = load_hpo_config(args.hpo_config)
    search_space = get_search_space(hpo_config, args.search_space_name)
    trial_cap = get_trial_cap(hpo_config, args.search_space_name)

    base_config = base_training_config(args)
    fixed_overrides = shared_fixed_command_overrides(hpo_config)

    base_config = {**base_config, **fixed_overrides}
    base_config["run_test"] = False
    base_config["search_stage"] = "tuning"
    base_config = normalize_frozen_distilbert_config(base_config)

    output_root = Path(args.trial_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    confirm_root = Path(args.confirm_output_root) if args.confirm_output_root else output_root / "confirm"
    final_root = Path(args.final_output_root) if args.final_output_root else output_root / "final"

    metric_name = str(args.metric_for_best_model)
    confirm_seeds = get_stage_seeds(
        hpo_config=hpo_config,
        stage="confirm",
        explicit_seeds=parse_int_list(args.confirm_seeds),
        fallback=[42, 43, 44],
    )
    final_seeds = get_stage_seeds(
        hpo_config=hpo_config,
        stage="final",
        explicit_seeds=parse_int_list(args.final_seeds),
        fallback=[42, 43, 44],
    )

    trial_overrides = build_trial_overrides(
        base_experiment_id=args.trial_id or "frozen_distilbert_hpo",
        method=args.method or "frozen_backbone",
        search_space=search_space,
        n_trials=args.n_trials,
        hpo_seed=args.hpo_seed,
        output_root=output_root.as_posix(),
        search_stage="tuning",
        trial_cap=trial_cap,
        allow_over_cap=args.allow_over_cap,
        fixed_overrides=fixed_overrides,
    )

    planned_tuning_configs = [
        normalize_frozen_distilbert_config({**base_config, **overrides})
        for overrides in trial_overrides
    ]

    state: dict[str, Any] = {
        "workflow": "frozen_distilbert_hpo_tune_confirm_final",
        "metric_for_selection": metric_name,
        "hpo_config": str(args.hpo_config),
        "search_space_name": args.search_space_name,
        "hpo_seed": args.hpo_seed,
        "n_trials": args.n_trials,
        "confirm_top_k": args.confirm_top_k,
        "confirm_seeds": confirm_seeds,
        "final_seeds": final_seeds,
        "weight_decay_policy": "from_shared_fixed_or_cli",
        "tuning_seed": int(base_config.get("seed", 42)),
        "planned_tuning_configs": planned_tuning_configs,
        "tuning_runs": [],
        "top_configs_after_tuning": [],
        "confirmation": [],
        "confirmation_summary": [],
        "selected_config": None,
        "final_runs": [],
        "final_summary": None,
        "runtime": {},
    }

    summary_path = output_root / args.workflow_summary_name
    write_json(summary_path, state)

    if args.dry_run_workflow:
        print(f"Dry run only. Wrote planned workflow to {summary_path}")
        return 0

    print(f"Starting frozen DistilBERT HPO workflow. Summary: {summary_path}", flush=True)

    # Stage 1: HPO tuning using seed 42/default training seed.
    for index, config in enumerate(planned_tuning_configs, start=1):
        print(f"\n=== Tuning trial {index}/{len(planned_tuning_configs)} ===", flush=True)
        run_record = run_one_config(config, metric_name=metric_name)
        run_record["rank_input_index"] = index
        state["tuning_runs"].append(run_record)
        write_json(output_root / "hpo_results_partial.json", state)

    sorted_tuning = sorted(
        state["tuning_runs"],
        key=lambda item: float(item["metric_value"]),
        reverse=True,
    )
    top_k = min(args.confirm_top_k, len(sorted_tuning))
    top_tuning = sorted_tuning[:top_k]
    state["top_configs_after_tuning"] = [
        {
            "rank": rank,
            "metric_name": metric_name,
            "metric_value": item["metric_value"],
            "hyperparameters": select_hyperparameters(item["config"]),
            "output_dir": item["config"].get("output_dir"),
            "trial_id": item["config"].get("trial_id"),
        }
        for rank, item in enumerate(top_tuning, start=1)
    ]
    write_json(output_root / "hpo_results_partial.json", state)

    # Stage 2: Confirm top configs by averaging confirmation seeds.
    for config_rank, tuning_item in enumerate(top_tuning, start=1):
        selected_hparams = select_hyperparameters(tuning_item["config"])
        config_runs = []
        for seed in confirm_seeds:
            trial_id = make_stage_trial_id(
                args.trial_id or "frozen_distilbert_hpo",
                config_rank=config_rank,
                seed=seed,
                stage="confirm",
            )
            confirm_config = normalize_frozen_distilbert_config(
                {
                    **base_config,
                    **selected_hparams,
                    "seed": int(seed),
                    "run_test": False,
                    "search_stage": "confirm",
                    "trial_id": trial_id,
                    "output_dir": (confirm_root / trial_id).as_posix(),
                }
            )
            print(
                f"\n=== Confirmation config {config_rank}/{top_k}, seed {seed} ===",
                flush=True,
            )
            run_record = run_one_config(confirm_config, metric_name=metric_name)
            run_record["config_rank"] = config_rank
            run_record["seed"] = int(seed)
            config_runs.append(run_record)
            state["confirmation"].append(run_record)
            write_json(output_root / "hpo_results_partial.json", state)

        state["confirmation_summary"].append(
            {
                "config_rank": config_rank,
                "hyperparameters": selected_hparams,
                **summarize_runs(config_runs, metric_name=metric_name),
            }
        )
        write_json(output_root / "hpo_results_partial.json", state)

    selected_confirmation = max(
        state["confirmation_summary"],
        key=lambda item: float(item["mean"]),
    )
    selected_hparams = dict(selected_confirmation["hyperparameters"])
    state["selected_config"] = {
        "reason": "highest_mean_confirmation_metric",
        "metric_name": metric_name,
        "confirmation_mean": selected_confirmation["mean"],
        "confirmation_std": selected_confirmation["std"],
        "hyperparameters": selected_hparams,
    }
    write_json(output_root / "hpo_results_partial.json", state)

    # Stage 3: Final normal training runs with the selected config on final seeds.
    final_run_records = []
    for seed in final_seeds:
        trial_id = make_stage_trial_id(
            args.trial_id or "frozen_distilbert_hpo",
            config_rank=int(selected_confirmation["config_rank"]),
            seed=seed,
            stage="final",
        )
        final_config = normalize_frozen_distilbert_config(
            {
                **base_config,
                **selected_hparams,
                "seed": int(seed),
                "run_test": bool(args.final_run_test),
                "search_stage": "final",
                "trial_id": trial_id,
                "output_dir": (final_root / trial_id).as_posix(),
                "max_train_samples": None,
                "max_eval_samples": None,
                "max_test_samples": None,
                "data_fraction": 1.0,
            }
        )
        print(f"\n=== Final run seed {seed} ===", flush=True)
        run_record = run_one_config(final_config, metric_name=metric_name)
        run_record["seed"] = int(seed)
        final_run_records.append(run_record)
        state["final_runs"].append(run_record)
        write_json(output_root / "hpo_results_partial.json", state)

    state["final_summary"] = {
        "selected_hyperparameters": selected_hparams,
        "eval": summarize_runs(final_run_records, metric_name=metric_name),
        "test_metrics_by_seed": [
            {
                "seed": record["seed"],
                "test_metrics": record["result"].get("test_metrics"),
                "output_dir": record["config"].get("output_dir"),
            }
            for record in final_run_records
        ],
    }
    state["runtime"] = {
        "workflow_time_sec": time.time() - workflow_start,
        "summary_path": summary_path.as_posix(),
    }

    write_json(summary_path, state)
    write_json(output_root / "hpo_results_partial.json", state)

    print("\nFinished frozen DistilBERT HPO workflow.", flush=True)
    print(json.dumps(json_safe(state["selected_config"]), indent=2, sort_keys=True), flush=True)
    print(f"Summary written to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
