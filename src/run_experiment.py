from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.registry import (
    DEFAULT_REGISTRY_PATH,
    build_experiment_command,
    format_command,
    load_experiment_registry,
    parse_override_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="List, preview, or run configured HateXplain experiments."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--include_planned", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override one script argument, e.g. --set learning_rate=3e-5",
    )

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hate-speech-ft")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--wandb_log_model",
        choices=("false", "end", "checkpoint"),
        default="false",
    )
    return parser.parse_args()


def print_experiment_list(registry, *, include_planned: bool) -> None:
    experiments = registry.experiments if include_planned else registry.ready_experiments()
    for spec in experiments:
        marker = "ready" if spec.is_ready else "planned"
        script_state = "script-ok" if spec.script_exists(REPO_ROOT) else "missing-script"
        print(
            f"{spec.experiment_id:32} {marker:8} {script_state:14} "
            f"{spec.method:18} {spec.stage:8} {spec.description}"
        )


def main() -> int:
    args = parse_args()
    registry = load_experiment_registry(args.config)

    if args.list:
        print_experiment_list(registry, include_planned=args.include_planned)
        return 0

    if not args.experiment:
        raise SystemExit("Pass --experiment NAME or use --list.")

    spec = registry.get(args.experiment)
    overrides = parse_override_pairs(args.overrides)
    try:
        command = build_experiment_command(
            spec,
            repo_root=REPO_ROOT,
            overrides=overrides,
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tags,
            wandb_mode=args.wandb_mode,
            wandb_log_model=args.wandb_log_model,
            python_executable=args.python,
        )
    except FileNotFoundError as exc:
        print(f"Cannot run experiment: {exc}", file=sys.stderr)
        print("Use --list --include_planned to inspect available experiments.", file=sys.stderr)
        return 2

    rendered = format_command(command)
    print(rendered)
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
