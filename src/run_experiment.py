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
    validate_direct_run_overrides,
)
from src.experiments.protocol import (
    format_protocol_report,
    validate_experiment_protocol,
)
from src.experiments.hpo import (
    DEFAULT_SEARCH_SPACE_PATH,
    build_seed_run_overrides,
    build_trial_overrides,
    default_search_space_name,
    get_config_hash_keys,
    get_seed_policy,
    get_search_space,
    get_time_cap_gpu_hours,
    get_trial_cap,
    load_hpo_config,
    merge_trial_overrides,
    SEED_RUN_PROTECTED_USER_OVERRIDE_KEYS,
    shared_fixed_command_overrides,
    validate_seed_run_base_stage,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="List, preview, or run configured HateXplain experiments."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--include_planned", action="store_true")
    parser.add_argument(
        "--validate_protocol",
        action="store_true",
        help="Validate catalog and HPO config against the final research protocol.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument(
        "--suggest_trials",
        type=int,
        default=0,
        help="Print deterministic HPO trial commands without running them.",
    )
    parser.add_argument(
        "--suggest_seed_runs",
        choices=("confirm", "final"),
        default=None,
        help=(
            "Print confirmation or final seed commands for a selected config. "
            "Pass selected hyperparameters with --set."
        ),
    )
    parser.add_argument(
        "--search_space",
        type=str,
        default=None,
        help="Search-space name from configs/search_spaces.json.",
    )
    parser.add_argument("--search_config", type=str, default=str(DEFAULT_SEARCH_SPACE_PATH))
    parser.add_argument("--hpo_seed", type=int, default=42)
    parser.add_argument("--trial_output_root", type=str, default="outputs/hpo")
    parser.add_argument(
        "--seed_output_root",
        type=str,
        default=None,
        help="Output root for --suggest_seed_runs. Defaults to outputs/confirm or outputs/final.",
    )
    parser.add_argument(
        "--allow_smoke_hpo",
        action="store_true",
        help="Allow HPO trial generation from a smoke experiment base.",
    )
    parser.add_argument(
        "--allow_over_cap",
        action="store_true",
        help="Allow --suggest_trials to exceed configs/search_spaces.json trial_caps.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Pass --overwrite_output_dir to method scripts for an intentional rerun.",
    )
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

    if args.validate_protocol:
        hpo_config = load_hpo_config(args.search_config)
        report = validate_experiment_protocol(registry, hpo_config, repo_root=REPO_ROOT)
        print(format_protocol_report(report))
        return 0 if report.is_valid else 2

    if not args.experiment:
        raise SystemExit("Pass --experiment NAME, --validate_protocol, or use --list.")

    try:
        spec = registry.get(args.experiment)
        overrides = parse_override_pairs(args.overrides)
        if args.overwrite_output_dir:
            overrides["overwrite_output_dir"] = True
    except (ValueError, KeyError) as exc:
        print(f"Cannot run experiment: {exc}", file=sys.stderr)
        print("Use --list --include_planned to inspect available experiments.", file=sys.stderr)
        return 2
    if args.suggest_trials < 0:
        raise SystemExit("--suggest_trials must be >= 0.")
    try:
        if args.suggest_seed_runs:
            search_config = load_hpo_config(args.search_config)
            seed_stage = args.suggest_seed_runs
            validate_seed_run_base_stage(spec.stage)
            search_space_name = args.search_space or default_search_space_name(spec.method)
            hash_keys = get_config_hash_keys(search_config, search_space_name)
            output_root = args.seed_output_root or f"outputs/{seed_stage}"
            seed_overrides = build_seed_run_overrides(
                base_experiment_id=spec.experiment_id,
                method=spec.method,
                seeds=get_seed_policy(search_config, seed_stage),
                output_root=output_root,
                search_stage=seed_stage,
                fixed_overrides=shared_fixed_command_overrides(search_config),
                trial_cap=get_trial_cap(search_config, search_space_name),
                time_cap_gpu_hours=get_time_cap_gpu_hours(
                    search_config,
                    search_space_name,
                ),
            )
            for trial_overrides in seed_overrides:
                merged_overrides = merge_trial_overrides(
                    base_args={**spec.args, "method": spec.method},
                    user_overrides=overrides,
                    trial_overrides=trial_overrides,
                    protected_user_override_keys=SEED_RUN_PROTECTED_USER_OVERRIDE_KEYS,
                    hash_keys=hash_keys,
                )
                command = build_experiment_command(
                    spec,
                    repo_root=REPO_ROOT,
                    overrides=merged_overrides,
                    use_wandb=args.use_wandb,
                    wandb_entity=args.wandb_entity,
                    wandb_project=args.wandb_project,
                    wandb_group=args.wandb_group,
                    wandb_tags=args.wandb_tags,
                    wandb_mode=args.wandb_mode,
                    wandb_log_model=args.wandb_log_model,
                    python_executable=args.python,
                )
                print(format_command(command))
            return 0

        if args.suggest_trials:
            search_config = load_hpo_config(args.search_config)
            search_space_name = args.search_space or default_search_space_name(spec.method)
            hash_keys = get_config_hash_keys(search_config, search_space_name)
            if spec.stage == "smoke" and not args.allow_smoke_hpo:
                raise SystemExit(
                    "Refusing to generate HPO trials from a smoke experiment because "
                    "its sample caps are for setup checks, not model selection. Use a "
                    "matching tuning experiment for real HPO, or pass "
                    "--allow_smoke_hpo for a smoke-only command test."
                )
            search_space = get_search_space(search_config, search_space_name)
            trials = build_trial_overrides(
                base_experiment_id=spec.experiment_id,
                method=spec.method,
                search_space=search_space,
                n_trials=args.suggest_trials,
                hpo_seed=args.hpo_seed,
                output_root=args.trial_output_root,
                search_stage="smoke" if spec.stage == "smoke" else "tuning",
                trial_cap=get_trial_cap(search_config, search_space_name),
                time_cap_gpu_hours=get_time_cap_gpu_hours(
                    search_config,
                    search_space_name,
                ),
                allow_over_cap=args.allow_over_cap,
                fixed_overrides=shared_fixed_command_overrides(search_config),
                hash_keys=hash_keys,
            )
            for trial_overrides in trials:
                merged_overrides = merge_trial_overrides(
                    base_args={**spec.args, "method": spec.method},
                    user_overrides=overrides,
                    trial_overrides=trial_overrides,
                    hash_keys=hash_keys,
                )
                command = build_experiment_command(
                    spec,
                    repo_root=REPO_ROOT,
                    overrides=merged_overrides,
                    use_wandb=args.use_wandb,
                    wandb_entity=args.wandb_entity,
                    wandb_project=args.wandb_project,
                    wandb_group=args.wandb_group,
                    wandb_tags=args.wandb_tags,
                    wandb_mode=args.wandb_mode,
                    wandb_log_model=args.wandb_log_model,
                    python_executable=args.python,
                )
                print(format_command(command))
            return 0

        config_hash_keys = None
        if spec.stage == "final":
            search_config = load_hpo_config(args.search_config)
            search_space_name = args.search_space or default_search_space_name(spec.method)
            config_hash_keys = get_config_hash_keys(search_config, search_space_name)
        validate_direct_run_overrides(spec, overrides)
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
            config_hash_keys=config_hash_keys,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
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
