from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.hpo import (
    add_config_hash_to_generated_identity,
    build_config_hash,
    build_config_hash_payload,
    validate_hash_safe_user_overrides,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "experiments.json"
METADATA_DEFAULT_KEYS = {
    "final_seeds",
    "selection_metric",
    "test_policy",
    "wandb_project",
}
STAGE_TAGS = {"smoke", "quick", "tuning", "confirm", "final"}
LOCAL_MODEL_ARTIFACT_ONLY_METHODS = {"bilstm", "tfidf-logreg"}
REMOVED_OVERRIDE_KEYS = {
    "full_determinism": (
        "The full_determinism switch was removed from the shared experiment "
        "interface. Use the normal seed fields for reproducibility control."
    )
}
DIRECT_RUN_PROTECTED_OVERRIDE_KEYS = {
    "search_stage",
    "trial_id",
    "config_hash",
    "hpo_seed",
    "hpo_trial_cap",
    "hpo_time_cap_gpu_hours",
    "run_test",
}
DIRECT_FINAL_PROTECTED_OVERRIDE_KEYS = {
    "seed",
    "data_fraction",
    "data_fraction_seed",
    "max_train_samples",
    "max_eval_samples",
    "max_test_samples",
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    status: str
    method: str
    family: str
    stage: str
    script: str
    description: str
    tags: tuple[str, ...]
    args: dict[str, Any]
    defaults: dict[str, Any]
    command_defaults: dict[str, Any]
    family_command_defaults: dict[str, Any]

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def script_path(self, repo_root: Path = REPO_ROOT) -> Path:
        return repo_root / self.script

    def script_exists(self, repo_root: Path = REPO_ROOT) -> bool:
        return self.script_path(repo_root).is_file()


class ExperimentRegistry:
    def __init__(
        self,
        experiments: list[ExperimentSpec],
        defaults: dict[str, Any],
        command_defaults: dict[str, Any],
        family_command_defaults: dict[str, dict[str, Any]],
    ) -> None:
        self.experiments = experiments
        self.defaults = defaults
        self.command_defaults = command_defaults
        self.family_command_defaults = family_command_defaults
        self._by_id = {experiment.experiment_id: experiment for experiment in experiments}

    def get(self, experiment_id: str) -> ExperimentSpec:
        try:
            return self._by_id[experiment_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._by_id))
            raise KeyError(
                f"Unknown experiment '{experiment_id}'. Available: {available}"
            ) from exc

    def ready_experiments(self) -> list[ExperimentSpec]:
        return [experiment for experiment in self.experiments if experiment.is_ready]


def load_experiment_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ExperimentRegistry:
    registry_path = Path(path)
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    defaults = dict(data.get("defaults") or {})
    raw_command_defaults = data.get("command_defaults")
    if raw_command_defaults is None:
        command_defaults = {
            key: value
            for key, value in defaults.items()
            if key not in METADATA_DEFAULT_KEYS
        }
    else:
        command_defaults = dict(raw_command_defaults)
    raw_family_command_defaults = {
        str(family): dict(values or {})
        for family, values in (data.get("family_command_defaults") or {}).items()
    }
    family_command_defaults = {
        family: _resolve_family_command_defaults(raw_family_command_defaults, family)
        for family in raw_family_command_defaults
    }
    experiments = []
    for experiment_id, raw in (data.get("experiments") or {}).items():
        family = str(raw.get("family", ""))
        per_family_defaults = family_command_defaults.get(family, {})
        args = {**command_defaults, **per_family_defaults, **dict(raw.get("args") or {})}
        experiments.append(
            ExperimentSpec(
                experiment_id=experiment_id,
                status=str(raw.get("status", "planned")),
                method=str(raw["method"]),
                family=family,
                stage=str(raw.get("stage", "")),
                script=str(raw["script"]),
                description=str(raw.get("description", "")),
                tags=tuple(str(tag) for tag in raw.get("tags", ())),
                args=args,
                defaults=defaults,
                command_defaults={**command_defaults, **per_family_defaults},
                family_command_defaults=per_family_defaults,
            )
        )
    return ExperimentRegistry(
        experiments,
        defaults,
        command_defaults,
        family_command_defaults,
    )


def _resolve_family_command_defaults(
    all_family_defaults: dict[str, dict[str, Any]],
    family: str,
    seen: tuple[str, ...] = (),
) -> dict[str, Any]:
    if family in seen:
        cycle = " -> ".join((*seen, family))
        raise ValueError(f"Cycle in family_command_defaults inheritance: {cycle}")
    values = dict(all_family_defaults.get(family, {}))
    parent = values.pop("inherits", None)
    if not parent:
        return values
    return {
        **_resolve_family_command_defaults(
            all_family_defaults,
            str(parent),
            (*seen, family),
        ),
        **values,
    }


def _parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    if value.startswith(("[", "{", '"')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_override_pairs(pairs: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"Override must use key=value format: {pair}")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override has an empty key: {pair}")
        if key in REMOVED_OVERRIDE_KEYS:
            raise ValueError(f"Unsupported override '{key}': {REMOVED_OVERRIDE_KEYS[key]}")
        overrides[key] = _parse_scalar(raw_value.strip())
    return overrides


def parse_override_text(text: str | None) -> dict[str, Any]:
    pairs = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pairs.append(line)
    return parse_override_pairs(pairs)


def validate_direct_run_overrides(
    spec: ExperimentSpec,
    overrides: dict[str, Any] | None,
) -> None:
    overrides = overrides or {}
    if spec.stage == "final" and overrides.get("run_test") is False:
        raise ValueError("Final-stage experiments must enable --run_test.")

    user_overrides = set(overrides)
    protected = set(DIRECT_RUN_PROTECTED_OVERRIDE_KEYS)
    if spec.stage == "final":
        protected.update(DIRECT_FINAL_PROTECTED_OVERRIDE_KEYS)
    blocked = sorted(user_overrides & protected)
    if blocked:
        raise ValueError(
            "Direct catalog runs do not allow overriding managed protocol fields: "
            f"{', '.join(blocked)}. Use catalog entries, HPO trial generation, or "
            "seed-run generation for stage/test/sample-policy changes."
        )
    validate_hash_safe_user_overrides(method=spec.method, user_overrides=overrides)


def _format_cli_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _append_cli_arg(command: list[str], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, _format_cli_value(value)])


def _effective_search_stage(spec: ExperimentSpec, args: dict[str, Any]) -> str:
    return str(args.get("search_stage") or spec.stage)


def _default_wandb_group(spec: ExperimentSpec, args: dict[str, Any]) -> str:
    base = f"{spec.method}-{_effective_search_stage(spec, args)}"
    config_hash = args.get("config_hash")
    return f"{base}-{config_hash}" if config_hash else base


def _default_wandb_tags(spec: ExperimentSpec, args: dict[str, Any]) -> str:
    effective_stage = _effective_search_stage(spec, args)
    tags: list[str] = []
    replaced_stage = False
    for tag in spec.tags:
        next_tag = effective_stage if tag in STAGE_TAGS else tag
        if tag in STAGE_TAGS:
            replaced_stage = True
        if next_tag not in tags:
            tags.append(next_tag)
    if not replaced_stage and effective_stage not in tags:
        tags.append(effective_stage)
    return ",".join(tags)


def _merge_wandb_tags(default_tags: str, extra_tags: str | None) -> str:
    tags: list[str] = []
    for raw_tags in (default_tags, extra_tags or ""):
        for raw_tag in raw_tags.split(","):
            tag = raw_tag.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return ",".join(tags)


def _with_default_config_hash(
    args: dict[str, Any],
    *,
    hash_keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if args.get("config_hash") or args.get("search_stage") != "final":
        return args
    hash_payload = build_config_hash_payload(args, hash_keys=hash_keys)
    return add_config_hash_to_generated_identity(
        {**args, "config_hash": build_config_hash(hash_payload)}
    )


def _validate_command_policy(args: dict[str, Any]) -> None:
    search_stage = args.get("search_stage")
    run_test = bool(args.get("run_test", False))
    if run_test and search_stage != "final":
        raise ValueError(
            "--run_test is only allowed for final-stage experiments. "
            f"Received search_stage={search_stage!r}."
        )
    if search_stage == "final" and not run_test:
        raise ValueError("Final-stage experiments must enable --run_test.")


def _validate_wandb_log_model_policy(spec: ExperimentSpec, wandb_log_model: str) -> None:
    if spec.method in LOCAL_MODEL_ARTIFACT_ONLY_METHODS and wandb_log_model != "false":
        raise ValueError(
            f"Experiment '{spec.experiment_id}' uses method '{spec.method}', which "
            "currently saves model artifacts locally only. Set --wandb_log_model false."
        )


def build_experiment_command(
    spec: ExperimentSpec,
    *,
    repo_root: Path = REPO_ROOT,
    overrides: dict[str, Any] | None = None,
    use_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: str | None = None,
    wandb_mode: str = "online",
    wandb_log_model: str = "false",
    python_executable: str | None = None,
    config_hash_keys: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if not spec.script_exists(repo_root):
        raise FileNotFoundError(
            f"Experiment '{spec.experiment_id}' points to missing script: {spec.script}"
        )
    validate_hash_safe_user_overrides(
        method=spec.method,
        user_overrides=overrides or {},
    )

    args = {
        "method": spec.method,
        "search_stage": spec.stage,
        "trial_id": spec.experiment_id,
        **spec.args,
        **(overrides or {}),
    }
    args = _with_default_config_hash(args, hash_keys=config_hash_keys)
    _validate_command_policy(args)
    _validate_wandb_log_model_policy(spec, wandb_log_model)
    command = [python_executable or sys.executable, spec.script]
    for key, value in args.items():
        _append_cli_arg(command, key, value)

    if use_wandb:
        command.append("--use_wandb")
        _append_cli_arg(command, "wandb_entity", wandb_entity)
        _append_cli_arg(command, "wandb_project", wandb_project)
        _append_cli_arg(
            command,
            "wandb_group",
            wandb_group or _default_wandb_group(spec, args),
        )
        combined_tags = _merge_wandb_tags(_default_wandb_tags(spec, args), wandb_tags)
        _append_cli_arg(command, "wandb_tags", combined_tags)
        _append_cli_arg(command, "wandb_mode", wandb_mode)
        _append_cli_arg(command, "wandb_log_model", wandb_log_model)

    return command


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)
