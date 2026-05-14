from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "experiments.json"
METADATA_DEFAULT_KEYS = {
    "final_seeds",
    "selection_metric",
    "test_policy",
    "wandb_project",
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
) -> list[str]:
    if not spec.script_exists(repo_root):
        raise FileNotFoundError(
            f"Experiment '{spec.experiment_id}' points to missing script: {spec.script}"
        )

    args = {
        "method": spec.method,
        "search_stage": spec.stage,
        "trial_id": spec.experiment_id,
        **spec.args,
        **(overrides or {}),
    }
    command = [python_executable or sys.executable, spec.script]
    for key, value in args.items():
        _append_cli_arg(command, key, value)

    if use_wandb:
        command.append("--use_wandb")
        _append_cli_arg(command, "wandb_entity", wandb_entity)
        _append_cli_arg(command, "wandb_project", wandb_project)
        _append_cli_arg(command, "wandb_group", wandb_group or spec.method)
        combined_tags = wandb_tags or ",".join(spec.tags)
        _append_cli_arg(command, "wandb_tags", combined_tags)
        _append_cli_arg(command, "wandb_mode", wandb_mode)
        _append_cli_arg(command, "wandb_log_model", wandb_log_model)

    return command


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)
