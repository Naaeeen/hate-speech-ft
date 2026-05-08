from __future__ import annotations

import json
import random
import hashlib
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_SPACE_PATH = REPO_ROOT / "configs" / "search_spaces.json"


def load_hpo_config(path: str | Path = DEFAULT_SEARCH_SPACE_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_search_space_name(method: str) -> str:
    return method.replace("-", "_")


def _choose_value(rng: random.Random, spec: Any) -> Any:
    if isinstance(spec, dict) and "values" in spec:
        values = spec["values"]
    elif isinstance(spec, list):
        values = spec
    else:
        return spec
    if not values:
        raise ValueError("Search-space value lists must not be empty.")
    return rng.choice(values)


def build_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def sample_search_space(
    search_space: dict[str, Any],
    *,
    seed: int,
    trial_index: int,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{trial_index}")
    overrides: dict[str, Any] = {}
    deferred_rules: dict[str, Any] = {}

    for key, spec in search_space.items():
        if key.endswith("_rule"):
            deferred_rules[key] = spec
            continue
        overrides[key] = _choose_value(rng, spec)

    if deferred_rules.get("lora_alpha_rule") == "alpha = 2 * r" and "lora_r" in overrides:
        overrides["lora_alpha"] = 2 * int(overrides["lora_r"])

    return overrides


def enumerate_search_space(search_space: dict[str, Any]) -> list[dict[str, Any]]:
    keys = []
    values_by_key = []
    deferred_rules: dict[str, Any] = {}

    for key, spec in search_space.items():
        if key.endswith("_rule"):
            deferred_rules[key] = spec
            continue
        keys.append(key)
        if isinstance(spec, dict) and "values" in spec:
            values = spec["values"]
        elif isinstance(spec, list):
            values = spec
        else:
            values = [spec]
        if not values:
            raise ValueError("Search-space value lists must not be empty.")
        values_by_key.append(values)

    combinations = []
    for values in product(*values_by_key):
        overrides = dict(zip(keys, values))
        if (
            deferred_rules.get("lora_alpha_rule") == "alpha = 2 * r"
            and "lora_r" in overrides
        ):
            overrides["lora_alpha"] = 2 * int(overrides["lora_r"])
        combinations.append(overrides)
    return combinations


def build_trial_overrides(
    *,
    base_experiment_id: str,
    method: str,
    search_space: dict[str, Any],
    n_trials: int,
    hpo_seed: int,
    output_root: str,
    search_stage: str = "tuning",
) -> list[dict[str, Any]]:
    trials = []
    method_part = default_search_space_name(method)
    combinations = enumerate_search_space(search_space)
    rng = random.Random(hpo_seed)
    rng.shuffle(combinations)
    for trial_index in range(n_trials):
        trial_id = f"{base_experiment_id}__{method_part}__trial{trial_index + 1:03d}"
        if trial_index < len(combinations):
            overrides = dict(combinations[trial_index])
        else:
            overrides = sample_search_space(
                search_space,
                seed=hpo_seed,
                trial_index=trial_index,
            )
        overrides.update(
            {
                "search_stage": search_stage,
                "trial_id": trial_id,
                "hpo_seed": hpo_seed,
                "config_hash": build_config_hash(overrides),
                "output_dir": f"{output_root.rstrip('/')}/{trial_id}",
            }
        )
        trials.append(overrides)
    return trials


def get_search_space(config: dict[str, Any], name: str) -> dict[str, Any]:
    spaces = config.get("search_spaces") or {}
    try:
        return dict(spaces[name])
    except KeyError as exc:
        available = ", ".join(sorted(spaces))
        raise KeyError(f"Unknown search space '{name}'. Available: {available}") from exc
