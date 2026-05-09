from __future__ import annotations

import json
import random
import hashlib
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_SPACE_PATH = REPO_ROOT / "configs" / "search_spaces.json"
TRIAL_METADATA_KEYS = {
    "search_stage",
    "trial_id",
    "hpo_seed",
    "output_dir",
    "config_hash",
    "overwrite_output_dir",
}
PROTECTED_USER_OVERRIDE_KEYS = {
    "search_stage",
    "trial_id",
    "hpo_seed",
    "output_dir",
    "config_hash",
}


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


def build_config_hash_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in TRIAL_METADATA_KEYS and value is not None
    }


def merge_trial_overrides(
    *,
    base_args: dict[str, Any],
    user_overrides: dict[str, Any],
    trial_overrides: dict[str, Any],
) -> dict[str, Any]:
    protected_user_keys = sorted(PROTECTED_USER_OVERRIDE_KEYS & set(user_overrides))
    if protected_user_keys:
        blocked = ", ".join(protected_user_keys)
        raise ValueError(
            f"HPO trial identity fields are managed by the launcher: {blocked}. "
            "Use --trial_output_root, --hpo_seed, or the experiment catalog instead."
        )

    merged = {**trial_overrides, **user_overrides}
    hash_payload = build_config_hash_payload({**base_args, **merged})
    merged["config_hash"] = build_config_hash(hash_payload)
    return merged


def get_trial_cap(config: dict[str, Any], search_space_name: str) -> int | None:
    caps = config.get("trial_caps") or {}
    cap = caps.get(search_space_name)
    return int(cap) if cap is not None else None


def shared_fixed_command_overrides(config: dict[str, Any]) -> dict[str, Any]:
    shared = config.get("shared_fixed") or {}
    command_keys = {
        "optim",
        "lr_scheduler_type",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "eval_strategy",
        "save_strategy",
        "load_best_model_at_end",
        "early_stopping_patience",
        "early_stopping_threshold",
        "mixed_precision",
        "gradient_checkpointing",
        "class_weighting",
    }
    return {key: shared[key] for key in command_keys if key in shared}


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
    trial_cap: int | None = None,
    allow_over_cap: bool = False,
    fixed_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if trial_cap is not None and n_trials > trial_cap and not allow_over_cap:
        raise ValueError(
            f"Requested {n_trials} trials for {method}, but the configured cap is "
            f"{trial_cap}. Pass an explicit over-cap option only for exploratory runs."
        )

    trials = []
    method_part = default_search_space_name(method)
    combinations = enumerate_search_space(search_space)
    rng = random.Random(hpo_seed)
    rng.shuffle(combinations)
    fixed_overrides = dict(fixed_overrides or {})
    for trial_index in range(n_trials):
        trial_id = f"{base_experiment_id}__{method_part}__trial{trial_index + 1:03d}"
        if trial_index < len(combinations):
            sampled_overrides = dict(combinations[trial_index])
        else:
            sampled_overrides = sample_search_space(
                search_space,
                seed=hpo_seed,
                trial_index=trial_index,
            )
        overrides = {**fixed_overrides, **sampled_overrides}
        overrides.update(
            {
                "search_stage": search_stage,
                "trial_id": trial_id,
                "hpo_seed": hpo_seed,
                "config_hash": build_config_hash(build_config_hash_payload(overrides)),
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
