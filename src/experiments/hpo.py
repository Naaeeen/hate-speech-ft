from __future__ import annotations

import json
import random
import hashlib
import re
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_SPACE_PATH = REPO_ROOT / "configs" / "search_spaces.json"
TRIAL_METADATA_KEYS = {
    "search_stage",
    "trial_id",
    "hpo_seed",
    "hpo_trial_cap",
    "hpo_time_cap_gpu_hours",
    "seed",
    "run_test",
    "output_dir",
    "config_hash",
    "overwrite_output_dir",
}
PROTECTED_USER_OVERRIDE_KEYS = {
    "search_stage",
    "trial_id",
    "hpo_seed",
    "hpo_trial_cap",
    "hpo_time_cap_gpu_hours",
    "seed",
    "output_dir",
    "config_hash",
}
SEED_RUN_PROTECTED_USER_OVERRIDE_KEYS = PROTECTED_USER_OVERRIDE_KEYS | {
    "run_test",
    "data_fraction",
    "data_fraction_seed",
    "max_train_samples",
    "max_eval_samples",
    "max_test_samples",
}
CONFIG_HASH_SUFFIX_PATTERN = re.compile(r"__[0-9a-f]{12}$")


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


def _canonical_ngram_range(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text) if text.startswith("[") else text.split(",")
        except json.JSONDecodeError:
            return value
    elif isinstance(value, (list, tuple)):
        parsed = value
    else:
        return value

    if len(parsed) != 2:
        return value
    try:
        return [int(parsed[0]), int(parsed[1])]
    except (TypeError, ValueError):
        return value


def _canonical_hash_value(key: str, value: Any) -> Any:
    if key == "ngram_range":
        return _canonical_ngram_range(value)
    if key == "C":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def build_config_hash_payload(
    payload: dict[str, Any],
    *,
    hash_keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    keys = hash_keys if hash_keys is not None else payload.keys()
    return {
        key: _canonical_hash_value(key, payload[key])
        for key in keys
        if key in payload
        and key not in TRIAL_METADATA_KEYS
        and payload[key] is not None
    }


def apply_deferred_rules(
    overrides: dict[str, Any],
    deferred_rules: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(overrides)
    if deferred_rules.get("lora_alpha_rule") == "alpha = 2 * r" and "lora_r" in resolved:
        resolved["lora_alpha"] = 2 * int(resolved["lora_r"])
    if (
        deferred_rules.get("stage1_lora_alpha_rule") == "alpha = 2 * r"
        and "stage1_lora_r" in resolved
    ):
        resolved["stage1_lora_alpha"] = 2 * int(resolved["stage1_lora_r"])
    return resolved


def merge_trial_overrides(
    *,
    base_args: dict[str, Any],
    user_overrides: dict[str, Any],
    trial_overrides: dict[str, Any],
    protected_user_override_keys: set[str] | None = None,
    hash_keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    protected_keys = protected_user_override_keys or PROTECTED_USER_OVERRIDE_KEYS
    protected_user_keys = sorted(protected_keys & set(user_overrides))
    if protected_user_keys:
        blocked = ", ".join(protected_user_keys)
        raise ValueError(
            f"Run identity/protocol fields are managed by the launcher: {blocked}. "
            "Use --trial_output_root, --seed_output_root, --hpo_seed, or the "
            "experiment catalog instead."
        )

    merged = {**trial_overrides, **user_overrides}
    hash_payload = build_config_hash_payload({**base_args, **merged}, hash_keys=hash_keys)
    merged["config_hash"] = build_config_hash(hash_payload)
    return add_config_hash_to_generated_identity(merged)


def add_config_hash_to_generated_identity(overrides: dict[str, Any]) -> dict[str, Any]:
    config_hash = overrides.get("config_hash")
    trial_id = overrides.get("trial_id")
    if not config_hash or not trial_id:
        return overrides

    original_trial_id = str(trial_id)
    trial_id_text = CONFIG_HASH_SUFFIX_PATTERN.sub("", original_trial_id)
    config_hash_text = str(config_hash)
    output_dir = overrides.get("output_dir")
    if original_trial_id.endswith(f"__{config_hash_text}") and (
        not output_dir or config_hash_text in str(output_dir)
    ):
        return overrides

    updated_trial_id = f"{trial_id_text}__{config_hash_text}"
    updated = dict(overrides)
    updated["trial_id"] = updated_trial_id

    output_dir = updated.get("output_dir")
    if output_dir:
        output_text = str(output_dir).rstrip("/\\")
        output_without_hash = CONFIG_HASH_SUFFIX_PATTERN.sub("", output_text)
        if output_without_hash.endswith(trial_id_text):
            updated["output_dir"] = (
                f"{output_without_hash[:-len(trial_id_text)]}{updated_trial_id}"
            )
        elif config_hash_text not in output_text:
            updated["output_dir"] = f"{output_text}__{config_hash_text}"

    return updated


def get_trial_cap(config: dict[str, Any], search_space_name: str) -> int | None:
    caps = config.get("trial_caps") or {}
    cap = caps.get(search_space_name)
    return int(cap) if cap is not None else None


def get_time_cap_gpu_hours(config: dict[str, Any], search_space_name: str) -> float | None:
    caps = config.get("time_caps_gpu_hours") or {}
    cap = caps.get(search_space_name)
    return float(cap) if cap is not None else None


def get_config_hash_keys(
    config: dict[str, Any],
    search_space_name: str,
) -> list[str] | None:
    hash_keys_by_space = config.get("config_hash_keys") or {}
    keys = hash_keys_by_space.get(search_space_name)
    if keys is None:
        return None
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError(
            f"config_hash_keys.{search_space_name} must be a list of strings."
        )
    return list(keys)


def get_seed_policy(config: dict[str, Any], stage: str) -> list[int]:
    policy_key_by_stage = {
        "confirm": "seeds_confirm",
        "final": "seeds_final",
    }
    try:
        policy_key = policy_key_by_stage[stage]
    except KeyError as exc:
        valid = ", ".join(sorted(policy_key_by_stage))
        raise ValueError(f"Unsupported seed-run stage '{stage}'. Valid: {valid}.") from exc

    seeds = (config.get("shared_fixed") or {}).get(policy_key)
    if not seeds:
        raise ValueError(f"Missing shared_fixed.{policy_key} in HPO config.")
    return [int(seed) for seed in seeds]


def validate_seed_run_base_stage(stage: str) -> None:
    if stage != "tuning":
        raise ValueError(
            "Seed-run generation must start from a tuning experiment so smoke "
            "sample caps and final-only test flags are not inherited."
        )


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

    return apply_deferred_rules(overrides, deferred_rules)


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
        combinations.append(apply_deferred_rules(overrides, deferred_rules))
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
    time_cap_gpu_hours: float | None = None,
    allow_over_cap: bool = False,
    fixed_overrides: dict[str, Any] | None = None,
    hash_keys: list[str] | tuple[str, ...] | None = None,
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
        trial_id = (
            f"{base_experiment_id}__{method_part}__hpo{hpo_seed}"
            f"__trial{trial_index + 1:03d}"
        )
        if trial_index < len(combinations):
            sampled_overrides = dict(combinations[trial_index])
        else:
            sampled_overrides = sample_search_space(
                search_space,
                seed=hpo_seed,
                trial_index=trial_index,
            )
        overrides = {**fixed_overrides, **sampled_overrides}
        config_hash = build_config_hash(
            build_config_hash_payload(overrides, hash_keys=hash_keys)
        )
        overrides.update(
            {
                "search_stage": search_stage,
                "trial_id": trial_id,
                "hpo_seed": hpo_seed,
                "hpo_trial_cap": trial_cap,
                "hpo_time_cap_gpu_hours": time_cap_gpu_hours,
                "config_hash": config_hash,
                "output_dir": f"{output_root.rstrip('/')}/{trial_id}",
            }
        )
        trials.append(add_config_hash_to_generated_identity(overrides))
    return trials


def build_seed_run_overrides(
    *,
    base_experiment_id: str,
    method: str,
    seeds: list[int],
    output_root: str,
    search_stage: str,
    fixed_overrides: dict[str, Any] | None = None,
    trial_cap: int | None = None,
    time_cap_gpu_hours: float | None = None,
) -> list[dict[str, Any]]:
    if search_stage not in {"confirm", "final"}:
        raise ValueError("--suggest_seed_runs supports only confirm or final.")

    method_part = default_search_space_name(method)
    fixed_overrides = dict(fixed_overrides or {})
    seed_runs = []
    for seed in seeds:
        trial_id = f"{base_experiment_id}__{method_part}__{search_stage}_seed{seed}"
        overrides = {
            **fixed_overrides,
            "search_stage": search_stage,
            "seed": seed,
            "data_fraction": 1.0,
            "max_train_samples": None,
            "max_eval_samples": None,
            "max_test_samples": None,
            "trial_id": trial_id,
            "hpo_trial_cap": trial_cap,
            "hpo_time_cap_gpu_hours": time_cap_gpu_hours,
            "output_dir": f"{output_root.rstrip('/')}/{trial_id}",
        }
        if search_stage == "final":
            overrides["run_test"] = True
        seed_runs.append(overrides)
    return seed_runs


def get_search_space(config: dict[str, Any], name: str) -> dict[str, Any]:
    spaces = config.get("search_spaces") or {}
    try:
        return dict(spaces[name])
    except KeyError as exc:
        available = ", ".join(sorted(spaces))
        raise KeyError(f"Unknown search space '{name}'. Available: {available}") from exc
