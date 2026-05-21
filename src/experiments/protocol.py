from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.hpo import (
    get_config_hash_keys,
    get_trial_cap,
    shared_fixed_command_overrides,
)
from src.experiments.registry import ExperimentRegistry, REPO_ROOT


@dataclass(frozen=True)
class ExpectedMethod:
    method_id: str
    search_space: str
    trial_cap: int
    catalog_stage: str = "template"
    search_space_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProtocolValidationReport:
    errors: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


EXPECTED_METHODS = (
    ExpectedMethod("tfidf-logreg", "tfidf_logreg", 12, catalog_stage="tuning"),
    ExpectedMethod("bilstm", "bilstm", 8, catalog_stage="tuning"),
    ExpectedMethod("random-init-distilbert", "random_init_distilbert", 4),
    ExpectedMethod("frozen-backbone", "frozen_backbone", 6),
    ExpectedMethod("partial-ft", "partial_ft", 6),
    ExpectedMethod("full-ft", "full_ft", 6, catalog_stage="tuning"),
    ExpectedMethod("lora", "lora", 6),
    ExpectedMethod("lp-ft", "lp_ft", 4, catalog_stage="tuning"),
    ExpectedMethod("efficient-head-ft", "efficient_head_ft", 4),
)

REQUIRED_SHARED_FIXED = {
    "selection_metric": "eval_f1_macro",
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.001,
    "mixed_precision": "none",
    "gradient_checkpointing": False,
    "class_weighting": "none",
    "seeds_search": [42],
    "seeds_confirm": [42, 43],
    "seeds_final": [42, 43, 44],
}

REQUIRED_TRANSFORMER_DEFAULTS = {
    "max_length": 128,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1_macro",
    "class_weighting": "none",
    "mixed_precision": "none",
    "gradient_checkpointing": False,
}

REQUIRED_SEARCH_SPACE_KEYS = {
    "tfidf_logreg": {"ngram_range", "C", "min_df"},
    "tfidf_lr": {"ngram_range", "C", "min_df"},
    "bilstm": {"hidden_size", "dropout", "learning_rate"},
    "random_init_distilbert": {"learning_rate", "num_train_epochs"},
    "frozen_backbone": {"head_learning_rate", "num_train_epochs"},
    "partial_ft": {"top_k_unfrozen_layers", "learning_rate"},
    "full_ft": {"learning_rate"},
    "lora": {
        "target_modules",
        "modules_to_save",
        "lora_r",
        "lora_alpha_rule",
        "lora_dropout",
        "learning_rate",
    },
    "lp_ft": {
        "stage1_head_learning_rate",
        "stage1_epochs",
        "stage2_learning_rate",
        "stage2_epochs",
    },
    "efficient_head_ft": {
        "stage1_learning_rate",
        "stage1_target_modules",
        "stage1_modules_to_save",
        "stage1_lora_r",
        "stage1_lora_alpha_rule",
        "stage1_lora_dropout",
        "stage2_learning_rate",
        "stage2_epochs",
    },
}


def validate_experiment_protocol(
    registry: ExperimentRegistry,
    hpo_config: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> ProtocolValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    _validate_registry_defaults(registry, errors)
    _validate_shared_fixed(hpo_config, errors)
    _validate_family_defaults(registry, hpo_config, errors)
    _validate_expected_methods(registry, hpo_config, errors)
    _validate_search_spaces(hpo_config, errors)
    _validate_experiment_entries(registry, repo_root, errors, warnings)

    return ProtocolValidationReport(errors=errors, warnings=warnings)


def format_protocol_report(report: ProtocolValidationReport) -> str:
    lines = ["Protocol validation: " + ("PASS" if report.is_valid else "FAIL")]
    if report.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in report.errors)
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report.warnings)
    if len(lines) == 1:
        lines.append("No errors or warnings.")
    return "\n".join(lines)


def _validate_registry_defaults(
    registry: ExperimentRegistry,
    errors: list[str],
) -> None:
    defaults = registry.defaults
    _expect_equal(
        defaults,
        "dataset_name",
        "Hate-speech-CNERG/hatexplain",
        errors,
        "defaults",
    )
    _expect_equal(defaults, "selection_metric", "f1_macro", errors, "defaults")
    _expect_equal(defaults, "test_policy", "final_only", errors, "defaults")
    _expect_equal(defaults, "final_seeds", [42, 43, 44], errors, "defaults")


def _validate_shared_fixed(hpo_config: dict[str, Any], errors: list[str]) -> None:
    shared = hpo_config.get("shared_fixed") or {}
    for key, expected in REQUIRED_SHARED_FIXED.items():
        _expect_equal(shared, key, expected, errors, "search_spaces.shared_fixed")


def _validate_family_defaults(
    registry: ExperimentRegistry,
    hpo_config: dict[str, Any],
    errors: list[str],
) -> None:
    transformer_defaults = registry.family_command_defaults.get("transformer", {})
    for key, expected in REQUIRED_TRANSFORMER_DEFAULTS.items():
        _expect_equal(
            transformer_defaults,
            key,
            expected,
            errors,
            "family_command_defaults.transformer",
        )

    shared_command_defaults = shared_fixed_command_overrides(hpo_config)
    for key, expected in shared_command_defaults.items():
        if key in transformer_defaults:
            _expect_equal(
                transformer_defaults,
                key,
                expected,
                errors,
                "family_command_defaults.transformer",
            )


def _validate_expected_methods(
    registry: ExperimentRegistry,
    hpo_config: dict[str, Any],
    errors: list[str],
) -> None:
    specs_by_method: dict[str, list[Any]] = {}
    for spec in registry.experiments:
        specs_by_method.setdefault(spec.method, []).append(spec)
    for expected_method in EXPECTED_METHODS:
        method_specs = specs_by_method.get(expected_method.method_id, [])
        if not method_specs:
            errors.append(
                f"Missing catalog experiment for method '{expected_method.method_id}'."
            )
        elif not any(spec.stage == expected_method.catalog_stage for spec in method_specs):
            errors.append(
                f"Method '{expected_method.method_id}' must have a catalog "
                f"{expected_method.catalog_stage} experiment."
            )
        _validate_trial_cap(hpo_config, expected_method.search_space, expected_method, errors)
        for alias in expected_method.search_space_aliases:
            _validate_trial_cap(hpo_config, alias, expected_method, errors)


def _validate_trial_cap(
    hpo_config: dict[str, Any],
    search_space: str,
    expected_method: ExpectedMethod,
    errors: list[str],
) -> None:
    actual_cap = get_trial_cap(hpo_config, search_space)
    if actual_cap != expected_method.trial_cap:
        errors.append(
            f"trial_caps.{search_space} must be {expected_method.trial_cap}; "
            f"found {actual_cap!r}."
        )


def _validate_search_spaces(hpo_config: dict[str, Any], errors: list[str]) -> None:
    spaces = hpo_config.get("search_spaces") or {}
    hash_keys_by_space = hpo_config.get("config_hash_keys") or {}
    time_caps = hpo_config.get("time_caps_gpu_hours") or {}
    for name, cap in time_caps.items():
        if cap is None:
            continue
        try:
            cap_value = float(cap)
        except (TypeError, ValueError):
            errors.append(f"time_caps_gpu_hours.{name} must be numeric; found {cap!r}.")
            continue
        if cap_value <= 0:
            errors.append(f"time_caps_gpu_hours.{name} must be positive; found {cap!r}.")

    for name, required_keys in REQUIRED_SEARCH_SPACE_KEYS.items():
        if name not in spaces:
            errors.append(f"Missing search space '{name}'.")
            continue
        missing_keys = sorted(required_keys - set(spaces[name]))
        if missing_keys:
            errors.append(
                f"search_spaces.{name} is missing keys: {', '.join(missing_keys)}."
            )
        try:
            hash_keys = get_config_hash_keys(hpo_config, name)
        except ValueError as exc:
            errors.append(str(exc))
            hash_keys = []
        if not hash_keys:
            errors.append(f"config_hash_keys.{name} must list effective config fields.")

    for name, keys in hash_keys_by_space.items():
        if name not in spaces:
            errors.append(f"config_hash_keys.{name} has no matching search space.")
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            errors.append(f"config_hash_keys.{name} must be a list of strings.")

    lora_space = spaces.get("lora") or {}
    _expect_equal(
        lora_space,
        "modules_to_save",
        [["pre_classifier", "classifier"]],
        errors,
        "search_spaces.lora",
    )
    _expect_contains(
        lora_space,
        "target_modules",
        ["q_lin", "v_lin"],
        errors,
        "search_spaces.lora",
    )
    _expect_contains(
        lora_space,
        "target_modules",
        ["q_lin", "k_lin", "v_lin", "out_lin"],
        errors,
        "search_spaces.lora",
    )

    efficient_head_space = spaces.get("efficient_head_ft") or {}
    _expect_equal(
        efficient_head_space,
        "stage1_modules_to_save",
        [["pre_classifier", "classifier"]],
        errors,
        "search_spaces.efficient_head_ft",
    )
    _expect_equal(
        efficient_head_space,
        "stage1_lora_alpha_rule",
        "alpha = 2 * r",
        errors,
        "search_spaces.efficient_head_ft",
    )


def _validate_experiment_entries(
    registry: ExperimentRegistry,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    seen_ids: set[str] = set()
    for spec in registry.experiments:
        if spec.experiment_id in seen_ids:
            errors.append(f"Duplicate experiment id: {spec.experiment_id}.")
        seen_ids.add(spec.experiment_id)

        if spec.is_ready and not spec.script_exists(repo_root):
            errors.append(
                f"Ready experiment '{spec.experiment_id}' points to missing script: "
                f"{spec.script}."
            )
        if spec.status not in {"ready", "planned"}:
            errors.append(
                f"Experiment '{spec.experiment_id}' has unsupported status "
                f"{spec.status!r}; use ready or planned."
            )
        if spec.args.get("run_test") and spec.stage != "final":
            errors.append(
                f"Experiment '{spec.experiment_id}' sets run_test outside final stage."
            )
        if spec.stage in {"tuning", "final"}:
            for cap_key in ("max_train_samples", "max_eval_samples", "max_test_samples"):
                if spec.args.get(cap_key) is not None:
                    errors.append(
                        f"Experiment '{spec.experiment_id}' is a {spec.stage} entry "
                        f"but sets {cap_key}; use smoke/quick for capped setup runs."
                    )
        if spec.stage == "final" and not spec.args.get("run_test"):
            errors.append(
                f"Final experiment '{spec.experiment_id}' does not enable run_test."
            )


def _expect_equal(
    values: dict[str, Any],
    key: str,
    expected: Any,
    errors: list[str],
    context: str,
) -> None:
    actual = values.get(key)
    if actual != expected:
        errors.append(f"{context}.{key} must be {expected!r}; found {actual!r}.")


def _expect_contains(
    values: dict[str, Any],
    key: str,
    expected_item: Any,
    errors: list[str],
    context: str,
) -> None:
    actual = values.get(key)
    if not isinstance(actual, list) or expected_item not in actual:
        errors.append(f"{context}.{key} must include {expected_item!r}; found {actual!r}.")
