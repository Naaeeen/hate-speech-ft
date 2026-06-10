"""Shared local result-file contract for every manual run.

All methods should leave behind the same small set of evidence files:
`resolved_config.json`, `metrics.json`, `runtime.json`, `result_summary.json`,
and prediction files when test evaluation is enabled. This file also protects
old outputs from being accidentally overwritten, which matters a lot when we
run one seed at a time in Colab.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


RESULT_FILE_NAMES = {
    "result_summary.json",
    "metrics.json",
    "runtime.json",
    "resolved_config.json",
    "eval_predictions.json",
    "test_predictions.json",
    "trainer_state.json",
}
MODEL_ARTIFACT_NAMES = {
    "config.json",
    "training_args.bin",
    "model.pt",
    "finalmodel.pt",
    "model.safetensors",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "model.joblib",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "added_tokens.json",
    "tokenizer",
}
STAGE_ARTIFACT_NAMES = {
    "stage1_linear_probe",
    "stage1_lora_head",
    "stage2_full_ft",
}
RUN_ARTIFACT_NAMES = RESULT_FILE_NAMES | MODEL_ARTIFACT_NAMES | STAGE_ARTIFACT_NAMES
CHECKPOINT_PREFIX = "checkpoint-"


def _json_safe(value: Any) -> Any:
    """Convert common ML objects into plain JSON-friendly values.

    Tensors, numpy arrays, dataclasses, and Paths are convenient while training,
    but annoying in result files. This normalizes them before we write JSON.
    """

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one pretty JSON file and return its path.

    A lot of the training code builds dictionaries with Paths, numpy scalars, or
    tensors inside. This helper quietly normalizes those first, so teammates can
    open the result files in Colab without seeing Python-only object reprs.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def validate_sample_selection_args(args: Any) -> None:
    """Check the manual sample-limit knobs before we touch the dataset.

    These options are mainly for tiny smoke runs. If they are invalid, it is
    nicer to fail right away than after downloading/tokenizing the dataset.
    """

    data_fraction = getattr(args, "data_fraction", None)
    if data_fraction is not None and not 0 < data_fraction <= 1:
        raise ValueError("data_fraction must be in the interval (0, 1].")

    for option_name in ("max_train_samples", "max_eval_samples", "max_test_samples"):
        value = getattr(args, option_name, None)
        if value is not None and value < 1:
            raise ValueError(f"{option_name} must be >= 1 when provided.")


def find_existing_run_artifacts(output_dir: str | Path) -> list[Path]:
    """Return run-owned files that would make this output folder unsafe to reuse."""

    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    return [
        child
        for child in output_path.iterdir()
        if child.name in RUN_ARTIFACT_NAMES or child.name.startswith(CHECKPOINT_PREFIX)
    ]


def clear_existing_run_artifacts(output_dir: str | Path) -> list[Path]:
    """Remove only files/folders that this training code knows it owns."""

    removed = []
    for artifact in find_existing_run_artifacts(output_dir):
        if artifact.is_dir():
            shutil.rmtree(artifact)
        else:
            artifact.unlink()
        removed.append(artifact)
    return removed


def prepare_output_dir_for_run(output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Create or clean the output directory for exactly one manual run."""

    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_path}")

    artifacts = find_existing_run_artifacts(output_path)
    if artifacts and not overwrite:
        preview = ", ".join(path.name for path in artifacts[:5])
        raise ValueError(
            f"Output directory '{output_dir}' already contains run artifacts "
            f"({preview}). Use a unique output_dir for a new run, or set "
            "overwrite_output_dir=True only when intentionally replacing artifacts."
        )
    if overwrite:
        clear_existing_run_artifacts(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_resolved_config(output_dir: str | Path, config: dict[str, Any]) -> Path:
    """Save the exact resolved config for this single run."""

    return write_json(Path(output_dir) / "resolved_config.json", config)


def write_result_files(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    eval_metrics: dict[str, Any],
    runtime_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
    model_selection: dict[str, Any] | None = None,
    prediction_paths: dict[str, str | Path] | None = None,
    artifact_paths: dict[str, str | Path] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the core result files used for manual aggregation later.

    We only write success outputs here. There is intentionally no failure
    summary machinery in the manual workflow: if a run fails, we fix/rerun it
    rather than treating the failed directory as research evidence.
    """

    output_path = Path(output_dir)
    metrics_payload = {
        "eval": eval_metrics,
        "test": test_metrics,
    }
    if extra_metrics:
        metrics_payload.update(extra_metrics)
    summary_payload = {
        "config": config,
        "metrics": metrics_payload,
        "runtime": runtime_metrics,
        "model_selection": model_selection or {},
        "artifacts": {
            "predictions": prediction_paths or {},
            "model": artifact_paths or {},
        },
    }

    return {
        "metrics": write_json(output_path / "metrics.json", metrics_payload),
        "runtime": write_json(output_path / "runtime.json", runtime_metrics),
        "summary": write_json(output_path / "result_summary.json", summary_payload),
    }
