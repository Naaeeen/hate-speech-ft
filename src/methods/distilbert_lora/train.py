"""Manual entrypoint for one DistilBERT LoRA run."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.distilbert_lora.config import build_experiment_config
from src.methods.distilbert_lora.manual_config import CONFIG as MANUAL_CONFIG
from src.methods.distilbert_lora.training import apply_lora_to_context
from src.methods.transformer_runner import run_single_stage_transformer


def main() -> None:
    """Run one LoRA experiment from `manual_config.py`."""

    args = SimpleNamespace(**MANUAL_CONFIG)
    run_single_stage_transformer(
        args,
        build_experiment_config=build_experiment_config,
        learning_rate_attr="learning_rate",
        prepare_context=apply_lora_to_context,
        params_label="Trainable LoRA params",
        train_message="Starting LoRA DistilBERT training",
    )


if __name__ == "__main__":
    main()
