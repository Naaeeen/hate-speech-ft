"""Manual entrypoint for one DistilBERT Full FT run.

The method owns its config and metadata builder. `run_single_stage_transformer`
owns the shared training/evaluation/output routine. Edit `manual_config.py`,
not this file, for research settings.
"""

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.distilbert_full.config import build_experiment_config
from src.methods.distilbert_full.manual_config import CONFIG as MANUAL_CONFIG
from src.methods.transformer_runner import run_single_stage_transformer


def main():
    """Run one full fine-tuning experiment from `manual_config.py`."""

    args = SimpleNamespace(**MANUAL_CONFIG)
    run_single_stage_transformer(
        args,
        build_experiment_config=build_experiment_config,
        learning_rate_attr="learning_rate",
    )


if __name__ == "__main__":
    main()
