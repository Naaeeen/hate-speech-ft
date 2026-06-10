"""Manual entrypoint for one Frozen DistilBERT run.

The shared runner handles the normal Transformer lifecycle. This file only adds
the frozen-backbone preparation step, where DistilBERT parameters stop training
and the classification head stays trainable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.frozen_distilbert.config import build_experiment_config
from src.methods.frozen_distilbert.manual_config import CONFIG as MANUAL_CONFIG
from src.methods.frozen_distilbert.training import set_frozen_backbone_trainability
from src.methods.transformer_runner import run_single_stage_transformer


def prepare_frozen_context(context, _args):
    """Freeze the backbone and return the updated Transformer context."""

    set_frozen_backbone_trainability(context.model)
    return context


def main() -> None:
    """Run one frozen-backbone experiment from `manual_config.py`."""

    args = SimpleNamespace(**MANUAL_CONFIG)
    run_single_stage_transformer(
        args,
        build_experiment_config=build_experiment_config,
        learning_rate_attr="head_learning_rate",
        prepare_context=prepare_frozen_context,
        params_label="Trainable head params",
        train_message="Starting frozen-backbone DistilBERT training",
    )


if __name__ == "__main__":
    main()
