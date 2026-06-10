"""Manual entrypoint for LP-FT.

LP-FT means "linear probe first, then full fine-tune." Stage 1 freezes the
backbone and trains the head. Stage 2 unfreezes the same model and keeps going.
Everything is still one manual run from one `manual_config.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.distilbert_lp_ft.config import (
    build_experiment_config,
)
from src.methods.distilbert_lp_ft.manual_config import CONFIG as MANUAL_CONFIG
from src.methods.distilbert_lp_ft.training import (
    STAGE1_DIR_NAME,
    STAGE2_DIR_NAME,
    set_full_finetune_trainability,
    set_linear_probe_trainability,
)
from src.methods.transformer_two_stage_runner import (
    TwoStagePlan,
    run_two_stage_transformer,
)
from src.utils.run_metadata import count_model_parameters


def build_stage_plan(context, _args) -> TwoStagePlan:
    """Prepare the freeze/unfreeze plan for the shared two-stage runner."""

    # Flip trainability a few times here only to count the two stages correctly
    # and then leave the model in the stage-1 state before training starts.
    set_linear_probe_trainability(context.model)
    stage1_trainable_params, total_params = count_model_parameters(context.model)
    set_full_finetune_trainability(context.model)
    stage2_trainable_params, _ = count_model_parameters(context.model)
    set_linear_probe_trainability(context.model)

    def build_stage2_context(_stage1_model):
        """Unfreeze the stage-1 model and continue with it for stage 2."""

        # Unlike Efficient-Head, LP-FT continues from the same stage-1 model.
        # We just unfreeze everything before stage 2 starts.
        set_full_finetune_trainability(context.model)
        return context

    return TwoStagePlan(
        stage1_context=context,
        stage1_trainable_params=stage1_trainable_params,
        stage2_trainable_params=stage2_trainable_params,
        total_params=total_params,
        build_stage2_context=build_stage2_context,
    )


def main() -> None:
    """Run one LP-FT two-stage experiment from `manual_config.py`."""

    args = SimpleNamespace(**MANUAL_CONFIG)
    run_two_stage_transformer(
        args,
        build_experiment_config=build_experiment_config,
        build_stage_plan=build_stage_plan,
        stage1_dir_name=STAGE1_DIR_NAME,
        stage2_dir_name=STAGE2_DIR_NAME,
        stage1_learning_rate_attr="stage1_head_learning_rate",
        stage1_epochs_attr="stage1_epochs",
        stage2_learning_rate_attr="stage2_learning_rate",
        stage2_epochs_attr="stage2_epochs",
        train_message="Starting LP+FT training",
        stage1_message="linear probing classification head",
        stage2_message="full fine-tuning all parameters",
    )


if __name__ == "__main__":
    main()
