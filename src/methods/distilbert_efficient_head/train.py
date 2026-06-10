"""Manual entrypoint for Efficient-Head FT.

This method is a little unusual, so here is the short version: stage 1 trains
LoRA adapters plus the classification head; stage 2 throws away the adapters
and freshens the DistilBERT backbone, but copies the trained head forward. The
manual config is the only place teammates should edit research settings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.distilbert_efficient_head.config import (
    STAGE1_DIR_NAME,
    STAGE2_DIR_NAME,
    build_experiment_config,
)
from src.methods.distilbert_efficient_head.manual_config import CONFIG as MANUAL_CONFIG
from src.methods.distilbert_efficient_head.training import (
    apply_stage1_lora_to_context,
    build_stage2_context,
)
from src.methods.transformer_two_stage_runner import (
    TwoStagePlan,
    run_two_stage_transformer,
)
from src.utils.run_metadata import count_model_parameters


def build_stage_plan(base_context, args) -> TwoStagePlan:
    """Prepare the two-stage Efficient-Head plan for the shared runner."""

    # Count the future full-FT parameter size before LoRA wraps the stage-1
    # model. That keeps the saved parameter counts comparable to Full FT.
    stage2_trainable_params, total_params = count_model_parameters(base_context.model)
    stage1_context = apply_stage1_lora_to_context(base_context, args)
    stage1_trainable_params, _stage1_total_params = count_model_parameters(
        stage1_context.model
    )

    def build_stage2_context_from_stage1(stage1_model):
        """Rebuild stage 2 from a fresh backbone plus the trained head."""

        # The stage-1 model gives us the trained head. The helper will rebuild a
        # fresh backbone and copy only that head into it.
        return build_stage2_context(stage1_model, base_context, args)

    return TwoStagePlan(
        stage1_context=stage1_context,
        stage1_trainable_params=stage1_trainable_params,
        stage2_trainable_params=stage2_trainable_params,
        total_params=total_params,
        build_stage2_context=build_stage2_context_from_stage1,
    )


def main() -> None:
    """Run one Efficient-Head two-stage experiment from `manual_config.py`."""

    args = SimpleNamespace(**MANUAL_CONFIG)
    run_two_stage_transformer(
        args,
        build_experiment_config=build_experiment_config,
        build_stage_plan=build_stage_plan,
        stage1_dir_name=STAGE1_DIR_NAME,
        stage2_dir_name=STAGE2_DIR_NAME,
        stage1_learning_rate_attr="stage1_learning_rate",
        stage1_epochs_attr="stage1_epochs",
        stage2_learning_rate_attr="stage2_learning_rate",
        stage2_epochs_attr="stage2_epochs",
        train_message="Starting efficient-head DistilBERT training",
        stage1_message="LoRA parameter-efficient head training",
        stage2_message="full fine-tuning fresh backbone with transferred head",
    )


if __name__ == "__main__":
    main()
