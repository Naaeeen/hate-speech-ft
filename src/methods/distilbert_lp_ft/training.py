"""Tiny trainability aliases for LP-FT.

Keeping these names local makes the method read like the experiment: linear
probe first, full fine-tune second.
"""

from __future__ import annotations

from src.methods.peft_utils import (
    set_all_parameters_trainable as set_full_finetune_trainability,
    set_classification_head_trainability as set_linear_probe_trainability,
)


STAGE1_DIR_NAME = "stage1_linear_probe"
STAGE2_DIR_NAME = "stage2_full_ft"
