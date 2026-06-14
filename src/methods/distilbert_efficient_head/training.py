"""Method-specific pieces for Efficient-Head FT.

Stage 1 trains the classification head. Stage 2 builds the final model from a
fresh DistilBERT backbone, then copies over the stage-1 head weights.
"""

from __future__ import annotations

from typing import Any

from src.methods.peft_utils import (
    apply_lora_to_context_model,
    extract_classification_head_state_dict,
    load_classification_head_state_dict,
    replace_context_model,
    set_all_parameters_trainable,
)


def apply_stage1_lora_to_context(context, args: Any):
    """Attach stage-1 LoRA adapters while keeping the head saveable/trainable."""

    return apply_lora_to_context_model(
        context,
        args,
        modules_to_save=args.stage1_modules_to_save,
        prefix="stage1_",
    )


def build_stage2_context(stage1_model, context, args: Any):
    """Build the stage-2 full-FT context from a fresh backbone plus stage-1 head."""

    from transformers import AutoModelForSequenceClassification

    # Only the classification head moves forward. The stage-1 backbone and LoRA
    # adapters are discarded so stage 2 measures "better head init",
    # not "continue training the adapter model".
    head_state = extract_classification_head_state_dict(stage1_model)
    stage2_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=context.num_labels,
        id2label=context.id2label,
        label2id=context.label2id,
    )
    if args.gradient_checkpointing:
        stage2_model.gradient_checkpointing_enable()
    load_classification_head_state_dict(stage2_model, head_state)
    set_all_parameters_trainable(stage2_model)
    return replace_context_model(context, stage2_model)
