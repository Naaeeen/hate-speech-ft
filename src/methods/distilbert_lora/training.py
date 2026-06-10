"""Method-specific LoRA context setup.

The shared Transformer runner loads the normal DistilBERT classifier first.
This helper then wraps that model with PEFT LoRA and checks that the
classification head is included in `modules_to_save`, so the task head is saved
with the adapter.
"""

from __future__ import annotations

from typing import Any

from src.methods.peft_utils import apply_lora_to_context_model


def apply_lora_to_context(context, args: Any):
    """Attach LoRA adapters to the shared Transformer run context."""

    return apply_lora_to_context_model(
        context,
        args,
        modules_to_save=args.modules_to_save,
    )
