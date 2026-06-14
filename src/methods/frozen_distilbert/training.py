"""Trainability helpers for frozen-backbone DistilBERT."""

from __future__ import annotations

from collections.abc import Iterable

from src.methods.peft_utils import set_classification_head_trainability


def _iter_backbone_modules(model) -> Iterable:
    """Yield backbone modules that stay in eval mode while the head trains."""

    seen = set()
    for module in (
        getattr(model, "base_model", None),
        getattr(model, getattr(model, "base_model_prefix", ""), None),
    ):
        if module is None or module is model:
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield module


def _set_backbone_eval_mode(model) -> None:
    for backbone_module in _iter_backbone_modules(model):
        backbone_module.eval()


def keep_frozen_backbone_in_eval_mode(model) -> None:
    """Keep frozen backbone dropout behavior stable during head-only training."""

    if getattr(model, "_frozen_backbone_train_patch_applied", False):
        _set_backbone_eval_mode(model)
        return

    original_train = model.train

    def train_with_frozen_backbone(mode: bool = True):
        """Call the original `train` and immediately restore backbone eval mode."""

        result = original_train(mode)
        _set_backbone_eval_mode(model)
        return result

    model.train = train_with_frozen_backbone
    model._frozen_backbone_train_patch_applied = True
    _set_backbone_eval_mode(model)


def set_frozen_backbone_trainability(model) -> None:
    """Freeze DistilBERT backbone weights and keep only the head trainable."""

    set_classification_head_trainability(model)
    keep_frozen_backbone_in_eval_mode(model)
