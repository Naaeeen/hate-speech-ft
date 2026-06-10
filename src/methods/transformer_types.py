"""Small typed containers passed around the Transformer helper stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.methods.transformer_data import TokenizedSplit
from src.utils.wandb_config import WandbSettings


@dataclass(frozen=True)
class HfRunSetup:
    """Environment/setup facts shared by one Transformer run."""

    gpu_type: str
    precision_policy: dict[str, Any]
    wandb_settings: WandbSettings


@dataclass(frozen=True)
class HfClassificationRun:
    """Everything the runner needs after dataset/model/tokenizer setup."""

    args: Any
    precision_policy: dict[str, Any]
    gpu_type: str
    model: Any
    tokenizer: Any
    data_collator: Any
    trainer_cls: Any
    class_weights: list[float] | None
    train_split: str
    eval_split: str
    test_split: str | None
    train_split_data: TokenizedSplit
    eval_split_data: TokenizedSplit
    test_split_data: TokenizedSplit | None
    id2label: dict[int, str]
    label2id: dict[str, int]
    num_labels: int

    @property
    def train_dataset(self) -> list[dict[str, Any]]:
        """Return the tokenized training rows passed to Trainer."""

        return self.train_split_data.dataset

    @property
    def eval_dataset(self) -> list[dict[str, Any]]:
        """Return the tokenized validation rows passed to Trainer."""

        return self.eval_split_data.dataset

    @property
    def test_dataset(self) -> list[dict[str, Any]] | None:
        """Return tokenized test rows, or `None` for validation-only runs."""

        return self.test_split_data.dataset if self.test_split_data else None

    def config_kwargs(self) -> dict[str, Any]:
        """Build split/accounting kwargs for method config builders."""

        return {
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "train_size": len(self.train_dataset),
            "eval_size": len(self.eval_dataset),
            "full_train_size": self.train_split_data.preprocessed_size,
            "full_eval_size": self.eval_split_data.preprocessed_size,
            "raw_train_size": self.train_split_data.raw_size,
            "raw_eval_size": self.eval_split_data.raw_size,
            "dropped_no_majority_train": (
                self.train_split_data.dropped_no_majority_count
            ),
            "dropped_no_majority_eval": self.eval_split_data.dropped_no_majority_count,
            "test_size": len(self.test_dataset) if self.test_dataset is not None else None,
            "full_test_size": (
                self.test_split_data.preprocessed_size if self.test_split_data else None
            ),
            "raw_test_size": self.test_split_data.raw_size if self.test_split_data else None,
            "dropped_no_majority_test": (
                self.test_split_data.dropped_no_majority_count
                if self.test_split_data
                else None
            ),
            "gpu_type": self.gpu_type,
            "class_weights": self.class_weights,
            "precision_policy": self.precision_policy,
        }
