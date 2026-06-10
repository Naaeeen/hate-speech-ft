"""Fixed DistilBERT tokenizer wrapper for the BiLSTM baseline.

The BiLSTM model itself is trained from scratch, but the token ids come from
`distilbert-base-uncased` so we do not maintain a separate train-split
vocabulary. The wrapper pads/truncates to `max_length` and returns the
non-padding tokenized length, including DistilBERT special tokens, so the LSTM
can ignore padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


TOKENIZER_NAME = "distilbert-base-uncased"


@dataclass
class StandardBiLSTMTokenizer:
    """Hard-coded DistilBERT tokenizer for the Bi-LSTM baseline."""

    max_length: int
    hf_tokenizer: "PreTrainedTokenizerBase | Any"

    @classmethod
    def create(cls, *, max_length: int) -> "StandardBiLSTMTokenizer":
        """Load the fixed DistilBERT tokenizer used by the BiLSTM baseline."""

        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            use_fast=True,
        )

        return cls(
            max_length=max_length,
            hf_tokenizer=hf_tokenizer,
        )

    @property
    def pad_id(self) -> int:
        """Return the tokenizer padding id used by the embedding layer."""

        pad_token_id = self.hf_tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("distilbert-base-uncased does not define pad_token_id.")
        return int(pad_token_id)

    @property
    def unk_id(self) -> int:
        """Return the tokenizer unknown-token id for metadata/debugging."""

        unk_token_id = self.hf_tokenizer.unk_token_id
        if unk_token_id is None:
            raise ValueError("distilbert-base-uncased does not define unk_token_id.")
        return int(unk_token_id)

    @property
    def vocab_size(self) -> int:
        """Return the fixed DistilBERT vocabulary size."""

        return len(self.hf_tokenizer)

    def encode(self, text: str) -> dict[str, list[int] | int]:
        """Encode text to padded ids plus non-padding tokenized length."""

        encoded = self.hf_tokenizer(
            str(text),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        input_ids = [int(value) for value in encoded["input_ids"]]
        attention_mask = [int(value) for value in encoded["attention_mask"]]

        length = max(1, sum(attention_mask))

        return {
            "input_ids": input_ids,
            "length": int(length),
        }

    def to_dict(self) -> dict[str, object]:
        """Return tokenizer metadata saved into `resolved_config.json`."""

        return {
            "tokenizer_name": TOKENIZER_NAME,
            "max_length": self.max_length,
            "pad_token": self.hf_tokenizer.pad_token,
            "pad_id": self.pad_id,
            "unk_token": self.hf_tokenizer.unk_token,
            "unk_id": self.unk_id,
            "vocab_size": self.vocab_size,
            "policy": "hardcoded_distilbert_base_uncased_tokenizer",
        }

    def save_pretrained(self, output_dir: str) -> None:
        """Save tokenizer files next to the final BiLSTM model."""

        self.hf_tokenizer.save_pretrained(output_dir)
