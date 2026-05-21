from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase


TOKENIZER_NAME = "distilbert-base-uncased"


@dataclass
class StandardBiLSTMTokenizer:
    """Hard-coded DistilBERT tokenizer for the Bi-LSTM baseline"""

    max_length: int
    hf_tokenizer: PreTrainedTokenizerBase

    @classmethod
    def create(cls, *, max_length: int) -> "StandardBiLSTMTokenizer":
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
        pad_token_id = self.hf_tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("distilbert-base-uncased does not define pad_token_id.")
        return int(pad_token_id)

    @property
    def unk_id(self) -> int:
        unk_token_id = self.hf_tokenizer.unk_token_id
        if unk_token_id is None:
            raise ValueError("distilbert-base-uncased does not define unk_token_id.")
        return int(unk_token_id)

    @property
    def vocab_size(self) -> int:
        return len(self.hf_tokenizer)

    def encode(self, text: str) -> dict[str, list[int] | int]:
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
        self.hf_tokenizer.save_pretrained(output_dir)