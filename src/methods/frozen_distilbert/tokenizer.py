from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "distilbert-base-uncased"


@dataclass
class FrozenDistilBertTokenizer:
    """Hard-coded DistilBERT tokenizer for frozen-backbone experiments."""

    max_length: int
    hf_tokenizer: PreTrainedTokenizerBase

    @classmethod
    def create(cls, *, max_length: int) -> "FrozenDistilBertTokenizer":
        hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        return cls(max_length=max_length, hf_tokenizer=hf_tokenizer)

    def encode(self, text: str) -> dict[str, list[int]]:
        encoded = self.hf_tokenizer(
            str(text),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        return {
            "input_ids": [int(value) for value in encoded["input_ids"]],
            "attention_mask": [int(value) for value in encoded["attention_mask"]],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": MODEL_NAME,
            "tokenizer_name": MODEL_NAME,
            "max_length": self.max_length,
            "pad_token": self.hf_tokenizer.pad_token,
            "pad_token_id": self.hf_tokenizer.pad_token_id,
            "unk_token": self.hf_tokenizer.unk_token,
            "unk_token_id": self.hf_tokenizer.unk_token_id,
            "vocab_size": len(self.hf_tokenizer),
            "policy": "hardcoded_distilbert_base_uncased_tokenizer",
        }

    def save_pretrained(self, output_dir: str) -> None:
        self.hf_tokenizer.save_pretrained(output_dir)
