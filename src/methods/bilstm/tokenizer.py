from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKENIZER_NAME = "bilstm-vocab"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
DEFAULT_MIN_FREQ = 2


@dataclass
class StandardBiLSTMTokenizer:
    """Word-level tokenizer/vocabulary for the Bi-LSTM baseline.

    The vocabulary is built only from the Bi-LSTM training split. This keeps the
    Bi-LSTM as a neural model trained from scratch instead of using a pretrained
    Transformer tokenizer/vocabulary.
    """

    max_length: int
    vocab: dict[str, int]
    min_freq: int = DEFAULT_MIN_FREQ

    @classmethod
    def create(
        cls,
        *,
        train_texts: Sequence[str],
        max_length: int,
        min_freq: int = DEFAULT_MIN_FREQ,
    ) -> "StandardBiLSTMTokenizer":
        if max_length < 1:
            raise ValueError("max_length must be >= 1.")
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1.")

        counter: Counter[str] = Counter()
        for text in train_texts:
            counter.update(cls.tokenize(text))

        vocab = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }

        # Deterministic ordering: frequent tokens first, then alphabetical order
        # for equal counts. This makes the vocabulary stable across runs.
        for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if freq >= min_freq and token not in vocab:
                vocab[token] = len(vocab)

        return cls(
            max_length=max_length,
            vocab=vocab,
            min_freq=min_freq,
        )

    @classmethod
    def from_pretrained(cls, input_dir: str | Path) -> "StandardBiLSTMTokenizer":
        input_path = Path(input_dir)
        with (input_path / "vocab.json").open("r", encoding="utf-8") as file:
            vocab = json.load(file)
        with (input_path / "tokenizer_config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        return cls(
            max_length=int(config["max_length"]),
            vocab={str(token): int(index) for token, index in vocab.items()},
            min_freq=int(config.get("min_freq", DEFAULT_MIN_FREQ)),
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        # The shared HateXplain text is already built from post_tokens with spaces.
        return str(text).split()

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> dict[str, list[int] | int]:
        tokens = self.tokenize(text)
        input_ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        input_ids = input_ids[: self.max_length]

        # pack_padded_sequence requires length >= 1. If
        # empty appears we encode it as a single <unk> token.
        if not input_ids:
            input_ids = [self.unk_id]

        length = len(input_ids)
        padding_length = self.max_length - length
        input_ids = input_ids + [self.pad_id] * padding_length

        return {
            "input_ids": [int(value) for value in input_ids],
            "length": int(length),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "tokenizer_name": TOKENIZER_NAME,
            "max_length": self.max_length,
            "pad_token": PAD_TOKEN,
            "pad_id": self.pad_id,
            "unk_token": UNK_TOKEN,
            "unk_id": self.unk_id,
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "policy": "word_level_vocabulary_built_from_training_split",
        }

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with (output_path / "vocab.json").open("w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2, sort_keys=True)

        with (output_path / "tokenizer_config.json").open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2, sort_keys=True)
