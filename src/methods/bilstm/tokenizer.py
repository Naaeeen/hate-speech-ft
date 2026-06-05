from __future__ import annotations

import json
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKENIZER_NAME = "bilstm-word"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
DEFAULT_MIN_FREQ = 2
DEFAULT_MAX_VOCAB_SIZE = 30000
DEFAULT_LOWERCASE = True


@dataclass(frozen=True)
class StandardBiLSTMTokenizer:
    """Train-split word vocabulary for the Bi-LSTM baseline.

    HateXplain text is already constructed from space-separated post_tokens.
    Building this vocabulary from the train split keeps Bi-LSTM independent of
    DistilBERT's pretrained WordPiece vocabulary while avoiding eval/test
    leakage.
    """

    max_length: int
    vocab: dict[str, int]
    min_freq: int = DEFAULT_MIN_FREQ
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE
    lowercase: bool = DEFAULT_LOWERCASE

    @classmethod
    def create(
        cls,
        *,
        train_records: Sequence[Mapping[str, Any]],
        max_length: int,
        min_freq: int = DEFAULT_MIN_FREQ,
        max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
        lowercase: bool = DEFAULT_LOWERCASE,
    ) -> "StandardBiLSTMTokenizer":
        if max_length < 1:
            raise ValueError("max_length must be >= 1.")
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1.")
        if max_vocab_size < 2:
            raise ValueError("max_vocab_size must be >= 2 for pad and unk tokens.")

        counter: Counter[str] = Counter()
        for record in train_records:
            counter.update(cls.tokenize(record.get("text", ""), lowercase=lowercase))

        vocab = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }
        available_slots = max_vocab_size - len(vocab)
        candidates = (
            token
            for token, frequency in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if frequency >= min_freq and token not in vocab
        )
        for token in candidates:
            if len(vocab) - 2 >= available_slots:
                break
            vocab[token] = len(vocab)

        return cls(
            max_length=max_length,
            vocab=vocab,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
            lowercase=lowercase,
        )

    @classmethod
    def from_pretrained(cls, input_dir: str | Path) -> "StandardBiLSTMTokenizer":
        input_path = Path(input_dir)
        with (input_path / "vocab.json").open("r", encoding="utf-8") as handle:
            raw_vocab = json.load(handle)
        with (input_path / "tokenizer_config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        vocab = {str(token): int(index) for token, index in raw_vocab.items()}
        tokenizer = cls(
            max_length=int(config["max_length"]),
            vocab=vocab,
            min_freq=int(config.get("min_freq", DEFAULT_MIN_FREQ)),
            max_vocab_size=int(config.get("max_vocab_size", len(vocab))),
            lowercase=bool(config.get("lowercase", DEFAULT_LOWERCASE)),
        )
        tokenizer._validate_loaded_config(config)
        return tokenizer

    @staticmethod
    def tokenize(text: Any, *, lowercase: bool = DEFAULT_LOWERCASE) -> list[str]:
        value = str(text)
        if lowercase:
            value = value.lower()
        return value.split()

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_checksum(self) -> str:
        return _vocab_checksum(self.vocab)

    def encode(self, text: str) -> dict[str, list[int] | int]:
        tokens = self.tokenize(text, lowercase=self.lowercase)
        input_ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        input_ids = input_ids[: self.max_length]

        if not input_ids:
            input_ids = [self.unk_id]

        length = len(input_ids)
        padding_length = self.max_length - length
        padded_input_ids = input_ids + [self.pad_id] * padding_length

        return {
            "input_ids": [int(value) for value in padded_input_ids],
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
            "vocab_checksum": self.vocab_checksum,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "lowercase": self.lowercase,
            "policy": "train_split_word_vocab",
        }

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with (output_path / "vocab.json").open("w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, ensure_ascii=False, indent=2, sort_keys=True)

        with (output_path / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)

    def _validate_loaded_config(self, config: Mapping[str, Any]) -> None:
        if config.get("tokenizer_name") != TOKENIZER_NAME:
            raise ValueError(
                f"Tokenizer config name must be {TOKENIZER_NAME!r}, "
                f"got {config.get('tokenizer_name')!r}."
            )
        if self.max_length < 1:
            raise ValueError("Tokenizer config max_length must be >= 1.")
        if self.min_freq < 1:
            raise ValueError("Tokenizer config min_freq must be >= 1.")
        if self.max_vocab_size < 2:
            raise ValueError("Tokenizer config max_vocab_size must be >= 2.")
        if len(self.vocab) > self.max_vocab_size:
            raise ValueError("Tokenizer vocab exceeds configured max_vocab_size.")
        if self.vocab.get(PAD_TOKEN) != 0:
            raise ValueError("Tokenizer vocab must map <pad> to id 0.")
        if self.vocab.get(UNK_TOKEN) != 1:
            raise ValueError("Tokenizer vocab must map <unk> to id 1.")

        expected_ids = set(range(len(self.vocab)))
        actual_ids = set(self.vocab.values())
        if actual_ids != expected_ids:
            raise ValueError("Tokenizer vocab ids must be contiguous from 0.")

        expected_fields = {
            "pad_token": PAD_TOKEN,
            "pad_id": self.pad_id,
            "unk_token": UNK_TOKEN,
            "unk_id": self.unk_id,
            "vocab_size": self.vocab_size,
            "vocab_checksum": self.vocab_checksum,
        }
        for field_name, expected_value in expected_fields.items():
            if config.get(field_name) != expected_value:
                raise ValueError(
                    f"Tokenizer config {field_name} mismatch: expected "
                    f"{expected_value!r}, got {config.get(field_name)!r}."
                )


def _vocab_checksum(vocab: Mapping[str, int]) -> str:
    payload = json.dumps(dict(vocab), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
