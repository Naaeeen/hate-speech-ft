import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.methods.bilstm.tokenizer import (
    PAD_TOKEN,
    UNK_TOKEN,
    StandardBiLSTMTokenizer,
)


class BiLSTMTokenizerTests(unittest.TestCase):
    def test_builds_train_only_word_vocab_with_deterministic_order(self):
        tokenizer = StandardBiLSTMTokenizer.create(
            train_records=[
                {"text": "Hello world world"},
                {"text": "hello data"},
                {"text": "rare token"},
            ],
            max_length=5,
            min_freq=2,
            max_vocab_size=4,
        )

        self.assertEqual(tokenizer.vocab[PAD_TOKEN], 0)
        self.assertEqual(tokenizer.vocab[UNK_TOKEN], 1)
        self.assertEqual(list(tokenizer.vocab), [PAD_TOKEN, UNK_TOKEN, "hello", "world"])
        self.assertEqual(tokenizer.vocab_size, 4)

        encoded = tokenizer.encode("HELLO evalonly world")

        self.assertEqual(
            encoded["input_ids"],
            [
                tokenizer.vocab["hello"],
                tokenizer.unk_id,
                tokenizer.vocab["world"],
                tokenizer.pad_id,
                tokenizer.pad_id,
            ],
        )
        self.assertEqual(encoded["length"], 3)

    def test_empty_text_encodes_as_single_unknown_token(self):
        tokenizer = StandardBiLSTMTokenizer.create(
            train_records=[{"text": "hello"}],
            max_length=3,
        )

        encoded = tokenizer.encode("")

        self.assertEqual(
            encoded,
            {
                "input_ids": [tokenizer.unk_id, tokenizer.pad_id, tokenizer.pad_id],
                "length": 1,
            },
        )

    def test_save_and_load_round_trip_preserves_vocab_and_config(self):
        tokenizer = StandardBiLSTMTokenizer.create(
            train_records=[
                {"text": "hello world"},
                {"text": "hello friend"},
            ],
            max_length=4,
            min_freq=1,
            max_vocab_size=5,
        )

        with TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            loaded = StandardBiLSTMTokenizer.from_pretrained(temp_dir)

            self.assertEqual(loaded.vocab, tokenizer.vocab)
            self.assertEqual(loaded.max_length, tokenizer.max_length)
            self.assertEqual(loaded.min_freq, tokenizer.min_freq)
            self.assertEqual(loaded.max_vocab_size, tokenizer.max_vocab_size)
            self.assertEqual(loaded.lowercase, tokenizer.lowercase)
            self.assertEqual(
                loaded.encode("HELLO missing"),
                tokenizer.encode("HELLO missing"),
            )

            config = json.loads(
                Path(temp_dir, "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["tokenizer_name"], "bilstm-word")
            self.assertEqual(config["policy"], "train_split_word_vocab")
            self.assertEqual(config["vocab_checksum"], tokenizer.vocab_checksum)

    def test_from_pretrained_rejects_vocab_checksum_mismatch(self):
        tokenizer = StandardBiLSTMTokenizer.create(
            train_records=[
                {"text": "hello world"},
                {"text": "hello friend"},
            ],
            max_length=4,
            min_freq=1,
        )

        with TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            vocab_path = Path(temp_dir, "vocab.json")
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab["tampered"] = vocab.pop("world")
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "vocab_checksum mismatch"):
                StandardBiLSTMTokenizer.from_pretrained(temp_dir)

    def test_from_pretrained_rejects_invalid_special_token_ids(self):
        tokenizer = StandardBiLSTMTokenizer.create(
            train_records=[{"text": "hello"}],
            max_length=4,
            min_freq=1,
        )

        with TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            vocab_path = Path(temp_dir, "vocab.json")
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab[PAD_TOKEN] = 1
            vocab[UNK_TOKEN] = 0
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "map <pad> to id 0"):
                StandardBiLSTMTokenizer.from_pretrained(temp_dir)


if __name__ == "__main__":
    unittest.main()
