import sys
import types
import unittest
from unittest.mock import Mock, patch

from src.methods.bilstm.tokenizer import TOKENIZER_NAME, StandardBiLSTMTokenizer


class FakeHfTokenizer:
    pad_token_id = 0
    unk_token_id = 100
    pad_token = "[PAD]"
    unk_token = "[UNK]"

    def __len__(self):
        return 30522

    def __call__(
        self,
        text,
        *,
        max_length,
        padding,
        truncation,
        return_attention_mask,
        add_special_tokens,
    ):
        self.last_call = {
            "text": text,
            "max_length": max_length,
            "padding": padding,
            "truncation": truncation,
            "return_attention_mask": return_attention_mask,
            "add_special_tokens": add_special_tokens,
        }
        return {
            "input_ids": [101, 7592, 102, 0, 0],
            "attention_mask": [1, 1, 1, 0, 0],
        }

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


class BiLSTMTokenizerTests(unittest.TestCase):
    def test_create_uses_distilbert_base_uncased_fast_tokenizer(self):
        fake_tokenizer = FakeHfTokenizer()
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=Mock(return_value=fake_tokenizer)
            )
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            tokenizer = StandardBiLSTMTokenizer.create(max_length=5)

        fake_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            TOKENIZER_NAME,
            use_fast=True,
        )
        self.assertIs(tokenizer.hf_tokenizer, fake_tokenizer)
        self.assertEqual(tokenizer.max_length, 5)

    def test_encode_uses_hf_padding_truncation_and_reports_attention_length(self):
        fake_tokenizer = FakeHfTokenizer()
        tokenizer = StandardBiLSTMTokenizer(max_length=5, hf_tokenizer=fake_tokenizer)

        encoded = tokenizer.encode("Hello world")

        self.assertEqual(
            fake_tokenizer.last_call,
            {
                "text": "Hello world",
                "max_length": 5,
                "padding": "max_length",
                "truncation": True,
                "return_attention_mask": True,
                "add_special_tokens": True,
            },
        )
        self.assertEqual(encoded["input_ids"], [101, 7592, 102, 0, 0])
        self.assertEqual(encoded["length"], 3)

    def test_to_dict_matches_main_tokenizer_policy(self):
        tokenizer = StandardBiLSTMTokenizer(max_length=5, hf_tokenizer=FakeHfTokenizer())

        self.assertEqual(
            tokenizer.to_dict(),
            {
                "tokenizer_name": "distilbert-base-uncased",
                "max_length": 5,
                "pad_token": "[PAD]",
                "pad_id": 0,
                "unk_token": "[UNK]",
                "unk_id": 100,
                "vocab_size": 30522,
                "policy": "hardcoded_distilbert_base_uncased_tokenizer",
            },
        )

    def test_save_pretrained_delegates_to_hf_tokenizer(self):
        fake_tokenizer = FakeHfTokenizer()
        tokenizer = StandardBiLSTMTokenizer(max_length=5, hf_tokenizer=fake_tokenizer)

        tokenizer.save_pretrained("out-tokenizer")

        self.assertEqual(fake_tokenizer.saved_to, "out-tokenizer")


if __name__ == "__main__":
    unittest.main()
