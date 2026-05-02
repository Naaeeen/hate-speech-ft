import unittest

from src.data.text_field_policy import (
    TEXT_FIELD_POLICY,
    TEXT_FIELD_USAGE,
    build_text_from_post_tokens,
    tokenize_hatexplain_text,
)


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [101, 102], "attention_mask": [1, 1]}


class TextFieldPolicyTests(unittest.TestCase):
    def test_build_text_joins_hatexplain_post_tokens_with_single_spaces(self):
        example = {"post_tokens": ["hello", ",", "world", "!"]}

        self.assertEqual(build_text_from_post_tokens(example), "hello , world !")

    def test_build_text_rejects_missing_post_tokens(self):
        with self.assertRaises(KeyError):
            build_text_from_post_tokens({"text": "hello world"})

    def test_tokenize_uses_shared_text_policy_and_model_tokenizer(self):
        tokenizer = RecordingTokenizer()
        example = {"post_tokens": ["can", "not", "speak", "properly"]}

        encoded = tokenize_hatexplain_text(example, tokenizer, max_length=128)

        self.assertEqual(encoded, {"input_ids": [101, 102], "attention_mask": [1, 1]})
        self.assertEqual(
            tokenizer.calls,
            [
                (
                    "can not speak properly",
                    {"truncation": True, "max_length": 128},
                )
            ],
        )

    def test_policy_text_documents_no_extra_dataset_level_cleaning(self):
        self.assertIn("post_tokens", TEXT_FIELD_POLICY)
        self.assertIn("No extra dataset-level cleaning", TEXT_FIELD_POLICY)

    def test_usage_text_explains_dataset_example_transformers_and_baselines(self):
        self.assertIn("example", TEXT_FIELD_USAGE)
        self.assertIn("load_dataset", TEXT_FIELD_USAGE)
        self.assertIn("raw HateXplain JSON", TEXT_FIELD_USAGE)
        self.assertIn("list of annotator dictionaries", TEXT_FIELD_USAGE)
        self.assertIn("ClassLabel", TEXT_FIELD_USAGE)
        self.assertIn("Transformer", TEXT_FIELD_USAGE)
        self.assertIn("TF-IDF", TEXT_FIELD_USAGE)
        self.assertIn("Bi-LSTM", TEXT_FIELD_USAGE)


if __name__ == "__main__":
    unittest.main()
