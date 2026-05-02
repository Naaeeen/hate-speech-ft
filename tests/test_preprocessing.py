import unittest

from src.data.preprocessing import (
    HATEXPLAIN_PREPROCESSING_POLICY,
    preprocess_hatexplain_example,
    preprocess_hatexplain_split,
    select_data_fraction,
    tokenize_preprocessed_record,
)


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [101, 2009, 102], "attention_mask": [1, 1, 1]}


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_example_builds_stable_model_record(self):
        example = {
            "id": "23107796_gab",
            "post_tokens": ["u", "really", "think"],
            "annotators": {"label": [0, 2, 2]},
            "rationales": [[0, 1, 0], [0, 0, 0]],
        }

        record = preprocess_hatexplain_example(example)

        self.assertEqual(
            record,
            {
                "id": "23107796_gab",
                "text": "u really think",
                "label": 2,
                "label_name": "offensive",
                "annotator_label_ids": [0, 2, 2],
                "annotator_label_names": ["hatespeech", "offensive", "offensive"],
                "token_count": 3,
                "has_majority_label": True,
            },
        )

    def test_preprocess_example_returns_none_for_no_majority_by_default(self):
        example = {
            "id": "ambiguous",
            "post_tokens": ["ambiguous", "post"],
            "annotators": {"label": [0, 1, 2]},
        }

        self.assertIsNone(preprocess_hatexplain_example(example))

    def test_preprocess_example_can_keep_undecided_samples_for_audit(self):
        example = {
            "id": "ambiguous",
            "post_tokens": ["ambiguous", "post"],
            "annotators": {"label": [0, 1, 2]},
        }

        record = preprocess_hatexplain_example(example, include_undecided=True)

        self.assertEqual(record["label"], -1)
        self.assertEqual(record["label_name"], "undecided")
        self.assertFalse(record["has_majority_label"])

    def test_preprocess_split_drops_no_majority_samples(self):
        examples = [
            {"id": "keep", "post_tokens": ["keep"], "annotators": {"label": [1, 1, 2]}},
            {"id": "drop", "post_tokens": ["drop"], "annotators": {"label": [0, 1, 2]}},
        ]

        records = preprocess_hatexplain_split(examples)

        self.assertEqual([record["id"] for record in records], ["keep"])

    def test_tokenize_preprocessed_record_adds_trainer_label_field(self):
        tokenizer = RecordingTokenizer()
        record = {
            "id": "23107796_gab",
            "text": "u really think",
            "label": 2,
            "label_name": "offensive",
        }

        tokenized = tokenize_preprocessed_record(record, tokenizer, max_length=128)

        self.assertEqual(tokenized["labels"], 2)
        self.assertEqual(
            tokenizer.calls,
            [("u really think", {"truncation": True, "max_length": 128})],
        )

    def test_select_data_fraction_is_deterministic_and_preserves_original_order(self):
        records = [{"id": str(i)} for i in range(10)]

        first = select_data_fraction(records, fraction=0.3, seed=42)
        second = select_data_fraction(records, fraction=0.3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(first, sorted(first, key=lambda record: int(record["id"])))

    def test_policy_documents_shared_dataset_level_decisions(self):
        self.assertIn("official train/validation/test splits", HATEXPLAIN_PREPROCESSING_POLICY)
        self.assertIn("majority vote", HATEXPLAIN_PREPROCESSING_POLICY)
        self.assertIn("rationales are not model input", HATEXPLAIN_PREPROCESSING_POLICY)


if __name__ == "__main__":
    unittest.main()
