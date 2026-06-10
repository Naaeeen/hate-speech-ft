import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.methods.transformer_data import (
    build_fixed_label_maps,
    build_tokenized_dataset_with_stats,
    resolve_eval_split_name,
)
from src.methods.distilbert_full.train import REPO_ROOT
from src.methods.predictions import save_prediction_file
from src.methods.transformer_trainer import (
    build_hf_trainer,
    compute_balanced_class_weights,
    resolve_class_weights,
)


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [101, 102], "attention_mask": [1, 1]}


class FakePredictionOutput:
    predictions = [[0.1, 0.8, 0.1], [2.0, 0.5, 0.1]]
    label_ids = [1, 0]


class RunDistilbertHatexplainTests(unittest.TestCase):
    def test_repo_root_points_to_project_root_after_method_move(self):
        self.assertTrue((REPO_ROOT / "README.md").is_file())
        self.assertTrue((REPO_ROOT / "src" / "methods" / "distilbert_full" / "train.py").is_file())

    def test_build_fixed_label_maps_uses_hatexplain_class_order(self):
        id2label, label2id, num_labels = build_fixed_label_maps()

        self.assertEqual(num_labels, 3)
        self.assertEqual(
            id2label,
            {0: "hatespeech", 1: "normal", 2: "offensive"},
        )
        self.assertEqual(
            label2id,
            {"hatespeech": 0, "normal": 1, "offensive": 2},
        )

    def test_build_tokenized_dataset_uses_shared_preprocessing_and_drops_undecided(self):
        tokenizer = RecordingTokenizer()
        examples = [
            {
                "id": "keep",
                "post_tokens": ["this", "is", "kept"],
                "annotators": {"label": [2, 2, 0]},
            },
            {
                "id": "drop",
                "post_tokens": ["this", "is", "dropped"],
                "annotators": {"label": [0, 1, 2]},
            },
        ]

        split = build_tokenized_dataset_with_stats(
            examples,
            tokenizer=tokenizer,
            max_length=128,
        )

        self.assertEqual(
            split.dataset,
            [{"input_ids": [101, 102], "attention_mask": [1, 1], "labels": 2}],
        )
        self.assertEqual(
            tokenizer.calls,
            [("this is kept", {"truncation": True, "max_length": 128})],
        )

    def test_build_tokenized_dataset_applies_sample_limit_after_shared_filtering(self):
        tokenizer = RecordingTokenizer()
        examples = [
            {"id": "a", "post_tokens": ["a"], "annotators": {"label": [0, 0, 2]}},
            {"id": "b", "post_tokens": ["b"], "annotators": {"label": [1, 1, 2]}},
        ]

        split = build_tokenized_dataset_with_stats(
            examples,
            tokenizer=tokenizer,
            max_length=128,
            max_samples=1,
        )

        self.assertEqual(len(split.dataset), 1)
        self.assertEqual(split.dataset[0]["labels"], 0)

    def test_build_tokenized_dataset_with_stats_tracks_raw_and_dropped_counts(self):
        tokenizer = RecordingTokenizer()
        examples = [
            {
                "id": "keep",
                "post_tokens": ["keep"],
                "annotators": {"label": [2, 2, 0]},
            },
            {
                "id": "drop",
                "post_tokens": ["drop"],
                "annotators": {"label": [0, 1, 2]},
            },
        ]

        split = build_tokenized_dataset_with_stats(
            examples,
            tokenizer=tokenizer,
            max_length=128,
        )

        self.assertEqual(split.raw_size, 2)
        self.assertEqual(split.preprocessed_size, 1)
        self.assertEqual(split.dropped_no_majority_count, 1)
        self.assertEqual(len(split.dataset), 1)
        self.assertEqual(split.records[0]["id"], "keep")

    def test_eval_split_resolution_never_falls_back_to_test(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            resolve_eval_split_name({"train": [], "test": []}, test_split_name="test")

        self.assertEqual(
            resolve_eval_split_name(
                {"train": [], "validation": [], "test": []},
                test_split_name="test",
            ),
            "validation",
        )

    def test_save_prediction_file_preserves_sample_identity_and_logits(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_predictions.json"
            records = [
                {"id": "a", "text": "first", "label": 1, "label_name": "normal"},
                {"id": "b", "text": "second", "label": 0, "label_name": "hatespeech"},
            ]

            saved = save_prediction_file(
                path,
                records=records,
                prediction_output=FakePredictionOutput(),
                id2label={0: "hatespeech", 1: "normal", 2: "offensive"},
            )

            payload = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(payload["count"], 2)
            self.assertEqual(payload["predictions"][0]["id"], "a")
            self.assertEqual(payload["predictions"][0]["predicted_label"], 1)
            self.assertEqual(payload["predictions"][0]["predicted_label_name"], "normal")
            self.assertEqual(payload["predictions"][1]["predicted_label"], 0)
            self.assertEqual(payload["predictions"][1]["logits"], [2.0, 0.5, 0.1])

    def test_build_hf_trainer_uses_processing_class_instead_of_removed_tokenizer_arg(self):
        captured_kwargs = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        tokenizer = object()

        context = type(
            "FakeContext",
            (),
            {
                "trainer_cls": FakeTrainer,
                "model": object(),
                "train_dataset": [],
                "eval_dataset": [],
                "tokenizer": tokenizer,
                "data_collator": object(),
            },
        )
        with patch(
            "src.methods.transformer_trainer.compute_metrics_fn",
            return_value=lambda _: {},
        ):
            trainer = build_hf_trainer(context, training_args=object())

        self.assertIsInstance(trainer, FakeTrainer)
        self.assertIs(captured_kwargs["processing_class"], tokenizer)
        self.assertNotIn("tokenizer", captured_kwargs)

    def test_build_hf_trainer_passes_callbacks_when_provided(self):
        captured_kwargs = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        callbacks = [object()]

        context = type(
            "FakeContext",
            (),
            {
                "trainer_cls": FakeTrainer,
                "model": object(),
                "train_dataset": [],
                "eval_dataset": [],
                "tokenizer": object(),
                "data_collator": object(),
            },
        )
        with patch(
            "src.methods.transformer_trainer.compute_metrics_fn",
            return_value=lambda _: {},
        ):
            build_hf_trainer(
                context,
                training_args=object(),
                callbacks=callbacks,
            )

        self.assertIs(captured_kwargs["callbacks"], callbacks)

    def test_balanced_class_weights_use_final_training_subset(self):
        dataset = [
            {"labels": 0},
            {"labels": 0},
            {"labels": 1},
            {"labels": 2},
        ]

        weights = compute_balanced_class_weights(dataset, num_labels=3)

        self.assertEqual(weights, [4 / 6, 4 / 3, 4 / 3])

    def test_balanced_class_weights_reject_missing_class(self):
        dataset = [{"labels": 0}, {"labels": 1}]

        with self.assertRaises(ValueError):
            compute_balanced_class_weights(dataset, num_labels=3)

    def test_resolve_class_weights_supports_global_switch(self):
        dataset = [{"labels": 0}, {"labels": 1}, {"labels": 2}]

        self.assertIsNone(
            resolve_class_weights(
                class_weighting="none",
                train_dataset=dataset,
                num_labels=3,
            )
        )
        self.assertEqual(
            resolve_class_weights(
                class_weighting="balanced",
                train_dataset=dataset,
                num_labels=3,
            ),
            [1.0, 1.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
