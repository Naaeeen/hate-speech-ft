import unittest

from src.run_distilbert_hatexplain import (
    build_fixed_label_maps,
    build_trainer,
    build_tokenized_dataset,
    compute_balanced_class_weights,
    resolve_class_weights,
)


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [101, 102], "attention_mask": [1, 1]}


class RunDistilbertHatexplainTests(unittest.TestCase):
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

        tokenized = build_tokenized_dataset(
            examples,
            tokenizer=tokenizer,
            max_length=128,
        )

        self.assertEqual(
            tokenized,
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

        tokenized = build_tokenized_dataset(
            examples,
            tokenizer=tokenizer,
            max_length=128,
            max_samples=1,
        )

        self.assertEqual(len(tokenized), 1)
        self.assertEqual(tokenized[0]["labels"], 0)

    def test_build_trainer_uses_processing_class_instead_of_removed_tokenizer_arg(self):
        captured_kwargs = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        tokenizer = object()

        trainer = build_trainer(
            trainer_cls=FakeTrainer,
            model=object(),
            training_args=object(),
            train_dataset=[],
            eval_dataset=[],
            tokenizer=tokenizer,
            data_collator=object(),
            compute_metrics=lambda _: {},
        )

        self.assertIsInstance(trainer, FakeTrainer)
        self.assertIs(captured_kwargs["processing_class"], tokenizer)
        self.assertNotIn("tokenizer", captured_kwargs)

    def test_build_trainer_passes_callbacks_when_provided(self):
        captured_kwargs = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        callbacks = [object()]

        build_trainer(
            trainer_cls=FakeTrainer,
            model=object(),
            training_args=object(),
            train_dataset=[],
            eval_dataset=[],
            tokenizer=object(),
            data_collator=object(),
            compute_metrics=lambda _: {},
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
