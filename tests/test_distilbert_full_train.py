import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.methods.distilbert_full.train import (
    REPO_ROOT,
    build_fixed_label_maps,
    build_training_arguments,
    build_trainer,
    build_tokenized_dataset,
    build_tokenized_dataset_with_stats,
    clear_existing_run_artifacts,
    compute_balanced_class_weights,
    find_existing_run_artifacts,
    resolve_eval_split_name,
    resolve_class_weights,
    save_prediction_file,
    validate_output_dir_for_run,
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
        self.assertTrue((REPO_ROOT / "configs" / "experiments.json").is_file())
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

    def test_build_training_arguments_filters_unsupported_kwargs(self):
        captured_kwargs = {}

        class FakeTrainingArguments:
            def __init__(self, output_dir, learning_rate):
                captured_kwargs.update(
                    {
                        "output_dir": output_dir,
                        "learning_rate": learning_rate,
                    }
                )

        args = build_training_arguments(
            FakeTrainingArguments,
            output_dir="outputs/example",
            learning_rate=2e-5,
            overwrite_output_dir=False,
        )

        self.assertIsInstance(args, FakeTrainingArguments)
        self.assertEqual(captured_kwargs["output_dir"], "outputs/example")
        self.assertEqual(captured_kwargs["learning_rate"], 2e-5)
        self.assertNotIn("overwrite_output_dir", captured_kwargs)

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

    def test_output_dir_guard_protects_existing_run_artifacts(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "result_summary.json").write_text("{}", encoding="utf-8")
            (output_dir / "test_predictions.json").write_text("[]", encoding="utf-8")
            (output_dir / "checkpoint-1").mkdir()
            (output_dir / "stage1_linear_probe").mkdir()

            artifacts = find_existing_run_artifacts(output_dir)

            self.assertIn(output_dir / "result_summary.json", artifacts)
            self.assertIn(output_dir / "test_predictions.json", artifacts)
            self.assertIn(output_dir / "checkpoint-1", artifacts)
            self.assertIn(output_dir / "stage1_linear_probe", artifacts)
            with self.assertRaisesRegex(ValueError, "already contains run artifacts"):
                validate_output_dir_for_run(output_dir, overwrite=False)
            validate_output_dir_for_run(output_dir, overwrite=True)

    def test_output_dir_guard_allows_empty_or_missing_directory(self):
        with TemporaryDirectory() as tmp:
            empty_dir = Path(tmp) / "empty"
            empty_dir.mkdir()
            missing_dir = Path(tmp) / "missing"

            validate_output_dir_for_run(empty_dir, overwrite=False)
            validate_output_dir_for_run(missing_dir, overwrite=False)

    def test_clear_existing_run_artifacts_removes_only_managed_outputs(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "result_summary.json").write_text("{}", encoding="utf-8")
            (output_dir / "model.safetensors").write_text("model", encoding="utf-8")
            (output_dir / "test_predictions.json").write_text("[]", encoding="utf-8")
            checkpoint = output_dir / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
            stage_dir = output_dir / "stage2_full_ft"
            stage_dir.mkdir()
            (stage_dir / "trainer_state.json").write_text("{}", encoding="utf-8")
            note = output_dir / "notes.txt"
            note.write_text("keep", encoding="utf-8")

            removed = clear_existing_run_artifacts(output_dir)

            self.assertEqual(
                {path.name for path in removed},
                {
                    "checkpoint-1",
                    "model.safetensors",
                    "result_summary.json",
                    "stage2_full_ft",
                    "test_predictions.json",
                },
            )
            self.assertFalse(checkpoint.exists())
            self.assertFalse(stage_dir.exists())
            self.assertFalse((output_dir / "model.safetensors").exists())
            self.assertTrue(note.exists())


if __name__ == "__main__":
    unittest.main()
