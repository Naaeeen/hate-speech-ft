import json
from types import SimpleNamespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.methods.tfidf_logreg import config as tfidf_config
from src.methods.tfidf_logreg import data as tfidf_data
from src.methods.tfidf_logreg import train
from src.methods.tfidf_logreg.manual_config import CONFIG as TFIDF_CONFIG
from src.methods.tfidf_logreg.training import (
    build_classification_metrics,
    parse_ngram_range,
)
from src.utils.wandb_config import build_wandb_settings_from_args


def make_example(post_id: str, label: int) -> dict:
    return {
        "id": post_id,
        "post_tokens": ["sample", post_id],
        "annotators": {
            "label": [label, label, (label + 1) % 3],
            "annotator_id": [1, 2, 3],
            "target": [[], [], []],
        },
        "rationales": [],
    }


class FakeDataset(dict):
    def __init__(self):
        super().__init__(
            {
                "train": [
                    make_example("train-0", 0),
                    make_example("train-1", 1),
                    make_example("train-2", 2),
                ],
                "validation": [
                    make_example("eval-0", 0),
                    make_example("eval-1", 1),
                    make_example("eval-2", 2),
                ],
                "test": [
                    make_example("test-0", 0),
                    make_example("test-1", 1),
                    make_example("test-2", 2),
                ],
            }
        )
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        return super().__getitem__(key)


class FakeTfidfVectorizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.vocabulary_ = {"sample": 0, "text": 1}


class FakeLogisticRegression:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.coef_ = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        self.intercept_ = [0.1, 0.2, 0.3]


class FakePipeline:
    def __init__(self, steps):
        self.named_steps = dict(steps)

    def fit(self, _texts, _labels):
        return self

    def predict(self, texts):
        return [index % 3 for index, _text in enumerate(texts)]

    def predict_proba(self, texts):
        return [[0.8, 0.1, 0.1] for _text in texts]


def fake_dump(_pipeline, path):
    Path(path).write_text("fake model", encoding="utf-8")


class FakeWandbConfig:
    def __init__(self):
        self.updates = []

    def update(self, payload, allow_val_change=False):
        self.updates.append((payload, allow_val_change))


class FakeWandbRun:
    def __init__(self):
        self.config = FakeWandbConfig()
        self.logs = []
        self.finished = False

    def log(self, payload):
        self.logs.append(payload)

    def finish(self):
        self.finished = True


class TfidfLogregTrainTests(unittest.TestCase):
    def test_manual_config_matches_selected_tfidf_run(self):
        args = SimpleNamespace(**TFIDF_CONFIG)

        self.assertEqual(args.method, "tfidf-logreg")
        self.assertEqual(args.run_name, "tfidf_logreg_final_seed42")
        self.assertEqual(args.output_dir, "outputs/tfidf_logreg_final_seed42")
        self.assertEqual(args.ngram_range, [1, 2])
        self.assertEqual(args.max_df, 0.9)
        self.assertTrue(args.sublinear_tf)

    def test_config_and_data_modules_record_shared_contract(self):
        args = SimpleNamespace(**{**TFIDF_CONFIG, "ngram_range": [1, 2], "use_wandb": False})
        split = tfidf_data.build_classical_split(
            [
                make_example("sample-0", 0),
                make_example("sample-1", 1),
            ],
            max_samples=1,
        )

        settings = build_wandb_settings_from_args(args)
        config = tfidf_config.build_experiment_config(
            args,
            ngram_range=parse_ngram_range(args.ngram_range),
            train_split="train",
            eval_split="validation",
            train_data=split,
            eval_data=split,
            gpu_type="cpu",
        )

        self.assertFalse(settings.enabled)
        self.assertEqual(config["model_name"], "tfidf-logreg")
        self.assertEqual(config["train_size"], 1)
        self.assertEqual(config["raw_train_size"], 2)
        self.assertEqual(config["dropped_no_majority_train"], 0)
        self.assertEqual(config["hyperparameters"]["ngram_range"], [1, 2])
        self.assertEqual(config["hyperparameters"]["max_df"], 0.9)
        self.assertTrue(config["hyperparameters"]["sublinear_tf"])
        self.assertEqual(config["test_policy"], "enabled_by_run_test")

    def test_tfidf_vectorizer_receives_manual_options(self):
        args = SimpleNamespace(
            **{
                **TFIDF_CONFIG,
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.9,
                "max_features": 20000,
                "sublinear_tf": True,
            }
        )

        pipeline = train.build_pipeline(
            TfidfVectorizer=FakeTfidfVectorizer,
            LogisticRegression=FakeLogisticRegression,
            Pipeline=FakePipeline,
            args=args,
            ngram_range=parse_ngram_range(args.ngram_range),
        )

        vectorizer = pipeline.named_steps["tfidf"]
        self.assertEqual(vectorizer.kwargs["max_df"], 0.9)
        self.assertEqual(vectorizer.kwargs["max_features"], 20000)
        self.assertTrue(vectorizer.kwargs["sublinear_tf"])

    def test_parse_ngram_range_uses_manual_config_lists_only(self):
        self.assertEqual(parse_ngram_range([1, 3]), (1, 3))
        with self.assertRaisesRegex(TypeError, "Python list"):
            parse_ngram_range("1,2")

        with self.assertRaises(ValueError):
            parse_ngram_range([2, 1])

    def test_rejects_invalid_data_fraction_before_dataset_load(self):
        args = SimpleNamespace(**{**TFIDF_CONFIG, "data_fraction": 0})

        with self.assertRaisesRegex(ValueError, "data_fraction"):
            train.validate_classical_args(args, parse_ngram_range(args.ngram_range))

    def test_rejects_invalid_max_df_before_dataset_load(self):
        args = SimpleNamespace(**{**TFIDF_CONFIG, "max_df": 0})

        with self.assertRaisesRegex(ValueError, "max_df"):
            train.validate_classical_args(args, parse_ngram_range(args.ngram_range))

    def test_classification_metrics_use_shared_key_names(self):
        metrics = build_classification_metrics(
            [0, 1, 2],
            [0, 2, 2],
            prefix="eval",
            label_id_to_name={0: "hatespeech", 1: "normal", 2: "offensive"},
        )

        self.assertIn("eval_f1_macro", metrics)
        self.assertIn("eval_accuracy", metrics)
        self.assertIn("eval_precision_hatespeech", metrics)
        self.assertEqual(metrics["eval_support_normal"], 1)

    def test_runtime_metrics_do_not_count_gpu_hours_for_cpu_baseline(self):
        runtime = tfidf_config.build_runtime_metrics(
            training_time_sec=60.0,
            gpu_type="NVIDIA A100-SXM4-80GB",
        )

        self.assertEqual(runtime["gpu_type"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(runtime["compute_device"], "cpu")
        self.assertAlmostEqual(runtime["training_time_hours"], 60.0 / 3600)
        self.assertIsNone(runtime["gpu_hours"])
        self.assertIsNone(runtime["peak_memory_mb"])
        self.assertIsNone(runtime["peak_memory_reserved_mb"])

    def run_fake_main(
        self,
        output_dir,
        *,
        run_name="tfidf_manual",
        run_test=False,
        use_wandb=False,
    ):
        dataset = FakeDataset()

        def fake_load_dataset(_name):
            return dataset

        fake_libraries = (
            fake_load_dataset,
            fake_dump,
            FakeTfidfVectorizer,
            FakeLogisticRegression,
            FakePipeline,
        )
        run_config = {
            **TFIDF_CONFIG,
            "run_name": run_name,
            "output_dir": str(output_dir),
            "ngram_range": [1, 2],
            "min_df": 1,
            "run_test": run_test,
            "use_wandb": use_wandb,
            "wandb_mode": "disabled" if use_wandb else "online",
        }

        with (
            patch.object(train, "MANUAL_CONFIG", run_config),
            patch.object(train, "load_libraries", return_value=fake_libraries),
            patch.object(train, "get_gpu_type", return_value="cpu"),
        ):
            train.main()
        return dataset

    def test_tuning_run_does_not_touch_test_split(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset = self.run_fake_main(output_dir, run_name="tfidf_validation")

            self.assertNotIn("test", dataset.accessed)
            summary = json.loads(
                (output_dir / "result_summary.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(summary["metrics"]["test"])
            self.assertEqual(summary["runtime"]["training_time_sec"] >= 0, True)
            self.assertIn("eval_f1_macro", summary["metrics"]["eval"])

    def test_final_run_saves_test_predictions_and_model(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset = self.run_fake_main(
                output_dir,
                run_name="tfidf_test",
                run_test=True,
            )

            self.assertIn("test", dataset.accessed)
            self.assertTrue((output_dir / "model.joblib").is_file())
            self.assertTrue((output_dir / "eval_predictions.json").is_file())
            self.assertTrue((output_dir / "test_predictions.json").is_file())
            summary = json.loads(
                (output_dir / "result_summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("test_f1_macro", summary["metrics"]["test"])
            self.assertEqual(
                summary["artifacts"]["predictions"]["test"],
                (output_dir / "test_predictions.json").as_posix(),
            )

    def test_success_run_writes_outputs_and_logs_final_metrics_to_wandb(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fake_run = FakeWandbRun()
            dataset = FakeDataset()

            def fake_load_dataset(_name):
                return dataset

            fake_libraries = (
                fake_load_dataset,
                fake_dump,
                FakeTfidfVectorizer,
                FakeLogisticRegression,
                FakePipeline,
            )
            run_config = {
                **TFIDF_CONFIG,
                "run_name": "tfidf_wandb_success",
                "output_dir": str(output_dir),
                "use_wandb": True,
                "wandb_mode": "disabled",
                "run_test": False,
            }

            with (
                patch.object(train, "MANUAL_CONFIG", run_config),
                patch.object(train, "load_libraries", return_value=fake_libraries),
                patch.object(train, "init_wandb_run", return_value=fake_run),
                patch.object(train, "get_gpu_type", return_value="cpu"),
            ):
                train.main()

            self.assertTrue((output_dir / "result_summary.json").is_file())
            self.assertTrue((output_dir / "model.joblib").is_file())
            self.assertTrue(
                any(payload.get("eval_f1_macro") == 1.0 for payload in fake_run.logs)
            )
            self.assertTrue(
                any(
                    payload.get("model_selection/best_metric") == 1.0
                    for payload in fake_run.logs
                )
            )
            self.assertTrue(fake_run.finished)


if __name__ == "__main__":
    unittest.main()
