import sys
import unittest
from unittest.mock import patch

import src.methods.frozen_distilbert.train as frozen_train
from src.methods.frozen_distilbert import args as frozen_args
from src.methods.frozen_distilbert import training


class FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def numel(self):
        return 1


class FakeModel:
    def __init__(self):
        self.items = [
            ("distilbert.embeddings.word_embeddings.weight", FakeParameter()),
            ("distilbert.transformer.layer.0.attention.q_lin.weight", FakeParameter()),
            ("pre_classifier.weight", FakeParameter()),
            ("classifier.bias", FakeParameter()),
        ]

    def named_parameters(self):
        return list(self.items)

    def parameters(self):
        return [parameter for _, parameter in self.items]


class FrozenDistilbertTrainTests(unittest.TestCase):
    def test_frozen_trainability_freezes_backbone_and_trains_head(self):
        model = FakeModel()

        training.set_frozen_backbone_trainability(model)

        state = {name: parameter.requires_grad for name, parameter in model.items}
        self.assertFalse(state["distilbert.embeddings.word_embeddings.weight"])
        self.assertFalse(state["distilbert.transformer.layer.0.attention.q_lin.weight"])
        self.assertTrue(state["pre_classifier.weight"])
        self.assertTrue(state["classifier.bias"])

    def test_parser_defaults_match_shared_hf_contract(self):
        with patch.object(sys, "argv", ["prog"]):
            args = frozen_args.parse_args()

        self.assertEqual(args.method, "frozen-backbone")
        self.assertEqual(args.model_name, "distilbert-base-uncased")
        self.assertTrue(args.load_best_model_at_end)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")
        self.assertEqual(args.head_learning_rate, 1e-4)
        self.assertEqual(args.per_device_train_batch_size, 8)

    def test_main_uses_shared_hf_workflow_and_frozen_model(self):
        calls = []

        class FakeTrainer:
            def train(self):
                calls.append("train")

        with patch.object(sys, "argv", ["prog"]):
            run_args = frozen_args.parse_args()
        run_args.run_test = True
        run_args.search_stage = "final"
        run_args.output_dir = "outputs/unit-test"

        class FakeContext:
            libraries = type(
                "FakeLibraries",
                (),
                {
                    "training_args_cls": object,
                    "early_stopping_callback_cls": object,
                },
            )()

            def __init__(self):
                self.model = FakeModel()
                self.tokenizer = object()
                self.test_dataset = [{"labels": 1}]

            def config_kwargs(self):
                return {
                    "train_split": "train",
                    "eval_split": "validation",
                    "train_size": 1,
                    "eval_size": 1,
                    "full_train_size": 1,
                    "full_eval_size": 1,
                    "raw_train_size": 1,
                    "raw_eval_size": 1,
                    "dropped_no_majority_train": 0,
                    "dropped_no_majority_eval": 0,
                    "test_size": 1,
                    "full_test_size": 1,
                    "raw_test_size": 1,
                    "dropped_no_majority_test": 0,
                    "gpu_type": "cpu",
                    "class_weights": None,
                    "precision_policy": {
                        "mixed_precision": "none",
                        "fp16": False,
                        "bf16": False,
                    },
                }

        setup = type(
            "FakeSetup",
            (),
            {
                "gpu_type": "cpu",
                "precision_policy": {
                    "mixed_precision": "none",
                    "fp16": False,
                    "bf16": False,
                },
                "experiment_config": {"setup_complete": False},
                "wandb_settings": object(),
            },
        )()
        trainer = FakeTrainer()

        with patch.object(frozen_train, "parse_args", return_value=run_args), patch.object(
            frozen_train,
            "initialize_hf_run",
            return_value=setup,
        ), patch.object(
            frozen_train,
            "start_hf_run",
            return_value=(setup.precision_policy, {"setup_complete": False}, None),
        ), patch.object(
            frozen_train,
            "prepare_hf_classification_run",
            return_value=FakeContext(),
        ), patch.object(
            frozen_train,
            "count_model_parameters",
            return_value=(2, 4),
        ), patch.object(
            frozen_train,
            "build_experiment_config",
            return_value={"setup_complete": True},
        ), patch.object(
            frozen_train,
            "write_config_snapshot",
        ), patch.object(
            frozen_train,
            "build_hf_training_arguments_from_args",
            return_value="training_args",
        ), patch.object(
            frozen_train,
            "build_early_stopping_callbacks",
            return_value=[],
        ), patch.object(
            frozen_train,
            "build_hf_trainer",
            return_value=trainer,
        ), patch.object(
            frozen_train,
            "evaluate_validation_and_optional_test",
            return_value=({"eval_f1_macro": 0.5}, {"test_f1_macro": 0.4}),
        ), patch.object(
            frozen_train,
            "build_model_selection_summary",
            return_value={"best_metric": 0.5},
        ), patch.object(
            frozen_train,
            "save_final_model",
            return_value={},
        ), patch.object(
            frozen_train,
            "save_final_predictions",
            return_value={},
        ) as save_final_predictions, patch.object(
            frozen_train,
            "build_runtime_metrics",
            return_value={"status": "completed"},
        ), patch.object(
            frozen_train,
            "write_success_outputs",
            return_value={},
        ) as write_success_outputs, patch.object(
            frozen_train,
            "print_run_report",
        ), patch.object(
            frozen_train,
            "finish_wandb_run",
        ), patch.object(
            frozen_train,
            "synchronize_cuda",
        ):
            frozen_train.main()

        self.assertIn("train", calls)
        save_final_predictions.assert_called_once()
        write_success_outputs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
