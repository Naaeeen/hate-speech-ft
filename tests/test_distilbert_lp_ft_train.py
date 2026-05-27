import sys
import unittest
from unittest.mock import patch

import src.methods.distilbert_lp_ft.train as lp_train
from src.methods.distilbert_lp_ft import args as lp_args
from src.methods.distilbert_lp_ft import training
from src.utils.wandb_config import WandbSettings


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


class DistilbertLpFtTrainTests(unittest.TestCase):
    def test_linear_probe_trainability_freezes_backbone_and_trains_head(self):
        model = FakeModel()

        training.set_linear_probe_trainability(model)

        state = {name: parameter.requires_grad for name, parameter in model.items}
        self.assertFalse(state["distilbert.embeddings.word_embeddings.weight"])
        self.assertFalse(state["distilbert.transformer.layer.0.attention.q_lin.weight"])
        self.assertTrue(state["pre_classifier.weight"])
        self.assertTrue(state["classifier.bias"])

    def test_full_finetune_trainability_unfreezes_everything(self):
        model = FakeModel()
        training.set_linear_probe_trainability(model)

        training.set_full_finetune_trainability(model)

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_parser_supports_legacy_batch_size_alias(self):
        with patch.object(sys, "argv", ["prog", "--batch_size", "16"]):
            args = lp_args.parse_args()

        self.assertEqual(training.resolve_train_batch_size(args), 16)
        self.assertEqual(training.resolve_eval_batch_size(args), 16)

    def test_parser_defaults_are_compatible_with_early_stopping_policy(self):
        with patch.object(sys, "argv", ["prog"]):
            args = lp_args.parse_args()

        self.assertTrue(args.load_best_model_at_end)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")

    def test_lp_ft_main_saves_final_outputs_from_stage2_trainer(self):
        calls = []

        class FakeTrainer:
            def __init__(self, name):
                self.name = name

            def train(self):
                calls.append((self.name, "train"))

            def evaluate(self, metric_key_prefix="eval", eval_dataset=None):
                calls.append((self.name, "evaluate", metric_key_prefix, eval_dataset))
                return {f"{metric_key_prefix}_f1_macro": 0.5}

        with patch.object(sys, "argv", ["prog"]):
            run_args = lp_args.parse_args()
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

            def __init__(self, args):
                self.args = args
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
                "wandb_settings": WandbSettings(enabled=True, project="unit-test"),
            },
        )()
        trainers = [FakeTrainer("stage1"), FakeTrainer("stage2")]

        with patch.object(lp_train, "parse_args", return_value=run_args), patch.object(
            lp_train,
            "initialize_hf_run",
            return_value=setup,
        ), patch.object(
            lp_train,
            "start_hf_run",
            return_value=(setup.precision_policy, {"setup_complete": False}, None),
        ), patch.object(
            lp_train,
            "prepare_hf_classification_run",
            return_value=FakeContext(run_args),
        ), patch.object(
            lp_train,
            "build_experiment_config",
            return_value={"setup_complete": True},
        ), patch.object(
            lp_train,
            "write_config_snapshot",
        ), patch.object(
            lp_train,
            "build_callbacks",
            return_value=[],
        ), patch.object(
            lp_train,
            "build_stage_training_arguments",
            side_effect=["stage1_args", "stage2_args"],
        ) as build_stage_training_arguments, patch.object(
            lp_train,
            "build_hf_trainer",
            side_effect=trainers,
        ), patch.object(
            lp_train,
            "build_model_selection_summary",
            return_value={"best_metric": 0.5},
        ), patch.object(
            lp_train,
            "save_final_model",
        ) as save_final_model, patch.object(
            lp_train,
            "save_final_predictions",
            return_value={},
        ) as save_final_predictions, patch.object(
            lp_train,
            "build_runtime_metrics",
            return_value={"status": "completed"},
        ), patch.object(
            lp_train,
            "write_success_outputs",
            return_value={},
        ) as write_success_outputs, patch.object(
            lp_train,
            "print_run_report",
        ), patch.object(
            lp_train,
            "finish_wandb_run",
        ), patch.object(
            lp_train,
            "synchronize_cuda",
        ):
            lp_train.main()

        self.assertIn(("stage1", "train"), calls)
        self.assertIn(("stage2", "train"), calls)
        save_final_model.assert_called_once()
        self.assertIs(save_final_model.call_args.args[0], trainers[1])
        save_final_predictions.assert_called_once()
        self.assertIs(save_final_predictions.call_args.args[1], trainers[1])
        write_success_outputs.assert_called_once()
        self.assertEqual(build_stage_training_arguments.call_count, 2)
        for call_args in build_stage_training_arguments.call_args_list:
            self.assertFalse(call_args.kwargs["wandb_settings"].enabled)
            self.assertEqual(call_args.kwargs["wandb_settings"].project, "unit-test")


if __name__ == "__main__":
    unittest.main()
