import sys
import unittest
from unittest.mock import patch

from src.methods import peft_utils
from src.utils.wandb_config import WandbSettings


class FakeTensor:
    def __init__(self, name):
        self.name = name

    def clone(self):
        return FakeTensor(f"{self.name}.clone")


class FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def numel(self):
        return 1


class FakeModel:
    def __init__(self):
        self.items = [
            ("distilbert.transformer.layer.0.attention.q_lin.weight", FakeParameter()),
            ("pre_classifier.weight", FakeParameter()),
            ("classifier.bias", FakeParameter()),
        ]
        self.loaded_state = None

    def named_parameters(self):
        return list(self.items)

    def parameters(self):
        return [parameter for _, parameter in self.items]

    def state_dict(self):
        return {
            "distilbert.transformer.layer.0.attention.q_lin.weight": FakeTensor("q"),
            "pre_classifier.weight": FakeTensor("pre"),
            "classifier.bias": FakeTensor("cls"),
        }

    def load_state_dict(self, state, strict=False):
        self.loaded_state = dict(state)
        return [], []


class DistilbertEfficientHeadTrainTests(unittest.TestCase):
    def test_extract_and_load_classification_head_only(self):
        source = FakeModel()
        target = FakeModel()

        head_state = peft_utils.extract_classification_head_state_dict(source)
        peft_utils.load_classification_head_state_dict(target, head_state)

        self.assertEqual(set(head_state), {"pre_classifier.weight", "classifier.bias"})
        self.assertEqual(set(target.loaded_state), set(head_state))

    def test_load_classification_head_rejects_partial_transfer(self):
        target = FakeModel()

        with self.assertRaisesRegex(ValueError, "Missing classification-head keys"):
            peft_utils.load_classification_head_state_dict(
                target,
                {"classifier.bias": FakeTensor("cls")},
            )

    def test_stage1_lora_rejects_modules_to_save_without_full_head_before_training(self):
        from src.methods.distilbert_efficient_head import training

        context = type("FakeContext", (), {"model": FakeModel()})()
        args = type("FakeArgs", (), {"stage1_modules_to_save": "classifier"})()

        with patch.object(training, "apply_lora_to_model") as apply_lora:
            with self.assertRaisesRegex(ValueError, "pre_classifier"):
                training.apply_stage1_lora_to_context(context, args)

        apply_lora.assert_not_called()

    def test_efficient_head_parser_defaults_match_two_stage_policy(self):
        from src.methods.distilbert_efficient_head import args as eh_args

        with patch.object(sys, "argv", ["prog"]):
            args = eh_args.parse_args()

        self.assertEqual(args.method, "efficient-head-ft")
        self.assertEqual(args.model_name, "distilbert-base-uncased")
        self.assertEqual(args.stage1_learning_rate, 3e-4)
        self.assertEqual(args.stage1_epochs, 5)
        self.assertEqual(args.stage2_learning_rate, 2e-5)
        self.assertEqual(args.stage2_epochs, 5)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")
        self.assertEqual(
            peft_utils.parse_module_names(args.stage1_target_modules),
            ["q_lin", "k_lin", "v_lin"],
        )

    def test_efficient_head_main_saves_outputs_from_stage2_trainer(self):
        import src.methods.distilbert_efficient_head.train as eh_train
        from src.methods.distilbert_efficient_head import args as eh_args

        calls = []

        class FakeTrainer:
            def __init__(self, name):
                self.name = name
                self.model = FakeModel()

            def train(self):
                calls.append((self.name, "train"))

            def evaluate(self, metric_key_prefix="eval", eval_dataset=None):
                calls.append((self.name, "evaluate", metric_key_prefix, eval_dataset))
                return {f"{metric_key_prefix}_f1_macro": 0.5}

        with patch.object(sys, "argv", ["prog"]):
            run_args = eh_args.parse_args()
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

            def __init__(self, model=None):
                self.args = run_args
                self.model = model or FakeModel()
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

        with patch.object(eh_train, "parse_args", return_value=run_args), patch.object(
            eh_train,
            "initialize_hf_run",
            return_value=setup,
        ), patch.object(
            eh_train,
            "start_hf_run",
            return_value=(setup.precision_policy, {"setup_complete": False}, None),
        ), patch.object(
            eh_train,
            "prepare_hf_classification_run",
            return_value=FakeContext(),
        ), patch.object(
            eh_train,
            "apply_stage1_lora_to_context",
            side_effect=lambda context, args: context,
        ), patch.object(
            eh_train,
            "build_stage2_context",
            return_value=FakeContext(),
        ), patch.object(
            eh_train,
            "count_model_parameters",
            side_effect=[(4, 4), (2, 6)],
        ), patch.object(
            eh_train,
            "build_experiment_config",
            return_value={"setup_complete": True},
        ) as build_experiment_config, patch.object(
            eh_train,
            "write_config_snapshot",
        ), patch.object(
            eh_train,
            "build_stage_training_arguments",
            side_effect=["stage1_args", "stage2_args"],
        ) as build_stage_training_arguments, patch.object(
            eh_train,
            "build_hf_trainer",
            side_effect=trainers,
        ), patch.object(
            eh_train,
            "build_callbacks",
            return_value=[],
        ), patch.object(
            eh_train,
            "build_model_selection_summary",
            return_value={"best_metric": 0.5},
        ), patch.object(
            eh_train,
            "save_final_model",
        ) as save_final_model, patch.object(
            eh_train,
            "save_final_predictions",
            return_value={},
        ) as save_final_predictions, patch.object(
            eh_train,
            "build_runtime_metrics",
            return_value={"status": "completed"},
        ), patch.object(
            eh_train,
            "write_success_outputs",
            return_value={},
        ) as write_success_outputs, patch.object(
            eh_train,
            "print_run_report",
        ), patch.object(
            eh_train,
            "finish_wandb_run",
        ), patch.object(
            eh_train,
            "synchronize_cuda",
        ):
            eh_train.main()

        self.assertIn(("stage1", "train"), calls)
        self.assertIn(("stage2", "train"), calls)
        build_config_kwargs = build_experiment_config.call_args.kwargs
        self.assertEqual(build_config_kwargs["stage1_trainable_params"], 2)
        self.assertEqual(build_config_kwargs["stage2_trainable_params"], 4)
        self.assertEqual(build_config_kwargs["total_params"], 4)
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
