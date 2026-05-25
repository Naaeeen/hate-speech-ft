import sys
import unittest
from unittest.mock import patch

import src.methods.peft_utils as peft_utils


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def numel(self):
        return 1


class FakeModel:
    def __init__(self):
        self.items = [
            ("distilbert.transformer.layer.0.attention.q_lin.weight", FakeParameter()),
            ("pre_classifier.weight", FakeParameter()),
            ("classifier.bias", FakeParameter()),
        ]

    def named_parameters(self):
        return list(self.items)

    def parameters(self):
        return [parameter for _, parameter in self.items]


class DistilbertLoraTrainTests(unittest.TestCase):
    def test_parse_module_names_accepts_json_and_csv(self):
        self.assertEqual(
            peft_utils.parse_module_names('["q_lin","v_lin"]'),
            ["q_lin", "v_lin"],
        )
        self.assertEqual(
            peft_utils.parse_module_names("q_lin,k_lin,v_lin"),
            ["q_lin", "k_lin", "v_lin"],
        )

    def test_parse_module_names_rejects_empty_values(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            peft_utils.parse_module_names("[]")

    def test_lora_parser_defaults_match_aaron_stage1_policy(self):
        from src.methods.distilbert_lora import args as lora_args

        with patch.object(sys, "argv", ["prog"]):
            args = lora_args.parse_args()

        self.assertEqual(args.method, "lora")
        self.assertEqual(args.model_name, "distilbert-base-uncased")
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.num_train_epochs, 5)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")
        self.assertEqual(args.mixed_precision, "none")
        self.assertEqual(
            peft_utils.parse_module_names(args.target_modules),
            ["q_lin", "k_lin", "v_lin"],
        )
        self.assertEqual(
            peft_utils.parse_module_names(args.modules_to_save),
            ["pre_classifier", "classifier"],
        )

    def test_lora_main_uses_shared_hf_workflow_and_peft_model(self):
        import src.methods.distilbert_lora.train as lora_train
        from src.methods.distilbert_lora import args as lora_args

        calls = []

        class FakeTrainer:
            def train(self):
                calls.append("train")

        with patch.object(sys, "argv", ["prog"]):
            run_args = lora_args.parse_args()
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

        with patch.object(lora_train, "parse_args", return_value=run_args), patch.object(
            lora_train,
            "initialize_hf_run",
            return_value=setup,
        ), patch.object(
            lora_train,
            "start_hf_run",
            return_value=(setup.precision_policy, {"setup_complete": False}, None),
        ), patch.object(
            lora_train,
            "prepare_hf_classification_run",
            return_value=FakeContext(),
        ), patch.object(
            lora_train,
            "apply_lora_to_context",
            side_effect=lambda context, args: context,
        ) as apply_lora, patch.object(
            lora_train,
            "count_model_parameters",
            return_value=(2, 4),
        ), patch.object(
            lora_train,
            "build_experiment_config",
            return_value={"setup_complete": True},
        ), patch.object(
            lora_train,
            "write_config_snapshot",
        ), patch.object(
            lora_train,
            "build_hf_training_arguments_from_args",
            return_value="training_args",
        ), patch.object(
            lora_train,
            "build_early_stopping_callbacks",
            return_value=[],
        ), patch.object(
            lora_train,
            "build_hf_trainer",
            return_value=trainer,
        ), patch.object(
            lora_train,
            "evaluate_validation_and_optional_test",
            return_value=({"eval_f1_macro": 0.5}, {"test_f1_macro": 0.4}),
        ), patch.object(
            lora_train,
            "build_model_selection_summary",
            return_value={"best_metric": 0.5},
        ), patch.object(
            lora_train,
            "save_final_model",
            return_value={},
        ), patch.object(
            lora_train,
            "save_final_predictions",
            return_value={},
        ) as save_final_predictions, patch.object(
            lora_train,
            "build_runtime_metrics",
            return_value={"status": "completed"},
        ), patch.object(
            lora_train,
            "write_success_outputs",
            return_value={},
        ) as write_success_outputs, patch.object(
            lora_train,
            "print_run_report",
        ), patch.object(
            lora_train,
            "finish_wandb_run",
        ), patch.object(
            lora_train,
            "synchronize_cuda",
        ):
            lora_train.main()

        self.assertIn("train", calls)
        apply_lora.assert_called_once()
        save_final_predictions.assert_called_once()
        write_success_outputs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
