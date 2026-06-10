from types import SimpleNamespace
import unittest
from unittest.mock import patch

import src.methods.peft_utils as peft_utils
from src.methods.distilbert_lora.manual_config import CONFIG as LORA_CONFIG


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

    def state_dict(self):
        return {
            "distilbert.transformer.layer.0.attention.q_lin.weight": object(),
            "pre_classifier.weight": object(),
            "classifier.bias": object(),
        }


class DistilbertLoraTrainTests(unittest.TestCase):
    def test_parse_module_names_accepts_manual_config_lists(self):
        self.assertEqual(
            peft_utils.parse_module_names(["q_lin", "v_lin"]),
            ["q_lin", "v_lin"],
        )
        with self.assertRaisesRegex(TypeError, "manual_config"):
            peft_utils.parse_module_names("q_lin,k_lin,v_lin")

    def test_parse_module_names_rejects_empty_values(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            peft_utils.parse_module_names([])

    def test_lora_manual_config_matches_selected_policy(self):
        args = SimpleNamespace(**LORA_CONFIG)

        self.assertEqual(args.method, "lora")
        self.assertEqual(args.model_name, "distilbert-base-uncased")
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.num_train_epochs, 4)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")
        self.assertEqual(args.mixed_precision, "none")
        self.assertEqual(
            peft_utils.parse_module_names(args.target_modules),
            ["q_lin", "k_lin", "v_lin", "out_lin"],
        )
        self.assertEqual(
            peft_utils.parse_module_names(args.modules_to_save),
            ["pre_classifier", "classifier"],
        )

    def test_lora_rejects_modules_to_save_without_full_head_before_training(self):
        from src.methods.distilbert_lora import training

        context = type("FakeContext", (), {"model": FakeModel()})()
        args = type("FakeArgs", (), {"modules_to_save": ["classifier"]})()

        with patch.object(peft_utils, "apply_lora_to_model") as apply_lora:
            with self.assertRaisesRegex(ValueError, "pre_classifier"):
                training.apply_lora_to_context(context, args)

        apply_lora.assert_not_called()

if __name__ == "__main__":
    unittest.main()
