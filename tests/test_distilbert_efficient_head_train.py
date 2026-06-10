from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.methods import peft_utils
from src.methods.distilbert_efficient_head.manual_config import (
    CONFIG as EFFICIENT_HEAD_CONFIG,
)


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

        with patch.object(peft_utils, "apply_lora_to_model") as apply_lora:
            with self.assertRaisesRegex(TypeError, "manual_config"):
                training.apply_stage1_lora_to_context(context, args)

        apply_lora.assert_not_called()

    def test_efficient_head_manual_config_matches_two_stage_policy(self):
        args = SimpleNamespace(**EFFICIENT_HEAD_CONFIG)

        self.assertEqual(args.method, "efficient-head-ft")
        self.assertEqual(args.model_name, "distilbert-base-uncased")
        self.assertEqual(args.stage1_learning_rate, 2e-4)
        self.assertEqual(args.stage1_epochs, 5)
        self.assertEqual(args.stage2_learning_rate, 2e-5)
        self.assertEqual(args.stage2_epochs, 3)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")
        self.assertEqual(
            peft_utils.parse_module_names(args.stage1_target_modules),
            ["q_lin", "v_lin"],
        )

if __name__ == "__main__":
    unittest.main()
