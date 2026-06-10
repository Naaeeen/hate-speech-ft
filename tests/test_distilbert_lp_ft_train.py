from types import SimpleNamespace
import unittest

from src.methods.distilbert_lp_ft import training
from src.methods.distilbert_lp_ft.manual_config import CONFIG as LP_FT_CONFIG


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

    def test_manual_config_is_compatible_with_early_stopping_policy(self):
        args = SimpleNamespace(**LP_FT_CONFIG)

        self.assertTrue(args.load_best_model_at_end)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")


if __name__ == "__main__":
    unittest.main()
