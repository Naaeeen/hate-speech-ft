from types import SimpleNamespace
import unittest

from src.methods.frozen_distilbert import training
from src.methods.frozen_distilbert.manual_config import CONFIG as FROZEN_CONFIG


class FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def numel(self):
        return 1


class FakeBackbone:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self


class FakeModel:
    def __init__(self):
        self.training = True
        self.base_model = FakeBackbone()
        self.base_model_prefix = "distilbert"
        self.distilbert = self.base_model
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

    def train(self, mode=True):
        self.training = mode
        self.base_model.training = mode
        return self


class FrozenDistilbertTrainTests(unittest.TestCase):
    def test_frozen_trainability_freezes_backbone_and_trains_head(self):
        model = FakeModel()

        training.set_frozen_backbone_trainability(model)

        state = {name: parameter.requires_grad for name, parameter in model.items}
        self.assertFalse(state["distilbert.embeddings.word_embeddings.weight"])
        self.assertFalse(state["distilbert.transformer.layer.0.attention.q_lin.weight"])
        self.assertTrue(state["pre_classifier.weight"])
        self.assertTrue(state["classifier.bias"])

    def test_frozen_backbone_stays_eval_after_model_train_call(self):
        model = FakeModel()

        training.set_frozen_backbone_trainability(model)
        model.train(True)

        self.assertTrue(model.training)
        self.assertFalse(model.base_model.training)

    def test_manual_config_matches_selected_transformer_contract(self):
        args = SimpleNamespace(**FROZEN_CONFIG)

        self.assertEqual(args.method, "frozen-backbone")
        self.assertEqual(args.model_name, "distilbert-base-uncased")
        self.assertTrue(args.load_best_model_at_end)
        self.assertEqual(args.metric_for_best_model, "eval_f1_macro")
        self.assertEqual(args.head_learning_rate, 3e-4)
        self.assertEqual(args.per_device_train_batch_size, 16)


if __name__ == "__main__":
    unittest.main()
