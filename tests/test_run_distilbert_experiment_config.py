import argparse
import unittest

from src.run_distilbert_hatexplain import build_experiment_config


class RunDistilbertExperimentConfigTests(unittest.TestCase):
    def test_experiment_config_logs_shared_policy_fields(self):
        args = argparse.Namespace(
            method="full-ft",
            search_stage="tuning",
            trial_id="trial-003",
            hpo_seed=2026,
            test_split_name="test",
            run_test=False,
            dataset_name="Hate-speech-CNERG/hatexplain",
            model_name="distilbert-base-uncased",
            seed=42,
            data_fraction=0.2,
            max_train_samples=None,
            max_eval_samples=None,
            max_test_samples=None,
            max_length=128,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            output_dir="outputs/example",
        )

        config = build_experiment_config(
            args,
            train_split="train",
            eval_split="validation",
            train_size=100,
            eval_size=50,
            full_train_size=500,
            full_eval_size=50,
            test_size=None,
            full_test_size=None,
            trainable_params=1000,
            total_params=2000,
        )

        self.assertEqual(config["search_stage"], "tuning")
        self.assertEqual(config["trial_id"], "trial-003")
        self.assertEqual(config["hpo_seed"], 2026)
        self.assertEqual(config["data_fraction"], 0.2)
        self.assertEqual(config["selection_metric"], "f1_macro")
        self.assertEqual(config["test_policy"], "final_only")
        self.assertEqual(config["warmup_ratio"], 0.06)
        self.assertEqual(config["output_dir"], "outputs/example")
        self.assertIs(config["run_test"], False)

    def test_test_evaluation_policy_blocks_non_final_runs(self):
        from src.run_distilbert_hatexplain import validate_test_evaluation_policy

        with self.assertRaises(ValueError):
            validate_test_evaluation_policy(search_stage="tuning", run_test=True)

        validate_test_evaluation_policy(search_stage="final", run_test=True)


if __name__ == "__main__":
    unittest.main()
