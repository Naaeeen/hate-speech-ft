import inspect
from types import SimpleNamespace
import unittest

from src.methods.distilbert_full.train import build_experiment_config
from src.methods import transformer_outputs
from src.methods import transformer_runner


class RunDistilbertExperimentConfigTests(unittest.TestCase):
    def test_experiment_config_logs_shared_policy_fields(self):
        args = SimpleNamespace(
            method="full-ft",
            run_name="full_ft_seed42_manual",
            test_split_name="test",
            run_test=False,
            dataset_name="Hate-speech-CNERG/hatexplain",
            model_name="distilbert-base-uncased",
            seed=42,
            data_fraction_seed=42,
            data_fraction=0.2,
            max_train_samples=None,
            max_eval_samples=None,
            max_test_samples=None,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            eval_steps=None,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            lower_is_better=False,
            no_save_final_model=False,
            overwrite_output_dir=False,
            fp16=False,
            mixed_precision="none",
            gradient_checkpointing=False,
            class_weighting="balanced",
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
            max_grad_norm=1.0,
            optim="adamw_torch",
            lr_scheduler_type="linear",
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
            raw_train_size=600,
            raw_eval_size=60,
            dropped_no_majority_train=100,
            dropped_no_majority_eval=10,
            test_size=None,
            full_test_size=None,
            raw_test_size=None,
            dropped_no_majority_test=None,
            trainable_params=1000,
            total_params=2000,
            class_weights=[1.0, 2.0, 0.5],
            precision_policy={"mixed_precision": "none", "fp16": False, "bf16": False},
        )

        self.assertEqual(config["run_name"], "full_ft_seed42_manual")
        self.assertEqual(config["data_fraction"], 0.2)
        self.assertEqual(config["effective_train_fraction"], 0.2)
        self.assertEqual(config["raw_train_size"], 600)
        self.assertEqual(config["full_train_size"], 500)
        self.assertEqual(config["dropped_no_majority_train"], 100)
        self.assertIn("post-load", config["split_accounting_policy"])
        self.assertEqual(config["raw_eval_size"], 60)
        self.assertEqual(config["dropped_no_majority_eval"], 10)
        self.assertEqual(config["selection_metric"], "f1_macro")
        self.assertEqual(config["test_policy"], "enabled_by_run_test")
        self.assertEqual(config["warmup_ratio"], 0.06)
        self.assertEqual(config["output_dir"], "outputs/example")
        self.assertIs(config["run_test"], False)
        self.assertEqual(config["hyperparameters"]["save_strategy"], "epoch")
        self.assertEqual(config["hyperparameters"]["class_weighting"], "balanced")
        self.assertEqual(config["class_weights"], [1.0, 2.0, 0.5])
        self.assertEqual(config["hyperparameters"]["max_grad_norm"], 1.0)
        self.assertEqual(config["hyperparameters"]["mixed_precision"], "none")
        self.assertIs(config["hyperparameters"]["load_best_model_at_end"], True)
        self.assertEqual(config["hyperparameters"]["early_stopping_patience"], 2)

    def test_experiment_config_separates_requested_and_effective_data_fraction(self):
        args = SimpleNamespace(
            method="full-ft",
            run_name="full_ft_smoke",
            test_split_name="test",
            run_test=False,
            dataset_name="Hate-speech-CNERG/hatexplain",
            model_name="distilbert-base-uncased",
            seed=42,
            data_fraction_seed=42,
            data_fraction=None,
            max_train_samples=64,
            max_eval_samples=64,
            max_test_samples=None,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            eval_steps=None,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            lower_is_better=False,
            no_save_final_model=False,
            overwrite_output_dir=False,
            fp16=False,
            mixed_precision="none",
            gradient_checkpointing=False,
            class_weighting="none",
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
            max_grad_norm=1.0,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            max_length=128,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            output_dir="outputs/smoke",
        )

        config = build_experiment_config(
            args,
            train_split="train",
            eval_split="validation",
            train_size=64,
            eval_size=64,
            full_train_size=500,
            full_eval_size=50,
            trainable_params=1000,
            total_params=2000,
            precision_policy={"mixed_precision": "none", "fp16": False, "bf16": False},
        )

        self.assertIsNone(config["data_fraction"])
        self.assertEqual(config["effective_train_fraction"], 64 / 500)
        self.assertEqual(config["hyperparameters"]["data_fraction"], None)

    def test_result_summary_is_written_after_final_model_save(self):
        source = inspect.getsource(transformer_runner.run_single_stage_transformer)

        self.assertLess(source.index("save_final_model("), source.index("write_success_outputs("))

        save_source = inspect.getsource(transformer_outputs.save_final_model)
        self.assertIn("trainer.save_model", save_source)
        self.assertIn("tokenizer.save_pretrained", save_source)

    def test_wandb_run_starts_after_full_config_is_built(self):
        source = inspect.getsource(transformer_runner.run_single_stage_transformer)

        self.assertLess(source.index("start_hf_run"), source.index("prepare_hf_classification_run"))
        self.assertLess(source.index("build_experiment_config("), source.index("init_wandb_run("))
        self.assertLess(source.index("init_wandb_run("), source.index("write_config_snapshot"))

if __name__ == "__main__":
    unittest.main()
