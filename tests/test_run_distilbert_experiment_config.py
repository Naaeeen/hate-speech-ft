import argparse
import unittest

from src.run_distilbert_hatexplain import build_experiment_config, build_setup_failure_config


class RunDistilbertExperimentConfigTests(unittest.TestCase):
    def test_experiment_config_logs_shared_policy_fields(self):
        args = argparse.Namespace(
            method="full-ft",
            search_stage="tuning",
            trial_id="trial-003",
            config_hash="abc123",
            hpo_seed=2026,
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
            fp16=False,
            mixed_precision="none",
            gradient_checkpointing=False,
            class_weighting="balanced",
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
            max_grad_norm=1.0,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            wandb_log_model="false",
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
            class_weights=[1.0, 2.0, 0.5],
            precision_policy={"mixed_precision": "none", "fp16": False, "bf16": False},
        )

        self.assertEqual(config["search_stage"], "tuning")
        self.assertEqual(config["trial_id"], "trial-003")
        self.assertEqual(config["config_hash"], "abc123")
        self.assertEqual(config["hpo_seed"], 2026)
        self.assertEqual(config["data_fraction"], 0.2)
        self.assertEqual(config["selection_metric"], "f1_macro")
        self.assertEqual(config["test_policy"], "final_only")
        self.assertEqual(config["warmup_ratio"], 0.06)
        self.assertEqual(config["output_dir"], "outputs/example")
        self.assertIs(config["run_test"], False)
        self.assertEqual(config["checkpoint_policy"]["save_strategy"], "epoch")
        self.assertEqual(config["training_policy"]["class_weighting"], "balanced")
        self.assertEqual(config["training_policy"]["class_weights"], [1.0, 2.0, 0.5])
        self.assertEqual(config["training_policy"]["max_grad_norm"], 1.0)
        self.assertEqual(config["global_switches"]["mixed_precision"], "none")
        self.assertIs(config["global_switches"]["weighted_ce"], True)
        self.assertEqual(
            config["checkpoint_policy"]["final_model_source"],
            "best_checkpoint",
        )
        self.assertIs(config["hyperparameters"]["load_best_model_at_end"], True)
        self.assertEqual(config["hyperparameters"]["early_stopping_patience"], 2)

    def test_setup_failure_config_records_global_switches_before_dataset_load(self):
        args = argparse.Namespace(
            method="full-ft",
            search_stage="tuning",
            trial_id="trial-setup",
            config_hash="abc123",
            hpo_seed=42,
            dataset_name="missing-dataset",
            model_name="distilbert-base-uncased",
            output_dir="outputs/setup-failure",
            seed=42,
            data_fraction_seed=99,
            fp16=False,
            mixed_precision="bf16",
            gradient_checkpointing=True,
            class_weighting="balanced",
            early_stopping_patience=2,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            warmup_ratio=0.06,
            weight_decay=0.01,
        )

        config = build_setup_failure_config(
            args,
            precision_policy={"mixed_precision": "bf16", "fp16": False, "bf16": True},
            gpu_type="A100",
        )

        self.assertIs(config["setup_complete"], False)
        self.assertEqual(config["runtime_context"]["gpu_type"], "A100")
        self.assertEqual(config["global_switches"]["mixed_precision"], "bf16")
        self.assertIs(config["global_switches"]["weighted_ce"], True)
        self.assertEqual(config["training_policy"]["class_weighting"], "balanced")

    def test_test_evaluation_policy_blocks_non_final_runs(self):
        from src.run_distilbert_hatexplain import validate_test_evaluation_policy

        with self.assertRaises(ValueError):
            validate_test_evaluation_policy(search_stage="tuning", run_test=True)

        validate_test_evaluation_policy(search_stage="final", run_test=True)

    def test_best_model_checkpoint_policy_requires_matching_save_and_eval(self):
        from src.run_distilbert_hatexplain import validate_checkpoint_policy

        args = argparse.Namespace(
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="steps",
            logging_steps=20,
            eval_steps=None,
            save_steps=500,
            early_stopping_patience=0,
            early_stopping_threshold=0.0,
            fp16=False,
            mixed_precision="none",
        )

        with self.assertRaises(ValueError):
            validate_checkpoint_policy(args)

        args.save_strategy = "epoch"
        validate_checkpoint_policy(args)

    def test_early_stopping_requires_best_model_selection(self):
        from src.run_distilbert_hatexplain import validate_checkpoint_policy

        args = argparse.Namespace(
            load_best_model_at_end=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20,
            eval_steps=None,
            save_steps=500,
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
            fp16=False,
            mixed_precision="none",
        )

        with self.assertRaises(ValueError):
            validate_checkpoint_policy(args)


if __name__ == "__main__":
    unittest.main()
