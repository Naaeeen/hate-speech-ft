import argparse
import inspect
import unittest

import src.methods.distilbert_full.train as run_distilbert
import src.methods.hf_sequence_classification as hf_workflow
from src.methods.distilbert_full.train import build_experiment_config, build_setup_failure_config


class RunDistilbertExperimentConfigTests(unittest.TestCase):
    def test_experiment_config_logs_shared_policy_fields(self):
        args = argparse.Namespace(
            method="full-ft",
            search_stage="tuning",
            trial_id="trial-003",
            config_hash="abc123",
            hpo_seed=2026,
            hpo_trial_cap=6,
            hpo_time_cap_gpu_hours=2.0,
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
            overwrite_output_dir=False,
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

        self.assertEqual(config["search_stage"], "tuning")
        self.assertEqual(config["trial_id"], "trial-003")
        self.assertEqual(config["config_hash"], "abc123")
        self.assertEqual(config["hpo_seed"], 2026)
        self.assertEqual(config["hpo_trial_cap"], 6)
        self.assertEqual(config["hpo_time_cap_gpu_hours"], 2.0)
        self.assertEqual(config["data_fraction"], 0.2)
        self.assertEqual(config["effective_train_fraction"], 0.2)
        self.assertEqual(config["raw_train_size"], 600)
        self.assertEqual(config["full_train_size"], 500)
        self.assertEqual(config["dropped_no_majority_train"], 100)
        self.assertIn("post-load", config["split_accounting_policy"])
        self.assertEqual(config["raw_eval_size"], 60)
        self.assertEqual(config["dropped_no_majority_eval"], 10)
        self.assertEqual(config["selection_metric"], "f1_macro")
        self.assertEqual(config["test_policy"], "final_only")
        self.assertEqual(config["warmup_ratio"], 0.06)
        self.assertEqual(config["output_dir"], "outputs/example")
        self.assertIs(config["run_test"], False)
        self.assertEqual(config["checkpoint_policy"]["save_strategy"], "epoch")
        self.assertIs(config["checkpoint_policy"]["overwrite_output_dir"], False)
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

    def test_experiment_config_separates_requested_and_effective_data_fraction(self):
        args = argparse.Namespace(
            method="full-ft",
            search_stage="smoke",
            trial_id="trial-smoke",
            config_hash=None,
            hpo_seed=None,
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
            fp16=False,
            mixed_precision="none",
            gradient_checkpointing=False,
            class_weighting="none",
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
            num_train_epochs=1,
            output_dir="outputs/smoke",
            overwrite_output_dir=False,
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

    def test_setup_failure_config_records_global_switches_before_dataset_load(self):
        args = argparse.Namespace(
            method="full-ft",
            search_stage="tuning",
            trial_id="trial-setup",
            config_hash="abc123",
            hpo_seed=42,
            hpo_trial_cap=6,
            hpo_time_cap_gpu_hours=2.0,
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
            overwrite_output_dir=False,
        )

        config = build_setup_failure_config(
            args,
            precision_policy={"mixed_precision": "bf16", "fp16": False, "bf16": True},
            gpu_type="A100",
        )

        self.assertIs(config["setup_complete"], False)
        self.assertEqual(config["hpo_time_cap_gpu_hours"], 2.0)
        self.assertEqual(config["runtime_context"]["gpu_type"], "A100")
        self.assertEqual(config["global_switches"]["mixed_precision"], "bf16")
        self.assertIs(config["global_switches"]["weighted_ce"], True)
        self.assertEqual(config["training_policy"]["class_weighting"], "balanced")
        self.assertIs(config["output_safety"]["overwrite_output_dir"], False)

    def test_test_evaluation_policy_blocks_non_final_runs(self):
        from src.methods.distilbert_full.train import validate_test_evaluation_policy

        with self.assertRaises(ValueError):
            validate_test_evaluation_policy(search_stage="tuning", run_test=True)
        with self.assertRaises(ValueError):
            validate_test_evaluation_policy(search_stage="final", run_test=False)

        validate_test_evaluation_policy(search_stage="final", run_test=True)

    def test_best_model_checkpoint_policy_requires_matching_save_and_eval(self):
        from src.methods.distilbert_full.train import validate_checkpoint_policy

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

    def test_completed_summary_is_written_after_final_model_save(self):
        source = inspect.getsource(run_distilbert.main)

        self.assertLess(source.index("save_final_model("), source.index('status="completed"'))
        self.assertLess(source.index("save_final_model("), source.index("write_success_outputs("))

        save_source = inspect.getsource(hf_workflow.save_final_model)
        self.assertIn("trainer.save_model", save_source)
        self.assertIn("tokenizer.save_pretrained", save_source)

    def test_wandb_run_starts_before_remote_setup_can_fail(self):
        source = inspect.getsource(run_distilbert.main)
        start_source = inspect.getsource(hf_workflow.start_hf_run)
        prepare_source = inspect.getsource(hf_workflow.prepare_hf_classification_run)

        self.assertLess(source.index("start_hf_run"), source.index("prepare_hf_classification_run"))
        self.assertIn("init_wandb_run", start_source)
        self.assertIn("load_dataset", prepare_source)
        self.assertLess(source.index("start_hf_run"), source.index("write_config_snapshot"))

    def test_early_stopping_requires_best_model_selection(self):
        from src.methods.distilbert_full.train import validate_checkpoint_policy

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
