"""Manual config for one DistilBERT Full FT run.

Usually change `seed`, `run_name`, and `output_dir` for the next final seed.
Change `learning_rate` or epochs only when you are intentionally rerunning an
HPO-style trial. Leave dataset, split, W&B, and checkpoint fields alone unless
the experiment plan says otherwise.
"""

CONFIG = {
    # Fixed method id written to result_summary.json and W&B.
    "method": "full-ft",
    # Human-readable run label. This is the name you search for in W&B.
    "run_name": "distilbert_full_final_seed42",
    # Hugging Face checkpoint to load. Keep this for comparable DistilBERT runs.
    "model_name": "distilbert-base-uncased",
    # Hugging Face dataset id. Keep this the same for comparable reruns.
    "dataset_name": "Hate-speech-CNERG/hatexplain",
    # Random seed for this one run. Change only one seed at a time.
    "seed": 42,
    # Only matters when data_fraction < 1.0. Full-data runs can leave it at 42.
    "data_fraction_seed": 42,
    "data_fraction": 1.0,
    # Debug caps. Leave as None for real runs, or results will not be comparable.
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_test_samples": None,
    # Folder where this run writes metrics.json, runtime.json, predictions, etc.
    "output_dir": "outputs/distilbert_full_final_seed42",
    # False protects older output folders from being overwritten by accident.
    "overwrite_output_dir": False,
    # True means also evaluate the test split and save test_predictions.json.
    "run_test": True,
    # Dataset split used when run_test=True.
    "test_split_name": "test",
    # Token length passed to the DistilBERT tokenizer; longer texts are truncated.
    "max_length": 128,
    # Main Full FT HPs. num_train_epochs is the field name Hugging Face Trainer uses.
    "learning_rate": 2e-5,
    "num_train_epochs": 4.0,
    # Batch size is per device/GPU, not total across all possible devices.
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    # Shared optimizer/scheduler settings from the reference setup.
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    # Trainer schedule. With "epoch", eval_steps/save_steps are basically backups.
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "steps",
    "logging_steps": 20,
    "eval_steps": None,
    "save_steps": 500,
    "save_total_limit": 1,
    # Pick the checkpoint with the best validation macro F1.
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1_macro",
    "lower_is_better": False,
    # Keep False for final runs so model/tokenizer artifacts are saved.
    "no_save_final_model": False,
    # "none" avoids fp16/bf16 differences when comparing reference numbers.
    "mixed_precision": "none",
    # Off for these small DistilBERT runs; enabling it changes memory/speed tradeoffs.
    "gradient_checkpointing": False,
    # "none" matches the reference setup; change only for a new class-weight study.
    "class_weighting": "none",
    # Stop if validation F1 stops improving enough across epochs.
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.001,
    # W&B project/entity choose the dashboard. mode can be online/offline/disabled.
    "use_wandb": True,
    "wandb_project": "hate-speech-ft",
    "wandb_entity": "hoangbachbach05-the-australian-national-university",
    "wandb_mode": "online",
}
