"""Manual config for one BiLSTM run.

Usually change `seed`, `run_name`, and `output_dir`. The settings below match
the new BiLSTM final reference config `c01881878157`.
"""

CONFIG = {
    # Fixed method id written to result_summary.json and W&B.
    "method": "bilstm",
    # Human-readable run label. This is the name you search for in W&B.
    "run_name": "bilstm_final_seed42",
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
    "output_dir": "outputs/bilstm_final_seed42",
    # False protects older output folders from being overwritten by accident.
    "overwrite_output_dir": False,
    # True means also evaluate the test split and save test_predictions.json.
    "run_test": True,
    # Number of word-token positions kept per example; longer texts are truncated.
    "max_length": 128,
    # Build the word vocab from train split only. min_freq drops very rare words.
    "tokenizer_min_freq": 2,
    # Hard cap on the train-split word vocabulary size.
    "max_vocab_size": 30000,
    # Optimizer and scheduler knobs for the custom PyTorch training loop.
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    # Save/evaluate by epoch so best checkpoint selection matches the reference setup.
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    # Best checkpoint is picked by validation macro F1.
    "metric_for_best_model": "eval_f1_macro",
    # Keep False for final runs so model.pt and tokenizer files are saved.
    "no_save_final_model": False,
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
    # Dataset split used when run_test=True.
    "test_split_name": "test",
    # "auto" uses CUDA in Colab when available, otherwise CPU.
    "device": "auto",
    # BiLSTM architecture HPs. These define the neural model size.
    "embedding_size": 100,
    "hidden_size": 256,
    "num_layers": 1,
    "dropout": 0.5,
    # Training HPs for the custom PyTorch loop.
    "learning_rate": 0.001,
    "batch_size": 64,
    "eval_batch_size": 128,
    "epochs": 10,
}
