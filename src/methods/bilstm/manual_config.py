"""Manual config for one BiLSTM run.

Usually change `seed`, `run_name`, and `output_dir`. The model-specific knobs
are `embedding_size`, `hidden_size`, `dropout`, and `learning_rate`. This is a
from-scratch PyTorch baseline, so `device="auto"` is fine for Colab unless you
are deliberately forcing CPU/GPU.
"""

CONFIG = {
    "method": "bilstm",
    "run_name": "bilstm_final_seed42",
    "dataset_name": "Hate-speech-CNERG/hatexplain",
    "seed": 42,
    "data_fraction_seed": 42,
    "data_fraction": 1.0,
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_test_samples": None,
    "output_dir": "outputs/bilstm_final_seed42",
    "overwrite_output_dir": False,
    "run_test": True,
    "max_length": 128,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "steps",
    "logging_steps": 20,
    "eval_steps": None,
    "save_steps": 500,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1_macro",
    "no_save_final_model": False,
    "class_weighting": "none",
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.001,
    "use_wandb": True,
    "wandb_project": "hate-speech-ft",
    "wandb_entity": "hoangbachbach05-the-australian-national-university",
    "wandb_mode": "online",
    "test_split_name": "test",
    "device": "auto",
    "embedding_size": 200,
    "hidden_size": 128,
    "num_layers": 1,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "eval_batch_size": 128,
    "epochs": 10,
}
