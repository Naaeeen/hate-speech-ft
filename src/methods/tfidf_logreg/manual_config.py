"""Manual config for one TF-IDF + Logistic Regression run.

Usually change `seed`, `run_name`, and `output_dir`. The method-specific knobs
are `ngram_range`, `min_df`, `max_df`, `max_features`, `sublinear_tf`, and `C`.
This baseline is CPU/sklearn, so there is no epoch loop or GPU checkpointing.
"""

CONFIG = {
    "method": "tfidf-logreg",
    "run_name": "tfidf_logreg_final_seed42",
    "dataset_name": "Hate-speech-CNERG/hatexplain",
    "seed": 42,
    "data_fraction_seed": 42,
    "data_fraction": 1.0,
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_test_samples": None,
    "output_dir": "outputs/tfidf_logreg_final_seed42",
    "overwrite_output_dir": False,
    "run_test": True,
    "no_save_final_model": False,
    "class_weighting": "none",
    "use_wandb": True,
    "wandb_project": "hate-speech-ft",
    "wandb_entity": "hoangbachbach05-the-australian-national-university",
    "wandb_mode": "online",
    "test_split_name": "test",
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.9,
    "max_features": 20000,
    "sublinear_tf": True,
    "C": 1.0,
}
