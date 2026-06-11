"""Manual config for one TF-IDF + Logistic Regression run.

Usually change `seed`, `run_name`, and `output_dir`. The method-specific knobs
are `ngram_range`, `min_df`, `max_df`, `max_features`, `sublinear_tf`, and `C`.
This baseline is CPU/sklearn, so there is no epoch loop or GPU checkpointing.
"""

CONFIG = {
    # Fixed method id. It is written to result_summary.json and W&B, so keep it
    # stable unless you are adding a genuinely new method.
    "method": "tfidf-logreg",
    # Human-readable run label. This is the name you search for in W&B.
    "run_name": "tfidf_logreg_final_seed42",
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
    "output_dir": "outputs/tfidf_logreg_final_seed42",
    # False protects older output folders from being overwritten by accident.
    "overwrite_output_dir": False,
    # True means also evaluate the test split and save test_predictions.json.
    "run_test": True,
    # Keep False for final runs so the sklearn pipeline is saved as model.joblib.
    "no_save_final_model": False,
    # "none" matches the reference setup; change only for a new class-weight study.
    "class_weighting": "none",
    # W&B project/entity choose the dashboard. mode can be online/offline/disabled.
    "use_wandb": True,
    "wandb_project": "hate-speech-ft",
    "wandb_entity": "hoangbachbach05-the-australian-national-university",
    "wandb_mode": "online",
    # Dataset split used when run_test=True.
    "test_split_name": "test",
    # TF-IDF feature settings. These are the main HPs for this baseline.
    "ngram_range": [1, 2],
    # min_df/max_df drop words that are too rare or too common.
    "min_df": 2,
    "max_df": 0.9,
    # Upper bound on vocabulary size after TF-IDF filtering.
    "max_features": 20000,
    # log-scaled term frequency; often helps sparse linear models.
    "sublinear_tf": True,
    # Logistic Regression regularization strength. Larger C means weaker regularization.
    "C": 1.0,
}
