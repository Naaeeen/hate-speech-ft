"""Manual entrypoint for the TF-IDF + Logistic Regression baseline.

This is the classical path: load the same HateXplain splits, fit one sklearn
pipeline on CPU, evaluate validation/test, save `model.joblib`, write the shared
JSON files, and log one W&B run if enabled. No epochs, no GPU training, no
launcher.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.results import (  # noqa: E402
    prepare_output_dir_for_run,
    write_resolved_config,
    write_result_files,
)
from src.utils.run_metadata import (  # noqa: E402
    get_gpu_type,
)
from src.methods.tfidf_logreg.manual_config import CONFIG as MANUAL_CONFIG  # noqa: E402
from src.methods.tfidf_logreg.config import (  # noqa: E402
    build_experiment_config,
    build_model_selection,
    build_runtime_metrics,
)
from src.methods.tfidf_logreg.data import (  # noqa: E402
    build_classical_data_splits,
    print_split_summary,
    records_to_xy,
    resolve_classical_split_names,
)
from src.methods.tfidf_logreg.reporting import (  # noqa: E402
    print_result_report,
    write_final_prediction_files,
)
from src.methods.tfidf_logreg.training import (  # noqa: E402
    build_classification_metrics,
    build_pipeline,
    get_model_stats,
    load_libraries,
    parse_ngram_range,
    validate_classical_args,
)
from src.methods.transformer_data import build_fixed_label_maps  # noqa: E402
from src.utils.wandb_config import (  # noqa: E402
    build_wandb_settings_from_args,
    finish_wandb_run,
    init_wandb_run,
    log_wandb,
    prefixed_wandb_scalars,
)


def main() -> None:
    """Run one TF-IDF manual experiment from `manual_config.py`."""

    args = SimpleNamespace(**MANUAL_CONFIG)
    output_dir = Path(args.output_dir)
    ngram_range = parse_ngram_range(args.ngram_range)

    gpu_type = get_gpu_type()

    validate_classical_args(args, ngram_range)
    prepare_output_dir_for_run(output_dir, overwrite=args.overwrite_output_dir)

    load_dataset, dump, TfidfVectorizer, LogisticRegression, Pipeline = load_libraries()
    random.seed(args.seed)

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    print("Available splits:", list(dataset.keys()))
    train_split, eval_split, test_split = resolve_classical_split_names(dataset, args)
    train_data, eval_data, test_data = build_classical_data_splits(
        dataset,
        args,
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
    )
    print_split_summary(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
    )

    id2label, _label2id, _num_labels = build_fixed_label_maps()

    x_train, y_train = records_to_xy(train_data.records)
    x_eval, y_eval = records_to_xy(eval_data.records)
    pipeline = build_pipeline(
        TfidfVectorizer=TfidfVectorizer,
        LogisticRegression=LogisticRegression,
        Pipeline=Pipeline,
        args=args,
        ngram_range=ngram_range,
    )

    print("\nTraining TF-IDF + Logistic Regression...")
    train_start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    training_time_sec = time.perf_counter() - train_start
    trainable_params, total_params, vocab_size = get_model_stats(pipeline)
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    eval_predictions = pipeline.predict(x_eval)
    eval_metrics = build_classification_metrics(
        y_eval,
        eval_predictions,
        prefix="eval",
        label_id_to_name=id2label,
    )

    test_metrics = None
    if args.run_test:
        if test_data is None:
            raise ValueError("Cannot run final test evaluation before loading test data.")
        x_test, y_test = records_to_xy(test_data.records)
        test_metrics = build_classification_metrics(
            y_test,
            pipeline.predict(x_test),
            prefix="test",
            label_id_to_name=id2label,
        )

    config = build_experiment_config(
        args,
        ngram_range=ngram_range,
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        gpu_type=gpu_type,
        trainable_params=trainable_params,
        total_params=total_params,
        vocab_size=vocab_size,
    )
    write_resolved_config(output_dir, config)

    model_artifact_paths = {}
    if args.no_save_final_model:
        print("\nSkipping final model save because no_save_final_model=True.")
    else:
        model_path = output_dir / "model.joblib"
        dump(pipeline, model_path)
        model_artifact_paths["model.joblib"] = model_path
        print(f"\nSaved final TF-IDF pipeline: {model_path}")

    prediction_paths = write_final_prediction_files(
        output_dir=output_dir,
        args=args,
        pipeline=pipeline,
        eval_data=eval_data,
        eval_predictions=eval_predictions,
        test_data=test_data,
        id2label=id2label,
    )
    runtime_metrics = build_runtime_metrics(
        training_time_sec=training_time_sec,
        gpu_type=gpu_type,
    )
    model_selection = build_model_selection(eval_metrics)
    result_paths = write_result_files(
        output_dir=output_dir,
        config=config,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        runtime_metrics=runtime_metrics,
        model_selection=model_selection,
        prediction_paths=prediction_paths,
        artifact_paths=model_artifact_paths,
    )
    wandb_run = init_wandb_run(build_wandb_settings_from_args(args), config=config)
    log_wandb(
        wandb_run,
        eval_metrics,
        *([test_metrics] if test_metrics is not None else []),
        runtime_metrics,
        {
            "model_selection": model_selection,
            **prefixed_wandb_scalars("model_selection", model_selection),
        },
    )
    finish_wandb_run(wandb_run)
    print_result_report(
        output_dir=output_dir,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        runtime_metrics=runtime_metrics,
        model_selection=model_selection,
        result_paths=result_paths,
        prediction_paths=prediction_paths,
    )


if __name__ == "__main__":
    main()
