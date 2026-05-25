from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.results import (  # noqa: E402
    write_failure_file,
    write_resolved_config,
    write_result_files,
)
from src.methods.common import (  # noqa: E402
    clear_existing_run_artifacts,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)
from src.methods.hf_common import (  # noqa: E402
    get_gpu_type,
)
from src.methods.tfidf_logreg.args import parse_args  # noqa: E402
from src.methods.tfidf_logreg.config import (  # noqa: E402
    build_experiment_config,
    build_model_selection,
    build_runtime_metrics as build_runtime_metrics_payload,
    resolve_wandb_settings,
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
    finish_wandb_run,
    init_wandb_run,
    log_wandb_best_effort,
    update_wandb_config_best_effort,
)


def build_runtime_metrics(
    *,
    training_time_sec: float | None,
    gpu_type: str,
    status: str,
    failure_phase: str | None = None,
) -> dict[str, Any]:
    return build_runtime_metrics_payload(
        training_time_sec=training_time_sec,
        gpu_type=gpu_type,
        status=status,
        failure_phase=failure_phase,
        peak_memory_mb=None,
        peak_memory_reserved_mb=None,
    )


def _validate_startup_args(args, ngram_range: tuple[int, int], output_dir: Path) -> None:
    validate_classical_args(args, ngram_range)
    validate_test_evaluation_policy(
        search_stage=args.search_stage,
        run_test=args.run_test,
    )
    validate_output_dir_for_run(
        output_dir,
        overwrite=args.overwrite_output_dir,
    )


def _prepare_output_dir(args, output_dir: Path) -> None:
    if args.overwrite_output_dir:
        removed_artifacts = clear_existing_run_artifacts(output_dir)
        if removed_artifacts:
            preview = ", ".join(path.name for path in removed_artifacts[:8])
            print(f"Cleared existing run artifacts from {output_dir}: {preview}")
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    output_dir = Path(args.output_dir)
    try:
        ngram_range = parse_ngram_range(args.ngram_range)
    except ValueError as exc:
        print(f"Cannot start run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    gpu_type = get_gpu_type()
    wandb_run = None
    run_artifacts_prepared = False
    config = build_experiment_config(
        args,
        ngram_range=ngram_range,
        gpu_type=gpu_type,
        setup_complete=False,
    )

    try:
        _validate_startup_args(args, ngram_range, output_dir)
    except ValueError as exc:
        print(f"Cannot start run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        wandb_run = init_wandb_run(resolve_wandb_settings(args), config=config)
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
            setup_complete=True,
        )

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
            setup_complete=True,
        )
        _prepare_output_dir(args, output_dir)
        run_artifacts_prepared = True
        write_resolved_config(output_dir, config)
        update_wandb_config_best_effort(wandb_run, config)
        log_wandb_best_effort(
            wandb_run,
            eval_metrics,
            *([test_metrics] if test_metrics is not None else []),
        )

        model_artifact_paths = {}
        if args.no_save_final_model:
            print("\nSkipping final model save because --no_save_final_model was set.")
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
            status="completed",
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
            status="completed",
        )
        log_wandb_best_effort(
            wandb_run,
            runtime_metrics,
            {"model_selection": model_selection},
        )
        print_result_report(
            output_dir=output_dir,
            eval_metrics=eval_metrics,
            test_metrics=test_metrics,
            runtime_metrics=runtime_metrics,
            result_paths=result_paths,
            prediction_paths=prediction_paths,
        )

    except Exception as exc:
        runtime_metrics = build_runtime_metrics(
            training_time_sec=time.perf_counter() - run_start,
            gpu_type=gpu_type,
            status="failed",
            failure_phase="setup_or_train_eval",
        )
        write_failure_file(
            output_dir,
            config=config,
            error=exc,
            runtime_metrics=runtime_metrics,
            clear_existing_artifacts=run_artifacts_prepared,
        )
        log_wandb_best_effort(
            wandb_run,
            {
                "status": "failed",
                "failure_phase": runtime_metrics["failure_phase"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        print(f"\nFailure summary: {output_dir / 'failure_summary.json'}")
        raise
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
