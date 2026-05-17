from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Get the root path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wandb # for WandB use
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.data.preprocessing import preprocess_hatexplain_split
from src.experiments.results import write_failure_file, write_resolved_config, write_result_files
from src.methods.common import (
    add_common_method_arguments,
    build_common_experiment_config,
    validate_output_dir_for_run,
    validate_test_evaluation_policy,
)

DEFAULT_METHOD_ID = "tfidf-logreg"
DEFAULT_METHOD_PACKAGE = "tfidf_logreg"
DEFAULT_DESCRIPTION = "TF-IDF + Logistic Regression baseline method."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)

    add_common_method_arguments(
        parser,
        defaults={
            "method": DEFAULT_METHOD_ID,
            "trial_id": f"{DEFAULT_METHOD_PACKAGE}_manual",
            "output_dir": f"outputs/{DEFAULT_METHOD_PACKAGE}",
        },
    )

    # arguments specifically for TF-IDF + LogReg method
    parser.add_argument("--ngram_range", type=str, default="1,2")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=1.0)

    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> dict[str, object]:
    return build_common_experiment_config(
        args,
        model_name="tfidf_logreg_baseline",
        tokenizer_name=None,
        hyperparameters={
            "ngram_range": args.ngram_range,
            "min_df": args.min_df,
            "max_features": args.max_features,
            "C": args.C,
            "seed": args.seed,
        },
        extra={
            "status": "ready",
            "training_policy": "classical"
        },
    )


def main() -> None:
    # initial setup for results recording
    args = parse_args()
    start = time.perf_counter()
    config = build_experiment_config(args)
    output_dir = Path(args.output_dir)

    validate_test_evaluation_policy(
        search_stage=args.search_stage,
        run_test=args.run_test,
    )
    validate_output_dir_for_run(
        output_dir,
        overwrite=args.overwrite_output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(output_dir, config)

    # initialise WandB run
    run = None
    if getattr(args, "use_wandb", False):
        run = wandb.init(
            project=getattr(args, "wandb_project", "hate-speech-ft"),
            entity=getattr(args, "wandb_entity", None),
            group=getattr(args, "wandb_group", "tfidf-logreg"),
            name=args.trial_id,
            config=config,
        )

    try:
        dataset = load_dataset(args.dataset_name)
        train_records = preprocess_hatexplain_split(dataset["train"])
        eval_records = preprocess_hatexplain_split(dataset["validation"])

        X_train = [" ".join(record['post_tokens']) if 'post_tokens' in record else record['text'] for record in
                   train_records]
        y_train = [record['label'] for record in train_records]

        X_eval = [" ".join(record['post_tokens']) if 'post_tokens' in record else record['text'] for record in
                  eval_records]
        y_eval = [record['label'] for record in eval_records]

        try:
            ngram_range_tuple = tuple(int(x.strip()) for x in args.ngram_range.split(","))
        except Exception as e:
            raise ValueError(
                f"Could not parse ngram_range '{args.ngram_range}'. Expected comma-separated format like '1,2'") from e

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=tuple(ngram_range_tuple),
                min_df=args.min_df,
                max_features=args.max_features
            )),
            ('clf', LogisticRegression(C=args.C, solver='liblinear', random_state=args.seed, max_iter=1000))
        ])

        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_eval)
        val_macro_f1 = f1_score(y_eval, val_preds, average='macro')
        val_acc = accuracy_score(y_eval, val_preds)

        val_metrics = {
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_acc
        }

        # log the WandB results
        if run:
            wandb.log(val_metrics)

        test_records = preprocess_hatexplain_split(dataset["test"])
        X_test = [" ".join(record['post_tokens']) if 'post_tokens' in record else record['text'] for record in
                  test_records]
        y_test = [record['label'] for record in test_records]

        test_preds = pipeline.predict(X_test)
        test_macro_f1 = f1_score(y_test, test_preds, average='macro')
        test_macro_acc = accuracy_score(y_test, test_preds)

        test_metrics = {
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_macro_acc
        }

        if run:
            wandb.log(test_metrics)

        runtime_metrics = {
            "total_duration_seconds": time.perf_counter() - start,
            "status": "completed"
        }

        model_selection = {}

        write_result_files(
            output_dir=output_dir,
            config=config,
            eval_metrics=val_metrics,
            runtime_metrics=runtime_metrics,
            test_metrics=test_metrics,
            model_selection=model_selection,
            status="completed"
        )

        if run:
            wandb.finish()

        print(f"Experiment finished successfully. Results saved to {output_dir}")

    except Exception as exc:
        if run:
            wandb.log({"status": "failed"})
            wandb.finish()

        write_failure_file(
            output_dir,
            config=config,
            error=exc,
            runtime_metrics={
                "status": "failed",
                "failure_phase": "training_or_evaluation",
                "training_time_sec": time.perf_counter() - start,
            },
        )
        print(f"Experiment failed! Error log dropped to {output_dir}/failure_summary.json", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()