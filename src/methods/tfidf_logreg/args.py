from __future__ import annotations

import argparse

from src.methods.common import add_common_method_arguments


DEFAULT_METHOD_ID = "tfidf-logreg"
DEFAULT_METHOD_PACKAGE = "tfidf_logreg"
DEFAULT_DESCRIPTION = "Run TF-IDF + Logistic Regression on HateXplain."


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
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument(
        "--ngram_range",
        type=str,
        default="1,2",
        help="Inclusive TF-IDF n-gram range, e.g. '1,2' or '[1,2]'.",
    )
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=1.0)
    return parser.parse_args()
