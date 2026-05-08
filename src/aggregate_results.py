from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.aggregate_results import (
    DEFAULT_METRICS,
    build_aggregate_report,
    write_aggregate_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate result_summary.json and failure_summary.json files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Output directories or summary JSON files to aggregate.",
    )
    parser.add_argument(
        "--output",
        default="outputs/aggregate_summary.json",
        help="Path for the aggregate JSON report.",
    )
    parser.add_argument(
        "--group_by",
        nargs="+",
        default=["method", "search_stage"],
        help="Run fields used to group mean/std summaries.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric to aggregate. Repeatable. Defaults to common eval/test metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = args.metric or list(DEFAULT_METRICS)
    report = build_aggregate_report(
        args.inputs,
        group_by=args.group_by,
        metrics=metrics,
    )
    output_path = write_aggregate_report(args.output, report)
    print(f"Wrote aggregate report: {output_path}")
    print(
        f"Runs: {report['total_runs']} "
        f"completed={report['completed_runs']} failed={report['failed_runs']}"
    )
    for group in report["groups"]:
        group_label = ", ".join(
            f"{key}={value}" for key, value in group["group"].items()
        )
        print(
            f"{group_label}: runs={group['runs']} "
            f"completed={group['completed']} failed={group['failed']}"
        )
        for metric, summary in group["metrics"].items():
            mean = summary["mean"]
            std = summary["std"]
            print(
                f"  {metric}: mean={mean:.6g} std={std:.6g} "
                f"n={summary['count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
