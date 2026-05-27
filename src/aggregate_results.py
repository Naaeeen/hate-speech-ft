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
    write_pareto_csvs,
)
from src.experiments.prediction_analysis import write_prediction_analysis_artifacts


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
        default=["method", "search_stage", "config_hash"],
        help="Run fields used to group mean/std summaries.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Metric to aggregate. Repeatable. Defaults to common eval/test, "
            "runtime, parameter, and best_epoch metrics."
        ),
    )
    parser.add_argument(
        "--write_pareto_csvs",
        action="store_true",
        help="Also write hpo_runs.csv, final_runs.csv, and method_summary.csv.",
    )
    parser.add_argument(
        "--csv_dir",
        default=None,
        help=(
            "Directory for Pareto CSV files. Defaults to the aggregate JSON "
            "output directory."
        ),
    )
    parser.add_argument(
        "--write_prediction_analysis",
        action="store_true",
        help=(
            "Also derive confusion matrices, AUROC summaries, and error examples "
            "from saved eval/test prediction files."
        ),
    )
    parser.add_argument(
        "--prediction_analysis_dir",
        default=None,
        help=(
            "Directory for prediction diagnostics. Defaults to a "
            "prediction_analysis directory beside the aggregate JSON output."
        ),
    )
    parser.add_argument(
        "--max_error_examples",
        type=int,
        default=50,
        help="Maximum misclassified examples to export per prediction file.",
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
    if args.write_pareto_csvs:
        csv_dir = Path(args.csv_dir) if args.csv_dir else output_path.parent
        csv_paths = write_pareto_csvs(csv_dir, report)
        for csv_path in csv_paths:
            print(f"Wrote Pareto CSV: {csv_path}")
    if args.write_prediction_analysis:
        analysis_dir = (
            Path(args.prediction_analysis_dir)
            if args.prediction_analysis_dir
            else output_path.parent / "prediction_analysis"
        )
        analysis_paths = write_prediction_analysis_artifacts(
            analysis_dir,
            report,
            max_error_examples=args.max_error_examples,
        )
        for analysis_path in analysis_paths.values():
            print(f"Wrote prediction analysis: {analysis_path}")
    print(
        f"Runs: {report['total_runs']} "
        f"completed={report['completed_runs']} failed={report['failed_runs']} "
        f"failed_oom={report['failed_oom_runs']}"
    )
    print(
        "Training time: "
        f"total={report.get('total_training_time_sec')} sec "
        f"hpo={report.get('hpo_total_training_time_sec')} sec"
    )
    for group in report["groups"]:
        group_label = ", ".join(
            f"{key}={value}" for key, value in group["group"].items()
        )
        print(
            f"{group_label}: runs={group['runs']} "
            f"completed={group['completed']} failed={group['failed']} "
            f"failed_oom={group['failed_oom']} "
            f"total_time_sec={group.get('total_training_time_sec')}"
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
