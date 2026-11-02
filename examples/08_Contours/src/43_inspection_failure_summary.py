"""
===============================================================================
File: 43_inspection_failure_summary.py
Project: 01_ImageProcessing
Day: 56
Topic: Failure Reason Rates and Frequency Ranking
Description:
    Reuse the Day 55 summary function to rank failure reasons and calculate
    their occurrence rates among failed objects.
Learning Objectives:
    - Reuse an existing CSV summary function.
    - Sort failure reasons by count and name.
    - Calculate occurrence rates using failed objects as the denominator.
    - Handle reports with no failed objects.
Author: Alex
Created Date: 2026-11-02
Last Modified: 2026-11-02
===============================================================================
"""

from importlib import import_module
from pathlib import Path


summarize_inspection_csv = import_module(
    "42_inspection_csv_summary"
).summarize_inspection_csv


def build_failure_ranking(summary):
    """Return failure reasons ranked by count, then alphabetically."""
    fail_count = summary["fail_count"]
    reason_counts = summary["reason_counts"]

    reasons = {"POSITION", "ANGLE", "SIZE"} | set(reason_counts)
    sorted_reasons = sorted(
        reasons,
        key=lambda reason: (-reason_counts[reason], reason),
    )

    ranking = []

    for reason in sorted_reasons:
        count = reason_counts[reason]
        rate = count / fail_count * 100 if fail_count else 0.0
        ranking.append((reason, count, rate))

    return ranking


def main():
    """Print failure reason counts and rates from the Day 54 report."""
    report_path = (
        Path(__file__).resolve().parent.parent
        / "output"
        / "day54_inspection_report.csv"
    )

    summary = summarize_inspection_csv(report_path)
    ranking = build_failure_ranking(summary)

    print("Failure Reason Ranking")
    print("-" * 48)
    print(f"Total objects : {summary['total_count']}")
    print(f"FAIL objects  : {summary['fail_count']}")
    print()
    print(f"{'Reason':<14}{'Count':>8}{'Rate among FAIL':>20}")

    for reason, count, rate in ranking:
        print(f"{reason:<14}{count:>8}{rate:>19.2f}%")

    print()
    print("Rate = reason count / FAIL objects * 100")
    print("An object may have multiple failure reasons.")
    print("Rates may add up to more than 100%.")


if __name__ == "__main__":
    main()
