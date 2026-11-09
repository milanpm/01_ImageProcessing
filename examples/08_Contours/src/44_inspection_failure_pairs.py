"""
===============================================================================
File: 44_inspection_failure_pairs.py
Project: 01_ImageProcessing
Day: 57
Topic: Inspection Failure Reason Co-occurrence
Description:
    Report how often two failure reasons occur in the same inspected object.
Learning Objectives:
    - Reuse pair counts from the inspection CSV summary.
    - Rank failure reason pairs by count and name.
    - Calculate pair occurrence rates among failed objects.
Author: Alex
Created Date: 2026-11-10
Last Modified: 2026-11-10
===============================================================================
"""

from importlib import import_module
from pathlib import Path


summarize_inspection_csv = import_module(
    "42_inspection_csv_summary"
).summarize_inspection_csv


def main():
    """Print failure reason pair counts and rates from the Day 54 report."""
    report_path = (
        Path(__file__).resolve().parent.parent
        / "output"
        / "day54_inspection_report.csv"
    )

    summary = summarize_inspection_csv(report_path)
    fail_count = summary["fail_count"]
    pair_counts = summary["pair_counts"]

    print("Failure Reason Co-occurrence")
    print("-" * 56)
    print(f"FAIL objects: {fail_count}")

    if not pair_counts:
        print("No failure reason pairs found.")
        return

    ranked_pairs = sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    print()
    print(f"{'Failure reason pair':<28}{'Count':>8}{'Rate among FAIL':>20}")

    for pair, count in ranked_pairs:
        pair_name = " + ".join(pair)
        rate = count / fail_count
        print(f"{pair_name:<28}{count:>8}{rate:>20.2%}")


if __name__ == "__main__":
    main()
