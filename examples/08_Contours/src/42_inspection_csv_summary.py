"""
===============================================================================
File: 42_inspection_csv_summary.py
Project: 01_ImageProcessing
Day: 55
Topic: Inspection CSV Summary
Description:
    Read an inspection CSV report and summarize inspection decisions
    and failure reason counts.
Learning Objectives:
    - Read CSV rows using csv.DictReader.
    - Count PASS and FAIL decisions.
    - Calculate the pass rate and count individual failure reasons.
Author: Alex
Created Date: 2026-10-23
Last Modified: 2026-10-23
===============================================================================
"""

import csv
from collections import Counter
from pathlib import Path


def summarize_inspection_csv(report_path):
    """Read a CSV report and return inspection summary statistics."""
    result_counts = Counter()
    reason_counts = Counter()

    with Path(report_path).open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"overall_result", "failure_reasons"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("CSV report is missing required columns.")

        for row in reader:
            overall_result = row["overall_result"].strip()

            if overall_result not in {"PASS", "FAIL"}:
                raise ValueError(
                    f"Invalid overall result at line {reader.line_num}: "
                    f"{overall_result!r}"
                )

            result_counts[overall_result] += 1

            if overall_result == "FAIL":
                reasons = {
                    reason.strip()
                    for reason in row["failure_reasons"].split(",")
                    if reason.strip() and reason.strip() != "NONE"
                }
                reason_counts.update(reasons)

    total_count = sum(result_counts.values())
    pass_count = result_counts["PASS"]
    fail_count = result_counts["FAIL"]
    pass_rate = pass_count / total_count * 100 if total_count else 0.0

    return {
        "total_count": total_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "reason_counts": reason_counts,
    }


def main():
    """Load the Day 54 report and print its summary."""
    report_path = (
        Path(__file__).resolve().parent.parent
        / "output"
        / "day54_inspection_report.csv"
    )

    summary = summarize_inspection_csv(report_path)

    print("Inspection CSV Summary")
    print("-" * 40)
    print(f"Source report : {report_path}")
    print(f"Total objects : {summary['total_count']}")
    print(f"PASS objects  : {summary['pass_count']}")
    print(f"FAIL objects  : {summary['fail_count']}")
    print(f"Pass rate     : {summary['pass_rate']:.2f}%")
    print()
    print("Failure reason counts")

    for reason in ("POSITION", "ANGLE", "SIZE"):
        print(f"{reason:<14}: {summary['reason_counts'][reason]}")

    print(
        f"{'Total reasons':<14}: "
        f"{sum(summary['reason_counts'].values())}"
    )


if __name__ == "__main__":
    main()
