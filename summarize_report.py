#!/usr/bin/env python3
"""Add simple, explainable review fields to a BirdScan species summary CSV.

This is post-processing only.  It never runs BioCLIP and never overwrites the
input CSV unless the caller explicitly supplies the same path for --output.
When present, the four species metadata columns are preserved as-is, including
blank optional Chinese or English names.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


CSV_ENCODING = "utf-8-sig"
REQUIRED_COLUMNS = ("top1_count", "top5_count", "max_score")
ADDED_COLUMNS = ("top5_only_count", "top5_top1_ratio", "confidence_level", "special_flag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add confidence and special-pattern review fields without changing the input CSV."
    )
    parser.add_argument("input_csv", type=Path, help="Path to species_summary.csv")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: species_summary_reviewed.csv beside the input)",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input_csv.with_name("species_summary_reviewed.csv")
    if args.input_csv.resolve() == args.output.resolve():
        parser.error("--output must not be the same file as the input CSV")
    return args


def parse_int(value: str, column: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {column} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"Row {row_number}: {column} must not be negative")
    return parsed


def parse_score(value: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: max_score must be numeric, got {value!r}") from exc
    if not 0 <= parsed <= 1:
        raise ValueError(f"Row {row_number}: max_score must be between 0 and 1")
    return parsed


def confidence_level(top1_count: int, max_score: float) -> str:
    """Classify confidence separately from special review patterns."""
    if (
        max_score >= 0.85
        or (top1_count >= 3 and max_score >= 0.65)
        or (top1_count >= 8 and max_score >= 0.50)
        or top1_count >= 20
    ):
        return "high"
    if max_score >= 0.60 or (top1_count >= 2 and max_score >= 0.45) or top1_count >= 5:
        return "medium"
    return "review"


def special_flag(top1_count: int, top5_count: int, max_score: float) -> str:
    """Return independent special-pattern flags, not a confidence estimate."""
    ratio = top5_count / max(top1_count, 1)
    flags = []
    if top5_count >= 20 and ratio >= 3:
        flags.append("mixed_candidate")
    if top1_count <= 2 and max_score >= 0.80:
        flags.append("rare_candidate")
    return ";".join(flags) if flags else "none"


def process(input_csv: Path, output_csv: Path) -> Counter[str]:
    if not input_csv.is_file():
        raise ValueError(f"Input CSV does not exist: {input_csv}")
    with input_csv.open("r", encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
        original_fields = list(reader.fieldnames)
        rows = list(reader)

    output_fields = original_fields + [column for column in ADDED_COLUMNS if column not in original_fields]
    counts: Counter[str] = Counter()
    reviewed_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        top1_count = parse_int(row["top1_count"], "top1_count", row_number)
        top5_count = parse_int(row["top5_count"], "top5_count", row_number)
        max_score = parse_score(row["max_score"], row_number)
        if top5_count < top1_count:
            raise ValueError(f"Row {row_number}: top5_count must be at least top1_count")
        ratio = top5_count / max(top1_count, 1)
        confidence = confidence_level(top1_count, max_score)
        special = special_flag(top1_count, top5_count, max_score)
        reviewed_rows.append({
            **row,
            "top5_only_count": str(top5_count - top1_count),
            "top5_top1_ratio": f"{ratio:.6f}",
            "confidence_level": confidence,
            "special_flag": special,
        })
        counts[confidence] += 1
        if special == "none":
            counts["none"] += 1
        for flag in ("mixed_candidate", "rare_candidate"):
            if flag in special.split(";"):
                counts[flag] += 1
        if special == "mixed_candidate;rare_candidate":
            counts["both_special"] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reviewed_rows)
    return counts


def main() -> int:
    args = parse_args()
    try:
        counts = process(args.input_csv, args.output)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote reviewed report: {args.output.resolve()}")
    for level in ("high", "medium", "review"):
        print(f"confidence_{level}: {counts[level]}")
    for flag in ("mixed_candidate", "rare_candidate", "none"):
        print(f"special_{flag}: {counts[flag]}")
    print(f"both_special_flags: {counts['both_special']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
