#!/usr/bin/env python3
"""Offline evaluation of fusion rules for original and crop BioCLIP predictions."""

from __future__ import annotations

import argparse
import csv
import math
import posixpath
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _directory in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import evaluate_crop_pipeline as matching


LOCK_THRESHOLDS = (0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20)
SUMMARY_FIELDS = [
    "dataset_name", "rule", "parameter", "evaluated_images", "correct_top1",
    "top1_accuracy", "rescued_count", "harmed_count", "no_result_count", "net_gain",
]
DETAIL_FIELDS = [
    "dataset_name", "source_file", "true_中文名", "baseline_species", "baseline_score",
    "crop_species", "crop_score", "crop_exists", "rule", "parameter",
    "predicted_species", "predicted_score", "correct", "no_result", "baseline_correct",
    "rescued", "harmed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline comparison of BioCLIP baseline/crop fusion rules; performs no inference."
    )
    parser.add_argument(
        "--dataset", action="append", nargs=4, metavar=("NAME", "GROUND_TRUTH", "BASELINE", "BEST_SCORE"),
        help="Dataset tuple; repeat for multiple sets. Each CSV path is explicit.",
    )
    parser.add_argument("--ground-truth", type=Path, help="Single-dataset ground_truth.csv")
    parser.add_argument("--baseline-predictions", type=Path, help="Single-dataset original predictions.csv")
    parser.add_argument("--aggregated-best-score", type=Path, help="Single-dataset aggregated_best_score.csv")
    parser.add_argument("--dataset-name", default="dataset", help="Name used with single-dataset inputs")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_score(value: Any, field: str, source: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        score = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} {text!r} for {source}") from exc
    if not math.isfinite(score):
        raise ValueError(f"Non-finite {field} {text!r} for {source}")
    return score


def prediction_rows(truth: list[dict[str, str]], basename_counts: dict[str, int],
                    rows: list[dict[str, str]], path: Path, is_baseline: bool,
                    latin_to_chinese: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    if is_baseline:
        matching.require_columns(rows, path, {"中文名"})
        prepared = matching.top1_rows(rows, path)
        name_fields, score_fields = ("中文名", "拉丁学名"), ("score",)
    else:
        matching.require_columns(rows, path, {"selected_species"})
        prepared = rows
        name_fields, score_fields = ("selected_species",), ("selected_score",)
    index = matching.SourceIndex(prepared, path)
    output: dict[str, dict[str, Any]] = {}
    latin_map = latin_to_chinese if latin_to_chinese is not None else matching.build_latin_map(rows)
    for gt in truth:
        found = index.lookup(gt["source_file"], basename_counts)
        row = found[0] if len(found) == 1 else None
        pred = matching.make_prediction(row, latin_map, name_fields, score_fields)
        output[gt["_key"]] = {
            "species": pred["name"],
            "score": parse_score(pred["score"], "score", gt["source_file"]),
            "crop_exists": False,
        }
        if not is_baseline and row is not None:
            crop_count = parse_score(row.get("crop_count", ""), "crop_count", gt["source_file"])
            output[gt["_key"]]["crop_exists"] = bool(
                (crop_count is not None and crop_count > 0) or pred["name"]
            )
    return output


def prediction_for(rule: str, baseline: dict[str, Any], crop: dict[str, Any], parameter: float | None) -> tuple[str, float | None]:
    base_name, base_score = baseline["species"], baseline["score"]
    crop_name, crop_score = crop["species"], crop["score"]
    if rule == "baseline_only":
        return base_name, base_score
    if rule == "crop_only":
        return (crop_name, crop_score) if crop["crop_exists"] else ("", None)
    if rule == "crop_with_fallback":
        return (crop_name, crop_score) if crop["crop_exists"] else (base_name, base_score)
    if rule == "baseline_lock":
        if base_score is not None and parameter is not None and base_score >= parameter:
            return base_name, base_score
        return (crop_name, crop_score) if crop["crop_exists"] else (base_name, base_score)
    if rule == "score_margin":
        if not crop["crop_exists"]:
            return base_name, base_score
        if crop_name == base_name:
            return crop_name, crop_score
        if (crop_score is not None and base_score is not None and parameter is not None
                and crop_score >= base_score + parameter):
            return crop_name, crop_score
        return base_name, base_score
    raise ValueError(f"Unknown rule: {rule}")


def evaluate_one(dataset: str, truth: list[dict[str, str]], baseline: dict[str, dict[str, Any]],
                 crop: dict[str, dict[str, Any]], rule: str, parameter: float | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct_count = rescued_count = harmed_count = no_result_count = 0
    details: list[dict[str, Any]] = []
    for gt in truth:
        key = gt["_key"]
        base = baseline[key]
        crop_row = crop[key]
        predicted, score = prediction_for(rule, base, crop_row, parameter)
        baseline_correct = matching.matches_truth(base["species"], gt["true_中文名"])
        has_result = bool(predicted)
        correct = bool(has_result and matching.matches_truth(predicted, gt["true_中文名"]))
        rescued = not baseline_correct and correct
        harmed = baseline_correct and not correct
        correct_count += int(correct)
        rescued_count += int(rescued)
        harmed_count += int(harmed)
        no_result_count += int(not has_result)
        details.append({
            "dataset_name": dataset, "source_file": gt["source_file"], "true_中文名": gt["true_中文名"],
            "baseline_species": base["species"], "baseline_score": score_text(base["score"]),
            "crop_species": crop_row["species"], "crop_score": score_text(crop_row["score"]),
            "crop_exists": int(crop_row["crop_exists"]), "rule": rule,
            "parameter": score_text(parameter), "predicted_species": predicted,
            "predicted_score": score_text(score), "correct": int(correct), "no_result": int(not has_result),
            "baseline_correct": int(baseline_correct), "rescued": int(rescued), "harmed": int(harmed),
        })
    total = len(truth)
    return ({
        "dataset_name": dataset, "rule": rule, "parameter": score_text(parameter),
        "evaluated_images": total, "correct_top1": correct_count,
        "top1_accuracy": f"{correct_count / total:.8f}" if total else "",
        "rescued_count": rescued_count, "harmed_count": harmed_count,
        "no_result_count": no_result_count, "net_gain": rescued_count - harmed_count,
    }, details)


def score_text(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def summarize_combined(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["rule"], row["parameter"])].append(row)
    combined = []
    for (rule, parameter), group in grouped.items():
        total = sum(int(row["evaluated_images"]) for row in group)
        correct = sum(int(row["correct_top1"]) for row in group)
        rescued = sum(int(row["rescued_count"]) for row in group)
        harmed = sum(int(row["harmed_count"]) for row in group)
        combined.append({
            "dataset_name": "combined", "rule": rule, "parameter": parameter,
            "evaluated_images": total, "correct_top1": correct,
            "top1_accuracy": f"{correct / total:.8f}" if total else "",
            "rescued_count": rescued, "harmed_count": harmed,
            "no_result_count": sum(int(row["no_result_count"]) for row in group),
            "net_gain": rescued - harmed,
        })
    return combined


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_datasets(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path]]:
    if args.dataset:
        if any((args.ground_truth, args.baseline_predictions, args.aggregated_best_score)):
            raise ValueError("Use either repeated --dataset tuples or the single-dataset path options, not both")
        names = [entry[0] for entry in args.dataset]
        if len(set(names)) != len(names):
            raise ValueError("Dataset names must be unique")
        return [(name, Path(gt), Path(base), Path(best)) for name, gt, base, best in args.dataset]
    values = (args.ground_truth, args.baseline_predictions, args.aggregated_best_score)
    if any(value is None for value in values):
        raise ValueError("Provide repeated --dataset tuples or all three single-dataset CSV paths")
    return [(args.dataset_name, args.ground_truth, args.baseline_predictions, args.aggregated_best_score)]


def run(args: argparse.Namespace) -> None:
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    fixed_rules = (("baseline_only", None), ("crop_only", None), ("crop_with_fallback", None))
    for dataset, gt_path, baseline_path, crop_path in resolve_datasets(args):
        truth, basename_counts = matching.truth_rows(gt_path)
        base_raw = matching.read_csv(baseline_path)
        crop_raw = matching.read_csv(crop_path)
        latin_map = matching.build_latin_map(base_raw, crop_raw)
        baseline = prediction_rows(truth, basename_counts, base_raw, baseline_path, True)
        crop = prediction_rows(truth, basename_counts, crop_raw, crop_path, False, latin_map)
        configurations = [*fixed_rules]
        configurations.extend(("baseline_lock", threshold) for threshold in LOCK_THRESHOLDS)
        configurations.extend(("score_margin", margin) for margin in MARGINS)
        for rule, parameter in configurations:
            summary, rule_details = evaluate_one(dataset, truth, baseline, crop, rule, parameter)
            summaries.append(summary)
            details.extend(rule_details)
    summaries.extend(summarize_combined(summaries))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fusion_summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(args.output_dir / "fusion_details.csv", DETAIL_FIELDS, details)
    print(f"Wrote {args.output_dir / 'fusion_summary.csv'}")
    print(f"Wrote {args.output_dir / 'fusion_details.csv'}")
    for row in summaries:
        if row["dataset_name"] == "combined" or len(resolve_datasets(args)) == 1:
            print(f"{row['dataset_name']}/{row['rule']}[{row['parameter']}]: accuracy={row['top1_accuracy']} net_gain={row['net_gain']}")


def main() -> int:
    try:
        run(parse_args())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
