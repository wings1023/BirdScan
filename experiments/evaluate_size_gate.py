#!/usr/bin/env python3
"""Offline evaluation of bbox-size gating between baseline and crop predictions.

Inputs may use historical best_score crops, not production's first-detection primary.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _directory in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import evaluate_crop_pipeline as matching


THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)
SUMMARY_FIELDS = [
    "dataset", "threshold", "evaluated_images", "correct_top1", "top1_accuracy",
    "rescued_count", "harmed_count", "net_gain", "fallback_no_detection_count",
]
DETAIL_FIELDS = [
    "dataset", "source_file", "true_中文名", "baseline_species", "baseline_score",
    "crop_species", "crop_score", "selected_crop", "detection_confidence",
    "bbox_area_ratio", "threshold", "selected_source", "final_species", "correct",
    "rescued", "harmed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline size-gate evaluation; reads existing CSVs and performs no inference."
    )
    parser.add_argument(
        "--dataset", action="append", nargs=5,
        metavar=("NAME", "GROUND_TRUTH", "BASELINE", "BEST_SCORE", "DETECTIONS"),
        help="Dataset tuple; repeat for multiple sets.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "reports" / "diagnostics" / "size_gate_evaluation")
    return parser.parse_args()


def parse_number(value: Any, field: str, source: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} {text!r} for {source}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field} {text!r} for {source}")
    return number


def dataset_records(name: str, gt_path: Path, baseline_path: Path,
                    best_path: Path, detections_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    truth, basename_counts = matching.truth_rows(gt_path)
    baseline_rows = matching.read_csv(baseline_path)
    best_rows = matching.read_csv(best_path)
    detection_rows = matching.read_csv(detections_path)
    matching.require_columns(baseline_rows, baseline_path, {"中文名"})
    matching.require_columns(best_rows, best_path, {"selected_species", "selected_crop_file"})
    matching.require_columns(
        detection_rows, detections_path,
        {"source_file", "crop_file", "confidence", "bbox_width", "bbox_height"},
    )

    latin_map = matching.build_latin_map(baseline_rows, best_rows)
    baseline_index = matching.SourceIndex(matching.top1_rows(baseline_rows, baseline_path), baseline_path)
    best_index = matching.SourceIndex(best_rows, best_path)
    detection_crop_keys: list[str] = []
    for row in detection_rows:
        crop_key = matching.normalize_path(row.get("crop_file", ""))
        if crop_key:
            detection_crop_keys.append(crop_key)
    crop_path_rows = [dict(row, source_file=row.get("crop_file", "")) for row in detection_rows]
    detection_index = matching.SourceIndex(crop_path_rows, detections_path)

    # Unique-basename fallback is allowed only when the selected crop references
    # and detection crop paths are each unambiguous.
    selected_crop_keys = [
        matching.normalize_path(row.get("selected_crop_file", "")) for row in best_rows
        if str(row.get("selected_crop_file", "") or "").strip()
    ]
    selected_basename_counts: Counter[str] = Counter()
    for key in selected_crop_keys:
        selected_basename_counts[Path(key).name] += 1
    detection_unique_counts: Counter[str] = Counter(Path(key).name for key in set(detection_crop_keys))
    crop_basename_counts = {
        basename: count for basename, count in selected_basename_counts.items()
        if count == 1 and detection_unique_counts.get(basename) == 1
    }

    output: list[dict[str, Any]] = []
    for gt in truth:
        baseline_matches = baseline_index.lookup(gt["source_file"], basename_counts)
        best_matches = best_index.lookup(gt["source_file"], basename_counts)
        baseline_row = baseline_matches[0] if len(baseline_matches) == 1 else None
        best_row = best_matches[0] if len(best_matches) == 1 else None
        base_pred = matching.make_prediction(
            baseline_row, latin_map, ("中文名", "拉丁学名"), ("score",)
        )
        crop_pred = matching.make_prediction(
            best_row, latin_map, ("selected_species",), ("selected_score",)
        )
        selected_crop = str((best_row or {}).get("selected_crop_file", "") or "").strip()
        detection: dict[str, str] | None = None
        if selected_crop:
            hits = detection_index.lookup(selected_crop, crop_basename_counts)
            if len(hits) == 1:
                detection = hits[0]
        area_ratio: float | None = None
        if detection is not None:
            image_width = parse_number(detection.get("image_width"), "image_width", gt["source_file"])
            image_height = parse_number(detection.get("image_height"), "image_height", gt["source_file"])
            bbox_width = parse_number(detection.get("bbox_width"), "bbox_width", gt["source_file"])
            bbox_height = parse_number(detection.get("bbox_height"), "bbox_height", gt["source_file"])
            if image_width is not None and image_height is not None:
                if image_width <= 0 or image_height <= 0:
                    raise ValueError(f"Non-positive image dimensions for {gt['source_file']}")
                if bbox_width is not None and bbox_height is not None:
                    area_ratio = bbox_width * bbox_height / (image_width * image_height)
            else:
                # inspect_auto_crop.py already stores this exact ratio because
                # its detections.csv does not include source image dimensions.
                area_ratio = parse_number(detection.get("bbox_area_ratio"), "bbox_area_ratio", gt["source_file"])
            if area_ratio is not None and not 0 <= area_ratio <= 1:
                raise ValueError(f"bbox_area_ratio outside [0, 1] for {gt['source_file']}: {area_ratio}")

        output.append({
            "dataset": name,
            "source_file": gt["source_file"],
            "true_中文名": gt["true_中文名"],
            "_key": gt["_key"],
            "baseline_species": base_pred["name"],
            "baseline_score": parse_number(base_pred["score"], "baseline_score", gt["source_file"]),
            "crop_species": crop_pred["name"],
            "crop_score": parse_number(crop_pred["score"], "crop_score", gt["source_file"]),
            "selected_crop": selected_crop if detection is not None else "",
            "detection_confidence": parse_number(
                detection.get("confidence") if detection else "", "detection_confidence", gt["source_file"]
            ),
            "bbox_area_ratio": area_ratio,
            "has_detection": detection is not None,
            "has_valid_crop": bool(detection is not None and crop_pred["name"]),
        })
    return output, basename_counts


def evaluate_threshold(records: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct_count = rescued_count = harmed_count = fallback_count = 0
    details = []
    for row in records:
        has_crop = row["has_valid_crop"]
        use_crop = bool(has_crop and row["bbox_area_ratio"] is not None
                        and row["bbox_area_ratio"] <= threshold)
        selected_source = "crop" if use_crop else "baseline"
        final_species = row["crop_species"] if use_crop else row["baseline_species"]
        truth = row["true_中文名"]
        baseline_correct = matching.matches_truth(row["baseline_species"], truth)
        correct = matching.matches_truth(final_species, truth)
        rescued = not baseline_correct and correct
        harmed = baseline_correct and not correct
        correct_count += int(correct)
        rescued_count += int(rescued)
        harmed_count += int(harmed)
        fallback_count += int(not row["has_detection"])
        details.append({
            "dataset": row["dataset"], "source_file": row["source_file"], "true_中文名": truth,
            "baseline_species": row["baseline_species"], "baseline_score": format_num(row["baseline_score"]),
            "crop_species": row["crop_species"], "crop_score": format_num(row["crop_score"]),
            "selected_crop": row["selected_crop"],
            "detection_confidence": format_num(row["detection_confidence"]),
            "bbox_area_ratio": format_num(row["bbox_area_ratio"]), "threshold": format_num(threshold),
            "selected_source": selected_source, "final_species": final_species,
            "correct": int(correct), "rescued": int(rescued), "harmed": int(harmed),
        })
    total = len(records)
    summary = {
        "dataset": records[0]["dataset"] if records else "",
        "threshold": format_num(threshold), "evaluated_images": total,
        "correct_top1": correct_count,
        "top1_accuracy": f"{correct_count / total:.8f}" if total else "",
        "rescued_count": rescued_count, "harmed_count": harmed_count,
        "net_gain": rescued_count - harmed_count,
        "fallback_no_detection_count": fallback_count,
    }
    return summary, details


def format_num(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def combined_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_threshold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_threshold[row["threshold"]].append(row)
    result = []
    for threshold, group in by_threshold.items():
        total = sum(int(row["evaluated_images"]) for row in group)
        correct = sum(int(row["correct_top1"]) for row in group)
        rescued = sum(int(row["rescued_count"]) for row in group)
        harmed = sum(int(row["harmed_count"]) for row in group)
        result.append({
            "dataset": "combined", "threshold": threshold, "evaluated_images": total,
            "correct_top1": correct, "top1_accuracy": f"{correct / total:.8f}" if total else "",
            "rescued_count": rescued, "harmed_count": harmed, "net_gain": rescued - harmed,
            "fallback_no_detection_count": sum(int(row["fallback_no_detection_count"]) for row in group),
        })
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset NAME GROUND_TRUTH BASELINE BEST_SCORE DETECTIONS is required")
    names = [item[0] for item in args.dataset]
    if len(set(names)) != len(names):
        raise ValueError("Dataset names must be unique")
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for name, gt, baseline, best, detections in args.dataset:
        records, _ = dataset_records(name, Path(gt), Path(baseline), Path(best), Path(detections))
        for threshold in THRESHOLDS:
            summary, details = evaluate_threshold(records, threshold)
            summary_rows.append(summary)
            detail_rows.extend(details)
    summary_rows.extend(combined_summaries(summary_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "size_gate_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(args.output_dir / "size_gate_details.csv", DETAIL_FIELDS, detail_rows)
    print(f"Wrote {args.output_dir / 'size_gate_summary.csv'}")
    print(f"Wrote {args.output_dir / 'size_gate_details.csv'}")


def main() -> int:
    try:
        run(parse_args())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
