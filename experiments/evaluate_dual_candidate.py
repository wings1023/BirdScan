#!/usr/bin/env python3
"""Offline evaluation of additional per-detection Top-1 candidates."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _directory in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import evaluate_crop_pipeline as matching


SUMMARY_FIELDS = [
    "threshold", "total_images", "detection_positive_images", "no_detection_images",
    "true_additional_species_recovered", "false_additional_species_emitted",
    "truth_coverage_total_count", "truth_coverage_total",
    "truth_coverage_detection_positive_count", "truth_coverage_detection_positive",
    "average_species_count_per_image",
]
DETAIL_FIELDS = [
    "threshold", "source_file", "true_中文名", "detection_status", "primary_species", "primary_score",
    "additional_species", "additional_species_scores", "true_additional_species_recovered",
    "false_additional_species_emitted", "truth_covered", "species_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate first-detection primary plus additional detection Top-1 candidates from CSVs."
    )
    parser.add_argument("--ground-truth", required=True, type=Path,
                        help="CSV with source_file and true_中文名 (pipe-separated for multiple species)")
    parser.add_argument("--aggregated-first-detection", required=True, type=Path,
                        help="Existing aggregated_first_detection.csv")
    parser.add_argument("--crop-detections", required=True, type=Path,
                        help="Existing detections.csv")
    parser.add_argument("--crop-predictions", required=True, type=Path,
                        help="Existing crop predictions.csv; rank-1 rows are used")
    parser.add_argument("--thresholds", required=True,
                        help="Comma-separated BioCLIP score thresholds, e.g. 0.70,0.80,0.90,0.95")
    parser.add_argument("--output-dir", required=True, type=Path)
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


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    rows = matching.read_csv(path)
    headers = set(rows[0]) if rows else set()
    if not headers:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            headers = set(csv.DictReader(handle).fieldnames or [])
    missing = required - headers
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return rows


def indexed_unique(index: matching.SourceIndex, source: str,
                   basename_counts: dict[str, int]) -> dict[str, str] | None:
    rows = index.lookup(source, basename_counts)
    return rows[0] if len(rows) == 1 else None


def detection_number(value: Any, source: str) -> int:
    number = parse_number(value, "detection_index", source)
    if number is None or not number.is_integer():
        raise ValueError(f"Invalid detection_index {value!r} for {source}")
    return int(number)


def collect_additional_candidates(
    source: str,
    primary_index: int,
    primary_species: str,
    detections_index: matching.SourceIndex,
    crop_rank1: dict[str, dict[str, str]],
    latin_to_chinese: dict[str, str],
    basename_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Return unique additional species, retaining each species' highest Top-1 score."""
    detections = detections_index.lookup(source, basename_counts)
    by_species: dict[str, dict[str, Any]] = {}
    for detection in detections:
        status = str(detection.get("status", "detected") or "").strip().casefold()
        if status not in ("", "detected"):
            continue
        number = detection_number(detection.get("detection_index"), source)
        if number == primary_index:
            continue
        crop_key = matching.normalize_path(detection.get("crop_file", ""))
        prediction = crop_rank1.get(crop_key)
        if prediction is None:
            continue
        species = matching.chinese_name(prediction, latin_to_chinese, "中文名", "拉丁学名")
        score = parse_number(prediction.get("score"), "score", source)
        if not species or species == primary_species or score is None:
            continue
        previous = by_species.get(species)
        if previous is None:
            by_species[species] = {"species": species, "score": score}
        elif score > previous["score"]:
            previous["score"] = score
    return list(by_species.values())


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = matching.parse_thresholds(args.thresholds)
    truth, basename_counts = matching.truth_rows(args.ground_truth)
    first_rows = read_rows(
        args.aggregated_first_detection,
        {"source_file", "selected_species", "selected_score", "selected_detection_index"},
    )
    detection_rows = read_rows(
        args.crop_detections, {"source_file", "status", "detection_index", "crop_file"}
    )
    crop_rows = read_rows(
        args.crop_predictions, {"crop_file", "rank", "status", "score", "中文名", "拉丁学名"}
    )
    first_index = matching.SourceIndex(first_rows, args.aggregated_first_detection)
    detections_index = matching.SourceIndex(detection_rows, args.crop_detections)

    crop_rank1: dict[str, dict[str, str]] = {}
    for row in crop_rows:
        status = str(row.get("status", "predicted") or "").strip().casefold()
        if status not in ("", "predicted") or str(row.get("rank", "")).strip() != "1":
            continue
        crop_key = matching.normalize_path(row.get("crop_file", ""))
        if not crop_key:
            continue
        if crop_key in crop_rank1:
            raise ValueError(f"Duplicate rank-1 crop prediction for crop_file: {row.get('crop_file')}")
        crop_rank1[crop_key] = row
    latin_to_chinese = matching.build_latin_map(crop_rows)

    images: list[dict[str, Any]] = []
    for gt in truth:
        primary_row = indexed_unique(first_index, gt["source_file"], basename_counts)
        primary_species = matching.chinese_name(
            primary_row or {}, latin_to_chinese, "selected_species"
        )
        primary_score = parse_number(
            (primary_row or {}).get("selected_score"), "primary_score", gt["source_file"]
        )
        # first_detection aggregation selects detection_index 1; the explicit value
        # keeps the linkage faithful if the saved aggregation records another index.
        primary_index = detection_number(
            (primary_row or {}).get("selected_detection_index", "1") or "1", gt["source_file"]
        )
        additional = collect_additional_candidates(
            gt["source_file"], primary_index, primary_species, detections_index, crop_rank1,
            latin_to_chinese, basename_counts,
        )
        images.append({
            "source_file": gt["source_file"],
            "true_中文名": gt["true_中文名"],
            "_accepted_species": gt["_accepted_species"],
            "primary_species": primary_species,
            "primary_score": primary_score,
            "detection_status": image_detection_status(
                detections_index, gt["source_file"], basename_counts,
                (primary_row or {}).get("reason", ""),
            ),
            "_additional": additional,
        })

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    total = len(images)
    for threshold in thresholds:
        true_recovered_count = false_emitted_count = covered_count = positive_covered_count = 0
        total_species_count = 0
        positive_count = sum(image["detection_status"] == "detected" for image in images)
        no_detection_count = sum(image["detection_status"] == "no_detection" for image in images)
        for image in images:
            accepted = image["_accepted_species"]
            emitted = [item for item in image["_additional"] if item["score"] >= threshold]
            additional_species = [item["species"] for item in emitted]
            additional_scores = [matching.format_threshold(item["score"]) for item in emitted]
            true_additional = [species for species in additional_species if species in accepted]
            false_additional = [species for species in additional_species if species not in accepted]
            candidates = set(additional_species)
            if image["primary_species"]:
                candidates.add(image["primary_species"])
            covered = bool(candidates & accepted)
            count = len(candidates)

            true_recovered_count += len(true_additional)
            false_emitted_count += len(false_additional)
            covered_count += int(covered)
            if image["detection_status"] == "detected":
                positive_covered_count += int(covered)
            total_species_count += count
            details.append({
                "threshold": matching.format_threshold(threshold),
                "source_file": image["source_file"],
                "true_中文名": image["true_中文名"],
                "detection_status": image["detection_status"],
                "primary_species": image["primary_species"],
                "primary_score": "" if image["primary_score"] is None else matching.format_threshold(image["primary_score"]),
                "additional_species": "|".join(additional_species),
                "additional_species_scores": "|".join(additional_scores),
                "true_additional_species_recovered": "|".join(true_additional),
                "false_additional_species_emitted": "|".join(false_additional),
                "truth_covered": int(covered),
                "species_count": count,
            })
        summaries.append({
            "threshold": matching.format_threshold(threshold),
            "total_images": total,
            "detection_positive_images": positive_count,
            "no_detection_images": no_detection_count,
            "true_additional_species_recovered": true_recovered_count,
            "false_additional_species_emitted": false_emitted_count,
            "truth_coverage_total_count": covered_count,
            "truth_coverage_total": f"{covered_count / total:.8f}" if total else "",
            "truth_coverage_detection_positive_count": positive_covered_count,
            "truth_coverage_detection_positive": (
                f"{positive_covered_count / positive_count:.8f}" if positive_count else ""
            ),
            "average_species_count_per_image": f"{total_species_count / total:.8f}" if total else "",
        })
    return summaries, details


def image_detection_status(
    detections_index: matching.SourceIndex,
    source: str,
    basename_counts: dict[str, int],
    aggregate_reason: str,
) -> str:
    detection_rows = detections_index.lookup(source, basename_counts)
    statuses = {str(row.get("status", "") or "").strip().casefold() for row in detection_rows}
    if "detected" in statuses:
        return "detected"
    if "no_detection" in statuses or aggregate_reason.strip().casefold() == "no_detection":
        return "no_detection"
    if "error" in statuses or aggregate_reason.strip().casefold() == "detection_error":
        return "error"
    return "no_detection"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        summaries, details = evaluate(args)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "dual_candidate_summary.csv", SUMMARY_FIELDS, summaries)
        write_csv(output_dir / "dual_candidate_details.csv", DETAIL_FIELDS, details)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    print(f"Evaluated {summaries[0]['total_images'] if summaries else 0} labeled images")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
