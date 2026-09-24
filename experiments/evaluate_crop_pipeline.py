#!/usr/bin/env python3
"""Offline evaluation of original-image and crop-aggregated BioCLIP results."""

from __future__ import annotations

import argparse
import csv
import math
import posixpath
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _directory in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


DEFAULT_THRESHOLDS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SUMMARY_FIELDS = [
    "strategy", "det_conf_threshold", "evaluated_images", "correct_top1", "top1_accuracy",
    "no_result_count", "rescued_count", "harmed_count",
]
DETAIL_FIELDS = [
    "source_file", "true_中文名", "strategy", "det_conf_threshold", "predicted_中文名",
    "score", "correct", "no_result", "baseline_prediction", "baseline_correct", "rescued", "harmed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original BioCLIP predictions and offline crop aggregation against a ground-truth CSV."
    )
    parser.add_argument("--ground-truth", required=True, type=Path,
                        help="CSV with source_file and true_中文名 columns")
    parser.add_argument("--baseline-predictions", required=True, type=Path,
                        help="Original-image BioCLIP 2.5 predictions.csv")
    parser.add_argument("--aggregated-first-detection", required=True, type=Path)
    parser.add_argument("--aggregated-best-score", required=True, type=Path)
    parser.add_argument("--crop-detections", required=True, type=Path,
                        help="Saved crop detection CSV; used to rebuild strategy C")
    parser.add_argument("--crop-predictions", required=True, type=Path,
                        help="Saved crop-level predictions CSV; no inference is run")
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)),
                        help="C detection-confidence thresholds, comma separated")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for the three evaluation CSV outputs")
    return parser.parse_args()


def normalize_path(value: Any) -> str:
    """Normalize separators, redundant path components, and case; never fuzzy-match."""
    text = unicodedata.normalize("NFC", str(value or "").strip()).replace("\\", "/")
    if not text:
        return ""
    normalized = posixpath.normpath(text)
    if normalized == ".":
        return ""
    return normalized.casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Input CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def require_columns(rows: list[dict[str, str]], path: Path, columns: set[str]) -> None:
    # DictReader keeps the header on an empty data set only via a separate read.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        headers = set(csv.DictReader(handle).fieldnames or [])
    missing = sorted(columns - headers)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def source_column(rows: list[dict[str, str]], path: Path) -> str:
    if rows and "source_file" in rows[0]:
        return "source_file"
    if rows and "file_name" in rows[0]:
        return "file_name"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        headers = set(csv.DictReader(handle).fieldnames or [])
    for name in ("source_file", "file_name"):
        if name in headers:
            return name
    raise ValueError(f"{path} must contain source_file or file_name")


class SourceIndex:
    """Index records by exact normalized path, with unique-basename fallback only."""

    def __init__(self, rows: list[dict[str, str]], path: Path):
        self.path = path
        self.by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.keys_by_basename: dict[str, set[str]] = defaultdict(set)
        self.source_field = source_column(rows, path)
        for row in rows:
            key = normalize_path(row.get(self.source_field, ""))
            if not key:
                continue
            self.by_key[key].append(row)
            self.keys_by_basename[posixpath.basename(key)].add(key)

    def lookup(self, source_file: str, gt_basename_counts: dict[str, int]) -> list[dict[str, str]]:
        key = normalize_path(source_file)
        if not key:
            return []
        if key in self.by_key:
            return self.by_key[key]
        basename = posixpath.basename(key)
        target_keys = self.keys_by_basename.get(basename, set())
        if len(target_keys) == 1 and gt_basename_counts.get(basename) == 1:
            return self.by_key[next(iter(target_keys))]
        return []


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--thresholds must be comma-separated numbers between 0 and 1") from exc
    if not thresholds or any(not math.isfinite(item) or not 0 <= item <= 1 for item in thresholds):
        raise ValueError("--thresholds must contain at least one number between 0 and 1")
    return list(dict.fromkeys(thresholds))


def format_threshold(value: float | None) -> str:
    return "" if value is None else format(value, ".8g")


def chinese_name(row: dict[str, str], latin_to_chinese: dict[str, str], *fields: str) -> str:
    for field in fields:
        value = str(row.get(field, "") or "").strip()
        if not value:
            continue
        return latin_to_chinese.get(value, value)
    return ""


def get_score(row: dict[str, str], *fields: str) -> str:
    for field in fields:
        value = str(row.get(field, "") or "").strip()
        if value:
            try:
                number = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid score {value!r} in column {field}") from exc
            if not math.isfinite(number):
                raise ValueError(f"Non-finite score in column {field}: {value!r}")
            return f"{number:.8f}"
    return ""


def build_latin_map(*datasets: list[dict[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for rows in datasets:
        for row in rows:
            latin = str(row.get("拉丁学名", "") or "").strip()
            chinese = str(row.get("中文名", "") or "").strip()
            if latin and chinese:
                mapping.setdefault(latin, chinese)
    return mapping


def top1_rows(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    field = source_column(rows, path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = normalize_path(row.get(field, ""))
        if key:
            grouped[key].append(row)
    result = []
    for key, candidates in grouped.items():
        if "rank" in candidates[0] and candidates[0].get("rank", "").strip():
            def rank_value(row: dict[str, str]) -> float:
                try:
                    return float(row.get("rank", ""))
                except ValueError:
                    return math.inf
            candidates = sorted(candidates, key=rank_value)
        result.append({**candidates[0], "_source_key": key})
    return result


def truth_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = read_csv(path)
    require_columns(rows, path, {"source_file", "true_中文名"})
    included: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        true_name = unicodedata.normalize("NFC", str(row.get("true_中文名", "") or "").strip())
        if not true_name:
            continue
        source = str(row.get("source_file", "") or "").strip()
        key = normalize_path(source)
        if not key:
            raise ValueError("A ground-truth row has a true_中文名 but an empty source_file")
        if key in seen:
            raise ValueError(f"Duplicate normalized source_file in ground truth: {source}")
        seen.add(key)
        included.append({"source_file": source, "true_中文名": true_name,
                         "_accepted_species": accepted_species(true_name), "_key": key})
    if not included:
        raise ValueError("Ground truth has no rows with a non-empty true_中文名")
    counts: dict[str, int] = defaultdict(int)
    for row in included:
        counts[posixpath.basename(row["_key"])] += 1
    return included, dict(counts)


def accepted_species(true_name: str) -> set[str]:
    """Return the non-empty species labels accepted by a ground-truth cell."""
    return {
        unicodedata.normalize("NFC", name.strip())
        for name in true_name.split("|")
        if name.strip()
    }


def matches_truth(prediction: str, true_name: str) -> bool:
    return bool(prediction and prediction in accepted_species(true_name))


def make_prediction(row: dict[str, str] | None, latin_to_chinese: dict[str, str],
                    name_fields: tuple[str, ...], score_fields: tuple[str, ...]) -> dict[str, str]:
    if row is None:
        return {"name": "", "score": ""}
    return {
        "name": chinese_name(row, latin_to_chinese, *name_fields),
        "score": get_score(row, *score_fields),
    }


def rebuild_c_predictions(
    truth: list[dict[str, str]], gt_basename_counts: dict[str, int],
    detection_rows: list[dict[str, str]], detection_path: Path,
    crop_prediction_rows: list[dict[str, str]], crop_prediction_path: Path,
    latin_to_chinese: dict[str, str], thresholds: list[float],
) -> dict[tuple[str, float], dict[str, str]]:
    detection_index = SourceIndex(detection_rows, detection_path)
    crop_rank1: dict[str, dict[str, str]] = {}
    for row in crop_prediction_rows:
        if str(row.get("status", "predicted")).strip().casefold() not in ("", "predicted"):
            continue
        if str(row.get("rank", "1")).strip() not in ("", "1"):
            continue
        crop_key = normalize_path(row.get("crop_file", ""))
        if not crop_key:
            continue
        if crop_key in crop_rank1:
            raise ValueError(f"Duplicate rank-1 crop prediction for crop_file: {row.get('crop_file')}")
        crop_rank1[crop_key] = row

    outputs: dict[tuple[str, float], dict[str, str]] = {}
    for gt in truth:
        source = gt["source_file"]
        gt_key = gt["_key"]
        source_detections = detection_index.lookup(source, gt_basename_counts)
        candidates: list[tuple[float, int, dict[str, str]]] = []
        for detection in source_detections:
            if str(detection.get("status", "detected")).strip().casefold() not in ("", "detected"):
                continue
            crop_key = normalize_path(detection.get("crop_file", ""))
            prediction = crop_rank1.get(crop_key)
            if prediction is None:
                continue
            confidence_text = str(detection.get("confidence", detection.get("detection_confidence", ""))).strip()
            try:
                confidence = float(confidence_text)
                detection_number = int(float(detection.get("detection_index", "0") or 0))
            except ValueError as exc:
                raise ValueError(f"Invalid detection confidence/index for {source}: {detection}") from exc
            if not math.isfinite(confidence):
                raise ValueError(f"Non-finite detection confidence for {source}: {confidence_text!r}")
            candidates.append((confidence, detection_number, prediction))

        for threshold in thresholds:
            eligible = [item for item in candidates if item[0] >= threshold]
            selected = max(eligible, key=lambda item: float(get_score(item[2], "score") or "-inf")) if eligible else None
            outputs[(gt_key, threshold)] = make_prediction(
                selected[2] if selected else None, latin_to_chinese,
                ("中文名", "拉丁学名"), ("score",),
            )
    return outputs


def metrics(rows: list[dict[str, str]], baseline_by_key: dict[str, dict[str, str]],
            prediction_by_key: dict[str, dict[str, str]], strategy: str,
            threshold: float | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct_count = no_result_count = rescued_count = harmed_count = 0
    details: list[dict[str, Any]] = []
    for gt in rows:
        key = gt["_key"]
        baseline = baseline_by_key.get(key, {"name": "", "score": ""})
        prediction = prediction_by_key.get(key, {"name": "", "score": ""})
        baseline_correct = matches_truth(baseline["name"], gt["true_中文名"])
        has_result = bool(prediction["name"])
        correct = bool(has_result and matches_truth(prediction["name"], gt["true_中文名"]))
        rescued = not baseline_correct and correct
        harmed = baseline_correct and not correct
        correct_count += int(correct)
        no_result_count += int(not has_result)
        rescued_count += int(rescued)
        harmed_count += int(harmed)
        details.append({
        "source_file": gt["source_file"],
        "true_中文名": gt["true_中文名"],
        "strategy": strategy,
        "det_conf_threshold": format_threshold(threshold),
            "predicted_中文名": prediction["name"],
            "score": prediction["score"],
            "correct": int(correct),
            "no_result": int(not has_result),
            "baseline_prediction": baseline["name"],
            "baseline_correct": int(baseline_correct),
            "rescued": int(rescued),
            "harmed": int(harmed),
        })
    evaluated = len(rows)
    summary = {
        "strategy": strategy,
        "det_conf_threshold": format_threshold(threshold),
        "evaluated_images": evaluated,
        "correct_top1": correct_count,
        "top1_accuracy": f"{correct_count / evaluated:.8f}" if evaluated else "",
        "no_result_count": no_result_count,
        "rescued_count": rescued_count,
        "harmed_count": harmed_count,
    }
    return summary, details


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    thresholds = parse_thresholds(args.thresholds)
    truth, gt_basename_counts = truth_rows(args.ground_truth)
    baseline_raw = read_csv(args.baseline_predictions)
    first_rows = read_csv(args.aggregated_first_detection)
    best_rows = read_csv(args.aggregated_best_score)
    detection_rows = read_csv(args.crop_detections)
    crop_prediction_rows = read_csv(args.crop_predictions)

    require_columns(baseline_raw, args.baseline_predictions, {"中文名"})
    require_columns(first_rows, args.aggregated_first_detection, {"selected_species"})
    require_columns(best_rows, args.aggregated_best_score, {"selected_species"})
    require_columns(detection_rows, args.crop_detections, {"status", "crop_file", "detection_index"})
    detection_headers = set(detection_rows[0]) if detection_rows else set()
    if "confidence" not in detection_headers and "detection_confidence" not in detection_headers:
        with args.crop_detections.open("r", encoding="utf-8-sig", newline="") as handle:
            detection_headers = set(csv.DictReader(handle).fieldnames or [])
        if "confidence" not in detection_headers and "detection_confidence" not in detection_headers:
            raise ValueError(f"{args.crop_detections} must contain confidence or detection_confidence")
    require_columns(crop_prediction_rows, args.crop_predictions, {"crop_file", "rank", "score"})

    baseline_rows = top1_rows(baseline_raw, args.baseline_predictions)
    baseline_index = SourceIndex(baseline_rows, args.baseline_predictions)
    first_index = SourceIndex(first_rows, args.aggregated_first_detection)
    best_index = SourceIndex(best_rows, args.aggregated_best_score)
    latin_to_chinese = build_latin_map(baseline_raw, crop_prediction_rows)

    baseline_by_key: dict[str, dict[str, str]] = {}
    first_by_key: dict[str, dict[str, str]] = {}
    best_by_key: dict[str, dict[str, str]] = {}
    for gt in truth:
        key = gt["_key"]
        baseline_matches = baseline_index.lookup(gt["source_file"], gt_basename_counts)
        first_matches = first_index.lookup(gt["source_file"], gt_basename_counts)
        best_matches = best_index.lookup(gt["source_file"], gt_basename_counts)
        baseline_by_key[key] = make_prediction(
            baseline_matches[0] if len(baseline_matches) == 1 else None,
            latin_to_chinese, ("中文名", "拉丁学名"), ("score",),
        )
        first_by_key[key] = make_prediction(
            first_matches[0] if len(first_matches) == 1 else None,
            latin_to_chinese, ("selected_species",), ("selected_score",),
        )
        best_by_key[key] = make_prediction(
            best_matches[0] if len(best_matches) == 1 else None,
            latin_to_chinese, ("selected_species",), ("selected_score",),
        )

    c_predictions = rebuild_c_predictions(
        truth, gt_basename_counts, detection_rows, args.crop_detections,
        crop_prediction_rows, args.crop_predictions, latin_to_chinese, thresholds,
    )

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    methods = [
        ("baseline", baseline_by_key, None),
        ("first_detection", first_by_key, None),
        ("best_score", best_by_key, None),
        ("det_filtered_best_score", {gt["_key"]: c_predictions[(gt["_key"], thresholds[0])] for gt in truth}, thresholds[0]),
    ]
    for name, predictions, threshold in methods:
        summary, details = metrics(truth, baseline_by_key, predictions, name, threshold)
        summary_rows.append(summary)
        detail_rows.extend(details)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predictions = {gt["_key"]: c_predictions[(gt["_key"], threshold)] for gt in truth}
        summary, details = metrics(truth, baseline_by_key, predictions, "det_filtered_best_score", threshold)
        threshold_rows.append(summary)
        if threshold != thresholds[0]:
            detail_rows.extend(details)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(output_dir / "threshold_accuracy_summary.csv", SUMMARY_FIELDS, threshold_rows)
    write_csv(output_dir / "evaluation_details.csv", DETAIL_FIELDS, detail_rows)

    print(f"Evaluated labeled ground-truth rows: {len(truth)}")
    print(f"Output directory: {output_dir}")
    for row in summary_rows:
        print(
            f"{row['strategy']} threshold={row['det_conf_threshold'] or 'n/a'}: "
            f"accuracy={row['top1_accuracy']}, no_result={row['no_result_count']}, "
            f"rescued={row['rescued_count']}, harmed={row['harmed_count']}"
        )


def main() -> int:
    args = parse_args()
    try:
        evaluate(args)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
