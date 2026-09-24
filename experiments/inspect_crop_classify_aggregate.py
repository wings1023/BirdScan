#!/usr/bin/env python3
"""Experiment: MegaDetector crops -> BioCLIP 2.5 -> original-image aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _directory in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import inspect_auto_crop as auto_crop
import scan_birds


DEFAULT_OUTPUT = REPO_ROOT / "reports" / "diagnostics" / "crop_aggregate"
AGGREGATE_MODES = ("first_detection", "best_score", "det_filtered_best_score")
DETECTION_FIELDS = [
    "source_file", "status", "detection_index", "confidence", "bbox_x", "bbox_y",
    "bbox_width", "bbox_height", "bbox_area_ratio", "margin", "crop_file",
]
PREDICTION_FIELDS = [
    "crop_file", "source_file", "detection_index", "detection_confidence", "status",
    "rank", *scan_birds.REQUIRED_COLUMNS, "score", "reason",
]
AGGREGATE_FIELDS = [
    "source_file", "crop_count", "valid_crop_count", "aggregate_mode", "det_conf_threshold",
    "selected_crop_file", "selected_detection_index", "selected_detection_confidence",
    "selected_species", "selected_score", "selected_rank2_species", "selected_rank2_score",
    "selected_rank3_species", "selected_rank3_score", "reason",
]
SWEEP_FIELDS = [
    "det_conf_threshold", "image_count", "no_detection_count", "detection_error_count", "no_valid_crop_count",
    "usable_image_count", "mean_selected_score", "median_selected_score", "selected_species_top10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated MegaDetector crop + BioCLIP 2.5 image-level aggregation experiment."
    )
    parser.add_argument("photo_dir", type=Path, help="Source photo directory (recursive JPG/JPEG/PNG)")
    parser.add_argument("--species-file", required=True, type=Path, help="Candidate species CSV or XLSX")
    parser.add_argument("--model", choices=("bioclip25",), default="bioclip25", help="Fixed experiment model")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-count", type=int, default=80)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Minimum MegaDetector animal confidence (passed to detector inference)")
    parser.add_argument("--det-confidence-threshold", type=float, default=0.0,
                        help="Detection confidence cutoff for det_filtered_best_score")
    parser.add_argument("--aggregate-mode", choices=AGGREGATE_MODES, default="best_score",
                        help="Primary strategy highlighted in the terminal; all three CSVs are always written")
    parser.add_argument("--det-threshold-sweep", default=None,
                        help="Optional comma-separated detection confidence cutoffs, e.g. 0,0.2,0.4,0.6")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[float] | None:
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.prompt_count < 1:
        parser.error("--prompt-count must be at least 1")
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("--margin must be finite and >= 0")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not math.isfinite(args.det_confidence_threshold) or not 0 <= args.det_confidence_threshold <= 1:
        parser.error("--det-confidence-threshold must be between 0 and 1")
    if not args.det_threshold_sweep:
        return None
    try:
        thresholds = [float(value.strip()) for value in args.det_threshold_sweep.split(",") if value.strip()]
    except ValueError:
        parser.error("--det-threshold-sweep must be comma-separated numbers between 0 and 1")
    if not thresholds or any(not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        parser.error("--det-threshold-sweep must contain values between 0 and 1")
    return list(dict.fromkeys(thresholds))


def safe_relative_stem(relative_path: Path) -> str:
    relative_path = Path(relative_path)
    safe_parents = [
        re.sub(r"[^A-Za-z0-9._-]+", "_", part)
        for part in relative_path.parent.parts
    ]
    return "__".join([*safe_parents, auto_crop.safe_stem(relative_path)])


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def detect_and_crop(
    image_paths: list[Path], photo_dir: Path, output_dir: Path, margin: float, threshold: float, device: str
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[tuple[str, str]]]:
    crop_dir = output_dir / "auto_crop" / "crops"
    preview_dir = output_dir / "auto_crop" / "preview"
    crop_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    detector = auto_crop.load_detector(device)
    detection_rows: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {str(path.relative_to(photo_dir)): [] for path in image_paths}
    failures: list[tuple[str, str]] = []

    for image_path in image_paths:
        relative = image_path.relative_to(photo_dir)
        source_file = relative.as_posix()
        try:
            result = detector.single_image_detection(str(image_path), det_conf_thres=threshold)
            animals = [item for item in auto_crop.extract_animals(result) if item[0] >= threshold]
            with Image.open(image_path) as opened:
                image = opened.copy()
        except Exception as exc:
            message = str(exc)
            failures.append((source_file, message))
            detection_rows.append({"source_file": source_file, "status": "error", "margin": margin})
            print(f"Detection error; continuing: {source_file}: {message}", file=sys.stderr)
            continue

        width_px, height_px = image.size
        preview_boxes: list[tuple[float, float, float, float]] = []
        for detection_index, (confidence, box) in enumerate(animals, start=1):
            x1, y1, x2, y2 = box
            x = max(0.0, min(float(width_px), x1))
            y = max(0.0, min(float(height_px), y1))
            right = max(x, min(float(width_px), x2))
            bottom = max(y, min(float(height_px), y2))
            bbox_width, bbox_height = right - x, bottom - y
            bounds = auto_crop.padded_box((x, y, right, bottom), margin, width_px, height_px)
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                failures.append((source_file, f"Invalid empty box for detection {detection_index}"))
                continue

            crop_name = (
                f"{safe_relative_stem(relative)}__det{detection_index:02d}__conf{confidence:.2f}"
                f"{image_path.suffix.lower()}"
            )
            crop_path = crop_dir / crop_name
            image.crop(bounds).save(crop_path)
            crop_relative = crop_path.relative_to(output_dir).as_posix()
            crop_record = {
                "crop_path": crop_path,
                "crop_file": crop_relative,
                "source_file": source_file,
                "detection_index": detection_index,
                "detection_confidence": confidence,
                "status": "pending",
            }
            by_source[source_file].append(crop_record)
            preview_boxes.append((x, y, right, bottom))
            detection_rows.append({
                "source_file": source_file,
                "status": "detected",
                "detection_index": detection_index,
                "confidence": f"{confidence:.6f}",
                "bbox_x": f"{x:.2f}",
                "bbox_y": f"{y:.2f}",
                "bbox_width": f"{bbox_width:.2f}",
                "bbox_height": f"{bbox_height:.2f}",
                "bbox_area_ratio": f"{bbox_width * bbox_height / (width_px * height_px):.8f}",
                "margin": margin,
                "crop_file": crop_relative,
            })

        if not animals:
            detection_rows.append({"source_file": source_file, "status": "no_detection", "margin": margin})
        if preview_boxes:
            preview_path = preview_dir / f"{safe_relative_stem(relative)}__preview.jpg"
            auto_crop.make_preview(image, preview_boxes, preview_path)

    write_csv(output_dir / "auto_crop" / "detections.csv", DETECTION_FIELDS, detection_rows)
    return detection_rows, by_source, failures


def enrich_crop_predictions(
    crops_by_source: dict[str, list[dict[str, Any]]], predictions: list[dict[str, Any]],
    failures: list[tuple[Path, str]], species: dict[str, dict[str, str]], output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    path_to_crop = {
        str(Path(record["crop_path"]).resolve()): record
        for crops in crops_by_source.values() for record in crops
    }
    rows_by_crop_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        prediction_path = str(Path(prediction["file_name"]).resolve())
        record = path_to_crop.get(prediction_path)
        if record is None:
            continue
        latin_name = prediction["classification"]
        crop_rows = rows_by_crop_path[prediction_path]
        crop_rows.append({
            **record,
            "status": "predicted",
            # pybioclip returns each image's Top-K in descending order without
            # a rank field; this experiment assigns rank by row order.
            "rank": len(crop_rows) + 1,
            "species": species[latin_name],
            "score": float(prediction["score"]),
        })

    failed_by_crop = {str(Path(path).resolve()): message for path, message in failures}
    enriched_rows: list[dict[str, Any]] = []
    for source_file, crops in crops_by_source.items():
        for crop in crops:
            resolved_crop = str(Path(crop["crop_path"]).resolve())
            ranked = sorted(rows_by_crop_path.get(resolved_crop, []), key=lambda row: row["rank"])
            if ranked:
                crop.update(status="predicted", predictions=ranked)
                for ranked_item in ranked:
                    enriched_rows.append({
                        "crop_file": crop["crop_file"],
                        "source_file": source_file,
                        "detection_index": crop["detection_index"],
                        "detection_confidence": f"{crop['detection_confidence']:.6f}",
                        "status": "predicted",
                        "rank": ranked_item["rank"],
                        **ranked_item["species"],
                        "score": f"{ranked_item['score']:.6f}",
                    })
            else:
                crop.update(status="error", predictions=[])
                enriched_rows.append({
                    "crop_file": crop["crop_file"],
                    "source_file": source_file,
                    "detection_index": crop["detection_index"],
                    "detection_confidence": f"{crop['detection_confidence']:.6f}",
                    "status": "error",
                    "reason": failed_by_crop.get(resolved_crop, "No prediction returned"),
                })
    write_csv(output_dir / "crop_predictions" / "predictions.csv", PREDICTION_FIELDS, enriched_rows)
    return enriched_rows


def _ranked(crop: dict[str, Any]) -> list[dict[str, Any]]:
    return crop.get("predictions", [])


def aggregate_one(
    source_file: str, crops: list[dict[str, Any]], mode: str, det_conf_threshold: float | None,
    detection_status: str | None = None,
) -> dict[str, Any]:
    predicted = [crop for crop in crops if _ranked(crop)]
    eligible = predicted
    reason = "max_bioclip_score"
    if mode == "first_detection":
        eligible = [crop for crop in predicted if int(crop["detection_index"]) == 1]
        reason = "det01"
    elif mode == "det_filtered_best_score":
        assert det_conf_threshold is not None
        eligible = [
            crop for crop in predicted
            if float(crop["detection_confidence"]) >= det_conf_threshold
        ]
        reason = "max_bioclip_score_after_det_filter"

    selected = None
    if eligible:
        selected = max(eligible, key=lambda crop: float(_ranked(crop)[0]["score"]))
    elif not crops:
        reason = "detection_error" if detection_status == "error" else "no_detection"
    else:
        reason = "no_valid_crop"

    output: dict[str, Any] = {
        "source_file": source_file,
        "crop_count": len(crops),
        "valid_crop_count": len(eligible),
        "aggregate_mode": mode,
        "det_conf_threshold": "" if det_conf_threshold is None else f"{det_conf_threshold:.6f}",
        "selected_crop_file": "",
        "selected_detection_index": "",
        "selected_detection_confidence": "",
        "selected_species": "",
        "selected_score": "",
        "selected_rank2_species": "",
        "selected_rank2_score": "",
        "selected_rank3_species": "",
        "selected_rank3_score": "",
        "reason": reason,
    }
    if selected is not None:
        topk = _ranked(selected)
        output.update({
            "selected_crop_file": selected["crop_file"],
            "selected_detection_index": selected["detection_index"],
            "selected_detection_confidence": f"{float(selected['detection_confidence']):.6f}",
            "selected_species": topk[0]["species"]["拉丁学名"],
            "selected_score": f"{float(topk[0]['score']):.6f}",
        })
        if len(topk) > 1:
            output["selected_rank2_species"] = topk[1]["species"]["拉丁学名"]
            output["selected_rank2_score"] = f"{float(topk[1]['score']):.6f}"
        if len(topk) > 2:
            output["selected_rank3_species"] = topk[2]["species"]["拉丁学名"]
            output["selected_rank3_score"] = f"{float(topk[2]['score']):.6f}"
    return output


def aggregate_all(
    source_files: list[str], crops_by_source: dict[str, list[dict[str, Any]]], mode: str,
    det_conf_threshold: float | None, detection_statuses: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        aggregate_one(
            source, crops_by_source.get(source, []), mode, det_conf_threshold,
            (detection_statuses or {}).get(source),
        )
        for source in source_files
    ]


def make_sweep_summary(
    threshold: float, aggregate_rows: list[dict[str, Any]], no_detection_sources: set[str],
    detection_error_sources: set[str] | None = None,
) -> dict[str, Any]:
    usable = [row for row in aggregate_rows if row["selected_species"]]
    scores = [float(row["selected_score"]) for row in usable]
    top_species = Counter(row["selected_species"] for row in usable)
    return {
        "det_conf_threshold": f"{threshold:.6f}",
        "image_count": len(aggregate_rows),
        "no_detection_count": sum(row["source_file"] in no_detection_sources for row in aggregate_rows),
        "detection_error_count": sum(
            row["source_file"] in (detection_error_sources or set()) for row in aggregate_rows
        ),
        "no_valid_crop_count": sum(row["reason"] == "no_valid_crop" for row in aggregate_rows),
        "usable_image_count": len(usable),
        "mean_selected_score": f"{statistics.mean(scores):.6f}" if scores else "",
        "median_selected_score": f"{statistics.median(scores):.6f}" if scores else "",
        "selected_species_top10": json.dumps(top_species.most_common(10), ensure_ascii=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    args = parse_args()
    threshold_sweep = validate_args(args, parser)
    photo_dir = args.photo_dir.expanduser().resolve()
    species_file = args.species_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crop_predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregated").mkdir(parents=True, exist_ok=True)

    try:
        species = scan_birds.load_species(species_file)
        image_paths = scan_birds.find_images(photo_dir)
    except (ValueError, OSError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    if not image_paths:
        print("No JPG, JPEG, or PNG images found.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    print(f"Images: {len(image_paths)}; candidate species: {len(species)}")
    print(f"Step A: MegaDetector V6 detection + crop on {args.device}...")
    try:
        detection_rows, crops_by_source, detection_failures = detect_and_crop(
            image_paths, photo_dir, output_dir, args.margin, args.threshold, args.device
        )
    except Exception as exc:
        print(f"Cannot load/run MegaDetector V6: {exc}", file=sys.stderr)
        return 2

    all_crops = [crop for crops in crops_by_source.values() for crop in crops]
    print(f"Detection crops: {len(all_crops)}; detection failures: {len(detection_failures)}")
    prediction_failures: list[tuple[Path, str]] = []
    predictions: list[dict[str, Any]] = []
    cache_status = "not used (no crops)"
    cache_path: Path | None = None
    if all_crops:
        try:
            print(f"Step B: loading BioCLIP 2.5 Huge on {args.device} ({args.prompt_count} prompts/species)...")
            classifier, candidate_seconds, cache_hit, cache_path = scan_birds.build_bioclip25_classifier(
                list(species), args.device, prompt_count=args.prompt_count
            )
            cache_status = "hit" if cache_hit else "miss / created"
            print(f"Candidate text cache {cache_status}: {cache_path} ({candidate_seconds:.3f}s)")
            crop_paths = [crop["crop_path"] for crop in all_crops]
            predictions, prediction_failures = scan_birds.predict_resiliently(
                classifier, crop_paths, min(5, len(species)), args.batch_size
            )
        except Exception as exc:
            print(f"BioCLIP 2.5 failed: {exc}", file=sys.stderr)
            prediction_failures = [(crop["crop_path"], str(exc)) for crop in all_crops]

    enrich_crop_predictions(
        crops_by_source, predictions, prediction_failures, species, output_dir
    )
    source_files = [path.relative_to(photo_dir).as_posix() for path in image_paths]
    no_detection_sources: set[str] = set()
    detection_error_sources: set[str] = set()
    detection_statuses: dict[str, str] = {}
    for row in detection_rows:
        source = row["source_file"]
        detection_statuses[source] = row["status"]
        if row["status"] == "no_detection":
            no_detection_sources.add(source)
        elif row["status"] == "error":
            detection_error_sources.add(source)
    # Include every crop, also crops whose BioCLIP inference failed.
    all_crop_records = crops_by_source
    selected_threshold: float | None = args.det_confidence_threshold
    for mode in AGGREGATE_MODES:
        mode_threshold = selected_threshold if mode == "det_filtered_best_score" else None
        rows = aggregate_all(source_files, all_crop_records, mode, mode_threshold, detection_statuses)
        filename = {
            "first_detection": "aggregated_first_detection.csv",
            "best_score": "aggregated_best_score.csv",
            "det_filtered_best_score": "aggregated_det_filtered_best_score.csv",
        }[mode]
        write_csv(output_dir / "aggregated" / filename, AGGREGATE_FIELDS, rows)

    sweep_rows: list[dict[str, Any]] = []
    if threshold_sweep is not None:
        for sweep_threshold in threshold_sweep:
            rows = aggregate_all(
                source_files, all_crop_records, "det_filtered_best_score", sweep_threshold,
                detection_statuses,
            )
            sweep_rows.append(
                make_sweep_summary(sweep_threshold, rows, no_detection_sources, detection_error_sources)
            )
    write_csv(output_dir / "aggregated" / "threshold_sweep_summary.csv", SWEEP_FIELDS, sweep_rows)

    summary_lines = [
        "MegaDetector crop + BioCLIP 2.5 + original-image aggregation experiment",
        f"Images: {len(image_paths)}",
        f"Detected crops: {len(all_crops)}",
        f"No-detection images: {len(no_detection_sources)}",
        f"Detection failures: {len(detection_error_sources)}",
        f"BioCLIP crop prediction failures: {len(prediction_failures)}",
        f"Model: bioclip25 ({scan_birds.BIOCLIP25_MODEL_STR})",
        f"Prompt count: {args.prompt_count}",
        f"Batch size: {args.batch_size}",
        f"Candidate text cache: {cache_status}{f' ({cache_path})' if cache_path else ''}",
        f"Primary aggregate mode: {args.aggregate_mode}",
        f"Detection confidence filter: {args.det_confidence_threshold:.6f}",
        f"Threshold sweep: {threshold_sweep if threshold_sweep is not None else 'not requested'}",
        f"Elapsed seconds: {time.perf_counter() - started:.3f}",
        f"Output directory: {output_dir}",
    ]
    if detection_failures:
        summary_lines.extend(["", "Detection failures:"])
        summary_lines.extend(f"- {source}: {message}" for source, message in detection_failures)
    if prediction_failures:
        summary_lines.extend(["", "BioCLIP crop prediction failures:"])
        summary_lines.extend(f"- {path}: {message}" for path, message in prediction_failures)
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nPrimary strategy: {args.aggregate_mode}")
    print(f"Experiment outputs written to {output_dir}")
    return 0 if not prediction_failures and not detection_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
