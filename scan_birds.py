#!/usr/bin/env python3
"""Batch bird identification with BioCLIP 2 and a user-supplied species list.

The script only reads image files.  It writes reports to the selected output
directory and never renames, moves, edits, or embeds metadata in source photos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_COLUMNS = ("鸟种编号", "中文名", "拉丁学名", "英文名称")
REQUIRED_VALUE_COLUMNS = ("鸟种编号", "拉丁学名")
CSV_ENCODING = "utf-8-sig"
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CACHE_FORMAT_VERSION = 1
CACHE_DIR = Path.home() / ".cache" / "birdscan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use BioCLIP 2 to identify birds in a photo folder. Source photos are read only."
    )
    parser.add_argument("photo_dir", type=Path, help="Folder to scan recursively for JPG/JPEG/PNG files")
    parser.add_argument("--species-file", required=True, type=Path, help="Candidate species CSV or .xlsx file")
    parser.add_argument("--top-k", type=int, default=5, help="Predictions saved per image (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per BioCLIP inference batch (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Top-1 uncertainty threshold (default: 0.5)")
    parser.add_argument("--device", default="mps", help="BioCLIP device, e.g. mps or cpu (default: mps)")
    parser.add_argument("--output-dir", type=Path, default=Path("bird_report"), help="Report directory (default: bird_report)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def column_index(cell_ref: str) -> int:
    """Return zero-based spreadsheet column index from an Excel cell reference."""
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter.upper()) - ord("A") + 1
    return value - 1


def read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet of a normal .xlsx file without an extra dependency."""
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{XLSX_NS}si"):
                shared_strings.append("".join(item.itertext()))

        workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheet = workbook_xml.find(f"{XLSX_NS}sheets/{XLSX_NS}sheet")
        if sheet is None:
            raise ValueError("Excel file has no worksheets")
        relation_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        relations = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        target = next(
            (item.attrib["Target"] for item in relations if item.attrib.get("Id") == relation_id), None
        )
        if not target:
            raise ValueError("Unable to locate the first worksheet in the Excel file")
        sheet_path = "xl/" + target.lstrip("/")
        sheet_root = ET.fromstring(workbook.read(sheet_path))

        table: list[list[str]] = []
        for row in sheet_root.findall(f".//{XLSX_NS}sheetData/{XLSX_NS}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{XLSX_NS}c"):
                index = column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{XLSX_NS}v")
                if cell_type == "inlineStr":
                    value = "".join(cell.itertext())
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text or "0")]
                else:
                    value = value_node.text or ""
                values[index] = value.strip()
            if values:
                width = max(values) + 1
                table.append([values.get(index, "") for index in range(width)])

    if not table:
        return []
    headers = table[0]
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(headers)}
        for row in table[1:]
        if any(row)
    ]


def load_species(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Species file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".xlsx":
        try:
            rows = read_xlsx_rows(path)
        except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
            raise ValueError(f"Cannot read Excel file {path}: {exc}") from exc
    else:
        raise ValueError("--species-file must be a CSV or .xlsx file")

    if not rows:
        raise ValueError("Species file contains no data rows")
    headers = set(rows[0])
    missing = [column for column in REQUIRED_COLUMNS if column not in headers]
    if missing:
        raise ValueError(f"Species file is missing required columns: {', '.join(missing)}")

    mapping: dict[str, dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        cleaned = {column: str(row.get(column, "")).strip() for column in REQUIRED_COLUMNS}
        empty = [column for column in REQUIRED_VALUE_COLUMNS if not cleaned[column]]
        if empty:
            raise ValueError(f"Species file row {number} has blank values: {', '.join(empty)}")
        latin_name = cleaned["拉丁学名"]
        if latin_name in mapping:
            raise ValueError(f"Duplicate Latin name (must be unique): {latin_name}")
        mapping[latin_name] = cleaned
    return mapping


def find_images(photo_dir: Path) -> list[Path]:
    if not photo_dir.is_dir():
        raise ValueError(f"Photo folder does not exist or is not a directory: {photo_dir}")
    return sorted(path for path in photo_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_images(paths: Iterable[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Check readability up front so a corrupt file cannot poison a batch."""
    from PIL import Image

    valid: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except Exception as exc:  # Pillow uses several format-specific exceptions.
            failed.append((path, f"Unreadable image: {exc}"))
    return valid, failed


def candidate_cache_path(model: str, pretrained: str | None, labels: list[str]) -> tuple[Path, dict[str, Any]]:
    """Build a cache key from the complete model configuration and label order."""
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "model": model,
        "pretrained": pretrained,
        "labels_sha256": hashlib.sha256(
            json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "label_count": len(labels),
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"candidate-embeddings-{key}.pt", metadata


def build_classifier(labels: list[str], device: str) -> tuple[Any, float, bool, Path]:
    """Load BioCLIP once and reuse a validated local candidate-text cache when possible."""
    import torch
    from bioclip import CustomLabelsClassifier
    from bioclip._constants import BIOCLIP_MODEL_STR
    from bioclip.predict import BaseClassifier, create_bioclip_tokenizer

    class _CachedCustomLabelsClassifier(CustomLabelsClassifier):
        def __init__(self, cls_ary: list[str], *, model_str: str, device: str) -> None:
            # CustomLabelsClassifier.__init__ always encodes every label.  Call its
            # BaseClassifier initializer instead, preserving its model/predict logic
            # while allowing a verified embedding tensor to be restored.
            BaseClassifier.__init__(self, model_str=model_str, device=device)
            self.tokenizer = create_bioclip_tokenizer(self.model_str)
            self.classes = [label.strip() for label in cls_ary]
            self.cache_path, metadata = candidate_cache_path(
                self.model_str, self.pretrained_str, self.classes
            )
            self.cache_hit = False
            started = time.perf_counter()
            try:
                cached = torch.load(self.cache_path, map_location="cpu", weights_only=True)
                if (
                    cached.get("metadata") == metadata
                    and tuple(cached["embeddings"].shape)[1] == len(self.classes)
                ):
                    self.txt_embeddings = cached["embeddings"].to(self.device)
                    self.cache_hit = True
                else:
                    raise ValueError("cache metadata or embedding shape does not match")
            except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError):
                self.txt_embeddings = self._get_txt_embeddings(self.classes)
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.cache_path.with_suffix(".tmp")
                torch.save(
                    {"metadata": metadata, "embeddings": self.txt_embeddings.detach().cpu()},
                    temporary_path,
                )
                temporary_path.replace(self.cache_path)
            self.candidate_seconds = time.perf_counter() - started

    classifier = _CachedCustomLabelsClassifier(labels, model_str=BIOCLIP_MODEL_STR, device=device)
    return classifier, classifier.candidate_seconds, classifier.cache_hit, classifier.cache_path


def predict_resiliently(classifier: Any, paths: list[Path], top_k: int, batch_size: int) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    """Predict batches, splitting a failed batch until the problematic image is isolated."""
    predictions: list[dict[str, Any]] = []
    failed: list[tuple[Path, str]] = []
    split_notice_printed = False

    def predict_group(group: list[Path]) -> None:
        nonlocal split_notice_printed
        if not group:
            return
        try:
            result = classifier.predict([str(path) for path in group], k=top_k, batch_size=len(group))
            predictions.extend(result)
        except Exception as exc:
            if len(group) == 1:
                failed.append((group[0], f"BioCLIP prediction failed: {exc}"))
                return
            if not split_notice_printed:
                print(f"Batch of {len(group)} failed; automatically retrying with smaller batches.")
                split_notice_printed = True
            midpoint = len(group) // 2
            predict_group(group[:midpoint])
            predict_group(group[midpoint:])

    total_batches = (len(paths) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(paths), batch_size), start=1):
        print(f"Processing batch {batch_number}/{total_batches}...")
        predict_group(paths[start : start + batch_size])
    return predictions, failed


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_dir: Path,
    photo_dir: Path,
    predictions: list[dict[str, Any]],
    species: dict[str, dict[str, str]],
    failures: list[tuple[Path, str]],
    scanned_count: int,
    threshold: float,
    elapsed_seconds: float,
    stage_timings: dict[str, float],
) -> float:
    report_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = ["file_name", "rank", *REQUIRED_COLUMNS, "score"]
    rows: list[dict[str, Any]] = []
    by_file: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        image_path = Path(item["file_name"])
        try:
            file_name = str(image_path.relative_to(photo_dir))
        except ValueError:
            file_name = str(image_path)
        latin_name = item["classification"]
        row = {"file_name": file_name, **species[latin_name], "score": f"{float(item['score']):.6f}"}
        by_file.setdefault(file_name, []).append(row)

    for file_name, file_rows in sorted(by_file.items()):
        for rank, row in enumerate(file_rows, start=1):
            rows.append({**row, "file_name": file_name, "rank": rank})
    write_csv(output_dir / "predictions.csv", prediction_fields, rows)

    top1 = {file_name: file_rows[0] for file_name, file_rows in by_file.items() if file_rows}
    top1_counts = Counter(row["拉丁学名"] for row in top1.values())
    top5_counts = Counter(row["拉丁学名"] for file_rows in by_file.values() for row in file_rows)
    best: dict[str, tuple[float, str]] = {}
    for file_name, file_rows in by_file.items():
        for row in file_rows:
            score = float(row["score"])
            name = row["拉丁学名"]
            if name not in best or score > best[name][0]:
                best[name] = (score, file_name)
    summary_rows = []
    for latin_name in sorted(top5_counts):
        score, image_name = best[latin_name]
        summary_rows.append({
            **species[latin_name],
            "top1_count": top1_counts[latin_name],
            "top5_count": top5_counts[latin_name],
            "max_score": f"{score:.6f}",
            "best_image": image_name,
        })
    summary_fields = [*REQUIRED_COLUMNS, "top1_count", "top5_count", "max_score", "best_image"]
    write_csv(output_dir / "species_summary.csv", summary_fields, summary_rows)

    uncertain_rows = [
        {"file_name": file_name, **row}
        for file_name, row in sorted(top1.items())
        if float(row["score"]) < threshold
    ]
    write_csv(output_dir / "uncertain.csv", ["file_name", *REQUIRED_COLUMNS, "score"], uncertain_rows)
    stage_timings["report_writing"] = time.perf_counter() - report_started
    elapsed_seconds += stage_timings["report_writing"]

    success_count = len(top1)
    summary = [
        f"Scanned photos: {scanned_count}",
        f"Successful: {success_count}",
        f"Failed: {len(failures)}",
        f"Distinct Top-1 species: {len(top1_counts)}",
        "",
        "Stage timings (seconds):",
        f"Species file read: {stage_timings['species_file']:.3f}",
        f"Model load: {stage_timings['model_load']:.3f}",
        f"Candidate Latin-name encoding/cache reuse: {stage_timings['candidate_text']:.3f}",
        f"Image inference: {stage_timings['image_inference']:.3f}",
        f"Report writing: {stage_timings['report_writing']:.3f}",
        f"Total elapsed seconds: {elapsed_seconds:.3f}",
        f"Average images/s: {success_count / elapsed_seconds:.2f}" if elapsed_seconds else "Average images/s: 0.00",
    ]
    if failures:
        summary.extend(["", "Failures:"])
        summary.extend(f"- {path}: {reason}" for path, reason in failures)
    (output_dir / "run_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return time.perf_counter() - report_started


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    stage_timings = {"species_file": 0.0, "model_load": 0.0, "candidate_text": 0.0, "image_inference": 0.0, "report_writing": 0.0}
    try:
        print("Loading species file...")
        stage_started = time.perf_counter()
        species = load_species(args.species_file)
        stage_timings["species_file"] = time.perf_counter() - stage_started
        print(f"Loaded {len(species)} candidate species in {stage_timings['species_file']:.3f}s.")
        image_paths = find_images(args.photo_dir)
        if not image_paths:
            raise ValueError("No JPG, JPEG, or PNG images found in the photo folder")
        valid_paths, failures = validate_images(image_paths)
        if not valid_paths:
            print("Writing reports...")
            write_reports(args.output_dir, args.photo_dir, [], species, failures, len(image_paths), args.threshold, time.perf_counter() - started, stage_timings)
            print("No readable images found; empty reports and run_summary.txt were written.", file=sys.stderr)
            return 2

        effective_top_k = min(args.top_k, len(species))
        print(f"Loading model (BioCLIP 2) on {args.device}...")
        model_started = time.perf_counter()
        classifier, candidate_seconds, cache_hit, cache_path = build_classifier(list(species), args.device)
        stage_timings["model_load"] = time.perf_counter() - model_started - candidate_seconds
        stage_timings["candidate_text"] = candidate_seconds
        print(f"Model loaded in {stage_timings['model_load']:.3f}s.")
        if cache_hit:
            print(f"Candidate cache hit: {cache_path} ({candidate_seconds:.3f}s to load).")
        else:
            print(f"Candidate cache miss. Encoded {len(species)} candidate species in {candidate_seconds:.3f}s.")
        print(f"Predicting {len(valid_paths)} readable images in batches of {args.batch_size}.")
        inference_started = time.perf_counter()
        predictions, prediction_failures = predict_resiliently(classifier, valid_paths, effective_top_k, args.batch_size)
        stage_timings["image_inference"] = time.perf_counter() - inference_started
        failures.extend(prediction_failures)
        print("Writing reports...")
        write_reports(args.output_dir, args.photo_dir, predictions, species, failures, len(image_paths), args.threshold, time.perf_counter() - started, stage_timings)
        total_seconds = time.perf_counter() - started
        print(f"Reports written to: {args.output_dir.resolve()}")
        print(f"Timings — model load: {stage_timings['model_load']:.3f}s; candidate text: {stage_timings['candidate_text']:.3f}s; image inference: {stage_timings['image_inference']:.3f}s; report writing: {stage_timings['report_writing']:.3f}s; total: {total_seconds:.3f}s.")
        return 0 if not prediction_failures else 2
    except (ValueError, OSError, csv.Error, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
