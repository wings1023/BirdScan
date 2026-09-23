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
BIOCLIP25_MODEL_STR = "hf-hub:imageomics/bioclip-2.5-vith14"
DEFAULT_SIZE_GATE = 0.08
DEFAULT_DETECTION_THRESHOLD = 0.15
DEFAULT_CROP_MARGIN = 0.20


def select_crop_result(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    """Select highest crop Top-1 score, then apply the bbox area gate."""
    if not candidates:
        return None, False
    best = max(candidates, key=lambda item: float(item["prediction"]["score"]))
    return best, float(best["record"]["area"]) <= DEFAULT_SIZE_GATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use BioCLIP 2 to identify birds in a photo folder. Source photos are read only."
    )
    parser.add_argument("photo_dir", type=Path, help="Folder to scan recursively for JPG/JPEG/PNG files")
    parser.add_argument("--species-file", required=True, type=Path, help="Candidate species CSV or .xlsx file")
    parser.add_argument("--top-k", type=int, default=3, help="Predictions saved per image (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per BioCLIP inference batch (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Top-1 uncertainty threshold (default: 0.5)")
    parser.add_argument("--device", default="mps", help="BioCLIP device, e.g. mps or cpu (default: mps)")
    parser.add_argument("--model", choices=("bioclip2", "bioclip25"), default="bioclip25", help="Model to use (default: bioclip25)")
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument("--crop", dest="crop", action="store_true", help="Enable MegaDetector crop classification (default)")
    crop_group.add_argument("--no-crop", dest="crop", action="store_false", help="Use original-image BioCLIP only")
    parser.set_defaults(crop=True)
    parser.add_argument("--prompt-count", type=int, default=80, help="BioCLIP 2.5 templates per species (default: 80)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Report directory (defaults under reports/ based on model)")
    args = parser.parse_args()
    from bioclip.predict import OPENA_AI_IMAGENET_TEMPLATE

    if args.prompt_count < 1:
        parser.error("--prompt-count must be at least 1")
    if args.prompt_count > len(OPENA_AI_IMAGENET_TEMPLATE):
        parser.error(f"--prompt-count cannot exceed the {len(OPENA_AI_IMAGENET_TEMPLATE)} available templates")
    if args.output_dir is None:
        report_name = "bird_report_bioclip25" if args.model == "bioclip25" else "bird_report"
        args.output_dir = Path(__file__).resolve().parent / "reports" / report_name
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


def candidate_cache_path(
    model: str,
    pretrained: str | None,
    labels: list[str],
    prompt_count: int | None = None,
) -> tuple[Path, dict[str, Any]]:
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
    if prompt_count is not None:
        metadata["prompt_count"] = prompt_count
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


def encode_bioclip25_candidate_embeddings(
    classifier: Any,
    labels: list[str],
    prompt_count: int = 80,
    batch_size: int = 32,
) -> Any:
    """Build the same per-label template mean as pybioclip, with bounded text batches."""
    import torch
    import torch.nn.functional as F
    from bioclip.predict import OPENA_AI_IMAGENET_TEMPLATE

    if not 1 <= prompt_count <= len(OPENA_AI_IMAGENET_TEMPLATE):
        raise ValueError(f"prompt_count must be between 1 and {len(OPENA_AI_IMAGENET_TEMPLATE)}")
    templates = OPENA_AI_IMAGENET_TEMPLATE[:prompt_count]
    all_label_features = []
    total = len(labels)
    print("Building BioCLIP 2.5 candidate text embeddings")
    print(f"species={total}")
    print(f"prompts_per_species={prompt_count}")
    print(f"total_texts={total * prompt_count}")
    with torch.no_grad():
        for start in range(0, total, batch_size):
            label_batch = labels[start : start + batch_size]
            prompts = [template(label) for label in label_batch for template in templates]
            prompt_features = []
            for prompt_start in range(0, len(prompts), batch_size):
                prompt_batch = prompts[prompt_start : prompt_start + batch_size]
                tokens = classifier.tokenizer(prompt_batch).to(classifier.device)
                features = classifier.model.encode_text(tokens)
                prompt_features.append(F.normalize(features, dim=-1))

            normalized_features = torch.cat(prompt_features).reshape(
                len(label_batch), len(templates), -1
            )
            label_features = normalized_features.mean(dim=1)
            label_features = F.normalize(label_features, dim=-1)
            all_label_features.append(label_features)
            completed = min(start + len(label_batch), total)
            print(f"BioCLIP 2.5 candidate text encoding: {completed}/{total}")

    return torch.cat(all_label_features, dim=0).T


def build_bioclip25_classifier(
    labels: list[str], device: str, prompt_count: int = 80
) -> tuple[Any, float, bool, Path]:
    """Load BioCLIP 2.5 via pybioclip's OpenCLIP-backed classifier path."""
    import torch
    from bioclip import CustomLabelsClassifier
    from bioclip.predict import BaseClassifier, create_bioclip_tokenizer

    class _CachedBioCLIP25Classifier(CustomLabelsClassifier):
        def __init__(self, cls_ary: list[str], *, device: str) -> None:
            # BaseClassifier delegates model and validation-transform loading to OpenCLIP.
            # Its HF Hub schema accepts BIOCLIP25_MODEL_STR directly.
            BaseClassifier.__init__(self, model_str=BIOCLIP25_MODEL_STR, device=device)
            self.tokenizer = create_bioclip_tokenizer(self.model_str)
            self.classes = [label.strip() for label in cls_ary]
            self.cache_path, metadata = candidate_cache_path(
                self.model_str, self.pretrained_str, self.classes, prompt_count=prompt_count
            )
            self.cache_hit = False
            started = time.perf_counter()
            txt_embeddings = None
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
                # Reuse and migrate the pre-prompt-count full-template cache, if present.
                # Older BioCLIP 2.5 caches were produced with all available templates.
                from bioclip.predict import OPENA_AI_IMAGENET_TEMPLATE

                if prompt_count == len(OPENA_AI_IMAGENET_TEMPLATE):
                    legacy_path, legacy_metadata = candidate_cache_path(
                        self.model_str, self.pretrained_str, self.classes
                    )
                    try:
                        legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
                        if (
                            legacy.get("metadata") == legacy_metadata
                            and tuple(legacy["embeddings"].shape)[1] == len(self.classes)
                        ):
                            txt_embeddings = legacy["embeddings"].to(self.device)
                            self.cache_hit = True
                    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError):
                        pass
                if txt_embeddings is None:
                    txt_embeddings = encode_bioclip25_candidate_embeddings(
                        self, self.classes, prompt_count=prompt_count, batch_size=32
                    )
                self.txt_embeddings = txt_embeddings
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.cache_path.with_suffix(".tmp")
                torch.save(
                    {"metadata": metadata, "embeddings": self.txt_embeddings.detach().cpu()},
                    temporary_path,
                )
                temporary_path.replace(self.cache_path)
            self.candidate_seconds = time.perf_counter() - started

    classifier = _CachedBioCLIP25Classifier(labels, device=device)
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
    diagnostics: dict[str, dict[str, Any]] | None = None,
) -> float:
    report_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = ["file_name", "rank", *REQUIRED_COLUMNS, "score", "final_source", "crop_used", "selected_crop_file", "detection_confidence", "bbox_area_ratio", "baseline_top1_species", "baseline_top1_score", "crop_top1_species", "crop_top1_score"]
    rows: list[dict[str, Any]] = []
    by_file: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        image_path = Path(item["file_name"])
        try:
            file_name = str(image_path.relative_to(photo_dir))
        except ValueError:
            file_name = str(image_path)
        latin_name = item["classification"]
        row = {"file_name": file_name, **species[latin_name], "score": f"{float(item['score']):.6f}", **(diagnostics or {}).get(file_name, {})}
        by_file.setdefault(file_name, []).append(row)

    for file_name, file_rows in sorted(by_file.items()):
        for rank, row in enumerate(file_rows, start=1):
            rows.append({**row, "file_name": file_name, "rank": rank})
    write_csv(output_dir / "predictions.csv", prediction_fields, rows)

    top1 = {file_name: file_rows[0] for file_name, file_rows in by_file.items() if file_rows}
    top1_counts = Counter(row["拉丁学名"] for row in top1.values())
    topk_counts = Counter(row["拉丁学名"] for file_rows in by_file.values() for row in file_rows)
    best: dict[str, tuple[float, str]] = {}
    for file_name, file_rows in by_file.items():
        for row in file_rows:
            score = float(row["score"])
            name = row["拉丁学名"]
            if name not in best or score > best[name][0]:
                best[name] = (score, file_name)
    summary_rows = []
    for latin_name in sorted(topk_counts):
        score, image_name = best[latin_name]
        summary_rows.append({
            **species[latin_name],
            "top1_count": top1_counts[latin_name],
            "topk_count": topk_counts[latin_name],
            "max_score": f"{score:.6f}",
            "best_image": image_name,
        })
    summary_fields = [*REQUIRED_COLUMNS, "top1_count", "topk_count", "max_score", "best_image"]
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
        model_label = "BioCLIP 2.5 Huge" if args.model == "bioclip25" else "BioCLIP 2"
        print(f"Loading model ({model_label}) on {args.device}...")
        model_started = time.perf_counter()
        if args.model == "bioclip25":
            classifier, candidate_seconds, cache_hit, cache_path = build_bioclip25_classifier(
                list(species), args.device, prompt_count=args.prompt_count
            )
        else:
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

        diagnostics: dict[str, dict[str, Any]] = {}
        if args.crop:
            from PIL import Image
            import megadetector_utils as md

            crop_root = args.output_dir / "diagnostics" / "crop_runtime"
            crop_root.mkdir(parents=True, exist_ok=True)
            baseline_by_file: dict[str, list[dict[str, Any]]] = {}
            for item in predictions:
                baseline_by_file.setdefault(Path(item["file_name"]).resolve().as_posix(), []).append(item)
            try:
                detector = md.load_detector(args.device)
            except Exception as exc:
                raise RuntimeError(f"MegaDetector V6 初始化失败（crop 默认开启）：{exc}") from exc

            crops: list[dict[str, Any]] = []
            per_source: dict[str, list[dict[str, Any]]] = {}
            for source in valid_paths:
                key = source.resolve().as_posix()
                per_source[key] = []
                try:
                    found = [d for d in md.extract_animals(detector.single_image_detection(str(source), det_conf_thres=DEFAULT_DETECTION_THRESHOLD)) if d[0] >= DEFAULT_DETECTION_THRESHOLD]
                    with Image.open(source) as opened:
                        image = opened.convert("RGB")
                    width, height = image.size
                    for index, (confidence, box) in enumerate(found, start=1):
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0.0, min(width, x1)), max(0.0, min(height, y1))
                        x2, y2 = max(x1, min(width, x2)), max(y1, min(height, y2))
                        area_ratio = (x2 - x1) * (y2 - y1) / (width * height)
                        bounds = md.padded_box((x1, y1, x2, y2), DEFAULT_CROP_MARGIN, width, height)
                        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                            continue
                        crop_path = crop_root / f"{hashlib.sha1(key.encode()).hexdigest()[:12]}__det{index:02d}{source.suffix.lower()}"
                        image.crop(bounds).save(crop_path)
                        record = {"path": crop_path, "source": key, "confidence": confidence, "area": area_ratio,
                                  "crop_file": crop_path.relative_to(args.output_dir).as_posix()}
                        per_source[key].append(record)
                        crops.append(record)
                except Exception as exc:
                    print(f"Crop detection failed; using baseline: {source}: {exc}", file=sys.stderr)

            crop_predictions: list[dict[str, Any]] = []
            crop_failures: list[tuple[Path, str]] = []
            if crops:
                crop_predictions, crop_failures = predict_resiliently(classifier, [c["path"] for c in crops], effective_top_k, args.batch_size)
            crop_rows_by_path: dict[str, list[dict[str, Any]]] = {}
            for item in crop_predictions:
                crop_rows_by_path.setdefault(Path(item["file_name"]).resolve().as_posix(), []).append(item)
            for source in valid_paths:
                key = source.resolve().as_posix()
                base_rows = baseline_by_file.get(key, [])
                baseline_top = base_rows[0] if base_rows else None
                candidates = []
                for record in per_source.get(key, []):
                    ranked_rows = crop_rows_by_path.get(record["path"].resolve().as_posix(), [])
                    if ranked_rows:
                        top1 = max(ranked_rows, key=lambda item: float(item["score"]))
                        candidates.append({"record": record, "prediction": top1})
                best, use_crop = select_crop_result(candidates)
                if use_crop:
                    selected_path = best["record"]["path"].resolve().as_posix()
                    chosen = [dict(p, file_name=str(source)) for p in crop_predictions if Path(p["file_name"]).resolve().as_posix() == selected_path]
                    chosen.sort(key=lambda p: -float(p["score"]))
                    predictions = [p for p in predictions if Path(p["file_name"]).resolve().as_posix() != key] + chosen
                record = best["record"] if best else None
                diagnostics[source.relative_to(args.photo_dir).as_posix()] = {
                    "final_source": "crop" if use_crop else "baseline", "crop_used": "true" if use_crop else "false",
                    "selected_crop_file": record["crop_file"] if record else "",
                    "detection_confidence": f"{record['confidence']:.6f}" if record else "",
                    "bbox_area_ratio": f"{record['area']:.8f}" if record else "",
                    "baseline_top1_species": baseline_top.get("classification", "") if baseline_top else "",
                    "baseline_top1_score": f"{float(baseline_top['score']):.6f}" if baseline_top else "",
                    "crop_top1_species": best["prediction"].get("classification", "") if best else "",
                    "crop_top1_score": f"{float(best['prediction']['score']):.6f}" if best else "",
                }
            if crop_failures:
                print(f"Crop classification failures: {len(crop_failures)}; affected images use baseline where no valid crop remains.", file=sys.stderr)
        else:
            diagnostics = {}
            baseline_rows: dict[str, dict[str, Any]] = {}
            for item in predictions:
                key = Path(item["file_name"]).resolve().as_posix()
                if key not in baseline_rows or float(item["score"]) > float(baseline_rows[key]["score"]):
                    baseline_rows[key] = item
            for source in valid_paths:
                baseline_top = baseline_rows.get(source.resolve().as_posix())
                diagnostics[source.relative_to(args.photo_dir).as_posix()] = {
                    "final_source": "baseline", "crop_used": "false",
                    "baseline_top1_species": baseline_top.get("classification", "") if baseline_top else "",
                    "baseline_top1_score": f"{float(baseline_top['score']):.6f}" if baseline_top else "",
                }
        print("Writing reports...")
        write_reports(args.output_dir, args.photo_dir, predictions, species, failures, len(image_paths), args.threshold, time.perf_counter() - started, stage_timings, diagnostics)
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
