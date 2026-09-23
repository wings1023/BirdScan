#!/usr/bin/env python3
"""Detect animals with MegaDetector V6 and save padded diagnostic crops.

Bounding boxes are recorded as pixel coordinates in the original image,
using (x, y, width, height) with the origin at the top-left.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _directory in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from megadetector_utils import (
    MODEL_VERSION, extract_animals, import_megadetector_v6_image_only,
    load_detector, padded_box,
)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "diagnostics" / "auto_crop"
CSV_FIELDS = [
    "source_file",
    "status",
    "detection_index",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_area_ratio",
    "margin",
    "crop_file",
]
FOCUS_NAMES = tuple(f"DSC032{i:02d}" for i in range(7, 16)) + ("DSC03210-cj",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect animal boxes with MegaDetector V6 and save padded crops. "
            "Coordinates in detections.csv are pixel coordinates in the source image."
        )
    )
    parser.add_argument("image_dir", type=Path, help="照片目录（递归读取 JPG/JPEG/PNG）")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--margin", type=float, default=0.20, help="bbox 四周扩展比例，默认 0.20")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="最低 detection confidence，默认 0.15（偏向保留小目标）",
    )
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def make_preview(image: Image.Image, boxes: list[tuple[float, float, float, float]], path: Path) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for index, box in enumerate(boxes, start=1):
        draw.rectangle(box, outline=(255, 40, 40), width=max(2, min(image.size) // 500))
        draw.text((box[0] + 3, box[1] + 3), f"{index}", fill=(255, 40, 40))
    preview.save(path)


def main() -> int:
    args = parse_args()
    if not args.image_dir.is_dir():
        print(f"照片目录不存在：{args.image_dir}", file=sys.stderr)
        return 2
    if not 0 <= args.margin:
        print("--margin 必须大于或等于 0", file=sys.stderr)
        return 2
    if not 0 <= args.threshold <= 1:
        print("--threshold 必须在 0 到 1 之间", file=sys.stderr)
        return 2

    images = sorted(
        (path for path in args.image_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: (path.name.casefold(), str(path).casefold()),
    )
    if not images:
        print(f"目录中没有 JPG/JPEG/PNG 图片：{args.image_dir}", file=sys.stderr)
        return 2

    crop_dir = args.output_dir / "crops"
    preview_dir = args.output_dir / "preview"
    crop_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = load_detector(args.device)
    except Exception as exc:
        print(f"无法加载 MegaDetector V6：{exc}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    image_detection_counts: list[int | None] = []
    confidences_all: list[float] = []
    failed_images = 0

    for image_path in images:
        relative = image_path.relative_to(args.image_dir)
        source_label = str(relative)
        try:
            result = model.single_image_detection(str(image_path), det_conf_thres=args.threshold)
            animals = [item for item in extract_animals(result) if item[0] >= args.threshold]
            with Image.open(image_path) as opened:
                image = opened.copy()
        except Exception as exc:
            print(f"检测失败，继续处理：{source_label}: {exc}", file=sys.stderr)
            rows.append({"source_file": source_label, "status": "error", "margin": args.margin})
            image_detection_counts.append(None)
            failed_images += 1
            continue

        image_width, image_height = image.size
        image_boxes: list[tuple[float, float, float, float]] = []
        image_confidences: list[float] = []
        if not animals:
            rows.append({"source_file": source_label, "status": "no_detection", "margin": args.margin})
        for index, (confidence, box) in enumerate(animals, start=1):
            x1, y1, x2, y2 = box
            x = max(0.0, min(float(image_width), x1))
            y = max(0.0, min(float(image_height), y1))
            right = max(x, min(float(image_width), x2))
            bottom = max(y, min(float(image_height), y2))
            width, height = right - x, bottom - y
            crop_bounds = padded_box((x, y, right, bottom), args.margin, image_width, image_height)
            if crop_bounds[2] <= crop_bounds[0] or crop_bounds[3] <= crop_bounds[1]:
                continue

            crop_name = f"{safe_stem(image_path)}__det{index:02d}__conf{confidence:.2f}{image_path.suffix.lower()}"
            crop_path = crop_dir / crop_name
            image.crop(crop_bounds).save(crop_path)
            image_boxes.append((x, y, right, bottom))
            image_confidences.append(confidence)
            confidences_all.append(confidence)
            rows.append({
                "source_file": source_label,
                "status": "detected",
                "detection_index": index,
                "confidence": f"{confidence:.6f}",
                "bbox_x": f"{x:.2f}",
                "bbox_y": f"{y:.2f}",
                "bbox_width": f"{width:.2f}",
                "bbox_height": f"{height:.2f}",
                "bbox_area_ratio": f"{(width * height) / (image_width * image_height):.8f}",
                "margin": args.margin,
                "crop_file": str(crop_path.relative_to(args.output_dir)),
            })
        image_detection_counts.append(len(image_confidences))
        if image_boxes:
            preview_path = preview_dir / f"{safe_stem(image_path)}__preview.jpg"
            make_preview(image, image_boxes, preview_path)

        if any(token.casefold() in image_path.stem.casefold() for token in FOCUS_NAMES):
            conf_text = ", ".join(f"{value:.3f}" for value in image_confidences) or "无"
            status = "animal detected" if image_confidences else "no_detection"
            print(f"重点样本 {image_path.name}: {status}; detections={len(image_confidences)}; confidence={conf_text}")

    csv_path = args.output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    successful_counts = [count for count in image_detection_counts if count is not None]
    detected_images = sum(count > 0 for count in successful_counts)
    no_detection_images = sum(count == 0 for count in successful_counts)
    total_detections = sum(successful_counts)
    multi_images = sum(count > 1 for count in successful_counts)
    print("\n检测统计")
    print(f"总图片数: {len(images)}")
    print(f"检测到 animal 的图片数: {detected_images}")
    print(f"no_detection 图片数: {no_detection_images}")
    print(f"检测失败图片数: {failed_images}")
    print(f"总 detection 数: {total_detections}")
    print(f"单图多 detection 数量: {multi_images}")
    if confidences_all:
        print(f"平均 detection confidence: {sum(confidences_all) / len(confidences_all):.4f}")
        print(f"最低 / 最高 confidence: {min(confidences_all):.4f} / {max(confidences_all):.4f}")
    else:
        print("平均 / 最低 / 最高 confidence: 无 detection")
    print(f"输出目录: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
