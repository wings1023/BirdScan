#!/usr/bin/env python3
"""Inspect pairwise BioCLIP image embedding similarity for a photo folder."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from scan_birds import find_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine similarity for BioCLIP 2 image embeddings."
    )
    parser.add_argument("photo_dir", type=Path, help="Folder containing JPG/JPEG/PNG photos")
    parser.add_argument("--device", default="mps", help="BioCLIP device, e.g. mps, cuda, or cpu (default: mps)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_similarity.csv"),
        help="CSV output path (default: ./embedding_similarity.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = find_images(args.photo_dir)
        if not paths:
            raise ValueError("No JPG, JPEG, or PNG images found in the photo folder")

        # Use the same BioCLIP model identifier, loader, transform and feature
        # method as scan_birds' CustomLabelsClassifier/BaseClassifier.
        from bioclip._constants import BIOCLIP_MODEL_STR
        from bioclip.predict import BaseClassifier

        classifier = BaseClassifier(model_str=BIOCLIP_MODEL_STR, device=args.device)
        classifier.eval()

        names = [path.relative_to(args.photo_dir).as_posix() for path in paths]
        feature_rows = []
        for path in paths:
            image = classifier.ensure_rgb_image(str(path))
            feature_rows.append(classifier.create_image_features([image], normalize=True)[0].detach())
        embeddings = torch.stack(feature_rows)
        similarities = embeddings @ embeddings.T
        matrix = similarities.detach().cpu().to(torch.float64).numpy()

        print("Files:")
        for index, name in enumerate(names, start=1):
            print(f"{index:>4}: {name}")
        print(f"\nNormalized image embedding tensor shape: {tuple(embeddings.shape)}")
        print("Cosine similarity matrix:")
        print("\t" + "\t".join(names))
        for name, row in zip(names, matrix):
            print(name + "\t" + "\t".join(f"{value:.3f}" for value in row))

        if len(names) > 1:
            off_diagonal = matrix[~(torch.eye(len(names), dtype=torch.bool).numpy())]
            print(f"\nWithin-group mean similarity (excluding self): {off_diagonal.mean():.6f}")
            print(f"Within-group minimum similarity: {off_diagonal.min():.6f}")
            print(f"Within-group maximum similarity: {off_diagonal.max():.6f}")
        else:
            print("\nWithin-group similarity statistics: undefined (only one image)")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_name", *names])
            for name, row in zip(names, matrix):
                writer.writerow([name, *(f"{value:.6f}" for value in row)])
        print(f"\nCSV written to: {args.output.resolve()}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
