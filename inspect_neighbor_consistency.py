#!/usr/bin/env python3
"""Compare embedding, filename-sequence, and hybrid photo neighborhoods."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether nearby or visually similar photos share BirdScan Top-1 predictions."
    )
    parser.add_argument("predictions_csv", type=Path, help="BirdScan predictions.csv")
    parser.add_argument("embedding_csv", type=Path, help="inspect_embeddings.py embedding similarity matrix CSV")
    parser.add_argument("--k", type=int, default=5, help="Neighbors per strategy (default: 5)")
    parser.add_argument("--window", type=int, default=10, help="Hybrid sequence radius (default: 10)")
    parser.add_argument("--output", type=Path, default=Path("neighbor_consistency.csv"))
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.window < 0:
        parser.error("--window must be non-negative")
    return args


def normalized_path(value: str) -> str:
    """Normalize separators, case, and redundant path components without fuzzy matching."""
    value = value.strip().replace("\\", "/")
    parts = [part for part in value.split("/") if part not in ("", ".")]
    return "/".join(parts).casefold()


def basename_key(value: str) -> str:
    return normalized_path(value).rsplit("/", 1)[-1]


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Predictions CSV does not exist: {path}")
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_name", "rank", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("predictions.csv must contain file_name, rank, and score columns")
        species_column = "中文名" if "中文名" in reader.fieldnames else "拉丁学名"
        for row in reader:
            if (row.get("rank") or "").strip() != "1":
                continue
            name = (row.get("file_name") or "").strip()
            key = normalized_path(name)
            if not key:
                raise ValueError("predictions.csv contains an empty file_name for rank 1")
            if key in result:
                raise ValueError(f"Duplicate rank-1 prediction path after normalization: {name}")
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid rank-1 score for {name}: {row.get('score')!r}") from exc
            species = (row.get(species_column) or "").strip()
            if not species:
                raise ValueError(f"Rank-1 prediction has blank {species_column}: {name}")
            result[key] = {
                "file_name": name,
                "species": species,
                "score": f"{score:.6f}",
            }
    if not result:
        raise ValueError("No rank-1 prediction rows found")
    return result


def load_similarity_matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    if not path.is_file():
        raise ValueError(f"Embedding CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2 or rows[0][0].strip().casefold() != "file_name":
        raise ValueError("Embedding CSV must be a square matrix with a file_name header column")
    names = [name.strip() for name in rows[0][1:]]
    if any(not name for name in names):
        raise ValueError("Embedding CSV contains a blank matrix column name")
    keys = [normalized_path(name) for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError("Embedding CSV contains duplicate column paths after normalization")
    if len(rows) != len(names) + 1:
        raise ValueError(f"Embedding matrix is not square: {len(names)} columns, {len(rows) - 1} data rows")
    matrix: list[list[float]] = []
    for index, row in enumerate(rows[1:]):
        if len(row) != len(names) + 1:
            raise ValueError(f"Embedding matrix row {index + 1} has {len(row) - 1} values; expected {len(names)}")
        if normalized_path(row[0]) != keys[index]:
            raise ValueError(f"Embedding row/column order mismatch at row {index + 1}: {row[0]}")
        try:
            matrix.append([float(value) for value in row[1:]])
        except ValueError as exc:
            raise ValueError(f"Embedding matrix contains a non-numeric value at row {index + 1}") from exc
    return names, matrix


def align_files(
    predictions: dict[str, dict[str, str]], matrix_names: list[str]
) -> tuple[list[tuple[int, str, dict[str, str]]], list[str], list[str]]:
    """Match exact normalized paths first, then only unique exact basenames."""
    pred_by_key = predictions
    matrix_keys = [normalized_path(name) for name in matrix_names]
    matrix_key_to_index = {key: index for index, key in enumerate(matrix_keys)}
    pairs: list[tuple[int, str, dict[str, str]]] = []
    unmatched_pred_keys = set(pred_by_key)
    unmatched_matrix_indices = set(range(len(matrix_names)))
    all_pred_basenames: dict[str, int] = {}
    all_matrix_basenames: dict[str, int] = {}
    for key in pred_by_key:
        basename = basename_key(key)
        all_pred_basenames[basename] = all_pred_basenames.get(basename, 0) + 1
    for name in matrix_names:
        basename = basename_key(name)
        all_matrix_basenames[basename] = all_matrix_basenames.get(basename, 0) + 1

    for pred_key in list(pred_by_key):
        if pred_key in matrix_key_to_index:
            index = matrix_key_to_index[pred_key]
            pairs.append((index, matrix_names[index], pred_by_key[pred_key]))
            unmatched_pred_keys.discard(pred_key)
            unmatched_matrix_indices.discard(index)

    pred_basenames: dict[str, list[str]] = {}
    matrix_basenames: dict[str, list[int]] = {}
    for key in unmatched_pred_keys:
        pred_basenames.setdefault(basename_key(key), []).append(key)
    for index in unmatched_matrix_indices:
        matrix_basenames.setdefault(basename_key(matrix_names[index]), []).append(index)
    for basename, pred_keys in pred_basenames.items():
        matrix_indices = matrix_basenames.get(basename, [])
        if (
            len(pred_keys) == 1
            and len(matrix_indices) == 1
            and all_pred_basenames.get(basename) == 1
            and all_matrix_basenames.get(basename) == 1
        ):
            pred_key, index = pred_keys[0], matrix_indices[0]
            pairs.append((index, matrix_names[index], pred_by_key[pred_key]))
            unmatched_pred_keys.discard(pred_key)
            unmatched_matrix_indices.discard(index)

    pairs.sort(key=lambda item: item[1].casefold())
    missing_embeddings = sorted((pred_by_key[key]["file_name"] for key in unmatched_pred_keys), key=str.casefold)
    missing_predictions = sorted((matrix_names[index] for index in unmatched_matrix_indices), key=str.casefold)
    return pairs, missing_embeddings, missing_predictions


def sequence_number(file_name: str) -> int | None:
    # Accept a trailing camera number, optionally followed by a suffix like -cj.
    match = re.search(r"(\d+)(?:-[A-Za-z]+)?$", Path(file_name.replace("\\", "/")).stem)
    return int(match.group(1)) if match else None


def json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def compute_neighbors(
    aligned: list[tuple[int, str, dict[str, str]]],
    matrix: list[list[float]],
    k: int,
    window: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    names = [item[1] for item in aligned]
    matrix_indices = [item[0] for item in aligned]
    predictions = [item[2] for item in aligned]
    sequence = [sequence_number(name) for name in names]
    missing_sequence = [name for name, number in zip(names, sequence) if number is None]
    if missing_sequence:
        print("Files without a parseable sequence number (excluded from B and C candidates):")
        for name in missing_sequence:
            print(f"  - {name}")

    details: list[dict[str, Any]] = []
    strategy_neighbors: dict[str, list[list[dict[str, Any]]]] = {"embedding": [], "sequence": [], "hybrid": []}
    for i, (matrix_i, name, prediction) in enumerate(aligned):
        all_other = [j for j in range(len(aligned)) if j != i]
        emb_ranked = sorted(all_other, key=lambda j: (-matrix[matrix_i][matrix_indices[j]], names[j].casefold()))[:k]
        seq_candidates = [j for j in all_other if sequence[i] is not None and sequence[j] is not None]
        seq_ranked = sorted(seq_candidates, key=lambda j: (abs(sequence[j] - sequence[i]), names[j].casefold()))[:k]
        hybrid_candidates = [j for j in seq_candidates if abs(sequence[j] - sequence[i]) <= window]
        hybrid_ranked = sorted(
            hybrid_candidates,
            key=lambda j: (-matrix[matrix_i][matrix_indices[j]], names[j].casefold()),
        )[:k]

        per_strategy: dict[str, dict[str, Any]] = {}
        for key, selected in (("embedding", emb_ranked), ("sequence", seq_ranked), ("hybrid", hybrid_ranked)):
            neighbors = [
                {
                    "file_name": names[j],
                    "species": predictions[j]["species"],
                    "similarity": matrix[matrix_i][matrix_indices[j]],
                    "index": j,
                }
                for j in selected
            ]
            strategy_neighbors[key].append(neighbors)
            distribution = Counter(item["species"] for item in neighbors)
            similarities = [item["similarity"] for item in neighbors]
            same_count = distribution.get(prediction["species"], 0)
            per_strategy[key] = {
                "neighbors": neighbors,
                "distribution": distribution,
                "neighbor_count": len(neighbors),
                "same_count": same_count,
                "same_ratio": same_count / len(neighbors) if neighbors else None,
                "nearest": max(similarities) if similarities else None,
                "mean": sum(similarities) / len(similarities) if similarities else None,
                # kth_similarity is the similarity of the last selected neighbor.
                "kth": similarities[-1] if similarities else None,
            }
        details.append({"name": name, "prediction": prediction, "strategies": per_strategy})
    return details, strategy_neighbors


def optional_number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_output(path: Path, details: list[dict[str, Any]]) -> None:
    fields = ["file_name", "self_species", "self_score"]
    for prefix in ("embedding", "sequence", "hybrid"):
        fields.extend([
            f"{prefix}_neighbor_count", f"{prefix}_neighbor_files", f"{prefix}_neighbor_species",
            f"{prefix}_neighbor_species_distribution",
            f"{prefix}_same_prediction_count", f"{prefix}_same_prediction_ratio",
            f"{prefix}_nearest_similarity", f"{prefix}_mean_similarity", f"{prefix}_kth_similarity",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in details:
            row: dict[str, Any] = {
                "file_name": item["name"],
                "self_species": item["prediction"]["species"],
                "self_score": item["prediction"]["score"],
            }
            for key, metrics in item["strategies"].items():
                neighbors = metrics["neighbors"]
                row.update({
                    f"{key}_neighbor_files": json_list([n["file_name"] for n in neighbors]),
                    f"{key}_neighbor_species": json_list([n["species"] for n in neighbors]),
                    f"{key}_neighbor_count": metrics["neighbor_count"],
                    f"{key}_neighbor_species_distribution": json.dumps(
                        {species: metrics["distribution"][species] for species in sorted(metrics["distribution"])},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    f"{key}_same_prediction_count": metrics["same_count"],
                    f"{key}_same_prediction_ratio": optional_number(metrics["same_ratio"]),
                    f"{key}_nearest_similarity": optional_number(metrics["nearest"]),
                    f"{key}_mean_similarity": optional_number(metrics["mean"]),
                    f"{key}_kth_similarity": optional_number(metrics["kth"]),
                })
            writer.writerow(row)


def print_strategy_summary(details: list[dict[str, Any]], k: int) -> None:
    print("\nDirectory-level strategy summary:")
    for key, title in (("embedding", "A. embedding-only"), ("sequence", "B. sequence-only"), ("hybrid", "C. sequence + embedding hybrid")):
        metrics = [item["strategies"][key] for item in details]
        with_neighbors = [m for m in metrics if m["neighbors"]]
        mean_ratio = sum(m["same_ratio"] for m in with_neighbors) / len(with_neighbors) if with_neighbors else None
        mean_nearest = sum(m["nearest"] for m in with_neighbors) / len(with_neighbors) if with_neighbors else None
        mean_topk = sum(m["mean"] for m in with_neighbors) / len(with_neighbors) if with_neighbors else None
        fully_consistent = sum(len(m["neighbors"]) == k and len(m["distribution"]) == 1 for m in metrics)
        # "Highly split" means no species reaches 60% of this image's selected neighbors.
        highly_split = sum(
            bool(m["neighbors"]) and max(m["distribution"].values()) / len(m["neighbors"]) < 0.60
            for m in metrics
        )
        print(f"\n{title}")
        print(f"  mean same_prediction_ratio: {optional_number(mean_ratio)}")
        print(f"  mean nearest_similarity: {optional_number(mean_nearest)}")
        print(f"  mean Top-K mean_similarity: {optional_number(mean_topk)}")
        print(f"  fully consistent Top-{k} predictions: {fully_consistent}")
        print(f"  highly split prediction groups (<60% for any species): {highly_split}")


def print_focus(details: list[dict[str, Any]]) -> None:
    focus_stems = {f"dsc032{i:02d}" for i in range(8, 16)}
    focus = [item for item in details if Path(item["name"].replace("\\", "/")).stem.casefold() in focus_stems
             or Path(item["name"].replace("\\", "/")).stem.casefold().startswith("dsc03210-cj")]
    print("\nFocus images (DSC03208 through DSC03215, including DSC03210-cj variants):")
    if not focus:
        print("  No matching files found.")
        return
    for item in focus:
        print(f"\n{item['name']}\nself prediction: {item['prediction']['species']} (score={item['prediction']['score']})")
        for key, tag in (("embedding", "A"), ("sequence", "B"), ("hybrid", "C")):
            metrics = item["strategies"][key]
            print(f"[{tag}] {key}-only neighbors" if key != "hybrid" else "[C] sequence + embedding hybrid neighbors")
            for rank, neighbor in enumerate(metrics["neighbors"], 1):
                print(f"  {rank}. {neighbor['file_name']}  similarity={neighbor['similarity']:.3f}  prediction={neighbor['species']}")
            dist = ", ".join(f"{species} {count}" for species, count in sorted(metrics["distribution"].items())) or "(none)"
            print(f"  neighbor prediction distribution: {dist}")
            print(f"  same_prediction_count={metrics['same_count']} same_prediction_ratio={optional_number(metrics['same_ratio'])}")
            print(f"  nearest_similarity={optional_number(metrics['nearest'])} mean_similarity={optional_number(metrics['mean'])} kth_similarity={optional_number(metrics['kth'])}")


def main() -> int:
    # Keep Chinese species/file names readable when launched from Windows shells.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        predictions = load_predictions(args.predictions_csv)
        matrix_names, matrix = load_similarity_matrix(args.embedding_csv)
        pairs, missing_embeddings, missing_predictions = align_files(predictions, matrix_names)
        print(f"Predictions rank-1 images: {len(predictions)}")
        print(f"Embedding matrix images: {len(matrix_names)}")
        print(f"Successfully matched: {len(pairs)}")
        print("Predictions without matching embedding entry:")
        for name in missing_embeddings or ["(none)"]:
            print(f"  - {name}")
        print("Embedding entries without matching rank-1 prediction:")
        for name in missing_predictions or ["(none)"]:
            print(f"  - {name}")
        if not pairs:
            raise ValueError("No images matched between predictions and embedding matrix")
        details, _ = compute_neighbors(pairs, matrix, args.k, args.window)
        write_output(args.output, details)
        print_strategy_summary(details, args.k)
        print_focus(details)
        print(f"\nPer-image CSV written to: {args.output.resolve()}")
        return 0
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
