"""Compare two YOLO label trees using class-aware one-to-one IoU matching."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import CLASS_NAMES
from .run import Box, IMAGE_SUFFIXES, parse_label


def iou(left: Box, right: Box) -> float:
    lx1, ly1 = left.cx - left.width / 2, left.cy - left.height / 2
    lx2, ly2 = left.cx + left.width / 2, left.cy + left.height / 2
    rx1, ry1 = right.cx - right.width / 2, right.cy - right.height / 2
    rx2, ry2 = right.cx + right.width / 2, right.cy + right.height / 2
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0.0 else 0.0


def match_boxes(reference: list[Box], prediction: list[Box], threshold: float) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference_box in enumerate(reference):
        for prediction_index, prediction_box in enumerate(prediction):
            if reference_box.class_id != prediction_box.class_id:
                continue
            overlap = iou(reference_box, prediction_box)
            if overlap >= threshold:
                candidates.append((overlap, reference_index, prediction_index))
    matched_reference: set[int] = set()
    matched_prediction: set[int] = set()
    for _, reference_index, prediction_index in sorted(candidates, reverse=True):
        if reference_index in matched_reference or prediction_index in matched_prediction:
            continue
        matched_reference.add(reference_index)
        matched_prediction.add(prediction_index)
    true_positive = len(matched_reference)
    return true_positive, len(prediction) - true_positive, len(reference) - true_positive


def metrics(counts: dict[str, int]) -> dict[str, int | float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    reference_root = args.reference.resolve()
    prediction_root = args.prediction.resolve()
    raw_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    images = 0
    exact_images = 0
    reference_issue_files = 0
    prediction_issue_files = 0

    for split in args.splits:
        image_dir = reference_root / split / "images"
        image_paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if args.prediction_files_only:
            predicted_stems = {
                path.stem for path in (prediction_root / split / "labels").glob("*.txt")
            }
            image_paths = [path for path in image_paths if path.stem in predicted_stems]
        for image_path in image_paths:
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            reference, reference_issues = parse_label(
                reference_root / split / "labels" / f"{image_path.stem}.txt"
            )
            prediction, prediction_issues = parse_label(
                prediction_root / split / "labels" / f"{image_path.stem}.txt"
            )
            images += 1
            reference_issue_files += bool(reference_issues)
            prediction_issue_files += bool(prediction_issues)
            image_exact = True
            for class_id in CLASS_NAMES:
                reference_class = [box for box in reference if box.class_id == class_id]
                prediction_class = [box for box in prediction if box.class_id == class_id]
                tp, fp, fn = match_boxes(reference_class, prediction_class, args.iou)
                counts = raw_counts[(split, class_id)]
                counts["tp"] += tp
                counts["fp"] += fp
                counts["fn"] += fn
                image_exact &= fp == 0 and fn == 0
            exact_images += image_exact

    split_results: dict[str, object] = {}
    overall_by_class: dict[int, dict[str, int]] = {
        class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in CLASS_NAMES
    }
    for split in args.splits:
        split_results[split] = {}
        for class_id, name in CLASS_NAMES.items():
            counts = raw_counts[(split, class_id)]
            split_results[split][name] = metrics(counts)
            for key in ("tp", "fp", "fn"):
                overall_by_class[class_id][key] += counts[key]

    overall_classes = {
        CLASS_NAMES[class_id]: metrics(counts) for class_id, counts in overall_by_class.items()
    }
    overall_counts = {
        key: sum(counts[key] for counts in overall_by_class.values())
        for key in ("tp", "fp", "fn")
    }
    return {
        "comparison_note": "predictions were evaluated with class-aware one-to-one IoU matching",
        "iou_threshold": args.iou,
        "images": images,
        "exact_match_images": exact_images,
        "exact_match_rate": exact_images / images if images else 1.0,
        "reference_issue_files_repaired_during_read": reference_issue_files,
        "prediction_issue_files": prediction_issue_files,
        "by_split": split_results,
        "overall_by_class": overall_classes,
        "overall": metrics(overall_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--prediction-files-only",
        action="store_true",
        help="evaluate only images having a prediction txt file (for pilot subsets)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
