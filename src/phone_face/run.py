"""Audit, render, train, and run the phone/face detector.

The source YOLO labels are never modified.  Every command writes to a separate
output directory so runs are resumable and easy to review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from . import CLASS_NAMES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR: phone red, face green


@dataclass(frozen=True)
class Box:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float
    confidence: float | None = None

    def xyxy(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x1 = round((self.cx - self.width / 2) * image_width)
        y1 = round((self.cy - self.height / 2) * image_height)
        x2 = round((self.cx + self.width / 2) * image_width)
        y2 = round((self.cy + self.height / 2) * image_height)
        return (
            max(0, min(image_width - 1, x1)),
            max(0, min(image_height - 1, y1)),
            max(0, min(image_width - 1, x2)),
            max(0, min(image_height - 1, y2)),
        )


def parse_label(path: Path) -> tuple[list[Box], list[str]]:
    boxes: list[Box] = []
    issues: list[str] = []
    if not path.exists():
        return boxes, ["missing_label"]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        # A source-export bug concatenated multiple ordinary five-field YOLO
        # rows onto one physical line in a small subset of labels. Recover all
        # rows while keeping the file in the review manifest.
        if len(parts) > 5 and len(parts) % 5 == 0:
            chunks = [parts[index : index + 5] for index in range(0, len(parts), 5)]
            issues.append(f"line_{line_number}:concatenated_{len(chunks)}_rows")
        elif len(parts) in (5, 6):
            chunks = [parts]
        else:
            issues.append(f"line_{line_number}:expected_5_or_6_fields")
            continue
        for chunk_index, chunk in enumerate(chunks, 1):
            issue_prefix = f"line_{line_number}"
            if len(chunks) > 1:
                issue_prefix += f"_row_{chunk_index}"
            try:
                class_id = int(chunk[0])
                values = [float(value) for value in chunk[1:]]
            except ValueError:
                issues.append(f"{issue_prefix}:non_numeric")
                continue
            cx, cy, width, height = values[:4]
            confidence = values[4] if len(values) == 5 else None
            if class_id not in CLASS_NAMES:
                issues.append(f"{issue_prefix}:unknown_class_{class_id}")
                continue
            if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
                issues.append(f"{issue_prefix}:coordinate_out_of_range")
                continue
            if width <= 0.0 or height <= 0.0:
                issues.append(f"{issue_prefix}:non_positive_size")
                continue
            boxes.append(Box(class_id, cx, cy, width, height, confidence))
    unique_boxes: list[Box] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    duplicate_count = 0
    for box in boxes:
        key = (box.class_id, box.cx, box.cy, box.width, box.height)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_boxes.append(box)
    if duplicate_count:
        issues.append(f"duplicate_rows_{duplicate_count}")
    return unique_boxes, issues


def iter_images(dataset: Path, splits: Iterable[str]) -> Iterable[tuple[str, Path, Path]]:
    for split in splits:
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield split, image_path, label_dir / f"{image_path.stem}.txt"


def draw_boxes(image, boxes: list[Box]):
    height, width = image.shape[:2]
    line_width = max(1, round(min(width, height) / 500))
    font_scale = max(0.45, min(width, height) / 900)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy(width, height)
        color = COLORS[box.class_id]
        label = CLASS_NAMES[box.class_id]
        if box.confidence is not None:
            label += f" {box.confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        text_y = max(text_height + baseline, y1)
        cv2.rectangle(
            image,
            (x1, text_y - text_height - baseline),
            (min(width - 1, x1 + text_width + 4), text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            line_width,
            cv2.LINE_AA,
        )
    return image


def write_data_yaml(dataset: Path) -> Path:
    yaml_path = dataset / "data.yaml"
    content = (
        f"path: {dataset.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: phone\n"
        "  1: face\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def normalize(args: argparse.Namespace) -> int:
    """Create a training-safe label tree without copying the image payload."""

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output == dataset:
        raise ValueError("normalized output must differ from the source dataset")
    output.mkdir(parents=True, exist_ok=True)
    totals = {"images": 0, "boxes": 0, "source_files_with_issues": 0}
    review_rows: list[dict[str, str]] = []

    for split in args.splits:
        source_images = dataset / split / "images"
        source_labels = dataset / split / "labels"
        if not source_images.is_dir():
            continue
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        clean_images = split_dir / "images"
        # Directory symlinks are unsuitable here: Ultralytics resolves them and
        # then derives label paths beside the *source* images. Individual hard
        # links preserve clean-dataset paths while consuming no extra payload.
        if clean_images.is_symlink():
            clean_images.unlink()
        clean_images.mkdir(parents=True, exist_ok=True)
        clean_labels = split_dir / "labels"
        clean_labels.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(source_images.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            clean_image = clean_images / image_path.name
            if not clean_image.exists():
                os.link(image_path, clean_image)
            source_label = source_labels / f"{image_path.stem}.txt"
            boxes, issues = parse_label(source_label)
            destination = clean_labels / source_label.name
            destination.write_text(
                "".join(
                    f"{box.class_id} {box.cx:.9g} {box.cy:.9g} "
                    f"{box.width:.9g} {box.height:.9g}\n"
                    for box in boxes
                ),
                encoding="utf-8",
            )
            totals["images"] += 1
            totals["boxes"] += len(boxes)
            if issues:
                totals["source_files_with_issues"] += 1
                review_rows.append(
                    {
                        "split": split,
                        "image": str(image_path),
                        "source_label": str(source_label),
                        "clean_label": str(destination),
                        "issues": ";".join(issues),
                    }
                )

    with (output / "source_label_issues.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["split", "image", "source_label", "clean_label", "issues"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    yaml_path = write_data_yaml(output)
    (output / "normalization_summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "data_yaml": str(yaml_path), **totals}, ensure_ascii=False))
    return 0


def audit_or_render(args: argparse.Namespace, render: bool) -> int:
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    totals = {"images": 0, "phone_boxes": 0, "face_boxes": 0, "empty": 0, "issues": 0}

    for index, (split, image_path, label_path) in enumerate(iter_images(dataset, args.splits), 1):
        boxes, issues = parse_label(label_path)
        image = cv2.imread(str(image_path))
        if image is None:
            issues.append("unreadable_image")
            width = height = 0
        else:
            height, width = image.shape[:2]
        phone_count = sum(box.class_id == 0 for box in boxes)
        face_count = sum(box.class_id == 1 for box in boxes)
        totals["images"] += 1
        totals["phone_boxes"] += phone_count
        totals["face_boxes"] += face_count
        totals["empty"] += not boxes
        totals["issues"] += bool(issues)
        rows.append(
            {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "width": width,
                "height": height,
                "phone_count": phone_count,
                "face_count": face_count,
                "status": "review" if issues else "ok",
                "issues": ";".join(issues),
            }
        )
        if render and image is not None:
            destination = output / split / "images" / image_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not args.resume or not destination.exists():
                cv2.imwrite(str(destination), draw_boxes(image, boxes))
        if args.limit and index >= args.limit:
            break

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    yaml_path = write_data_yaml(dataset)
    print(json.dumps({"output": str(output), "data_yaml": str(yaml_path), **totals}, ensure_ascii=False))
    return 0


def train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    dataset = args.dataset.resolve()
    yaml_path = args.data_yaml.resolve() if args.data_yaml else write_data_yaml(dataset)
    model = YOLO(str(args.model))
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        close_mosaic=args.close_mosaic,
        multi_scale=args.multi_scale,
        mosaic=args.mosaic,
        scale=args.scale,
        translate=args.translate,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        plots=True,
    )
    return 0


def predict(args: argparse.Namespace) -> int:
    from .video import VIDEO_SUFFIXES, predict_video

    if args.command == "video" or args.source.suffix.lower() in VIDEO_SUFFIXES:
        return predict_video(args)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    thresholds = {
        0: args.phone_conf if args.phone_conf is not None else args.conf,
        1: args.face_conf if args.face_conf is not None else args.conf,
    }
    inference_conf = min(thresholds.values())
    save_dir = args.output.resolve() / args.name
    labels_dir = save_dir / "labels"
    save_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    detected = 0
    results = model.predict(
        source=str(args.source.resolve()), conf=inference_conf, iou=args.iou,
        imgsz=args.imgsz, device=args.device, stream=True, verbose=False,
    )
    for result in results:
        keep = [
            index for index, (confidence, class_id) in enumerate(
                zip(result.boxes.conf.tolist(), result.boxes.cls.tolist())
            )
            if confidence >= thresholds.get(int(class_id), args.conf)
        ]
        result.boxes = result.boxes[keep]
        output_image = save_dir / Path(result.path).name
        rendered_boxes = [
            Box(
                class_id=int(class_id),
                cx=float(xywhn[0]),
                cy=float(xywhn[1]),
                width=float(xywhn[2]),
                height=float(xywhn[3]),
                confidence=float(confidence),
            )
            for xywhn, confidence, class_id in zip(
                result.boxes.xywhn.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            )
        ]
        rendered = draw_boxes(result.orig_img.copy(), rendered_boxes)
        if not cv2.imwrite(str(output_image), rendered):
            raise OSError(f"failed to save prediction image: {output_image}")
        label_path = labels_dir / f"{Path(result.path).stem}.txt"
        if label_path.exists():
            label_path.unlink()
        if keep:
            result.save_txt(str(label_path), save_conf=True)
            detected += 1
        count += 1
    print(json.dumps({
        "output": str(save_dir), "images": count, "images_with_detections": detected,
        "phone_conf": thresholds[0], "face_conf": thresholds[1],
    }, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phone/face YOLO workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("audit", "render"):
        child = subparsers.add_parser(command)
        child.add_argument("dataset", type=Path)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--splits", nargs="+", default=["train", "val", "test"])
        child.add_argument("--limit", type=int, default=0)
        child.add_argument("--resume", action="store_true")
        child.set_defaults(func=lambda args, render=command == "render": audit_or_render(args, render))

    child = subparsers.add_parser("normalize")
    child.add_argument("dataset", type=Path)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    child.set_defaults(func=normalize)

    child = subparsers.add_parser("train")
    child.add_argument("dataset", type=Path)
    child.add_argument("--model", type=Path, required=True)
    child.add_argument("--data-yaml", type=Path)
    child.add_argument("--project", type=Path, default=Path("outputs/phone_face_yolo"))
    child.add_argument("--name", default="yolo26m_phone_face")
    child.add_argument("--epochs", type=int, default=100)
    child.add_argument("--patience", type=int, default=20)
    child.add_argument("--imgsz", type=int, default=640)
    child.add_argument("--batch", type=int, default=32)
    child.add_argument("--device", default="1")
    child.add_argument("--workers", type=int, default=8)
    child.add_argument("--multi-scale", type=float, default=0.0)
    child.add_argument("--close-mosaic", type=int, default=10)
    child.add_argument("--mosaic", type=float, default=1.0)
    child.add_argument("--scale", type=float, default=0.5)
    child.add_argument("--translate", type=float, default=0.1)
    child.add_argument("--optimizer", default="auto")
    child.add_argument("--lr0", type=float, default=0.01)
    child.add_argument("--lrf", type=float, default=0.01)
    child.add_argument("--warmup-epochs", type=float, default=3.0)
    child.add_argument("--seed", type=int, default=20260912)
    child.add_argument("--exist-ok", action="store_true")
    child.set_defaults(func=train)

    child = subparsers.add_parser("predict", aliases=["video"])
    child.add_argument("source", type=Path)
    child.add_argument("--model", type=Path, required=True)
    child.add_argument("--output", type=Path, default=Path("outputs/phone_face_predictions"))
    child.add_argument("--name", default="predict")
    child.add_argument("--conf", type=float, default=0.20)
    child.add_argument("--phone-conf", type=float)
    child.add_argument("--face-conf", type=float)
    child.add_argument("--iou", type=float, default=0.60)
    child.add_argument("--imgsz", type=int, default=640)
    child.add_argument("--device", default="1")
    child.add_argument("--pose-model", type=Path, help="Optional YOLO pose model for phone/person association")
    child.add_argument("--pose-conf", type=float, default=0.35)
    child.add_argument("--smooth-alpha", type=float, default=0.65, help="New-box weight for temporal smoothing")
    child.add_argument("--line-width", type=int, default=1)
    child.add_argument("--exist-ok", action="store_true")
    child.set_defaults(func=predict)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
