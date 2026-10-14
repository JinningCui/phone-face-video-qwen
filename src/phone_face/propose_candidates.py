"""Generate phone/face candidates without reading ground-truth labels.

Phones come from the cached COCO-pretrained YOLO26 model. Faces come from two
OpenCV Haar cascades (frontal and profile, including mirrored profile). The
output is consumed by the controlled Qwen reviewer, which only selects IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from .run import IMAGE_SUFFIXES


def iou(left: dict, right: dict) -> float:
    lx1, ly1, lx2, ly2 = left["bbox"]
    rx1, ry1, rx2, ry2 = right["bbox"]
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0 else 0.0


def nms(candidates: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if any(
            candidate["class_id"] == prior["class_id"] and iou(candidate, prior) >= threshold
            for prior in kept
        ):
            continue
        kept.append(candidate)
    return kept


def face_candidates(image) -> list[dict]:
    height, width = image.shape[:2]
    scale = min(3.0, max(1.0, 640.0 / min(height, width)))
    working = cv2.resize(image, (round(width * scale), round(height * scale)))
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    root = Path(cv2.data.haarcascades)
    detectors = [
        (cv2.CascadeClassifier(str(root / "haarcascade_frontalface_default.xml")), "haar_frontal", 0.55),
        (cv2.CascadeClassifier(str(root / "haarcascade_profileface.xml")), "haar_profile", 0.42),
    ]
    found: list[dict] = []
    for detector, source, confidence in detectors:
        for mirrored in (False, True) if source == "haar_profile" else (False,):
            target = cv2.flip(gray, 1) if mirrored else gray
            boxes = detector.detectMultiScale(
                target,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(18, 18),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            target_width = target.shape[1]
            for x, y, box_width, box_height in boxes:
                if mirrored:
                    x = target_width - x - box_width
                found.append(
                    {
                        "class_id": 1,
                        "bbox": [x / scale / width, y / scale / height,
                                 (x + box_width) / scale / width,
                                 (y + box_height) / scale / height],
                        "confidence": confidence,
                        "source": source + ("_mirrored" if mirrored else ""),
                    }
                )
    return nms(found, 0.35)


def person_head_candidates(box: list[float], confidence: float) -> list[dict]:
    """Turn a COCO person box into several conservative head/face priors."""

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    center = (x1 + x2) / 2
    variants = []
    for width_ratio, height_ratio in ((0.34, 0.18), (0.44, 0.23)):
        half_width = width * width_ratio / 2
        variants.append(
            {
                "class_id": 1,
                "bbox": [max(0.0, center - half_width), max(0.0, y1),
                         min(1.0, center + half_width), min(1.0, y1 + height * height_ratio)],
                "confidence": max(0.58, min(0.72, confidence * 0.80)),
                "source": "yolo_person_head_prior",
            }
        )
    return variants


def draw_candidates(image, candidates: list[dict]):
    height, width = image.shape[:2]
    for index, candidate in enumerate(candidates, 1):
        x1, y1, x2, y2 = candidate["bbox"]
        points = tuple(round(value) for value in (x1 * width, y1 * height, x2 * width, y2 * height))
        color = (0, 165, 255) if candidate["class_id"] == 0 else (0, 220, 0)
        cv2.rectangle(image, points[:2], points[2:], color, 2)
        label = f"C{index} {'phone' if candidate['class_id'] == 0 else 'face'}"
        cv2.putText(image, label, (points[0], max(14, points[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        candidate["candidate_id"] = f"C{index}"
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--phone-conf", type=float, default=0.03)
    args = parser.parse_args()
    image_dir = args.images.resolve()
    if image_dir.name != "images":
        raise ValueError("input must be an image-only directory named 'images'")
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        paths = paths[: args.limit]
    output = args.output.resolve()
    (output / "candidates").mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    phone_class_ids = [key for key, name in model.names.items() if name == "cell phone"]
    person_class_ids = [key for key, name in model.names.items() if name == "person"]
    if not phone_class_ids:
        raise ValueError("model has no COCO 'cell phone' class")
    results = model.predict(
        source=[str(path) for path in paths],
        classes=phone_class_ids + person_class_ids,
        conf=args.phone_conf,
        iou=0.6,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )
    for index, (path, result) in enumerate(zip(paths, results), 1):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        candidates = face_candidates(image)
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxyn.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                normalized = [float(value) for value in xyxy]
                if int(class_id) in phone_class_ids:
                    candidates.append(
                        {"class_id": 0, "bbox": normalized, "confidence": float(confidence),
                         "source": "yolo26m_coco"}
                    )
                elif int(class_id) in person_class_ids:
                    candidates.extend(person_head_candidates(normalized, float(confidence)))
        candidates = nms(candidates, 0.72)
        phones = sorted(
            (item for item in candidates if item["class_id"] == 0),
            key=lambda item: item["confidence"], reverse=True,
        )[:8]
        faces = sorted(
            (item for item in candidates if item["class_id"] == 1),
            key=lambda item: (item["source"] == "yolo_person_head_prior", item["confidence"]),
            reverse=True,
        )[:14]
        candidates = phones + faces
        overlay = draw_candidates(image.copy(), candidates)
        record = {"image": path.name, "width": width, "height": height, "candidates": candidates}
        (output / "candidates" / f"{path.stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        cv2.imwrite(str(output / "overlays" / path.name), overlay)
        print(f"[{index}/{len(paths)}] {path.name} candidates={len(candidates)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
