"""Leakage-free Qwen phone/face annotation agent.

This module only accepts an image directory. It never discovers or reads YOLO
ground-truth label paths. Evaluation is intentionally implemented elsewhere.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI

from . import CLASS_NAMES
from .run import Box, COLORS, IMAGE_SUFFIXES, draw_boxes


SYSTEM_PROMPT = """你是目标检测标注智能体。只检测两类真实物体：
class_id=0 是手机（智能手机、功能手机；允许手持、遮挡和侧视）；
class_id=1 是可见人脸（正脸、侧脸、部分遮挡、被打码但脸部位置清楚）。
不要把平板、遥控器、相机、纯屏幕画面、后脑勺或衣服标成目标。
必须找出图中每一个目标，不能只给最显眼的一个。
bbox_2d 必须是 [x1,y1,x2,y2]，坐标相对于整张输入图归一化到 0..1000，且 x1<x2、y1<y2。
只输出 JSON 对象，不要 Markdown：
{"detections":[{"class_id":0,"bbox_2d":[0,0,1000,1000],"confidence":0.0}],"uncertain":false}
没有目标时 detections 为空数组。confidence 表示你对类别和边界共同正确的置信度。"""

REVIEW_PROMPT = """你是独立复核员。请重新查看原图，并审查下方调查员候选。
补上漏掉的每一部手机和每一张人脸；删除误检；收紧过大的框并补足被截断的框。
最终仍按系统规定的同一 JSON 协议输出完整检测列表，而不是修改说明。

调查员输出：
{investigator}
"""


@dataclass
class AgentResult:
    image: str
    detections: list[Box]
    investigator_raw: str
    reviewer_raw: str
    error: str
    seconds: float


_thread_state = threading.local()


def client_for(base_url: str, api_key: str) -> OpenAI:
    client = getattr(_thread_state, "client", None)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0, max_retries=2)
        _thread_state.client = client
    return client


def image_data_url(path: Path, max_side: int) -> str:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("unreadable image")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("JPEG encoding failed")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def boxes_from_response(text: str) -> list[Box]:
    value = extract_json(text)
    detections = value.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError("detections must be an array")
    boxes: list[Box] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        try:
            class_id = int(item["class_id"])
            coords = [float(number) for number in item["bbox_2d"]]
            confidence = float(item.get("confidence", 0.5))
        except (KeyError, TypeError, ValueError):
            continue
        if class_id not in CLASS_NAMES or len(coords) != 4:
            continue
        x1, y1, x2, y2 = [max(0.0, min(1000.0, number)) / 1000.0 for number in coords]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            Box(
                class_id=class_id,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                width=x2 - x1,
                height=y2 - y1,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )
    return boxes


def overlap(left: Box, right: Box) -> float:
    lx1, ly1 = left.cx - left.width / 2, left.cy - left.height / 2
    lx2, ly2 = left.cx + left.width / 2, left.cy + left.height / 2
    rx1, ry1 = right.cx - right.width / 2, right.cy - right.height / 2
    rx2, ry2 = right.cx + right.width / 2, right.cy + right.height / 2
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def adjudicate(boxes: list[Box], confidence_threshold: float, nms_iou: float) -> list[Box]:
    accepted: list[Box] = []
    ordered = sorted(boxes, key=lambda box: box.confidence or 0.0, reverse=True)
    for box in ordered:
        if (box.confidence or 0.0) < confidence_threshold:
            continue
        if any(box.class_id == kept.class_id and overlap(box, kept) >= nms_iou for kept in accepted):
            continue
        accepted.append(box)
    return accepted


def call_agent(
    image_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    max_side: int,
    confidence: float,
    nms_iou: float,
) -> AgentResult:
    started = time.monotonic()
    investigator_raw = reviewer_raw = ""
    error = ""
    try:
        data_url = image_data_url(image_path, max_side)
        client = client_for(base_url, api_key)
        common = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }
        first = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "调查阶段：检测所有手机和人脸。"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            **common,
        )
        investigator_raw = first.choices[0].message.content or ""
        investigator_boxes = boxes_from_response(investigator_raw)

        second = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": REVIEW_PROMPT.format(investigator=investigator_raw),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            **common,
        )
        reviewer_raw = second.choices[0].message.content or ""
        reviewer_boxes = boxes_from_response(reviewer_raw)
        # The two visual passes are independent evidence sources. A reviewer
        # occasionally over-corrects by returning an empty list, so deterministic
        # adjudication takes their union and applies class-aware NMS.
        final_boxes = investigator_boxes + reviewer_boxes
        final_boxes = adjudicate(final_boxes, confidence, nms_iou)
    except Exception as exc:  # recorded per image so the batch remains resumable
        error = f"{type(exc).__name__}: {exc}"
        final_boxes = []
    return AgentResult(
        image=image_path.name,
        detections=final_boxes,
        investigator_raw=investigator_raw,
        reviewer_raw=reviewer_raw,
        error=error,
        seconds=time.monotonic() - started,
    )


def save_result(result: AgentResult, image_dir: Path, output: Path) -> None:
    stem = Path(result.image).stem
    label_dir = output / "labels"
    overlay_dir = output / "images"
    trajectory_dir = output / "trajectories"
    label_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{stem}.txt").write_text(
        "".join(
            f"{box.class_id} {box.cx:.8f} {box.cy:.8f} {box.width:.8f} "
            f"{box.height:.8f} {(box.confidence or 0.0):.6f}\n"
            for box in result.detections
        ),
        encoding="utf-8",
    )
    image = cv2.imread(str(image_dir / result.image))
    if image is not None:
        cv2.imwrite(str(overlay_dir / result.image), draw_boxes(image, result.detections))
    (trajectory_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "image": result.image,
                "investigator_raw": result.investigator_raw,
                "reviewer_raw": result.reviewer_raw,
                "detections": [box.__dict__ for box in result.detections],
                "error": result.error,
                "seconds": result.seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen phone/face annotation agent")
    parser.add_argument("images", type=Path, help="image directory only; labels are never read")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    image_dir = args.images.resolve()
    output = args.output.resolve()
    if image_dir.name != "images":
        raise ValueError("input must be an image-only directory named 'images'")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.resume:
        images = [path for path in images if not (output / "trajectories" / f"{path.stem}.json").exists()]
    if args.limit:
        images = images[: args.limit]
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                call_agent,
                path,
                args.base_url,
                args.api_key,
                args.model,
                args.max_side,
                args.confidence,
                args.nms_iou,
            ): path
            for path in images
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            save_result(result, image_dir, output)
            rows.append(
                {
                    "image": result.image,
                    "phone_count": sum(box.class_id == 0 for box in result.detections),
                    "face_count": sum(box.class_id == 1 for box in result.detections),
                    "seconds": round(result.seconds, 3),
                    "status": "error" if result.error else "ok",
                    "error": result.error,
                }
            )
            print(f"[{completed}/{len(images)}] {result.image} boxes={len(result.detections)} {result.error}", flush=True)
    rows.sort(key=lambda row: str(row["image"]))
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image", "phone_count", "face_count", "seconds", "status", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"images": len(rows), "errors": sum(row["status"] == "error" for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
