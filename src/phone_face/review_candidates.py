"""Controlled Qwen reviewer: the VLM may select candidate IDs, not move boxes."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from openai import OpenAI

from .qwen_agent import extract_json, image_data_url
from .run import Box, IMAGE_SUFFIXES, draw_boxes


SYSTEM = """你是受控目标检测证据审查员。原图上已画出编号候选框：橙色 phone，绿色 face。
逐个判断候选框是否确实覆盖对应类别。框不必完美，但必须覆盖目标主体且类别正确。
不要因为遮挡、侧脸、低清、打码而拒绝真实目标；不要把平板、遥控器、衣服、后脑勺或背景纹理当目标。
你只能选择输入中存在的 candidate_id，不能修改坐标。
只输出 JSON：{"accepted_ids":["C1"],"missing_classes":["phone"],"summary":""}。
missing_classes 只表示肉眼明确看到遗漏但候选中没有相应框，可为空。"""

REVIEW = """请独立复查所有候选。上一位调查员给出的结果如下，但你必须以图像证据为准：
{first}
输出你自己的 accepted_ids、missing_classes 和简短 summary。"""

ADJUDICATE = """你是最终裁决员。调查员与复核员发生分歧。请逐个查看编号框，只保留确实覆盖
phone 或 face 的候选。不要因为前两人选择或拒绝而盲从。输出同一 JSON 协议。
调查员：{first}
复核员：{second}
候选表：{candidates}"""


_state = threading.local()


def client(base_url: str, api_key: str) -> OpenAI:
    value = getattr(_state, "client", None)
    if value is None:
        value = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0, max_retries=2)
        _state.client = value
    return value


def selected_ids(text: str, allowed: set[str]) -> set[str]:
    value = extract_json(text)
    ids = value.get("accepted_ids", [])
    if not isinstance(ids, list):
        raise ValueError("accepted_ids must be an array")
    return {str(item) for item in ids if str(item) in allowed}


def review_one(
    image_path: Path,
    candidate_path: Path,
    overlay_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    max_side: int,
) -> dict:
    started = time.monotonic()
    record = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = record["candidates"]
    allowed = {item["candidate_id"] for item in candidates}
    if not candidates:
        return {"image": image_path.name, "boxes": [], "first": "", "second": "",
                "error": "", "seconds": time.monotonic() - started}
    try:
        raw_url = image_data_url(image_path, max_side)
        overlay_url = image_data_url(overlay_path, max_side)
        candidate_table = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
        api = client(base_url, api_key)
        common = dict(model=model, temperature=0.0, max_tokens=700,
                      response_format={"type": "json_object"})
        first_response = api.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": "调查阶段。候选表：" + candidate_table},
                    {"type": "image_url", "image_url": {"url": raw_url}},
                    {"type": "image_url", "image_url": {"url": overlay_url}},
                ]},
            ], **common,
        )
        first = first_response.choices[0].message.content or ""
        first_ids = selected_ids(first, allowed)
        second_response = api.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": REVIEW.format(first=first) + "\n候选表：" + candidate_table},
                    {"type": "image_url", "image_url": {"url": raw_url}},
                    {"type": "image_url", "image_url": {"url": overlay_url}},
                ]},
            ], **common,
        )
        second = second_response.choices[0].message.content or ""
        second_ids = selected_ids(second, allowed)
        dense_unanimous_accept = (
            len(candidates) >= 8
            and len(first_ids) >= max(6, round(len(candidates) * 0.75))
        )
        if first_ids == second_ids and not dense_unanimous_accept:
            accepted = first_ids
            third = ""
        else:
            third_response = api.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": ADJUDICATE.format(
                            first=first, second=second, candidates=candidate_table)},
                        {"type": "image_url", "image_url": {"url": raw_url}},
                        {"type": "image_url", "image_url": {"url": overlay_url}},
                    ]},
                ], **common,
            )
            third = third_response.choices[0].message.content or ""
            accepted = selected_ids(third, allowed)
        boxes = []
        for item in candidates:
            if item["candidate_id"] not in accepted:
                continue
            x1, y1, x2, y2 = item["bbox"]
            votes = int(item["candidate_id"] in first_ids) + int(item["candidate_id"] in second_ids)
            confidence = max(float(item["confidence"]), 0.85 if votes == 2 else 0.55)
            boxes.append(Box(item["class_id"], (x1 + x2) / 2, (y1 + y2) / 2,
                             x2 - x1, y2 - y1, confidence))
        error = ""
    except Exception as exc:
        first = locals().get("first", "")
        second = locals().get("second", "")
        third = locals().get("third", "")
        boxes = []
        error = f"{type(exc).__name__}: {exc}"
    return {"image": image_path.name, "boxes": boxes, "first": first, "second": second,
            "third": locals().get("third", ""),
            "error": error, "seconds": time.monotonic() - started}


def save(result: dict, image_dir: Path, output: Path) -> None:
    stem = Path(result["image"]).stem
    for directory in ("labels", "images", "trajectories"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    boxes: list[Box] = result["boxes"]
    (output / "labels" / f"{stem}.txt").write_text(
        "".join(f"{box.class_id} {box.cx:.8f} {box.cy:.8f} {box.width:.8f} "
                f"{box.height:.8f} {(box.confidence or 0):.6f}\n" for box in boxes),
        encoding="utf-8",
    )
    image = cv2.imread(str(image_dir / result["image"]))
    if image is not None:
        cv2.imwrite(str(output / "images" / result["image"]), draw_boxes(image, boxes))
    payload = {**result, "boxes": [box.__dict__ for box in boxes]}
    (output / "trajectories" / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()
    image_dir, candidate_root, output = args.images.resolve(), args.candidates.resolve(), args.output.resolve()
    candidate_files = sorted((candidate_root / "candidates").glob("*.json"))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for candidate_path in candidate_files:
            record = json.loads(candidate_path.read_text(encoding="utf-8"))
            image_path = image_dir / record["image"]
            future = executor.submit(review_one, image_path, candidate_path,
                                     candidate_root / "overlays" / record["image"],
                                     args.base_url, args.api_key, args.model, args.max_side)
            futures[future] = image_path
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            save(result, image_dir, output)
            rows.append({"image": result["image"], "boxes": len(result["boxes"]),
                         "seconds": round(result["seconds"], 3),
                         "status": "error" if result["error"] else "ok", "error": result["error"]})
            print(f"[{index}/{len(futures)}] {result['image']} boxes={len(result['boxes'])} {result['error']}", flush=True)
    rows.sort(key=lambda item: item["image"])
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "boxes", "seconds", "status", "error"])
        writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
