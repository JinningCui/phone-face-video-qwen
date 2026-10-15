"""Profile the live page's decode, detection, capture review and Qwen stages."""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from pathlib import Path

import cv2

from src.phone_face.capture import CaptureReviewer
from src.phone_face.live import LiveDetector


def summary(values):
    ordered = sorted(values)
    return {
        "seconds": round(sum(values), 3),
        "mean_ms": round(1000 * sum(values) / len(values), 2) if values else 0,
        "p95_ms": round(1000 * ordered[int(.95 * (len(ordered) - 1))], 2) if values else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("video_test/test.mp4"))
    parser.add_argument("--frames", type=int, default=724)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    detector = LiveDetector(Path("weights/phone_face/best.pt"),
                            Path("weights/yolov8n-pose.pt"))
    reviewer = CaptureReviewer()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise OSError(f"Cannot open {args.video}")
    timings = {name: [] for name in (
        "decode", "render", "detect_predict", "pose_predict", "consider", "qwen_review")}
    lock = threading.Lock()

    def timed_predict(model, label):
        original = model.predict

        def wrapped(*args, **kwargs):
            began = time.perf_counter()
            result = original(*args, **kwargs)
            with lock:
                timings[label].append(time.perf_counter() - began)
            return result

        model.predict = wrapped

    timed_predict(detector.model, "detect_predict")
    if detector.pose_model:
        timed_predict(detector.pose_model, "pose_predict")
    for method_name in ("_classify", "_classify_face"):
        original = getattr(reviewer, method_name)

        def timed_classify(*args, _original=original):
            began = time.perf_counter()
            try:
                return _original(*args)
            finally:
                with lock:
                    timings["qwen_review"].append(time.perf_counter() - began)

        setattr(reviewer, method_name, timed_classify)
    session_id = uuid.uuid4().hex
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    began = time.perf_counter()
    frames = 0
    while frames < args.frames:
        step = time.perf_counter()
        ok, frame = capture.read()
        timings["decode"].append(time.perf_counter() - step)
        if not ok:
            break
        step = time.perf_counter()
        detector.render(frame)
        timings["render"].append(time.perf_counter() - step)
        step = time.perf_counter()
        reviewer.consider(session_id, *detector.last_observation,
                          frame_time=frames / fps)
        timings["consider"].append(time.perf_counter() - step)
        frames += 1
    read_render_seconds = time.perf_counter() - began
    step = time.perf_counter()
    reviewer.tasks.join()
    join_seconds = time.perf_counter() - step
    total_seconds = time.perf_counter() - began
    state = reviewer.snapshot(session_id)
    report = {"frames": frames, "fps": fps,
              "read_render_seconds": round(read_render_seconds, 3),
              "qwen_join_seconds": round(join_seconds, 3),
              "total_seconds": round(total_seconds, 3),
              "qwen_candidates": reviewer.sessions[session_id]["next_sequence"],
              "people": len(state["people"]), "captures": len(state["captures"]),
              "stages": {name: summary(values) for name, values in timings.items()}}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
