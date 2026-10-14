"""Real-time phone/face detection for USB webcams and network video sources."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

from .video import _associate_phone, _draw_detection, _pose_people, _smooth_detections


class LiveDetector:
    def __init__(self, model_path: Path, pose_path: Path, device: str | None = None):
        from ultralytics import YOLO

        self.device = device or os.environ.get("DEVICE", "cpu")
        self.imgsz = int(os.environ.get("LIVE_IMG_SIZE", "1280"))
        self.phone_conf = float(os.environ.get("LIVE_PHONE_CONF", os.environ.get("PHONE_CONF", "0.12")))
        self.face_conf = float(os.environ.get("FACE_CONF", "0.35"))
        self.pose_conf = float(os.environ.get("POSE_CONF", "0.35"))
        self.alpha = float(os.environ.get("SMOOTH_ALPHA", "0.55"))
        self.line_width = max(1, int(os.environ.get("LINE_WIDTH", "1")))
        self.max_width = max(320, int(os.environ.get("LIVE_MAX_WIDTH", "1920")))
        self.jpeg_quality = min(95, max(40, int(os.environ.get("LIVE_JPEG_QUALITY", "80"))))
        if self.imgsz <= 0 or not all(0 <= x <= 1 for x in
                                      (self.phone_conf, self.face_conf, self.pose_conf)):
            raise ValueError("Invalid live inference settings")
        self.model = YOLO(str(model_path))
        names = dict(enumerate(self.model.names)) if isinstance(self.model.names, list) else self.model.names
        if names != {0: "phone", 1: "face"}:
            raise ValueError(f"Expected phone/face classes, got {names}")
        self.pose_model = YOLO(str(pose_path)) if os.environ.get("LIVE_POSE", "1") != "0" else None
        self.previous = []
        self.last_frame_at = None

    def render(self, frame):
        source_frame = frame
        height, width = frame.shape[:2]
        # Detection and face capture must never consume the reduced JPEG preview.
        # Ultralytics still letterboxes to imgsz internally for model inference.
        result = self.model.predict(
            source=source_frame, conf=min(self.phone_conf, self.face_conf),
            imgsz=self.imgsz, device=self.device, verbose=False,
        )[0]
        detections = []
        thresholds = {0: self.phone_conf, 1: self.face_conf}
        names = {0: "phone", 1: "face"}
        for xyxy, confidence, class_id in zip(result.boxes.xyxy.tolist(),
                                               result.boxes.conf.tolist(),
                                               result.boxes.cls.tolist()):
            class_id = int(class_id)
            if class_id not in thresholds or confidence < thresholds[class_id]:
                continue
            detections.append({"class_id": class_id, "class_name": names[class_id],
                               "confidence": float(confidence), "xyxy": list(map(float, xyxy))})
        people = []
        if self.pose_model is not None:
            pose_result = self.pose_model.predict(
                source=source_frame, conf=self.pose_conf, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )[0]
            people = _pose_people(pose_result, self.pose_conf)
        faces = [d for d in detections if d["class_id"] == 1]
        for phone in (d for d in detections if d["class_id"] == 0):
            _associate_phone(phone, faces, people)
        self.previous = _smooth_detections(detections, self.previous, self.alpha)
        self.last_observation = (source_frame, detections, people)
        scale = min(1.0, self.max_width / width)
        if scale < 1:
            preview = cv2.resize(source_frame, (self.max_width, round(height * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            preview = source_frame
        rendered = preview.copy()
        for detection in detections:
            shown = detection if scale == 1 else detection | {
                "xyxy": [value * scale for value in detection["xyxy"]]}
            _draw_detection(rendered, shown, self.line_width)
        now = time.monotonic()
        fps = 0.0 if self.last_frame_at is None else 1.0 / max(now - self.last_frame_at, 0.001)
        self.last_frame_at = now
        cv2.putText(rendered, f"LIVE  {fps:.1f} FPS", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        ok, jpeg = cv2.imencode(".jpg", rendered,
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise OSError("Could not encode live frame")
        return jpeg.tobytes()


def open_camera(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif source.startswith("/dev/video"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    elif source.startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    elif source.startswith(("rtmp://", "rtmps://")):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise OSError(f"无法打开摄像头或视频源：{source}")
    if source.isdigit() or source.startswith("/dev/video"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.environ.get("CAMERA_WIDTH", "1280")))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.environ.get("CAMERA_HEIGHT", "720")))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
