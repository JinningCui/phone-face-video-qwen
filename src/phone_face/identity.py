"""Small CPU face embedding model used to reconnect a person after re-entry."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR = ROOT / 'weights/face_reid/face_detection_yunet_2023mar.onnx'
DEFAULT_RECOGNIZER = ROOT / 'weights/face_reid/face_recognition_sface_2021dec.onnx'
if not DEFAULT_DETECTOR.is_file() or not DEFAULT_RECOGNIZER.is_file():
    # Older installations placed the same two files beside the phone/face YOLO model.
    DEFAULT_DETECTOR = ROOT / 'weights/phone_face/face_detection_yunet_2023mar.onnx'
    DEFAULT_RECOGNIZER = ROOT / 'weights/phone_face/face_recognition_sface_2021dec.onnx'


def _landmarks_frontal(face, max_nose_offset=.28):
    """Use YuNet's eye and nose landmarks to reject obvious profile faces."""
    right_eye = np.asarray(face[4:6], dtype=np.float32)
    left_eye = np.asarray(face[6:8], dtype=np.float32)
    nose = np.asarray(face[8:10], dtype=np.float32)
    eye_axis = left_eye - right_eye
    eye_distance = float(np.linalg.norm(eye_axis))
    if eye_distance < 1:
        return False
    midpoint = (left_eye + right_eye) / 2
    offset = abs(float(np.dot(nose - midpoint, eye_axis))) / (eye_distance ** 2)
    return offset <= max_nose_offset


class FaceIdentity:
    def __init__(self):
        detector_path = Path(os.environ.get('FACE_REID_DETECTOR', DEFAULT_DETECTOR))
        recognizer_path = Path(os.environ.get('FACE_REID_MODEL', DEFAULT_RECOGNIZER))
        self.available = detector_path.is_file() and recognizer_path.is_file()
        self.lock = threading.Lock()
        if self.available:
            self.detector = cv2.FaceDetectorYN.create(
                str(detector_path), '', (320, 320), score_threshold=.7)
            self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), '')
        else:
            self.detector = None
            self.recognizer = None
            print('Face re-identification disabled: YuNet/SFace weights missing', flush=True)

    def feature(self, face_crop):
        """Return a unit SFace embedding; never change the image sent to users."""
        if not self.available or face_crop is None or face_crop.size == 0:
            return None
        height, width = face_crop.shape[:2]
        if min(height, width) < 40:
            return None
        scale = min(4.0, max(1.0, 320 / min(height, width)))
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(face_crop, size, interpolation=cv2.INTER_CUBIC) if scale > 1 else face_crop
        with self.lock:
            self.detector.setInputSize(size)
            _, faces = self.detector.detect(image)
            if faces is None:
                return None
            # The YOLO crop should contain one face. Prefer the largest central detection.
            cx, cy = size[0] / 2, size[1] / 2
            face = max(faces, key=lambda row: float(row[2] * row[3]) /
                       (1 + ((row[0] + row[2] / 2 - cx) / size[0]) ** 2 +
                        ((row[1] + row[3] / 2 - cy) / size[1]) ** 2))
            aligned = self.recognizer.alignCrop(image, face)
            vector = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None

    def is_frontal_png(self, face_png):
        """Return False for a detected profile; None if no landmarks are available."""
        if not self.available or not face_png:
            return None
        image = cv2.imdecode(np.frombuffer(face_png, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        if min(height, width) < 40:
            return None
        scale = min(4.0, max(1.0, 320 / min(height, width)))
        size = (round(width * scale), round(height * scale))
        work = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) if scale > 1 else image
        with self.lock:
            self.detector.setInputSize(size)
            _, faces = self.detector.detect(work)
        if faces is None:
            return None
        cx, cy = size[0] / 2, size[1] / 2
        face = max(faces, key=lambda row: float(row[2] * row[3]) /
                   (1 + ((row[0] + row[2] / 2 - cx) / size[0]) ** 2 +
                    ((row[1] + row[3] / 2 - cy) / size[1]) ** 2))
        return _landmarks_frontal(face)
