"""Conservative same-session face re-identification for returning subjects."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[2] / 'weights' / 'face_reid'
YUNET = MODEL_ROOT / 'face_detection_yunet_2023mar.onnx'
SFACE = MODEL_ROOT / 'face_recognition_sface_2021dec.onnx'


class FaceReidentifier:
    def __init__(self, yunet: Path = YUNET, sface: Path = SFACE):
        self.detector = cv2.FaceDetectorYN.create(str(yunet), '', (320, 320),
                                                 score_threshold=.75)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), '')
        self.lock = threading.Lock()

    def extract(self, crop):
        """Return a unit-length face vector, or None if alignment is uncertain."""
        if crop is None or min(crop.shape[:2]) < 60:
            return None
        face_image = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_LINEAR)
        try:
            with self.lock:
                _, faces = self.detector.detect(face_image)
                if faces is None or not len(faces):
                    return None
                # The YOLO crop should contain just one face; prefer the central one.
                face = min(faces, key=lambda item:
                           (item[0] + item[2] / 2 - 160) ** 2 +
                           (item[1] + item[3] / 2 - 160) ** 2)
                aligned = self.recognizer.alignCrop(face_image, face)
                vector = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        except cv2.error:
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-6 else None


def choose_returning_track(tracks, embedding, assigned, now, threshold=None, margin=None):
    """Reuse an ID only when one confirmed track has an unambiguous match."""
    if embedding is None:
        return None
    threshold = float(os.environ.get('FACE_REID_SIMILARITY', '.75')) if threshold is None else threshold
    margin = float(os.environ.get('FACE_REID_MARGIN', '.12')) if margin is None else margin
    scores = []
    for track in tracks:
        # An active face elsewhere in the current frame cannot be the returner.
        # A dormant ID is considered only after spatial tracking has expired.
        if (track['id'] in assigned or not track['images'] or
                now - track['seen'] < 12):
            continue
        gallery = track.get('embeddings', [])
        if gallery:
            scores.append((max(float(np.dot(embedding, known)) for known in gallery), track))
    scores.sort(key=lambda item: item[0], reverse=True)
    if (not scores or scores[0][0] < threshold or
            (len(scores) > 1 and scores[0][0] - scores[1][0] < margin)):
        return None
    return scores[0][1]
