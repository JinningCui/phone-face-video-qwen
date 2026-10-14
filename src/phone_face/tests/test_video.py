import json
from types import SimpleNamespace

import cv2
import numpy as np
import unittest
import tempfile
from pathlib import Path

from src.phone_face.run import build_parser
from src.phone_face.video import predict_video


def setup_video(tmp_path):
    source = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 12, (64, 48))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    model = tmp_path / 'model.pt'
    model.touch()
    return build_parser().parse_args(['video', str(source), '--model', str(model),
                                     '--output', str(tmp_path / 'out'), '--phone-conf', '.5'])


class FakeModel:
    names = {0: 'phone', 1: 'face'}

    def __init__(self, path):
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        classes = [0, 1] if self.calls != 2 else []
        boxes = SimpleNamespace(cls=np.array(classes), conf=np.array([.3, .9] if classes else []),
                                xywhn=np.array([[.5, .5, .4, .4]] * len(classes)),
                                xyxy=np.array([[19, 14, 45, 34]] * len(classes)))
        return [SimpleNamespace(boxes=boxes)]


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_video_frames_timestamps_and_class_thresholds(self):
        tmp_path = Path(self.temp.name)
        args = setup_video(tmp_path)
        assert predict_video(args, FakeModel) == 0
        output = args.output / args.name
        rows = [json.loads(line) for line in (output / 'detections.jsonl').read_text().splitlines()]
        assert len(rows) == 4
        assert [r['frame_index'] for r in rows] == [0, 1, 2, 3]
        assert rows[-1]['timestamp_seconds'] == 3 / 12
        assert rows[1]['detections'] == []
        assert [d['class_name'] for d in rows[0]['detections']] == ['face']
        cap = cv2.VideoCapture(str(output / 'input_detected.mp4'))
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
        assert cap.get(cv2.CAP_PROP_FPS) == 12
        cap.release()
        assert len(decoded) == 4
        assert decoded[0].sum() > 0
        summary = json.loads((output / 'summary.json').read_text())
        assert summary['detection_counts'] == {'phone': 0, 'face': 3}
        with self.assertRaises(FileExistsError):
            predict_video(args, FakeModel)


    def test_invalid_input_and_model(self):
        tmp_path = Path(self.temp.name)
        args = setup_video(tmp_path)
        args.source = tmp_path / 'missing.mp4'
        with self.assertRaises(FileNotFoundError):
            predict_video(args, FakeModel)
        args.source.write_bytes(b'not a video')
        with self.assertRaisesRegex(ValueError, 'Cannot open'):
            predict_video(args, FakeModel)


    def test_reject_wrong_classes(self):
        tmp_path = Path(self.temp.name)
        args = setup_video(tmp_path)
        class WrongModel(FakeModel):
            names = {0: 'person'}
        with self.assertRaisesRegex(ValueError, 'Expected phone/face'):
            predict_video(args, WrongModel)
