"""Ordering and throughput checks using deterministic fake inference."""
import os
import time
import unittest
from unittest.mock import patch

import numpy as np


from src.phone_face import capture


def run_case(urls):
    with patch.dict(os.environ, {'QWEN_VL_URLS': ','.join(urls),
                                 'QWEN_REVIEW_WORKERS': '1'}):
        reviewer = capture.CaptureReviewer()
    session_id = 'test'
    track_id = 'person'
    reviewer.sessions[session_id] = {
        'tracks': {track_id: {'id': track_id, 'pending': 8, 'images': [],
                              'center': (0, 0), 'seen': time.monotonic()}},
        'last_frame': time.monotonic(), 'status': '',
        'next_sequence': 8, 'next_commit': 0, 'completed': {},
    }
    endpoints = set()
    committed = []

    def classify(person, _face, _phone):
        endpoints.add(reviewer.worker_local.url)
        # Make frame 0 finish last to exercise ordered publication.
        time.sleep(.18 if person == b'0' else .09)
        return True

    def store(_track, _face, score, _signature, _frontal=False):
        committed.append(score)

    reviewer._classify = classify
    started = time.monotonic()
    with patch.object(capture, '_store_ranked', store):
        for index in range(8):
            reviewer.tasks.put((session_id, index, track_id, str(index).encode(),
                                b'face', b'phone', index, index, time.monotonic(), False))
        reviewer.tasks.join()
    return time.monotonic() - started, committed, endpoints, reviewer


class ParallelTests(unittest.TestCase):
    def test_consider_does_not_discard_when_queue_is_full(self):
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        y, x = np.indices((80, 80))
        frame[40:120, 90:170] = ((x + y) % 2 * 190 + 30)[:, :, None]
        detections = [{'class_id': 1, 'xyxy': [90, 40, 170, 120]},
                      {'class_id': 0, 'xyxy': [145, 135, 175, 165]}]
        people = [{'box': [60, 20, 220, 220], 'keypoints': [(9, (160, 150))]}]
        with patch.dict(os.environ, {'QWEN_VL_URLS': 'http://0,http://1,http://2,http://3',
                                     'QWEN_REVIEW_QUEUE_SIZE': '1',
                                     'CAPTURE_MIN_FACE_SIDE': '64',
                                     'CAPTURE_MIN_SHARPNESS': '22'}):
            reviewer = capture.CaptureReviewer()
            reviewer.frame_interval = 0
            reviewer.interval = 0
            reviewer._classify = lambda *_: (time.sleep(.05) or True)
            reviewer._classify_face = lambda *_: {'face': True, 'frontal': True}
            reviewer.identity.feature = lambda *_: None
            samples = iter(range(9))
            def observations(*_):
                index = next(samples)
                return iter([{'center': (130, 80), 'width': 80,
                              'quality': (100 + index, frame[31:129, 81:179]),
                              'context': frame, 'phone_crop': frame[120:180, 120:180]}])
            signatures = [255 << (8 * index) for index in range(9)]
            with patch.object(capture, '_frame_observations', side_effect=observations), \
                    patch.object(capture, '_face_signature', side_effect=signatures):
                for _ in range(9):
                    reviewer.consider('test', frame, detections, people)
            reviewer.tasks.join()
        session = reviewer.sessions['test']
        self.assertEqual(session['next_sequence'], 8)
        self.assertEqual(session['next_commit'], 8)
        self.assertEqual(next(iter(session['tracks'].values()))['pending'], 0)

    def test_four_endpoints_and_order(self):
        serial_time, serial_order, _, _ = run_case(['http://only-one'])
        parallel_time, order, endpoints, reviewer = run_case(
            [f'http://gpu-{index}' for index in range(4)])
        self.assertEqual(serial_order, list(range(8)))
        self.assertEqual(order, list(range(8)))
        self.assertEqual(len(endpoints), 4)
        self.assertEqual(reviewer.sessions['test']['next_commit'], 8)
        self.assertEqual(reviewer.sessions['test']['tracks']['person']['pending'], 0)
        self.assertLess(parallel_time, serial_time * .6)
        print(f'fake inference: one endpoint {serial_time:.3f}s; four endpoints '
              f'{parallel_time:.3f}s (no GPU/model used)')


if __name__ == '__main__':
    unittest.main()
