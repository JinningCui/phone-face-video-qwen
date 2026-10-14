import unittest
import json
import os
from unittest.mock import patch

import numpy as np

from src.phone_face.capture import CaptureReviewer, _candidates, _store_ranked


class CaptureTests(unittest.TestCase):
    def setUp(self):
        # Legacy association fixtures use 80px faces; production defaults now
        # require at least 96px and sharper source pixels.
        settings = patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '64',
                                          'CAPTURE_MIN_SHARPNESS': '22'})
        settings.start()
        self.addCleanup(settings.stop)
        self.frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        y, x = np.indices((80, 80))
        self.frame[40:120, 90:170] = ((x + y) % 2 * 190 + 30)[:, :, None]
        self.detections = [
            {'class_id': 1, 'xyxy': [90, 40, 170, 120]},
            {'class_id': 0, 'xyxy': [145, 135, 175, 165]},
        ]
        self.people = [{'box': [60, 20, 220, 220], 'keypoints': [(9, (160, 150))]}]

    def test_requires_wrist_near_phone_and_clear_face(self):
        self.assertEqual(len(list(_candidates(self.frame, self.detections, self.people))), 1)
        far_wrist = [{'box': [60, 20, 220, 220], 'keypoints': [(9, (70, 200))]}]
        self.assertEqual(len(list(_candidates(self.frame, self.detections, far_wrist))), 0)
        clipped = [dict(self.detections[0], xyxy=[0, 40, 80, 120]), self.detections[1]]
        self.assertEqual(len(list(_candidates(self.frame, clipped, self.people))), 0)
        low_conf = [dict(self.detections[0], confidence=.4), self.detections[1]]
        self.assertEqual(len(list(_candidates(self.frame, low_conf, self.people))), 0)
        covered = [self.detections[0], dict(self.detections[1], xyxy=[130, 95, 160, 130])]
        hand_at_face = [{'box': [60, 20, 220, 220], 'keypoints': [(9, (150, 115))]}]
        self.assertEqual(len(list(_candidates(self.frame, covered, hand_at_face))), 0)

    def test_production_minimum_rejects_small_face(self):
        with patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '96'}):
            self.assertEqual(len(list(_candidates(self.frame, self.detections, self.people))), 0)

    def test_qwen_confirmation_publishes_face_image(self):
        reviewer = CaptureReviewer()
        reviewer._classify = lambda person, face, phone: True
        session = 'a' * 32
        reviewer.consider(session, self.frame, self.detections, self.people)
        reviewer.tasks.join()
        snapshot = reviewer.snapshot(session)
        self.assertEqual(len(snapshot['captures']), 1)
        self.assertTrue(reviewer.image(session, snapshot['captures'][0]['id']).startswith(b'\xff\xd8'))

    def test_phone_is_assigned_only_to_nearest_pose_wrist(self):
        frame = self.frame.copy()
        frame[40:120, 180:260] = frame[40:120, 90:170]
        detections = self.detections + [
            {'class_id': 1, 'xyxy': [180, 40, 260, 120]},
        ]
        people = [
            {'box': [60, 20, 210, 220],
             'keypoints': [(0, (130, 80)), (9, (160, 150))]},
            {'box': [130, 20, 300, 220],
             'keypoints': [(0, (220, 80)), (9, (185, 150))]},
        ]
        candidates = list(_candidates(frame, detections, people))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], [90, 40, 170, 120])

    def test_qwen35_uses_non_thinking_response(self):
        reviewer = CaptureReviewer()
        reviewer.model = 'Qwen3.5-4B'
        reply = {'choices': [{'message': {'content':
            '{"real_phone":true,"shooting":true,"visible_shooting_evidence":true,'
            '"valid_face":true,"unoccluded":true,"clear":true}'}}]}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, *_): return json.dumps(reply).encode()
        with patch.dict(os.environ, {'QWEN_DISABLE_THINKING': '1'}), \
                patch('urllib.request.urlopen', return_value=Response()) as urlopen:
            self.assertTrue(reviewer._classify(b'person', b'face', b'phone'))
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})

    def test_top_five_keeps_high_scores_and_replaces_similar_frame(self):
        track = {'images': []}
        for index in range(9):
            _store_ranked(track, bytes([index]), index + 1, 255 << (index * 8))
        self.assertEqual(len(track['images']), 5)
        self.assertEqual([item['score'] for item in track['images']], list(range(9, 4, -1)))
        best_id = track['images'][0]['id']
        _store_ranked(track, b'better', 20, track['images'][0]['signature'])
        self.assertEqual(len(track['images']), 5)
        self.assertEqual(track['images'][0]['id'], best_id)
        self.assertEqual(track['images'][0]['version'], 2)


if __name__ == '__main__':
    unittest.main()
