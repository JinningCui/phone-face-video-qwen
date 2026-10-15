import unittest
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.phone_face.capture import CaptureReviewer, _candidates, _face_quality, _store_ranked
from src.phone_face.identity import _landmarks_frontal


class CaptureTests(unittest.TestCase):
    def test_landmark_gate_rejects_profile_face(self):
        frontal = np.zeros(15, dtype=np.float32)
        frontal[4:10] = [25, 20, 75, 20, 50, 45]
        profile = frontal.copy()
        profile[8] = 16
        self.assertTrue(_landmarks_frontal(frontal))
        self.assertFalse(_landmarks_frontal(profile))
        tilted = frontal.copy()
        tilted[4:10] = [25, 10, 65, 40, 33, 45]
        self.assertTrue(_landmarks_frontal(tilted))

    def setUp(self):
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

    def test_lowlight_face_reaches_quality_review(self):
        frame = np.zeros((220, 220, 3), dtype=np.uint8)
        rows, columns = np.indices((120, 120))
        frame[50:170, 50:170] = (((rows + columns) % 2) * 44)[:, :, None]
        face = {'xyxy': [50, 50, 170, 170], 'confidence': .9}
        with patch.dict(os.environ, {'CAPTURE_MIN_FACE_BRIGHTNESS': '18'}):
            self.assertIsNotNone(_face_quality(frame, face))
        with patch.dict(os.environ, {'CAPTURE_MIN_FACE_BRIGHTNESS': '24'}):
            self.assertIsNone(_face_quality(frame, face))

    def test_qwen_confirmation_publishes_face_image(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer._classify = lambda person, face, phone: True
        session = 'a' * 32
        reviewer.consider(session, self.frame, self.detections, self.people)
        self.assertEqual(len(reviewer.snapshot(session)['captures']), 0)
        reviewer.consider(session, self.frame, self.detections, self.people)
        reviewer.tasks.join()
        snapshot = reviewer.snapshot(session)
        self.assertEqual(len(snapshot['captures']), 1)
        self.assertTrue(reviewer.image(session, snapshot['captures'][0]['id']).startswith(b'\x89PNG\r\n\x1a\n'))

    def test_visible_gallery_capture_is_published_for_ftp(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'PHONE_FACE_EXPORT_DIR': directory,
                                         'CAPTURE_MIN_FACE_SIDE': '64',
                                         'CAPTURE_MIN_SHARPNESS': '22'}):
            reviewer = CaptureReviewer()
            reviewer.frame_interval = 0
            reviewer._classify = lambda *_: {'shooting': True, 'face': True,
                                              'frontal': False}
            session = 'a' * 32
            reviewer.consider(session, self.frame, self.detections, self.people)
            reviewer.consider(session, self.frame, self.detections, self.people)
            reviewer.tasks.join()
            state = reviewer.snapshot(session)
            self.assertEqual(len(state['captures']), 1)
            self.assertFalse(state['captures'][0]['frontal'])
            person = state['people'][0]['id']
            manifest = Path(directory) / session / person / 'manifest.json'
            self.assertTrue(manifest.is_file())
            self.assertTrue((manifest.parent / '01.png').is_file())

    @patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '64',
                             'CAPTURE_MIN_SHARPNESS': '22'})
    def test_all_qwen_images_are_lossless_png(self):
        expected = list(_candidates(self.frame, self.detections, self.people))[0][1:4]
        received = []
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer._classify = lambda *images: received.extend(images) or True
        reviewer.consider('png-inputs', self.frame, self.detections, self.people)
        reviewer.consider('png-inputs', self.frame, self.detections, self.people)
        reviewer.tasks.join()
        self.assertEqual(len(received), 3)
        for payload, original in zip(received, expected):
            self.assertTrue(payload.startswith(b'\x89PNG\r\n\x1a\n'))
            decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(decoded, original))

    @patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '64',
                             'CAPTURE_MIN_SHARPNESS': '22'})
    def test_new_capture_wakes_gallery_listener(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer._classify = lambda *_: True
        session = 'b' * 32
        revisions = []
        listener = threading.Thread(target=lambda: revisions.append(
            reviewer.wait_for_revision(session, 0, timeout=3)))
        listener.start()
        time.sleep(.05)
        reviewer.consider(session, self.frame, self.detections, self.people)
        reviewer.consider(session, self.frame, self.detections, self.people)
        reviewer.tasks.join()
        listener.join(timeout=3)
        self.assertFalse(listener.is_alive())
        self.assertEqual(len(revisions), 1)
        self.assertGreaterEqual(revisions[0], 1)
        self.assertEqual(reviewer.snapshot(session)['revision'], 2)

    @patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '64',
                             'CAPTURE_MIN_SHARPNESS': '22'})
    def test_face_crop_keeps_source_pixels(self):
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        y, x = np.indices((80, 80))
        frame[40:120, 90:170] = ((x + y) % 2 * 100 + 30)[:, :, None]
        quality = _face_quality(frame, {'xyxy': [90, 40, 170, 120], 'confidence': .9})
        self.assertIsNotNone(quality)
        _, crop = quality
        self.assertTrue(np.array_equal(crop, frame[31:129, 81:179]))

    @patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '64',
                             'CAPTURE_MIN_SHARPNESS': '22'})
    def test_short_action_uses_clear_face_from_before(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        received = []
        reviewer._classify = lambda *images: received.append(images) or True
        face_only, phone_only = self.detections[:1], self.detections[1:]
        reviewer.consider('face-before', self.frame, face_only, self.people)
        reviewer.consider('face-before', self.frame, phone_only, self.people)
        self.assertEqual(received, [])
        reviewer.consider('face-before', self.frame, phone_only, self.people)
        reviewer.tasks.join()
        self.assertEqual(len(received), 1)
        expected = _face_quality(self.frame, face_only[0])[1]
        face = cv2.imdecode(np.frombuffer(received[0][1], np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.array_equal(face, expected))
        self.assertEqual(len(reviewer.snapshot('face-before')['captures']), 1)

    @patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '64',
                             'CAPTURE_MIN_SHARPNESS': '22'})
    def test_short_action_waits_for_clear_face_after(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer._classify = lambda *_: True
        reviewer._classify_face = lambda *_: {'face': True, 'frontal': True}
        face_only, phone_only = self.detections[:1], self.detections[1:]
        reviewer.consider('face-after', self.frame, phone_only, self.people)
        reviewer.consider('face-after', self.frame, phone_only, self.people)
        reviewer.tasks.join()
        self.assertEqual(len(reviewer.snapshot('face-after')['captures']), 0)
        reviewer.consider('face-after', self.frame, face_only, self.people)
        reviewer.tasks.join()
        self.assertEqual(len(reviewer.snapshot('face-after')['captures']), 1)

    def test_action_selects_highest_quality_face_from_buffer(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer.identity.feature = lambda *_: None
        received = []
        reviewer._classify = lambda *images: received.append(images) or True
        low = np.full((100, 100, 3), 35, dtype=np.uint8)
        high = np.full((100, 100, 3), 180, dtype=np.uint8)
        observations = iter([(10, low, False), (20, high, False),
                             (0, None, True), (0, None, True)])
        def observe(*_):
            score, crop, action = next(observations)
            return iter([{'center': (130, 80), 'width': 80,
                          'quality': (score, crop) if crop is not None else None,
                          'context': self.frame if action else None,
                          'phone_crop': self.frame[120:180, 120:180] if action else None}])
        with patch('src.phone_face.capture._frame_observations', side_effect=observe):
            for _ in range(4):
                reviewer.consider('best-face', self.frame, self.detections, self.people)
        reviewer.tasks.join()
        self.assertEqual(len(received), 1)
        decoded = cv2.imdecode(np.frombuffer(received[0][1], np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.array_equal(decoded, high))

    def test_video_timestamps_limit_face_window(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer.identity.feature = lambda *_: None
        reviewer._classify = lambda *_: True
        reviewer._classify_face = lambda *_: {'face': True, 'frontal': True}
        face = self.frame[31:129, 81:179]
        observations = iter([(face, False), (None, True), (None, True),
                             (face, False)])
        def observe(*_):
            crop, action = next(observations)
            return iter([{'center': (130, 80), 'width': 80,
                          'quality': (100, crop) if crop is not None else None,
                          'context': self.frame if action else None,
                          'phone_crop': face if action else None}])
        with patch('src.phone_face.capture._frame_observations', side_effect=observe):
            for video_time in (0, 2, 2.1):
                reviewer.consider('video-time', self.frame, self.detections,
                                  self.people, frame_time=video_time)
            reviewer.tasks.join()
            self.assertEqual(reviewer.snapshot('video-time')['captures'], [])
            reviewer.consider('video-time', self.frame, self.detections,
                              self.people, frame_time=2.2)
        reviewer.tasks.join()
        self.assertEqual(len(reviewer.snapshot('video-time')['captures']), 1)

    def test_confirmed_shooter_is_followed_until_frontal_face(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = reviewer.interval = 0
        reviewer.identity.feature = lambda *_: None
        x = np.arange(100, dtype=np.uint8)
        profile = np.repeat(x[None, :, None], 100, axis=0).repeat(3, axis=2)
        frontal = np.repeat(x[::-1][None, :, None], 100, axis=0).repeat(3, axis=2)
        calls = []
        followup_faces = []
        def classify(*images):
            calls.append(images)
            return {'shooting': True, 'face': True, 'frontal': False}
        reviewer._classify = classify
        reviewer._classify_face = lambda face: (followup_faces.append(face) or
                                                {'face': True, 'frontal': True})
        observations = iter([(profile, True), (profile, True), (None, False),
                             (frontal, False)])
        def observe(*_):
            face, action = next(observations)
            return iter([{'center': (130, 80), 'width': 80,
                          'quality': (100, face) if face is not None else None,
                          'context': self.frame if action else None,
                          'phone_crop': self.frame[120:180, 120:180] if action else None}])
        with patch('src.phone_face.capture._frame_observations', side_effect=observe):
            reviewer.consider('follow', self.frame, self.detections, self.people,
                              frame_time=0)
            reviewer.consider('follow', self.frame, self.detections, self.people,
                              frame_time=.1)
            reviewer.tasks.join()
            provisional = reviewer.snapshot('follow')['captures']
            self.assertEqual(len(provisional), 1)
            self.assertFalse(provisional[0]['frontal'])
            self.assertIn('继续等待正脸', reviewer.snapshot('follow')['status'])
            reviewer.consider('follow', self.frame, self.detections, self.people,
                              frame_time=14.9)
            reviewer.consider('follow', self.frame, self.detections, self.people,
                              frame_time=29.8)
        reviewer.tasks.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(followup_faces), 1)
        result = reviewer.snapshot('follow')
        self.assertEqual(len(result['captures']), 2)
        self.assertTrue(result['captures'][0]['frontal'])
        saved = reviewer.image('follow', result['captures'][0]['id'])
        decoded = cv2.imdecode(np.frombuffer(saved, np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.array_equal(decoded, frontal))

    def test_same_face_reentry_keeps_person_and_different_face_stays_separate(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer.interval = 0
        reviewer._classify = lambda *_: True
        reviewer._classify_face = lambda *_: {'face': True, 'frontal': True}
        features = iter([np.array([1., 0.], dtype=np.float32),
                         np.array([1., 0.], dtype=np.float32),
                         np.array([1., 0.], dtype=np.float32),
                         np.array([0., 1.], dtype=np.float32)])
        reviewer.identity.feature = lambda *_: next(features)
        face = self.frame[31:129, 81:179].copy()
        def observation(center, phone):
            return {'center': center, 'width': 80, 'quality': (100, face),
                    'context': self.frame if phone else None,
                    'phone_crop': face if phone else None}
        observations = iter([observation((130, 80), True),
                             observation((130, 80), True),
                             observation((250, 80), False),
                             observation((250, 80), False)])
        with patch('src.phone_face.capture._frame_observations',
                   side_effect=lambda *_: iter([next(observations)])):
            reviewer.consider('reentry', self.frame, self.detections, self.people)
            reviewer.consider('reentry', self.frame, self.detections, self.people)
            reviewer.tasks.join()
            first_id = reviewer.snapshot('reentry')['people'][0]['id']
            reviewer.sessions['reentry']['tracks'][first_id]['seen'] -= 20
            reviewer.consider('reentry', self.frame, self.detections, self.people)
            reviewer.tasks.join()
            self.assertEqual([person['id'] for person in reviewer.snapshot('reentry')['people']],
                             [first_id])
            reviewer.consider('reentry', self.frame, self.detections, self.people)
            reviewer.tasks.join()
        self.assertEqual(len(reviewer.snapshot('reentry')['people']), 1)
        self.assertEqual(len(reviewer.sessions['reentry']['tracks']), 2)

    @patch.dict(os.environ, {'CAPTURE_MIN_FACE_SIDE': '64',
                             'CAPTURE_MIN_SHARPNESS': '22'})
    def test_stable_track_reuses_recent_face_embedding_but_reentry_refreshes(self):
        reviewer = CaptureReviewer()
        reviewer.frame_interval = 0
        reviewer.reid_refresh_interval = .5
        feature = np.array([1., 0.], dtype=np.float32)
        with patch.object(reviewer.identity, 'feature', return_value=feature) as extract:
            for _ in range(3):
                reviewer.consider('embedding-cache', self.frame,
                                  self.detections[:1], self.people)
            self.assertEqual(extract.call_count, 2)
            track = next(iter(reviewer.sessions['embedding-cache']['tracks'].values()))
            track['seen'] -= 2
            reviewer.consider('embedding-cache', self.frame,
                              self.detections[:1], self.people)
            self.assertEqual(extract.call_count, 3)

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
            '"valid_face":true,"unoccluded":true,"clear":true,'
            '"frontal_face":true}'}}]}
        face_reply = {'choices': [{'message': {'content':
            '{"usable_face":true,"frontal_unoccluded":true}'}}]}
        class Response:
            def __init__(self, value): self.value = value
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, *_): return json.dumps(self.value).encode()
        with patch.dict(os.environ, {'QWEN_DISABLE_THINKING': '1'}), \
                patch('urllib.request.urlopen', side_effect=[Response(reply),
                      Response(face_reply)]) as urlopen:
            self.assertEqual(reviewer._classify(b'person', b'face', b'phone'),
                             {'shooting': True, 'face': True, 'frontal': True})
        self.assertEqual(urlopen.call_count, 2)
        payload = json.loads(urlopen.call_args_list[0].args[0].data)
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        for part in payload['messages'][0]['content'][1:]:
            self.assertTrue(part['image_url']['url'].startswith('data:image/png;base64,'))
        second = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(second['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(len(second['messages'][0]['content']), 2)

    def test_face_only_review_can_reject_occluded_face_without_losing_action(self):
        reviewer = CaptureReviewer()
        reply = {'choices': [{'message': {'content':
            '{"real_phone":true,"shooting":true,"visible_shooting_evidence":true,'
            '"valid_face":true,"unoccluded":true,"clear":true,'
            '"frontal_face":true}'}}]}
        face_reply = {'choices': [{'message': {'content':
            '{"usable_face":false,"frontal_unoccluded":false}'}}]}
        class Response:
            def __init__(self, value): self.value = value
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, *_): return json.dumps(self.value).encode()
        with patch('urllib.request.urlopen', side_effect=[Response(reply),
                    Response(face_reply)]):
            self.assertEqual(reviewer._classify(b'person', b'face', b'phone'),
                             {'shooting': True, 'face': False, 'frontal': False})

    def test_profile_is_kept_while_frontal_has_higher_rank(self):
        track = {'images': []}
        _store_ranked(track, b'profile', 200, 0xFF, frontal=False)
        _store_ranked(track, b'front', 100, 0xFF00, frontal=True)
        self.assertEqual([item['image'] for item in track['images']],
                         [b'front', b'profile'])

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
