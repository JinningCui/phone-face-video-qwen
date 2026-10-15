"""Review filming candidates and keep the five strongest face frames per track."""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import cv2

from .identity import FaceIdentity
from .export_spool import publish_gallery, retire_gallery


MAX_CAPTURES_PER_TRACK = 5


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _face_phone_overlap(face_box, phone_box):
    x1, y1, x2, y2 = face_box
    px1, py1, px2, py2 = phone_box
    area = max(1, (x2-x1) * (y2-y1))
    overlap = max(0, min(x2, px2)-max(x1, px1)) * max(0, min(y2, py2)-max(y1, py1))
    return overlap / area


def _face_signature(crop):
    gray = cv2.cvtColor(cv2.resize(crop, (9, 8)), cv2.COLOR_BGR2GRAY)
    bits = (gray[:, 1:] > gray[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _store_ranked(track, image, score, signature, frontal=False):
    images = track['images']
    duplicate = next((item for item in images
                      if (item['signature'] ^ signature).bit_count() <= 5), None)
    if duplicate is not None:
        if (frontal, score) <= (duplicate.get('frontal', False), duplicate['score']):
            return
        duplicate.update(image=image, score=score, signature=signature,
                         frontal=frontal,
                         version=duplicate['version'] + 1)
    else:
        images.append({'id': uuid.uuid4().hex[:12], 'score': score, 'image': image,
                       'signature': signature, 'frontal': frontal, 'version': 1})
    images.sort(key=lambda item: (item.get('frontal', False), item['score']), reverse=True)
    del images[MAX_CAPTURES_PER_TRACK:]


def _identity_similarity(track, feature):
    samples = track.get('identity_samples', ())
    return max((float(feature @ sample['feature']) for sample in samples), default=None)


def _remember_identity(track, feature, score):
    if feature is None:
        return
    samples = track.setdefault('identity_samples', [])
    for sample in samples:
        if float(feature @ sample['feature']) > .985:
            if score > sample['score']:
                sample.update(feature=feature, score=score)
            return
    samples.append({'feature': feature, 'score': score})
    samples.sort(key=lambda sample: sample['score'], reverse=True)
    del samples[5:]


def _track_similarity(left, right):
    return max((_identity_similarity(right, sample['feature']) or -1)
               for sample in left.get('identity_samples', ())) if left.get('identity_samples') else -1


def _merge_ranked_images(target, source):
    merged = []
    for image in sorted(target['images'] + source['images'],
                        key=lambda item: (item.get('frontal', False), item['score']),
                        reverse=True):
        if any((item['signature'] ^ image['signature']).bit_count() <= 5
               for item in merged):
            continue
        merged.append(image)
        if len(merged) == MAX_CAPTURES_PER_TRACK:
            break
    target['images'] = merged


def _face_quality(frame, face):
    box = face['xyxy']
    confidence = float(face.get('confidence', 1))
    if confidence < float(os.environ.get('CAPTURE_FACE_CONF', '.62')):
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    side = min(x2 - x1, y2 - y1)
    if side < int(os.environ.get('CAPTURE_MIN_FACE_SIDE', '96')) or not .65 <= (x2 - x1) / max(1, y2 - y1) <= 1.5:
        return None
    margin = max(5, int(side * .12))
    if x1 < margin or y1 < margin or x2 > width - margin or y2 > height - margin:
        return None
    core = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    min_brightness = float(os.environ.get('CAPTURE_MIN_FACE_BRIGHTNESS', '18'))
    if brightness < min_brightness or brightness > 235 or contrast < 15:
        return None
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < float(os.environ.get('CAPTURE_MIN_SHARPNESS', '40')):
        return None
    # Keep the source pixels for Qwen, the gallery and FTP exports.
    crop = frame[y1-margin:y2+margin, x1-margin:x2+margin].copy()
    score = (confidence * math.sqrt(min(sharpness, 700)) * side *
             min(brightness / 75, 1) * min(contrast / 35, 1))
    return score, crop


def _frame_observations(frame, detections, people):
    """Associate each pose with a face and a held phone independently.

    A clear face may occur before or after the brief phone action; requiring
    both in one frame would discard those useful frames.
    """
    phones = [d for d in detections if d['class_id'] == 0]
    faces = [d for d in detections if d['class_id'] == 1]
    height, width = frame.shape[:2]
    person_phones = {index: [] for index in range(len(people))}
    for phone in phones:
        cx, cy = _center(phone['xyxy'])
        choices = []
        for index, person in enumerate(people):
            px1, py1, px2, py2 = person['box']
            size = max(1, py2 - py1)
            wrists = [point for key, point in person['keypoints'] if key in (9, 10)]
            if not wrists or not (px1-.12*size <= cx <= px2+.12*size and
                                  py1-.1*size <= cy <= py2+.1*size):
                continue
            distance = min(math.hypot(cx-x, cy-y) for x, y in wrists) / size
            if distance <= .25:
                choices.append((distance, index))
        if choices:
            person_phones[min(choices)[1]].append(phone)
    used_faces = set()
    for index, person in enumerate(people):
        px1, py1, px2, py2 = person['box']
        size = max(1, py2 - py1)
        head = next((point for key, point in person['keypoints'] if key == 0), None)
        best = None
        for face in faces:
            fx, fy = _center(face['xyxy'])
            face_key = tuple(round(v) for v in face['xyxy'])
            if face_key in used_faces or not (px1 <= fx <= px2 and py1 <= fy <= py1 + .63*size):
                continue
            head_distance = math.hypot(fx-head[0], fy-head[1]) if head else abs(fx-(px1+px2)/2)
            if head and head_distance > .15*size:
                continue
            priority = (head_distance, -face.get('confidence', 0))
            if best is None or priority < best[0]:
                best = (priority, face, face_key)
        face = best[1] if best else None
        if best:
            used_faces.add(best[2])
        quality = _face_quality(frame, face) if face else None
        center = _center(face['xyxy']) if face else (head or ((px1+px2)/2, py1+.16*size))
        matching_phones = person_phones[index]
        phone = max(matching_phones, key=lambda d: d.get('confidence', 0)) if matching_phones else None
        context = phone_crop = None
        if phone is not None:
            margin = .08 * size
            x1 = max(0, int(min(px1, phone['xyxy'][0]) - margin))
            y1 = max(0, int(min(py1, phone['xyxy'][1]) - margin))
            x2 = min(width, int(max(px2, phone['xyxy'][2]) + margin))
            y2 = min(height, int(max(py2, phone['xyxy'][3]) + margin))
            if x2 <= x1 or y2 <= y1:
                phone = None
        if phone is not None:
            context = frame[y1:y2, x1:x2].copy()
            boxes = [(phone['xyxy'], (0, 255, 255))]
            if face is not None:
                boxes.insert(0, (face['xyxy'], (255, 255, 0)))
            for box, color in boxes:
                bx1, by1, bx2, by2 = (int(round(v)) for v in box)
                cv2.rectangle(context, (bx1-x1, by1-y1), (bx2-x1, by2-y1), color, 2)
            bx1, by1, bx2, by2 = phone['xyxy']
            pad = max(12, int(max(bx2-bx1, by2-by1) * .35))
            ax1, ay1 = max(0, int(bx1-pad)), max(0, int(by1-pad))
            ax2, ay2 = min(width, int(bx2+pad)), min(height, int(by2+pad))
            phone_crop = frame[ay1:ay2, ax1:ax2].copy()
            crop_h, crop_w = phone_crop.shape[:2]
            if max(crop_h, crop_w) < 256:
                factor = 256 / max(crop_h, crop_w)
                phone_crop = cv2.resize(phone_crop,
                                        (round(crop_w*factor), round(crop_h*factor)),
                                        interpolation=cv2.INTER_CUBIC)
        if face is not None or phone is not None:
            yield {'center': center, 'width': (face['xyxy'][2]-face['xyxy'][0]) if face else
                   max(40, (px2-px1)*.35), 'quality': quality,
                   'face_box': face['xyxy'] if face else None,
                   'context': context, 'phone_crop': phone_crop,
                   'phone_box': phone['xyxy'] if phone else None,
                   'phone_confidence': float(phone.get('confidence', 0)) if phone else 0}


def _candidates(frame, detections, people):
    """Legacy same-frame candidates retained for the throughput benchmark."""
    for observation in _frame_observations(frame, detections, people):
        if (observation['quality'] and observation['context'] is not None and
                _face_phone_overlap(observation['face_box'],
                                    observation['phone_box']) < .02):
            score, face_crop = observation['quality']
            yield (observation['face_box'], observation['context'], face_crop,
                   observation['phone_crop'], score)


class CaptureReviewer:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.identity = FaceIdentity()
        self.reid_threshold = float(os.environ.get('FACE_REID_THRESHOLD', '.42'))
        self.reid_merge_threshold = float(os.environ.get('FACE_REID_MERGE_THRESHOLD', '.62'))
        self.reid_refresh_interval = max(0., float(os.environ.get(
            'FACE_REID_REFRESH_INTERVAL', '.5')))
        self.url = os.environ.get('QWEN_VL_URL', 'http://127.0.0.1:8001/v1/chat/completions')
        self.urls = [url.strip() for url in os.environ.get('QWEN_VL_URLS', '').split(',')
                     if url.strip()] or [self.url]
        self.parallel = len(self.urls) > 1
        self.worker_local = threading.local()
        self.tasks = queue.Queue(maxsize=max(1, int(os.environ.get('QWEN_REVIEW_QUEUE_SIZE', '32'))))
        self.model = os.environ.get('QWEN_VL_MODEL', 'Qwen3.5-4B')
        export_dir = os.environ.get('PHONE_FACE_EXPORT_DIR', '').strip()
        self.export_root = Path(export_dir) if export_dir else None
        self.interval = max(.1, float(os.environ.get(
            'QWEN_REVIEW_INTERVAL', '.25')))
        self.frame_interval = max(0, float(os.environ.get(
            'QWEN_FRAME_INTERVAL', '0')))
        self.action_window = max(.1, float(os.environ.get('CAPTURE_ACTION_WINDOW', '.8')))
        self.face_window = max(.1, float(os.environ.get('CAPTURE_FACE_WINDOW', '1.5')))
        self.frontal_follow = max(1., float(os.environ.get('CAPTURE_FRONTAL_FOLLOW', '15')))
        self.follow_review_interval = max(self.interval, float(os.environ.get(
            'CAPTURE_FOLLOW_REVIEW_INTERVAL', '.75')))
        self.action_frames = max(2, int(os.environ.get('CAPTURE_ACTION_FRAMES', '2')))
        workers = (len(self.urls) if self.parallel else
                   min(3, max(1, int(os.environ.get('QWEN_REVIEW_WORKERS', '2')))))
        for index in range(workers):
            threading.Thread(target=self._worker, args=(self.urls[index % len(self.urls)],), daemon=True,
                             name=f'capture-reviewer-{index}').start()

    def _merge_tracks(self, session, target, source):
        if target['id'] == source['id']:
            return target
        before = [(image['id'], image['version']) for image in target['images']]
        had_source_images = bool(source['images'])
        _merge_ranked_images(target, source)
        target['pending'] += source['pending']
        target['last_review'] = max(target['last_review'], source['last_review'])
        target['last_feature_at'] = max(target.get('last_feature_at', 0),
                                        source.get('last_feature_at', 0))
        target['stable_matches'] = 0
        target['created'] = min(target.get('created', 0), source.get('created', 0))
        if source['seen'] > target['seen']:
            target['center'], target['seen'] = source['center'], source['seen']
        for sample in source.get('identity_samples', ()):
            _remember_identity(target, sample['feature'], sample['score'])
        target['face_samples'] = sorted(target['face_samples'] + source['face_samples'],
                                        key=lambda item: item['at'])[-24:]
        target['submitted_faces'] = (target['submitted_faces'] +
                                     source['submitted_faces'])[-30:]
        target['action_hits'] = sorted(target['action_hits'] + source['action_hits'])[-12:]
        if source['action_at'] > target['action_at']:
            target['action_at'] = source['action_at']
            target['action_context'] = source['action_context']
            target['action_phone'] = source['action_phone']
        target['action_until'] = max(target['action_until'], source['action_until'])
        target['confirmed_until'] = max(target['confirmed_until'], source['confirmed_until'])
        target['action_pending'] += source['action_pending']
        target['last_action_review'] = max(target['last_action_review'],
                                           source['last_action_review'])
        session['aliases'][source['id']] = target['id']
        del session['tracks'][source['id']]
        after = [(image['id'], image['version']) for image in target['images']]
        if had_source_images or after != before:
            session['revision'] += 1
            self.changed.notify_all()
            if self.export_root is not None and session.get('id'):
                try:
                    publish_gallery(self.export_root, session['id'], target['id'],
                                    target['images'])
                    retire_gallery(self.export_root, session['id'], source['id'])
                except Exception as exc:
                    session['status'] = f'FTP 待上传目录写入失败：{str(exc)[:120]}'
        return target

    def _match_track(self, session, assigned, feature, center, width, now):
        tracks = [track for track in session['tracks'].values()
                  if track['id'] not in assigned]
        similarities = []
        if feature is not None:
            for track in tracks:
                similarity = _identity_similarity(track, feature)
                if similarity is not None:
                    similarities.append((similarity, track))
            similarities.sort(key=lambda item: item[0], reverse=True)
            strong = [(similarity, track) for similarity, track in similarities
                      if similarity >= self.reid_threshold]
            if strong:
                if (len(strong) == 1 or strong[0][0] - strong[1][0] >= .05 or
                        _track_similarity(strong[0][1], strong[1][1]) >=
                        self.reid_merge_threshold):
                    # Retain the earliest gallery ID when two fragments are already
                    # clearly the same face.
                    close = [track for similarity, track in strong
                             if similarity >= strong[0][0] - .18 and
                             (track is strong[0][1] or
                              _track_similarity(track, strong[0][1]) >=
                              self.reid_merge_threshold)]
                    return min(close, key=lambda track: track.get('created', now))
        choices = []
        for track in tracks:
            if now - track['seen'] >= 12:
                continue
            distance = math.hypot(center[0] - track['center'][0],
                                  center[1] - track['center'][1])
            if distance >= max(90, 1.2 * width):
                continue
            similarity = _identity_similarity(track, feature) if feature is not None else None
            if similarity is not None and similarity < .25:
                continue
            choices.append((distance, track))
        return min(choices, key=lambda item: item[0])[1] if choices else None

    def _stable_spatial_track(self, session, assigned, center, width, now):
        """Find an unambiguous continuous track that needs no new face embedding."""
        distances = sorted(((math.hypot(center[0] - track['center'][0],
                                        center[1] - track['center'][1]), track)
                            for track in session['tracks'].values()
                            if track['id'] not in assigned and now - track['seen'] < 1),
                           key=lambda item: item[0])
        if not distances or distances[0][0] > max(35, .55 * width):
            return None
        if len(distances) > 1 and distances[1][0] <= max(90, 1.2 * width):
            return None
        return distances[0][1]

    def consider(self, session_id, frame, detections, people, frame_time=None):
        if not people:
            return
        now = time.monotonic()
        event_at = now if frame_time is None else float(frame_time)
        with self.lock:
            for old_id, old in list(self.sessions.items()):
                if now - old['last_frame'] > 3600:
                    del self.sessions[old_id]
            session = self.sessions.setdefault(session_id, {'id': session_id,
                                                             'tracks': {}, 'last_frame': 0,
                                                             'status': '等待持手机者',
                                                             'revision': 0,
                                                             'aliases': {},
                                                             'next_sequence': 0,
                                                             'next_commit': 0,
                                                             'completed': {}})
            if now - session['last_frame'] < self.frame_interval:
                return
            session['last_frame'] = now
            for old_id, old in list(session['tracks'].items()):
                if not old['images'] and not old['pending'] and now - old['seen'] > 30:
                    del session['tracks'][old_id]
        assigned = set()
        for observation in _frame_observations(frame, detections, people):
            quality = observation['quality']
            score, face_crop = quality if quality is not None else (0, None)
            with self.lock:
                hint = self._stable_spatial_track(
                    session, assigned, observation['center'], observation['width'], now)
                need_feature = (face_crop is not None and
                                (hint is None or hint.get('stable_matches', 0) < 1 or
                                 now - hint.get('last_feature_at', -1e9)
                                 >= self.reid_refresh_interval))
            feature = self.identity.feature(face_crop) if need_feature else None
            task = None
            with self.lock:
                tracks = session['tracks']
                center = observation['center']
                width = observation['width']
                track = self._match_track(session, assigned, feature, center, width, now)
                if track is None:
                    track_id = uuid.uuid4().hex[:12]
                    track = {'id': track_id, 'center': center, 'seen': now,
                             'created': now, 'last_review': 0, 'pending': 0,
                             'images': [], 'identity_samples': [],
                             'last_feature_at': -1e9,
                             'stable_matches': 0,
                             'face_samples': [], 'submitted_faces': [],
                             'action_hits': [], 'action_until': 0,
                             'confirmed_until': -1,
                             'action_pending': 0, 'last_action_review': 0,
                             'action_at': 0, 'action_context': None,
                             'action_phone': None}
                    tracks[track_id] = track
                track['stable_matches'] = (track.get('stable_matches', 0) + 1
                                           if hint is track else 0)
                track['center'], track['seen'] = center, now
                if need_feature:
                    track['last_feature_at'] = now
                if track['confirmed_until'] >= event_at:
                    track['confirmed_until'] = event_at + self.frontal_follow
                _remember_identity(track, feature, score)
                if feature is not None:
                    for other in list(tracks.values()):
                        if other is track or other['id'] in assigned:
                            continue
                        similarity = _track_similarity(track, other)
                        recent_fragment = (not other.get('identity_samples') and
                                           now - other.get('created', now) < 5 and
                                           math.hypot(center[0]-other['center'][0],
                                                      center[1]-other['center'][1]) <
                                           max(90, 1.2 * width) and
                                           (not other['images'] or
                                            (face_crop is not None and
                                             min((_face_signature(face_crop) ^ image['signature']).bit_count()
                                                 for image in other['images']) <= 10)))
                        if similarity < self.reid_merge_threshold and not recent_fragment:
                            continue
                        if (now - other['seen'] < 1 and
                                math.hypot(center[0]-other['center'][0],
                                           center[1]-other['center'][1]) >
                                max(90, 1.2 * width)):
                            continue
                        older, newer = sorted((track, other),
                                              key=lambda item: item.get('created', now))
                        track = self._merge_tracks(session, older, newer)
                assigned.add(track['id'])
                track['face_samples'] = [sample for sample in track['face_samples']
                                         if event_at - sample['at'] <= self.face_window]
                if face_crop is not None:
                    track['face_samples'].append({'at': event_at, 'score': score,
                                                  'crop': face_crop,
                                                  'signature': _face_signature(face_crop)})
                    track['face_samples'] = track['face_samples'][-24:]
                track['action_hits'] = [at for at in track['action_hits']
                                        if event_at - at <= self.action_window]
                if observation['context'] is not None:
                    track['action_hits'].append(event_at)
                    if len(track['action_hits']) >= self.action_frames:
                        track['action_until'] = event_at + self.face_window
                        if track['confirmed_until'] >= event_at:
                            track['confirmed_until'] = event_at + self.frontal_follow
                        track['action_at'] = event_at
                        track['action_context'] = observation['context']
                        track['action_phone'] = observation['phone_crop']
                if event_at > max(track['action_until'], track['confirmed_until']):
                    track['action_context'] = track['action_phone'] = None
                    continue
                if track['action_context'] is None:
                    continue
                review_interval = (self.follow_review_interval
                                   if track['confirmed_until'] >= event_at and track['images']
                                   else self.interval)
                if ((not self.parallel and track['pending']) or
                        (now - track['last_review'] < review_interval and
                         not (track['confirmed_until'] >= event_at and
                              not any(item.get('frontal', False)
                                      for item in track['images'])))):
                    continue
                eligible = [sample for sample in track['face_samples']
                            if not any((sample['signature'] ^ done['signature']).bit_count() <= 5
                                       and sample['score'] <= done['score'] * 1.1
                                       for done in track['submitted_faces'])]
                chosen = max(eligible, key=lambda sample: sample['score']) if eligible else None
                if chosen is not None:
                    score, face_crop, signature = (chosen['score'], chosen['crop'],
                                                   chosen['signature'])
                    if (len(track['images']) >= MAX_CAPTURES_PER_TRACK and
                            all(item.get('frontal', False) for item in track['images']) and
                            score <= track['images'][-1]['score']):
                        continue
                elif (event_at <= track['action_until'] and
                      event_at > track['confirmed_until'] and
                      not track['action_pending'] and
                      now - track['last_action_review'] >= self.action_window):
                    score, face_crop, signature = 0, None, None
                else:
                    continue
                followup_only = chosen is not None and track['confirmed_until'] >= event_at
                if followup_only:
                    ok_person, person_png = True, None
                    ok_phone, phone_png = True, None
                else:
                    ok_person, person_png = cv2.imencode(
                        '.png', track['action_context'], [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    ok_phone, phone_png = cv2.imencode(
                        '.png', track['action_phone'], [cv2.IMWRITE_PNG_COMPRESSION, 3])
                ok_face, face_png = (cv2.imencode('.png', face_crop,
                                                 [cv2.IMWRITE_PNG_COMPRESSION, 3])
                                     if face_crop is not None else (True, None))
                if not ok_person or not ok_face or not ok_phone:
                    continue
                sequence = session['next_sequence']
                session['next_sequence'] += 1
                task = (session_id, sequence, track['id'],
                        person_png.tobytes() if person_png is not None else None,
                        face_png.tobytes() if face_png is not None else None,
                        phone_png.tobytes() if phone_png is not None else None,
                        score, signature, event_at, followup_only)
                track['pending'] += 1
                track['last_review'] = now
                if chosen is None:
                    track['action_pending'] += 1
                    track['last_action_review'] = now
                    session['status'] = 'Qwen 正在核对短暂拍摄动作'
                else:
                    track['submitted_faces'].append({'signature': signature, 'score': score})
                    track['submitted_faces'] = track['submitted_faces'][-30:]
                    session['status'] = ('Qwen 正在核对人脸质量' if followup_only else
                                         'Qwen 正在核对拍摄动作和正脸质量')
            # Backpressure rather than silently discarding a selected frame.
            # This must be outside self.lock so workers can finish while the queue is full.
            self.tasks.put(task)

    def _classify_face(self, face_png):
        """Review a new face after its owner's shooting action is confirmed."""
        face_payload = {
            'model': self.model, 'temperature': 0, 'max_tokens': 80,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': (
                    '只判断这张脸部特写。usable_face 表示真人脸清楚可辨认，'
                    '允许侧脸，但明显模糊或被手、手机遮住大部分五官算 false。'
                    'frontal_unoccluded 表示接近正对镜头、双眼鼻子嘴巴完整可见；'
                    '任何手、手机或物体挡住嘴巴、鼻子、眼睛都必须是 false。'
                    '只返回一行 JSON：'
                    '{"usable_face":true/false,"frontal_unoccluded":true/false}。')},
                {'type': 'image_url', 'image_url': {'url':
                    'data:image/png;base64,' + base64.b64encode(face_png).decode('ascii')}}
            ]}],
        }
        if os.environ.get('QWEN_DISABLE_THINKING') == '1':
            face_payload['chat_template_kwargs'] = {'enable_thinking': False}
        url = getattr(self.worker_local, 'url', self.url)
        face_request = urllib.request.Request(
            url, json.dumps(face_payload).encode(),
            {'Content-Type': 'application/json'})
        with urllib.request.urlopen(face_request, timeout=40) as response:
            face_content = json.load(response)['choices'][0]['message'].get('content') or ''
        usable = re.search(r'"usable_face"\s*:\s*(true|false)', face_content, re.I)
        frontal = re.search(r'"frontal_unoccluded"\s*:\s*(true|false)', face_content, re.I)
        face_ok = usable is not None and usable.group(1).lower() == 'true'
        return {'face': face_ok,
                'frontal': (face_ok and frontal is not None and
                            frontal.group(1).lower() == 'true')}

    def _classify(self, person_png, face_png, phone_png):
        def image_part(image):
            return {'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(image).decode('ascii')}}
        if face_png is None:
            image_layout = ('第一张图是目标人物举手机时的画面，黄色框标出关联手机；'
                            '第二张图是手机局部放大图。本次没有提供人脸特写，'
                            '只判断该人物是否正在对外拍照或录像，人脸相关字段全部填 false。')
            images = [image_part(person_png), image_part(phone_png)]
        else:
            image_layout = ('第一张图是目标人物举手机动作时的画面，青色框如存在则标出人脸，'
                            '黄色框标出关联手机；第二张图是同一跟踪对象在动作前后选出的'
                            '人脸，可能与动作相隔数秒；第三张图是手机局部放大图。'
                            '第二张不要求出现举手机动作。')
            images = [image_part(person_png), image_part(face_png), image_part(phone_png)]
        payload = {
            'model': self.model, 'temperature': 0, 'max_tokens': 160,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': (
                    image_layout + '框只指示位置，不是拍摄行为的证据。严格判断：'
                    '1.手机局部图必须能确认是真实手机；若只是手指、衣服或无法确认，算否。'
                    '再核对手机是否由青色框人物自己手持；若属于旁边的人，算否。'
                    '若目标人物举起手机，且手机正面或背面的宽面'
                    '大致平行于画面、朝向拍摄场景，这就是对外拍照或录像的可见证据，算是；'
                    '不要求看见相机应用界面。低头刷手机、打电话、仅拿手机但没有对外取景姿态算否。'
                    '2.若提供人脸特写，valid_face 表示可辨认的真人脸，侧脸也可以；'
                    'clear 表示不明显模糊；unoccluded 表示主要五官未被遮挡；'
                    'frontal_face 单独表示正脸或接近正脸、双眼鼻子嘴部清楚可见。'
                    '背影、假人脸或不确定则 valid_face 为 false。'
                    '若无特写，人脸字段均为 false。'
                    '只返回一行JSON，格式为'
                    '{"real_phone":true/false,"shooting":true/false,'
                    '"visible_shooting_evidence":true/false,'
                    '"valid_face":true/false,'
                    '"unoccluded":true/false,"clear":true/false,'
                    '"frontal_face":true/false}。')},
                *images,
            ]}],
        }
        if os.environ.get('QWEN_DISABLE_THINKING') == '1':
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        url = getattr(self.worker_local, 'url', self.url)
        request = urllib.request.Request(url, json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=40) as response:
            data = json.load(response)
        content = data['choices'][0]['message'].get('content') or ''
        flags = {}
        for key in ('real_phone', 'shooting', 'visible_shooting_evidence', 'valid_face',
                    'unoccluded', 'clear', 'frontal_face'):
            match = re.search(r'"' + key + r'"\s*:\s*(true|false)', content, re.I)
            if match is None:
                raise ValueError('Qwen 未返回完整的人脸质量判断')
            flags[key] = match.group(1).lower() == 'true'
        face_ok = face_png is not None and all(flags[key] for key in
                            ('valid_face', 'clear'))
        frontal_ok = False
        if face_ok:
            # A separate single-image question prevents the action image from
            # persuading Qwen to accept a face whose mouth is covered.
            try:
                reviewed_face = self._classify_face(face_png)
                face_ok = reviewed_face['face']
                frontal_ok = reviewed_face['frontal'] and flags['frontal_face']
            except Exception:
                face_ok = False
                frontal_ok = False
        return {'shooting': all(flags[key] for key in
                                ('real_phone', 'shooting', 'visible_shooting_evidence')),
                'face': face_ok, 'frontal': frontal_ok}

    def _worker(self, url):
        self.worker_local.url = url
        while True:
            session_id, sequence, track_id, person, face, phone, score, signature, event_at, followup_only = self.tasks.get()
            error = None
            try:
                verdict = ({'shooting': True, **self._classify_face(face)}
                           if followup_only else self._classify(person, face, phone))
                if isinstance(verdict, bool):
                    verdict = {'shooting': verdict, 'face': verdict, 'frontal': verdict}
                verdict.setdefault('frontal', verdict['face'])
                if face is not None and verdict['frontal']:
                    landmark_frontal = self.identity.is_frontal_png(face)
                    if landmark_frontal is False:
                        verdict['frontal'] = False
            except Exception as exc:
                verdict, error = {'shooting': False, 'face': False,
                                  'frontal': False}, str(exc)[:180]
            with self.lock:
                session = self.sessions.get(session_id)
                if session:
                    session['completed'][sequence] = (track_id, face, score, signature,
                                                      event_at, verdict, error)
                    while session['next_commit'] in session['completed']:
                        item = session['completed'].pop(session['next_commit'])
                        session['next_commit'] += 1
                        ready_track_id, ready_face, ready_score, ready_signature, ready_at, ready_verdict, ready_error = item
                        aliases = session.get('aliases', {})
                        while ready_track_id in aliases:
                            ready_track_id = aliases[ready_track_id]
                        track = session['tracks'].get(ready_track_id)
                        if track:
                            track['pending'] -= 1
                            if ready_face is None:
                                track['action_pending'] = max(0, track.get('action_pending', 0) - 1)
                            was_confirmed = track.get('confirmed_until', -1) >= ready_at
                            if ready_verdict['shooting']:
                                track['confirmed_until'] = max(
                                    track.get('confirmed_until', -1),
                                    ready_at + self.frontal_follow)
                                if not was_confirmed:
                                    session['revision'] = session.get('revision', 0) + 1
                                    self.changed.notify_all()
                            ready_accepted = (ready_face is not None and
                                              ready_verdict['face'] and
                                              (ready_verdict['shooting'] or was_confirmed))
                            if ready_accepted:
                                before = [(image['id'], image['version'])
                                          for image in track['images']]
                                _store_ranked(track, ready_face, ready_score,
                                              ready_signature, ready_verdict['frontal'])
                                after = [(image['id'], image['version'])
                                         for image in track['images']]
                                if after != before:
                                    session['revision'] += 1
                                    self.changed.notify_all()
                                    if self.export_root is not None:
                                        try:
                                            publish_gallery(self.export_root, session_id,
                                                            track['id'], track['images'])
                                        except Exception as exc:
                                            ready_error = f'FTP 待上传目录写入失败：{str(exc)[:120]}'
                            session['status'] = ('Qwen 服务错误：' + ready_error if ready_error else
                                                 '已更新拍摄者正脸 Top-5 截图' if ready_accepted and ready_verdict['frontal'] else
                                                 '已截取拍摄者人脸，继续等待正脸' if ready_accepted else
                                                 '已锁定拍摄者，持续等待清晰正脸' if ready_verdict['shooting']
                                                 or was_confirmed else
                                                 '已过滤非拍摄或无效画面')
            self.tasks.task_done()

    def snapshot(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return {'status': '等待实时画面', 'revision': 0,
                        'people': [], 'captures': []}
            tracks = [t for t in session['tracks'].values() if t['images']]
            tracks.sort(key=lambda t: t['seen'], reverse=True)
            people = []
            captures = []
            for track in tracks[:16]:
                ranked = []
                for rank, item in enumerate(track['images'], 1):
                    record = {'id': item['id'], 'person_id': track['id'], 'rank': rank,
                              'version': item['version'], 'score': round(item['score'], 1),
                              'frontal': item.get('frontal', False)}
                    ranked.append(record)
                    captures.append(record)
                people.append({'id': track['id'], 'captures': ranked})
            return {'status': session['status'], 'revision': session['revision'],
                    'people': people, 'captures': captures}

    def wait_for_revision(self, session_id, after, timeout=25):
        with self.changed:
            self.changed.wait_for(
                lambda: self.sessions.get(session_id, {}).get('revision', 0) > after,
                timeout=timeout)
            return self.sessions.get(session_id, {}).get('revision', 0)

    def image(self, session_id, capture_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                for track in session['tracks'].values():
                    for item in track['images']:
                        if item['id'] == capture_id:
                            return item['image']
            return None
