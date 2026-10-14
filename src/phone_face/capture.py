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

import cv2


MAX_CAPTURES_PER_TRACK = 5


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _face_phone_overlap(face_box, phone_box):
    x1, y1, x2, y2 = face_box
    px1, py1, px2, py2 = phone_box
    area = max(1, (x2-x1) * (y2-y1))
    overlap = max(0, min(x2, px2)-max(x1, px1)) * max(0, min(y2, py2)-max(y1, py1))
    return overlap / area


def _enhance_dark_face(crop, brightness):
    if brightness >= 95:
        return crop.copy()
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(crop, .25, enhanced, .75, 0)


def _enhance_face(crop, brightness, sharpness):
    """Improve accepted crops gently without inventing facial detail."""
    if brightness < 95:
        crop = cv2.fastNlMeansDenoisingColored(crop, None, 3, 3, 7, 21)
    crop = _enhance_dark_face(crop, brightness)
    if sharpness < 180:
        softened = cv2.GaussianBlur(crop, (0, 0), .8)
        crop = cv2.addWeighted(crop, 1.22, softened, -.22, 0)
    short_side = min(crop.shape[:2])
    target = max(0, int(os.environ.get('CAPTURE_DISPLAY_MIN_SIDE', '192')))
    if 0 < short_side < target:
        factor = min(2.0, target / short_side)
        crop = cv2.resize(crop, (round(crop.shape[1] * factor),
                                 round(crop.shape[0] * factor)),
                          interpolation=cv2.INTER_LANCZOS4)
    return crop


def _face_signature(crop):
    gray = cv2.cvtColor(cv2.resize(crop, (9, 8)), cv2.COLOR_BGR2GRAY)
    bits = (gray[:, 1:] > gray[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _store_ranked(track, image, score, signature):
    images = track['images']
    duplicate = next((item for item in images
                      if (item['signature'] ^ signature).bit_count() <= 5), None)
    if duplicate is not None:
        if score <= duplicate['score']:
            return
        duplicate.update(image=image, score=score, signature=signature,
                         version=duplicate['version'] + 1)
    else:
        images.append({'id': uuid.uuid4().hex[:12], 'score': score, 'image': image,
                       'signature': signature, 'version': 1})
    images.sort(key=lambda item: item['score'], reverse=True)
    del images[MAX_CAPTURES_PER_TRACK:]


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
    if brightness < 24 or brightness > 235 or contrast < 15:
        return None
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < float(os.environ.get('CAPTURE_MIN_SHARPNESS', '40')):
        return None
    crop = frame[y1-margin:y2+margin, x1-margin:x2+margin]
    crop = _enhance_face(crop, brightness, sharpness)
    score = (confidence * math.sqrt(min(sharpness, 700)) * side *
             min(brightness / 75, 1) * min(contrast / 35, 1))
    return score, crop


def _candidates(frame, detections, people):
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
        matching_phones = person_phones[index]
        if not matching_phones:
            continue
        px1, py1, px2, py2 = person['box']
        size = max(1, py2 - py1)
        head = next((point for key, point in person['keypoints'] if key == 0), None)
        best = None
        for face in faces:
            fx, fy = _center(face['xyxy'])
            face_key = tuple(round(v) for v in face['xyxy'])
            if face_key in used_faces or not (px1 <= fx <= px2 and py1 <= fy <= py1 + .63*size):
                continue
            quality = _face_quality(frame, face)
            if quality is None:
                continue
            score, face_crop = quality
            head_distance = math.hypot(fx-head[0], fy-head[1]) if head else abs(fx-(px1+px2)/2)
            if head and head_distance > .15*size:
                continue
            priority = (head_distance, -score)
            if best is None or priority < best[0]:
                best = (priority, score, face, face_crop, face_key)
        if best is None:
            continue
        _, score, face, face_crop, face_key = best
        used_faces.add(face_key)
        visible_phones = [phone for phone in matching_phones
                          if _face_phone_overlap(face['xyxy'], phone['xyxy']) < .02]
        if not visible_phones:
            continue
        phone = max(visible_phones, key=lambda d: d.get('confidence', 0))
        margin = .08 * size
        x1 = max(0, int(min(px1, phone['xyxy'][0]) - margin))
        y1 = max(0, int(min(py1, phone['xyxy'][1]) - margin))
        x2 = min(width, int(max(px2, phone['xyxy'][2]) + margin))
        y2 = min(height, int(max(py2, phone['xyxy'][3]) + margin))
        if x2 > x1 and y2 > y1:
            context = frame[y1:y2, x1:x2].copy()
            for box, color in ((face['xyxy'], (255, 255, 0)),
                               (phone['xyxy'], (0, 255, 255))):
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
            yield face['xyxy'], context, face_crop, phone_crop, score


class CaptureReviewer:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.tasks = queue.Queue(maxsize=16)
        self.url = os.environ.get('QWEN_VL_URL', 'http://127.0.0.1:8001/v1/chat/completions')
        self.model = os.environ.get('QWEN_VL_MODEL', 'Qwen3.5-4B')
        self.interval = max(.7, float(os.environ.get('QWEN_REVIEW_INTERVAL', '1.5')))
        workers = min(3, max(1, int(os.environ.get('QWEN_REVIEW_WORKERS', '2'))))
        for index in range(workers):
            threading.Thread(target=self._worker, daemon=True,
                             name=f'capture-reviewer-{index}').start()

    def consider(self, session_id, frame, detections, people):
        if not people:
            return
        now = time.monotonic()
        with self.lock:
            for old_id, old in list(self.sessions.items()):
                if now - old['last_frame'] > 3600:
                    del self.sessions[old_id]
            session = self.sessions.setdefault(session_id, {'tracks': {}, 'last_frame': 0,
                                                             'status': '等待持手机者'})
            if now - session['last_frame'] < .4:
                return
            session['last_frame'] = now
            for old_id, old in list(session['tracks'].items()):
                if not old['images'] and not old['pending'] and now - old['seen'] > 30:
                    del session['tracks'][old_id]
        assigned = set()
        for box, person_crop, face_crop, phone_crop, score in _candidates(frame, detections, people):
            fx, fy = _center(box)
            with self.lock:
                tracks = session['tracks']
                choices = [t for t in tracks.values() if t['id'] not in assigned and
                           now - t['seen'] < 12 and
                           math.hypot(fx-t['center'][0], fy-t['center'][1]) <
                           max(90, 1.2 * (box[2]-box[0]))]
                track = min(choices, key=lambda t: math.hypot(fx-t['center'][0],
                                                              fy-t['center'][1])) if choices else None
                if track is None:
                    track_id = uuid.uuid4().hex[:12]
                    track = {'id': track_id, 'center': (fx, fy), 'seen': now,
                             'last_review': 0, 'pending': False, 'images': []}
                    tracks[track_id] = track
                assigned.add(track['id'])
                track['center'], track['seen'] = (fx, fy), now
                if track['pending'] or now - track['last_review'] < self.interval:
                    continue
                if (len(track['images']) >= MAX_CAPTURES_PER_TRACK and
                        score <= track['images'][-1]['score']):
                    continue
                ok_person, person_jpeg = cv2.imencode('.jpg', person_crop,
                                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                ok_face, face_jpeg = cv2.imencode('.jpg', face_crop,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 96])
                ok_phone, phone_jpeg = cv2.imencode('.jpg', phone_crop,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not ok_person or not ok_face or not ok_phone:
                    continue
                try:
                    self.tasks.put_nowait((session_id, track['id'], person_jpeg.tobytes(),
                                           face_jpeg.tobytes(), phone_jpeg.tobytes(), score,
                                           _face_signature(face_crop)))
                except queue.Full:
                    session['status'] = 'Qwen 审核排队中'
                    continue
                track['pending'], track['last_review'] = True, now
                session['status'] = 'Qwen 正在核对拍摄动作和人脸质量'

    def _classify(self, person_jpeg, face_jpeg, phone_jpeg):
        def image_part(jpeg):
            return {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii')}}
        payload = {
            'model': self.model, 'temperature': 0, 'max_tokens': 160,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': (
                    '第一张图中青色框标出目标人物的人脸，黄色框标出关联手机；框只指示位置，'
                    '不是拍摄行为的证据。第二张图是该人的人脸特写，第三张图是黄色框放大图。严格判断：'
                    '1.第三张图必须能确认黄色框中是真实手机；若只是手指、衣服或无法确认，算否。'
                    '再核对手机是否由青色框人物自己手持；若属于旁边的人，算否。'
                    '若目标人物举起手机，且手机正面或背面的宽面'
                    '大致平行于画面、朝向拍摄场景，这就是对外拍照或录像的可见证据，算是；'
                    '不要求看见相机应用界面。低头刷手机、打电话、仅拿手机但没有对外取景姿态算否。'
                    '2.第二张是否确实是一张完整、可辨认、未被手机/手/物体遮挡且不明显模糊的人脸；'
                    '背影、局部五官、假人脸或不确定算否。'
                    '只返回一行JSON，格式为'
                    '{"real_phone":true/false,"shooting":true/false,'
                    '"visible_shooting_evidence":true/false,'
                    '"valid_face":true/false,'
                    '"unoccluded":true/false,"clear":true/false}。')},
                image_part(person_jpeg), image_part(face_jpeg), image_part(phone_jpeg),
            ]}],
        }
        if os.environ.get('QWEN_DISABLE_THINKING') == '1':
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        request = urllib.request.Request(self.url, json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=40) as response:
            data = json.load(response)
        content = data['choices'][0]['message'].get('content') or ''
        flags = {}
        for key in ('real_phone', 'shooting', 'visible_shooting_evidence', 'valid_face',
                    'unoccluded', 'clear'):
            match = re.search(r'"' + key + r'"\s*:\s*(true|false)', content, re.I)
            if match is None:
                raise ValueError('Qwen 未返回完整的人脸质量判断')
            flags[key] = match.group(1).lower() == 'true'
        return all(flags.values())

    def _worker(self):
        while True:
            session_id, track_id, person, face, phone, score, signature = self.tasks.get()
            error = None
            try:
                accepted = self._classify(person, face, phone)
            except Exception as exc:
                accepted, error = False, str(exc)[:180]
            with self.lock:
                session = self.sessions.get(session_id)
                track = session['tracks'].get(track_id) if session else None
                if track:
                    track['pending'] = False
                    if accepted:
                        _store_ranked(track, face, score, signature)
                    session['status'] = ('Qwen 服务错误：' + error if error else
                                         '已更新拍摄者 Top-5 截图' if accepted else
                                         '已过滤非拍摄、模糊或遮挡画面')
            self.tasks.task_done()

    def snapshot(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return {'status': '等待实时画面', 'people': [], 'captures': []}
            tracks = [t for t in session['tracks'].values() if t['images']]
            tracks.sort(key=lambda t: t['seen'], reverse=True)
            people = []
            captures = []
            for track in tracks[:16]:
                ranked = []
                for rank, item in enumerate(track['images'], 1):
                    record = {'id': item['id'], 'person_id': track['id'], 'rank': rank,
                              'version': item['version'], 'score': round(item['score'], 1)}
                    ranked.append(record)
                    captures.append(record)
                people.append({'id': track['id'], 'captures': ranked})
            return {'status': session['status'], 'people': people, 'captures': captures}

    def image(self, session_id, capture_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                for track in session['tracks'].values():
                    for item in track['images']:
                        if item['id'] == capture_id:
                            return item['image']
            return None
