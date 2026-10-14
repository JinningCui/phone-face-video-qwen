"""Stream a local video through the phone/face detector with optional pose association."""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpeg', '.mpg', '.wmv', '.ts', '.osv'}


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _smooth_detections(detections, previous, alpha):
    """Greedily match same-class boxes and apply an exponential moving average."""
    used = set()
    current = []
    for detection in detections:
        raw = detection['xyxy']
        best = None
        best_distance = float('inf')
        for index, (class_id, old_box) in enumerate(previous):
            if index in used or class_id != detection['class_id']:
                continue
            distance = math.hypot(_center(raw)[0] - _center(old_box)[0],
                                 _center(raw)[1] - _center(old_box)[1])
            limit = max(raw[2] - raw[0], raw[3] - raw[1], 40.0) * 2.0
            if distance < best_distance and distance <= limit:
                best, best_distance = index, distance
        if best is None:
            smoothed = list(map(float, raw))
        else:
            used.add(best)
            old_box = previous[best][1]
            smoothed = [alpha * float(new) + (1.0 - alpha) * float(old)
                        for new, old in zip(raw, old_box)]
        detection['xyxy'] = smoothed
        current.append((detection['class_id'], smoothed))
    return current


def _pose_people(pose_result, confidence):
    if pose_result is None or pose_result.boxes is None or pose_result.keypoints is None:
        return []
    boxes = pose_result.boxes.xyxy.tolist()
    points = pose_result.keypoints.xy.tolist()
    point_conf = pose_result.keypoints.conf.tolist() if pose_result.keypoints.conf is not None else None
    people = []
    for index, (box, keypoints) in enumerate(zip(boxes, points)):
        if float(pose_result.boxes.conf[index]) < confidence:
            continue
        visible = []
        for keypoint_index, point in enumerate(keypoints):
            score = 1.0 if point_conf is None else float(point_conf[index][keypoint_index])
            if score >= confidence:
                visible.append((keypoint_index, (float(point[0]), float(point[1]))))
        people.append({'box': list(map(float, box)), 'keypoints': visible})
    return people


def _associate_phone(phone, faces, people):
    """Return behavior state and the closest face/wrist anchor for visualization."""
    cx, cy = _center(phone['xyxy'])
    best_face = None
    best_face_distance = float('inf')
    for face in faces:
        fx, fy = _center(face['xyxy'])
        face_height = max(1.0, face['xyxy'][3] - face['xyxy'][1])
        distance = math.hypot(cx - fx, cy - fy) / face_height
        if distance < best_face_distance:
            best_face, best_face_distance = face, distance
    if best_face is not None and best_face_distance <= 1.6:
        phone['state'] = 'near_face'
        phone['anchor'] = list(_center(best_face['xyxy']))
        return

    best_wrist = None
    best_wrist_distance = float('inf')
    for person in people:
        x1, y1, x2, y2 = person['box']
        person_height = max(1.0, y2 - y1)
        # COCO pose keypoints: 9 left wrist, 10 right wrist.
        for keypoint_index, point in person['keypoints']:
            if keypoint_index not in (9, 10):
                continue
            distance = math.hypot(cx - point[0], cy - point[1]) / person_height
            if distance < best_wrist_distance:
                best_wrist, best_wrist_distance = point, distance
    if best_wrist is not None and best_wrist_distance <= 0.22:
        phone['state'] = 'in_hand'
        phone['anchor'] = list(best_wrist)
        return
    phone['state'] = 'detected'


def _draw_detection(image, detection, line_width):
    x1, y1, x2, y2 = [int(round(value)) for value in detection['xyxy']]
    class_name = detection['class_name']
    color = (0, 0, 255) if class_name == 'phone' else (0, 220, 0)
    label = f"{class_name} {detection['confidence']:.2f}"
    if class_name == 'phone' and detection.get('state') not in (None, 'detected'):
        label += f" {detection['state']}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_width)
    top = max(0, y1 - th - baseline - 4)
    cv2.rectangle(image, (x1, top), (min(image.shape[1] - 1, x1 + tw + 5), y1), color, -1)
    cv2.putText(image, label, (x1 + 2, max(th + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), line_width, cv2.LINE_AA)


def _encode_compatible(video_path):
    source_mp4v = video_path.with_name(video_path.stem + '.mp4v.mp4')
    video_path.replace(source_mp4v)
    try:
        subprocess.run([
            'ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-i', str(source_mp4v),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
            '-pix_fmt', 'yuv420p', '-an',
            '-movflags', '+faststart', str(video_path),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        source_mp4v.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        video_path.unlink(missing_ok=True)
        detail = exc.stderr.decode(errors='replace')[-1200:] if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        raise RuntimeError(f'H.264 encoding failed; intermediate video retained at {source_mp4v}: {detail}') from exc


def predict_video(args, model_factory=None):
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Video file not found: {source}')
    if source.suffix.lower() == '.osv':
        from .osv import predict_osv
        return predict_osv(args, model_factory=model_factory)
    if not args.model.is_file():
        raise FileNotFoundError(f'Model not found: {args.model}')
    thresholds = {0: args.phone_conf if args.phone_conf is not None else args.conf,
                  1: args.face_conf if args.face_conf is not None else args.conf}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (*thresholds.values(), args.iou)):
        raise ValueError('Confidence and IoU must be between 0 and 1')
    if args.imgsz <= 0:
        raise ValueError('imgsz must be positive')
    smooth_alpha = float(getattr(args, 'smooth_alpha', 0.65))
    line_width = max(1, int(getattr(args, 'line_width', 1)))
    pose_model_path = getattr(args, 'pose_model', None)
    pose_conf = float(getattr(args, 'pose_conf', 0.35))
    if not 0 < smooth_alpha <= 1 or not 0 <= pose_conf <= 1:
        raise ValueError('smooth-alpha must be in (0, 1] and pose-conf in [0, 1]')
    output = args.output.resolve() / args.name
    video_path = output / f'{source.stem}_detected.mp4'
    records_path = output / 'detections.jsonl'
    summary_path = output / 'summary.json'
    if not args.exist_ok and any(p.exists() for p in (video_path, records_path, summary_path)):
        raise FileExistsError(f'Output already exists: {output}; choose --name or use --exist-ok')
    cap = cv2.VideoCapture(str(source))
    writer = None
    started = time.perf_counter()
    frames = detected = 0
    totals = {'phone': 0, 'face': 0}
    state_totals = {'near_face': 0, 'in_hand': 0, 'detected': 0}
    previous_boxes = []
    pose_model = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {source}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid frame rate; convert it to a standard MP4 first')
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Video contains no decodable frames: {source}')
        height, width = frame.shape[:2]
        out_width, out_height = width + width % 2, height + height % 2
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        model = model_factory(str(args.model))
        names = dict(enumerate(model.names)) if isinstance(model.names, list) else model.names
        if names != {0: 'phone', 1: 'face'}:
            raise ValueError(f'Expected phone/face model classes, got {names}')
        if pose_model_path:
            if not Path(pose_model_path).is_file():
                raise FileNotFoundError(f'Pose model not found: {pose_model_path}')
            if model_factory is None:
                from ultralytics import YOLO
                model_factory = YOLO
            pose_model = model_factory(str(pose_model_path))
        output.mkdir(parents=True, exist_ok=True)
        summary_path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                 (out_width, out_height))
        if not writer.isOpened():
            raise OSError(f'Cannot create video: {video_path}')
        with records_path.open('w', encoding='utf-8') as records:
            while ok:
                if frame.shape[:2] != (height, width):
                    raise ValueError('Video resolution changed during decoding')
                result = model.predict(source=frame, conf=min(thresholds.values()),
                                       iou=args.iou, imgsz=args.imgsz, device=args.device,
                                       verbose=False)[0]
                detections = []
                for xyxy, confidence, class_id in zip(result.boxes.xyxy.tolist(),
                                                        result.boxes.conf.tolist(),
                                                        result.boxes.cls.tolist()):
                    class_id = int(class_id)
                    if class_id not in thresholds or confidence < thresholds[class_id]:
                        continue
                    detections.append({'class_id': class_id, 'class_name': names[class_id],
                                       'confidence': float(confidence), 'xyxy': list(map(float, xyxy))})
                faces = [d for d in detections if d['class_name'] == 'face']
                phones = [d for d in detections if d['class_name'] == 'phone']
                people = []
                if pose_model is not None:
                    pose_result = pose_model.predict(source=frame, conf=pose_conf,
                                                     imgsz=args.imgsz, device=args.device,
                                                     verbose=False)[0]
                    people = _pose_people(pose_result, pose_conf)
                for phone in phones:
                    _associate_phone(phone, faces, people)
                    state_totals[phone['state']] += 1
                previous_boxes = _smooth_detections(detections, previous_boxes, smooth_alpha)
                for detection in detections:
                    totals[detection['class_name']] += 1
                rendered = frame.copy()
                for detection in detections:
                    _draw_detection(rendered, detection, line_width)
                if (out_height, out_width) != (height, width):
                    rendered = cv2.copyMakeBorder(rendered, 0, out_height-height, 0,
                                                  out_width-width, cv2.BORDER_CONSTANT)
                writer.write(rendered)
                serializable = []
                for d in detections:
                    item = {k: v for k, v in d.items() if k != 'anchor'}
                    if d.get('anchor') is not None:
                        item['anchor'] = d['anchor']
                    serializable.append(item)
                records.write(json.dumps({'frame_index': frames, 'timestamp_seconds': frames / fps,
                                          'detections': serializable, 'pose_people': len(people)},
                                         ensure_ascii=False) + '\n')
                frames += 1
                detected += bool(detections)
                if frames % 100 == 0:
                    print(f'Processed {frames} frames ({frames / fps:.1f}s of video)', flush=True)
                ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise OSError('Video encoder produced no output')
    _encode_compatible(video_path)
    elapsed = time.perf_counter() - started
    summary = {'input': str(source), 'video': str(video_path), 'records': str(records_path),
               'model': str(args.model.resolve()), 'pose_model': str(Path(pose_model_path).resolve()) if pose_model_path else None,
               'frames': frames, 'fps': fps, 'width': out_width, 'height': out_height,
               'duration_seconds': frames / fps, 'frames_with_detections': detected,
               'detection_counts': totals, 'phone_state_counts': state_totals,
               'phone_conf': thresholds[0], 'face_conf': thresholds[1], 'pose_conf': pose_conf,
               'smooth_alpha': smooth_alpha, 'line_width': line_width,
               'elapsed_seconds': elapsed, 'processing_fps': frames / elapsed,
               'audio_preserved': False}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 0
