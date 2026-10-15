"""Small browser UI for the existing image and video inference commands."""
from __future__ import annotations

import json
import html
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .run import IMAGE_SUFFIXES
from .video import VIDEO_SUFFIXES
from .capture import CaptureReviewer, MAX_CAPTURES_PER_TRACK


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("WEB_OUTPUT_DIR", ROOT / "outputs")).resolve()
UPLOAD_ROOT = OUTPUT_ROOT / "web_uploads"
JOB_ROOT = OUTPUT_ROOT / "web_jobs"
DEFAULT_VIDEO = ROOT / "video_test" / "test.mp4"
LOWLIGHT_VIDEO = ROOT / "video_test" / "20261003_C8920.MP4"
LOWLIGHT_ANNOTATED = OUTPUT_ROOT / "phone_face_video" / "lowlight_20261003" / "20261003_C8920_detected.mp4"
DEMO_VIDEOS = {"test": (DEFAULT_VIDEO, "test.mp4", "24"),
               "lowlight": (LOWLIGHT_VIDEO, "20261003_C8920.MP4 · 低光", "31")}
LIVE_VIDEO_SUFFIXES = VIDEO_SUFFIXES - {".osv"}
MODEL = ROOT / "weights" / "phone_face" / "best.pt"
POSE_MODEL = ROOT / "weights" / "yolov8n-pose.pt"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_GB", "4")) * 1024**3
SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
IDENTIFIER = re.compile(r"^[0-9a-f]{32}$")
UPLOADS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
LIVE_LOCK = threading.Lock()
DEMO_SLOTS = threading.BoundedSemaphore(3)
BROWSER_FRAME_LOCK = threading.Lock()
BROWSER_DETECTOR = None
CAPTURE_REVIEWER = CaptureReviewer()
LIVE_SESSION = re.compile(r"^[0-9a-f]{32}$")
TRACK_ID = re.compile(r"^[0-9a-f]{12}$")


def _uploaded_video(upload_id: str) -> Path | None:
    if not IDENTIFIER.fullmatch(upload_id):
        return None
    upload_dir = UPLOAD_ROOT / upload_id
    if not upload_dir.is_dir():
        return None
    files = [path for path in upload_dir.iterdir()
             if path.is_file() and path.suffix.lower() in LIVE_VIDEO_SUFFIXES]
    return files[0] if len(files) == 1 else None


def _file_info(path: Path, job_id: str) -> dict:
    return {
        "name": path.name,
        "kind": "video" if path.suffix.lower() in VIDEO_SUFFIXES else
                "image" if path.suffix.lower() in IMAGE_SUFFIXES else "data",
        "size": path.stat().st_size,
        "url": f"/api/jobs/{job_id}/files/{path.name}",
    }


def _run_job(job_id: str, items: list[dict]) -> None:
    job = JOBS[job_id]
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(items, 1):
            source = Path(item["path"])
            kind = "video" if source.suffix.lower() in VIDEO_SUFFIXES else "image"
            name = f"{index:03d}_{uuid.uuid4().hex[:8]}"
            with LOCK:
                job["current"] = f"{index}/{len(items)} · {item['name']}"
            command = [
                sys.executable, "-m", "src.phone_face.run",
                "video" if kind == "video" else "predict", str(source),
                "--model", str(MODEL), "--output", str(job_dir), "--name", name,
                "--device", os.environ.get("DEVICE", "cpu"),
                "--imgsz", os.environ.get("IMG_SIZE", "960"),
                "--phone-conf", os.environ.get("PHONE_CONF", "0.35"),
                "--face-conf", os.environ.get("FACE_CONF", "0.35"),
            ]
            if kind == "video":
                command.extend([
                    "--pose-model", str(POSE_MODEL),
                    "--pose-conf", os.environ.get("POSE_CONF", "0.35"),
                    "--smooth-alpha", os.environ.get("SMOOTH_ALPHA", "0.55"),
                    "--line-width", os.environ.get("LINE_WIDTH", "1"),
                ])
            log_path = job_dir / f"{name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                with LOCK:
                    job["errors"].append({"name": item["name"],
                                          "message": log_path.read_text(errors="replace")[-3000:]})
                continue
            result_dir = job_dir / name
            files = sorted(p for p in result_dir.iterdir() if p.is_file())
            with LOCK:
                job["results"].append({
                    "source": item["name"], "kind": kind,
                    "files": [_file_info(p, job_id) | {"url":
                        f"/api/jobs/{job_id}/files/{name}/{p.name}"} for p in files],
                })
        with LOCK:
            job["status"] = "done" if not job["errors"] else "completed_with_errors"
            job["current"] = ""
    except Exception as exc:
        with LOCK:
            job["status"] = "failed"
            job["errors"].append({"name": "job", "message": str(exc)})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def do_GET(self) -> None:
        request_url = urlsplit(self.path)
        path = request_url.path
        if path == "/dashboard":
            self._send(200, (Path(__file__).with_name("web_index.html")).read_bytes(),
                       "text/html; charset=utf-8")
            return
        if path in ("/", "/demo", "/demo.html"):
            params = parse_qs(request_url.query)
            upload_id = params.get("upload", [""])[0]
            if upload_id:
                uploaded = _uploaded_video(upload_id)
                if uploaded is None:
                    self._error(404, "上传视频不存在或格式不支持实时播放")
                    return
                title, duration, source_query = uploaded.name, "上传后逐帧识别", f"upload={upload_id}"
                video_key = ""
            else:
                video_key = params.get("video", ["test"])[0]
                if video_key not in DEMO_VIDEOS:
                    self._error(400, "未知演示视频")
                    return
                video_path, title, seconds = DEMO_VIDEOS[video_key]
                if not video_path.is_file():
                    self._error(404, "演示视频不存在，请先上传视频")
                    return
                duration, source_query = f"{seconds} 秒视频 · 逐帧处理", f"video={video_key}"
            session_id = uuid.uuid4().hex
            body = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
            body = (body.replace("__SESSION_ID__", session_id)
                    .replace("__SOURCE_QUERY__", source_query)
                    .replace("__VIDEO_TITLE__", html.escape(title))
                    .replace("__VIDEO_DURATION__", duration)
                    .replace("__RESULT_LINK__", ' · <a href="/api/demo/annotated?video=lowlight">下载完整打框 MP4</a>'
                             if video_key == "lowlight" and LOWLIGHT_ANNOTATED.is_file() else ""))
            self._send(200, body.encode(),
                       "text/html; charset=utf-8")
            return
        if path == "/api/demo/annotated":
            if (parse_qs(request_url.query).get("video", [""])[0] == "lowlight"
                    and LOWLIGHT_ANNOTATED.is_file()):
                self._serve_file(LOWLIGHT_ANNOTATED)
            else:
                self._error(404, "打框视频不存在")
            return
        if path == "/api/demo/gallery":
            session_id = parse_qs(request_url.query).get("session", [""])[0]
            if not LIVE_SESSION.fullmatch(session_id):
                self._error(400, "无效演示会话")
                return
            state = CAPTURE_REVIEWER.snapshot(session_id)
            groups = []
            for person_index, person in enumerate(state['people'], 1):
                cards = []
                for capture in person['captures']:
                    image_url = (f"/api/live/captures/{session_id}/{capture['id']}.png"
                                 f"?v={capture['version']}")
                    cards.append(f'<a class="card" href="{image_url}" download="shooter_{capture["id"]}.png">'
                                 f'<img src="{image_url}" alt="对象 {person_index} 的人脸截图">'
                                 f'<small>Top {capture["rank"]} · '
                                 f'{"正脸" if capture["frontal"] else "侧脸／待正脸"} · '
                                 f'质量分 {capture["score"]}</small></a>')
                groups.append(f'<section class="person"><h3>跟踪对象 {person_index} · {len(cards)}/{MAX_CAPTURES_PER_TRACK} 张</h3>'
                              '<div class="cards">' + ''.join(cards) + '</div></section>')
            body = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                    '<style>*{box-sizing:border-box}body{margin:0;background:#111e2c;color:#dcecf4;'
                    'font:14px system-ui,sans-serif}.status{color:#a9c2d0;margin:0 0 14px}'
                    '.person{margin-bottom:22px}.person h3{font-size:15px;margin:0 0 10px}'
                    '.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}'
                    '.card{display:block;padding:8px;border:1px solid #344d60;background:#182b3a;'
                    'border-radius:12px;color:inherit;text-decoration:none}.card img{display:block;'
                    'width:100%;height:200px;object-fit:scale-down;background:#071018;border-radius:7px}'
                    '.card small{display:block;margin-top:8px}.empty{border:1px dashed #40586c;'
                    'padding:28px 12px;border-radius:12px;text-align:center;color:#a7bdcb}'
                    '</style></head><body><p class="status">' + html.escape(state['status']) +
                    f' · 跟踪对象 {len(groups)} 位 · 截图 {len(state["captures"])} 张</p>' +
                    (''.join(groups) if groups else
                     '<div class="empty">等待识别到拍摄者的人脸；后续会继续寻找更清晰的正脸。</div>') +
                    '<script>'
                    f'let revision={state["revision"]};'
                    f'const changes="/api/demo/changes?session={session_id}&after=";'
                    'async function watch(){for(;;){try{'
                    'const response=await fetch(changes+revision,{cache:"no-store"});'
                    'if(!response.ok)throw Error("changes unavailable");'
                    'const data=await response.json();'
                    'if(data.revision>revision){location.reload();return;}'
                    'revision=data.revision;'
                    '}catch(error){await new Promise(resolve=>setTimeout(resolve,1000));}'
                    '}}watch();'
                    '</script></body></html>')
            self._send(200, body.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/demo/changes":
            params = parse_qs(request_url.query)
            session_id = params.get("session", [""])[0]
            if not LIVE_SESSION.fullmatch(session_id):
                self._error(400, "无效演示会话")
                return
            try:
                after = max(0, int(params.get("after", ["0"])[0]))
            except ValueError:
                self._error(400, "无效截图版本")
                return
            revision = CAPTURE_REVIEWER.wait_for_revision(session_id, after)
            self._json(200, {"revision": revision})
            return
        if path == "/api/health":
            camera_source = os.environ.get("CAMERA_SOURCE", "/dev/video0")
            self._json(200, {"ok": True, "default_video": DEFAULT_VIDEO.exists(),
                             "model": MODEL.exists(), "pose_model": POSE_MODEL.exists(),
                             "device": os.environ.get("DEVICE", "cpu"),
                             "camera_source": camera_source,
                             "network_source": os.environ.get("NETWORK_SOURCE", ""),
                             "camera_present": (Path(camera_source).exists()
                                                if camera_source.startswith("/dev/") else True)})
            return
        if path == "/api/live/stream":
            override = parse_qs(request_url.query, keep_blank_values=True).get("source", [None])[0]
            if override is not None:
                parsed_source = urlsplit(override)
                if (len(override) > 1024 or parsed_source.scheme not in ("rtsp", "rtsps", "rtmp", "rtmps")
                        or not parsed_source.hostname):
                    self._error(400, "请输入有效的 RTSP 或 RTMP 视频流地址")
                    return
            session_id = parse_qs(request_url.query).get("session", [""])[0]
            self._live_stream(override, session_id if LIVE_SESSION.fullmatch(session_id) else None)
            return
        if path == "/api/demo/stream":
            params = parse_qs(request_url.query)
            session_id = params.get("session", [""])[0]
            if not LIVE_SESSION.fullmatch(session_id):
                self._error(400, "无效演示会话")
                return
            upload_id = params.get("upload", [""])[0]
            if upload_id:
                video_path = _uploaded_video(upload_id)
            else:
                video_key = params.get("video", ["test"])[0]
                video_path = DEMO_VIDEOS[video_key][0] if video_key in DEMO_VIDEOS else None
            if video_path is None or not video_path.is_file():
                self._error(404, "演示视频不存在")
                return
            self._live_stream(str(video_path), session_id, demo=True)
            return
        if path == "/api/live/captures":
            session_id = parse_qs(request_url.query).get("session", [""])[0]
            if not LIVE_SESSION.fullmatch(session_id):
                self._error(400, "无效实时会话")
                return
            self._json(200, CAPTURE_REVIEWER.snapshot(session_id))
            return
        parts = path.strip("/").split("/")
        if (len(parts) == 5 and parts[:3] == ["api", "live", "captures"]
                and LIVE_SESSION.fullmatch(parts[3]) and
                parts[4].endswith('.png') and
                TRACK_ID.fullmatch(parts[4].removesuffix('.png'))):
            image = CAPTURE_REVIEWER.image(parts[3], parts[4].removesuffix('.png'))
            self._send(200, image, "image/png") if image else self._error(404, "截图不存在")
            return
        if len(parts) == 3 and parts[:2] == ["api", "jobs"] and IDENTIFIER.fullmatch(parts[2]):
            with LOCK:
                job = JOBS.get(parts[2])
                data = json.loads(json.dumps(job)) if job else None
            self._json(200, data) if data else self._error(404, "任务不存在")
            return
        if (len(parts) == 6 and parts[:2] == ["api", "jobs"]
                and parts[3] == "files" and IDENTIFIER.fullmatch(parts[2])):
            job_id, run_name, filename = parts[2], parts[4], unquote(parts[5])
            if not re.fullmatch(r"[0-9]{3}_[\w.-]+", run_name) or Path(filename).name != filename:
                self._error(400, "无效文件名")
                return
            file_path = JOB_ROOT / job_id / run_name / filename
            if not file_path.is_file():
                self._error(404, "文件不存在")
                return
            self._serve_file(file_path)
            return
        self._error(404, "页面不存在")

    def _live_stream(self, source_override: str | None = None,
                     session_id: str | None = None, demo: bool = False) -> None:
        stream_slot = DEMO_SLOTS if demo else LIVE_LOCK
        if not stream_slot.acquire(blocking=False):
            self._error(409, "演示视频正在被其他用户处理，请稍后刷新" if demo else
                        "摄像头正在被另一位浏览器用户使用")
            return
        cap = None
        headers_started = False
        try:
            from .live import LatestFrameReader, LiveDetector, open_camera

            source = source_override or os.environ.get("CAMERA_SOURCE", "/dev/video0")
            detector = LiveDetector(MODEL, POSE_MODEL)
            cap = open_camera(source)
            ok, frame = cap.read()
            if not ok:
                self._error(503, f"无法从摄像头读取画面：{source}")
                return
            latest = None
            sequence = 0
            live_source = (source.isdigit() or source.startswith(
                ("/dev/video", "rtsp://", "rtsps://", "rtmp://", "rtmps://")))
            if not demo and live_source and os.environ.get("LIVE_DROP_STALE_FRAMES", "1") != "0":
                latest = LatestFrameReader(cap, frame)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            headers_started = True
            self.close_connection = True
            if demo:
                import cv2
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                duration = max(1.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
                started = time.monotonic()
                next_frame_at = started
                demo_frame_index = 0
            while ok:
                if demo and time.monotonic() - started >= max(90.0, duration * 2):
                    break
                jpeg = detector.render(frame)
                if session_id:
                    CAPTURE_REVIEWER.consider(
                        session_id, *detector.last_observation,
                        frame_time=demo_frame_index / fps if demo else None)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                 str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
                if demo:
                    demo_frame_index += 1
                    next_frame_at += 1 / fps
                    delay = next_frame_at - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        next_frame_at = time.monotonic()
                else:
                    if latest is None:
                        ok, frame = cap.read()
                    else:
                        ok, frame, sequence = latest.next(sequence)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.log_error("Live stream failed: %s", exc)
            if not headers_started:
                self._error(503, str(exc))
        finally:
            if 'latest' in locals() and latest is not None:
                latest.close()
            if cap is not None:
                cap.release()
            stream_slot.release()

    def _serve_file(self, file_path: Path) -> None:
        size = file_path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range", "")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match:
                self._error(416, "无效范围")
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            elif match.group(2):
                start = max(0, size - int(match.group(2)))
            if start >= size or end < start:
                self._error(416, "超出文件范围")
                return
            end = min(end, size - 1)
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", mimetypes.guess_type(file_path.name)[0] or
                         "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with file_path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/upload":
            self._upload()
        elif path == "/api/jobs":
            self._create_job()
        elif path == "/api/live/frame":
            self._browser_frame()
        else:
            self._error(404, "接口不存在")

    def _browser_frame(self) -> None:
        global BROWSER_DETECTOR
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "需要 Content-Length")
            return
        if length <= 0 or length > int(os.environ.get('BROWSER_FRAME_MAX_MB', '32')) * 1024 * 1024:
            self._error(413, "摄像头画面过大或为空")
            return
        import cv2
        import numpy as np
        from .live import LiveDetector

        raw = self.rfile.read(length)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._error(400, "无法解码摄像头画面")
            return
        try:
            with BROWSER_FRAME_LOCK:
                if BROWSER_DETECTOR is None:
                    BROWSER_DETECTOR = LiveDetector(MODEL, POSE_MODEL)
                jpeg = BROWSER_DETECTOR.render(frame)
                session_id = self.headers.get("X-Live-Session", "")
                if LIVE_SESSION.fullmatch(session_id):
                    CAPTURE_REVIEWER.consider(session_id, *BROWSER_DETECTOR.last_observation)
        except Exception as exc:
            self.log_error("Browser camera inference failed: %s", exc)
            self._error(500, str(exc))
            return
        self._send(200, jpeg, "image/jpeg")

    def _upload(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "需要 Content-Length")
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._error(413, "文件为空或超过大小限制")
            return
        name = Path(unquote(self.headers.get("X-Filename", ""))).name
        if not name or Path(name).suffix.lower() not in SUFFIXES:
            self._error(415, "只支持图片和视频文件")
            return
        upload_id = uuid.uuid4().hex
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        upload_dir = UPLOAD_ROOT / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / name
        remaining = length
        try:
            with destination.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("上传中断")
                    output.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        with LOCK:
            UPLOADS[upload_id] = {"path": str(destination), "name": name}
        self._json(200, {"id": upload_id, "name": name, "size": length})

    def _create_job(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 0 or length > 1024 * 1024:
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            ids = payload.get("upload_ids", [])
            if not isinstance(ids, list) or len(ids) > 100 or any(
                    not isinstance(item, str) or not IDENTIFIER.fullmatch(item) for item in ids):
                raise ValueError
        except (ValueError, TypeError, json.JSONDecodeError):
            self._error(400, "请求格式错误")
            return
        with LOCK:
            if ids and any(item not in UPLOADS for item in ids):
                self._error(404, "上传文件不存在，请重新选择")
                return
            if not ids and not DEFAULT_VIDEO.is_file():
                self._error(404, "默认测试视频不存在，请先上传文件")
                return
            items = ([UPLOADS[item] for item in ids] if ids else
                     [{"path": str(DEFAULT_VIDEO), "name": "test.mp4"}])
            job_id = uuid.uuid4().hex
            JOBS[job_id] = {"id": job_id, "status": "running", "current": "准备中",
                            "total": len(items), "results": [], "errors": []}
        threading.Thread(target=_run_job, args=(job_id, items), daemon=True).start()
        self._json(202, {"id": job_id})


def main() -> None:
    if not MODEL.is_file() or not POSE_MODEL.is_file():
        raise SystemExit("缺少 YOLO 或姿态模型，请检查 weights 目录")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "7860"))
    print(f"Web UI listening on http://{host}:{port}", flush=True)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Web UI stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
