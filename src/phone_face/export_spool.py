"""Publish reviewed frontal face crops for the host-side FTP uploader."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path


SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
PERSON_ID = re.compile(r"[0-9a-f]{12}\Z")
MIN_EXPORT_FACES = 1
MAX_EXPORT_FACES = 5


def _replace_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_faces(root: Path, session_id: str, person_id: str,
                  images: list[dict]) -> bool:
    """Publish 1–5 improving JPEG crops without replacing earlier ranks."""
    if not SESSION_ID.fullmatch(session_id) or not PERSON_ID.fullmatch(person_id):
        raise ValueError("Invalid session or person identifier")
    selected = images[:MAX_EXPORT_FACES]
    if len(selected) < MIN_EXPORT_FACES:
        return False
    scores = [float(item["score"]) for item in selected]
    if (not all(math.isfinite(score) for score in scores) or
            any(later <= earlier for earlier, later in zip(scores, scores[1:]))):
        raise ValueError("Export face scores must strictly increase")
    directory = root / session_id / person_id
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for rank, item in enumerate(selected, 1):
        image = item["image"]
        name = f"{rank:02d}.jpg"
        digest = hashlib.sha256(image).hexdigest()
        entries.append({"name": name, "sha256": digest, "score": scores[rank - 1]})
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("policy") == "improving_append_only_v1":
            old_entries = previous.get("images", [])
            if len(entries) < len(old_entries) or entries[:len(old_entries)] != old_entries:
                raise ValueError("Previously exported face images cannot be replaced")
            if len(entries) == len(old_entries):
                return False
    for rank, item in enumerate(selected, 1):
        image = item["image"]
        name = f"{rank:02d}.jpg"
        digest = entries[rank - 1]["sha256"]
        target = directory / name
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            _replace_bytes(target, image)
    manifest = {"session_id": session_id, "person_id": person_id,
                "policy": "improving_append_only_v1",
                "count": len(entries), "images": entries}
    _replace_bytes(manifest_path,
                   json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode())
    return True
