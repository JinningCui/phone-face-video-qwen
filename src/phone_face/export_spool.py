"""Atomically mirror each person's visible Top-5 PNG captures for FTP upload."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path


SESSION_ID = re.compile(r'[0-9a-f]{32}\Z')
PERSON_ID = re.compile(r'[0-9a-f]{12}\Z')
MAX_IMAGES = 5


def _replace(path: Path, data: bytes) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_gallery(root: Path, session_id: str, person_id: str,
                    images: list[dict]) -> bool:
    """Publish the gallery state; FTP may update ranks as better faces arrive."""
    if not SESSION_ID.fullmatch(session_id) or not PERSON_ID.fullmatch(person_id):
        raise ValueError('Invalid session or person identifier')
    selected = images[:MAX_IMAGES]
    if not selected:
        return False
    directory = root / session_id / person_id
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for rank, item in enumerate(selected, 1):
        image = item['image']
        if not image.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('FTP gallery export requires PNG source pixels')
        name = f'{rank:02d}.png'
        digest = hashlib.sha256(image).hexdigest()
        entries.append({'name': name, 'sha256': digest,
                        'score': float(item['score']),
                        'frontal': bool(item.get('frontal', False))})
        target = directory / name
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            _replace(target, image)
    manifest = {'session_id': session_id, 'person_id': person_id,
                'policy': 'gallery_top5_png_v1', 'count': len(entries),
                'images': entries}
    manifest_path = directory / 'manifest.json'
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    if manifest_path.is_file() and manifest_path.read_bytes() == encoded:
        return False
    _replace(manifest_path, encoded)
    return True


def retire_gallery(root: Path, session_id: str, person_id: str) -> None:
    """Stop exporting a track after it has merged into another gallery group."""
    if not SESSION_ID.fullmatch(session_id) or not PERSON_ID.fullmatch(person_id):
        raise ValueError('Invalid session or person identifier')
    (root / session_id / person_id / 'manifest.json').unlink(missing_ok=True)
