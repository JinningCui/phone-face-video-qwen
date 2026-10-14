#!/usr/bin/env python3
"""Upload each newly accepted face capture through the host's FTP route."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
from io import BytesIO
import json
import math
import os
import re
import sys
import time
import uuid
from ftplib import FTP, error_perm
from pathlib import Path


ID32 = re.compile(r'[0-9a-f]{32}\Z')
ID12 = re.compile(r'[0-9a-f]{12}\Z')
IMAGE_NAME = re.compile(r'0[1-5]\.jpg\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
SESSION_NAME = re.compile(r'session_\d{8}_\d{6}(?:_\d{2,})?\Z')
PERSON_NAME = re.compile(r'person_\d{2,}\Z')


def credentials_from_example(path: Path) -> tuple[str, str, str]:
    """Read literal HOST and ftp.login arguments without executing ftp.py."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    host = username = password = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'HOST'
                for target in node.targets):
            host = ast.literal_eval(node.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'login' and len(node.args) >= 2):
            username = ast.literal_eval(node.args[0])
            password = ast.literal_eval(node.args[1])
    if not all(isinstance(value, str) and value for value in
               (host, username, password)):
        raise ValueError('ftp.py must contain literal HOST and ftp.login credentials')
    return host, username, password


def ensure_dir(ftp: FTP, path: str) -> None:
    ftp.cwd('/')
    for part in path.strip('/').split('/'):
        if not part:
            continue
        if part in ('.', '..'):
            raise ValueError('Invalid FTP directory')
        try:
            ftp.cwd(part)
        except error_perm:
            ftp.mkd(part)
            ftp.cwd(part)


def _manifest(directory: Path) -> dict | None:
    path = directory / 'manifest.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if (not isinstance(data, dict) or
            not ID32.fullmatch(data.get('session_id', '')) or
            not ID12.fullmatch(data.get('person_id', '')) or
            directory.name != data['person_id'] or
            directory.parent.name != data['session_id']):
        return None
    entries = data.get('images', [])
    if (not isinstance(entries, list) or not 1 <= len(entries) <= 5 or
            data.get('count') != len(entries)):
        return None
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            return None
        name, digest = entry.get('name', ''), entry.get('sha256', '')
        if name != f'{index:02d}.jpg' or not IMAGE_NAME.fullmatch(name) or not SHA256.fullmatch(digest):
            return None
    if data.get('policy') == 'improving_append_only_v1':
        scores = [entry.get('score') for entry in entries]
        if (any(not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) for score in scores) or
                any(later <= earlier for earlier, later in zip(scores, scores[1:]))):
            return None
    return data


def _save_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(data, sort_keys=True), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _named_target(state_root: Path, remote_root: str, data: dict,
                  manifest_path: Path) -> str:
    """Keep one readable session/person mapping across polls and restarts."""
    mapping_path = state_root / 'directory_names.json'
    try:
        mappings = json.loads(mapping_path.read_text(encoding='utf-8'))
        if not isinstance(mappings, dict):
            mappings = {}
    except (OSError, ValueError):
        mappings = {}
    session_id, person_id = data['session_id'], data['person_id']
    session = mappings.get(session_id)
    if (not isinstance(session, dict) or
            not isinstance(session.get('name'), str) or
            not SESSION_NAME.fullmatch(session['name'])):
        stamp = datetime.fromtimestamp(manifest_path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
        base = f'session_{stamp}'
        used = {entry.get('name') for entry in mappings.values() if isinstance(entry, dict)}
        name = base
        number = 2
        while name in used:
            name = f'{base}_{number:02d}'
            number += 1
        session = {'name': name, 'people': {}}
        mappings[session_id] = session
    people = session.get('people')
    if not isinstance(people, dict):
        people = session['people'] = {}
    person_name = people.get(person_id)
    if not isinstance(person_name, str) or not PERSON_NAME.fullmatch(person_name):
        used = set(people.values())
        number = 1
        while f'person_{number:02d}' in used:
            number += 1
        person_name = people[person_id] = f'person_{number:02d}'
    _save_json(mapping_path, mappings)
    return remote_root.rstrip('/') + '/' + session['name'] + '/' + person_name


def sync_once(spool: Path, example: Path, remote_root: str,
              state_root: Path | None = None) -> tuple[int, int]:
    if not remote_root.startswith('/'):
        raise ValueError('FTP root must be an absolute path')
    host, username, password = credentials_from_example(example)
    state_root = state_root or spool.parent / 'ftp_upload_state'
    state_root.mkdir(parents=True, exist_ok=True)
    ready = uploaded = 0
    for manifest_path in sorted(spool.glob('*/*/manifest.json')):
        directory = manifest_path.parent
        data = _manifest(directory)
        if data is None:
            continue
        ready += 1
        digests = {item['name']: item['sha256'] for item in data['images']}
        state_path = state_root / f"{data['session_id']}-{data['person_id']}.json"
        try:
            state = json.loads(state_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            state = {}
        previous_images = state.get('images')
        if not isinstance(previous_images, dict):
            previous_images = {}
        # Existing transfers keep their original destination, including legacy
        # UUID paths, so a Top-5 update never creates a duplicate person folder.
        target_dir = state.get('remote_dir') or _named_target(
            state_root, remote_root, data, manifest_path)
        if state.get('remote_dir') == target_dir and state.get('images') == digests:
            continue
        # Manifest publication is atomic. Verify and transfer only new ranks;
        # older ranks remain unchanged in the append-only export policy.
        pending = []
        for entry in data['images']:
            name, digest = entry['name'], entry['sha256']
            if previous_images.get(name) == digest:
                continue
            try:
                content = (directory / name).read_bytes()
            except OSError:
                pending = []
                break
            if hashlib.sha256(content).hexdigest() != digest:
                pending = []
                break
            pending.append((name, content))
            if data.get('policy') == 'improving_append_only_v1':
                break
        if not pending:
            continue
        with FTP() as ftp:
            ftp.connect(host, int(os.environ.get('FTP_PORT', '21')), timeout=30)
            ftp.login(username, password)
            ftp.set_pasv(True)
            ensure_dir(ftp, target_dir)
            for name, content in pending:
                temporary = '.' + name + '.uploading'
                ftp.storbinary(f'STOR {temporary}', BytesIO(content), blocksize=1024 * 1024)
                try:
                    ftp.rename(temporary, name)
                except error_perm:
                    ftp.delete(name)
                    ftp.rename(temporary, name)
        uploaded_images = (previous_images | {name: digests[name] for name, _ in pending}
                           if data.get('policy') == 'improving_append_only_v1' else digests)
        state = {'remote_dir': target_dir, 'images': uploaded_images}
        _save_json(state_path, state)
        uploaded += 1
        print(f'Uploaded {len(pending)} new face captures: {target_dir}', flush=True)
    return ready, uploaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--spool', type=Path, required=True)
    parser.add_argument('--ftp-example', type=Path, required=True)
    parser.add_argument('--remote-root', default='/20261010')
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            ready, uploaded = sync_once(args.spool, args.ftp_example,
                                        args.remote_root, args.state_dir)
            if args.once:
                print(f'{ready} ready groups, {uploaded} uploaded', flush=True)
                return
        except Exception as exc:
            print(f'FTP upload failed: {exc}', file=sys.stderr, flush=True)
            if args.once:
                raise SystemExit(1)
        time.sleep(max(0.5, args.interval))


if __name__ == '__main__':
    main()
