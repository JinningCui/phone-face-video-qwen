#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PHONE_FACE_EXPORT_DIR="${PHONE_FACE_EXPORT_DIR:-$ROOT/outputs/ftp_spool}"
FTP_CONFIG_FILE="${FTP_CONFIG_FILE:-$ROOT/ftp.py}"
mkdir -p "$PHONE_FACE_EXPORT_DIR" "$ROOT/outputs/ftp_upload_state"

python scripts/ftp_upload_faces.py --ftp-example "$FTP_CONFIG_FILE" --check-config
python scripts/ftp_upload_faces.py \
  --spool "$PHONE_FACE_EXPORT_DIR" \
  --ftp-example "$FTP_CONFIG_FILE" \
  --state-dir "$ROOT/outputs/ftp_upload_state" &
ftp_pid=$!
trap 'kill "$ftp_pid" 2>/dev/null || true' EXIT

python -m src.phone_face.web
