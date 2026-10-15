#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
export DEVICE=${DEVICE:-3}
export LIVE_POSE_DEVICE=${LIVE_POSE_DEVICE:-3}
export LIVE_DROP_STALE_FRAMES=${LIVE_DROP_STALE_FRAMES:-1}
export QWEN_VL_MODEL=${QWEN_VL_MODEL:-Qwen3.5-4B}
export QWEN_DISABLE_THINKING=${QWEN_DISABLE_THINKING:-1}
export QWEN_VL_URLS=${QWEN_VL_URLS:-http://127.0.0.1:8101/v1/chat/completions,http://127.0.0.1:8102/v1/chat/completions,http://127.0.0.1:8103/v1/chat/completions}
export PHONE_FACE_EXPORT_DIR=${PHONE_FACE_EXPORT_DIR:-outputs/ftp_spool}
exec "${VISION_PYTHON_BIN:-python}" -m src.phone_face.web
