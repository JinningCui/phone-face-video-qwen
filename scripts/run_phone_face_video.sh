#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory; every default path is relative to the project root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 INPUT_VIDEO [OUTPUT_NAME]" >&2
  echo "Example: $0 video_test/test.mp4 pose_smooth" >&2
  exit 2
fi

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

INPUT_VIDEO="$(resolve_path "$1")"
VIDEO_NAME="${2:-$(basename "${1%.*}")}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-weights/phone_face/best.pt}"
POSE_MODEL_PATH="${POSE_MODEL_PATH:-weights/yolov8n-pose.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phone_face_video}"
PHONE_CONF="${PHONE_CONF:-0.35}"
FACE_CONF="${FACE_CONF:-0.35}"
POSE_CONF="${POSE_CONF:-0.35}"
IMG_SIZE="${IMG_SIZE:-960}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.55}"
LINE_WIDTH="${LINE_WIDTH:-1}"
DEVICE="${DEVICE:-cpu}"

MODEL_ABS="$(resolve_path "$MODEL_PATH")"
POSE_MODEL_ABS="$(resolve_path "$POSE_MODEL_PATH")"
OUTPUT_ABS="$(resolve_path "$OUTPUT_DIR")"

[[ -f "$INPUT_VIDEO" ]] || { echo "Input video not found: $INPUT_VIDEO" >&2; exit 1; }
[[ -f "$MODEL_ABS" ]] || { echo "Phone/face model not found: $MODEL_ABS" >&2; exit 1; }
[[ -f "$POSE_MODEL_ABS" ]] || { echo "Pose model not found: $POSE_MODEL_ABS" >&2; exit 1; }

exec "$PYTHON_BIN" -m src.phone_face.run video "$INPUT_VIDEO" \
  --model "$MODEL_ABS" \
  --pose-model "$POSE_MODEL_ABS" \
  --device "$DEVICE" \
  --imgsz "$IMG_SIZE" \
  --phone-conf "$PHONE_CONF" \
  --face-conf "$FACE_CONF" \
  --pose-conf "$POSE_CONF" \
  --smooth-alpha "$SMOOTH_ALPHA" \
  --line-width "$LINE_WIDTH" \
  --output "$OUTPUT_ABS" \
  --name "$VIDEO_NAME"
