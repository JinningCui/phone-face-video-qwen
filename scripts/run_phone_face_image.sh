#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 INPUT_IMAGE [OUTPUT_NAME]" >&2
  exit 2
fi

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

INPUT_IMAGE="$(resolve_path "$1")"
IMAGE_NAME="${2:-$(basename "${1%.*}")}"
MODEL_ABS="$(resolve_path "${MODEL_PATH:-weights/phone_face/best.pt}")"
OUTPUT_ABS="$(resolve_path "${OUTPUT_DIR:-outputs/phone_face_images}")"

[[ -f "$INPUT_IMAGE" ]] || { echo "Input image not found: $INPUT_IMAGE" >&2; exit 1; }
[[ -f "$MODEL_ABS" ]] || { echo "Model not found: $MODEL_ABS" >&2; exit 1; }

exec "${PYTHON_BIN:-python}" -m src.phone_face.run predict "$INPUT_IMAGE" \
  --model "$MODEL_ABS" \
  --device "${DEVICE:-cpu}" \
  --imgsz "${IMG_SIZE:-960}" \
  --phone-conf "${PHONE_CONF:-0.35}" \
  --face-conf "${FACE_CONF:-0.35}" \
  --output "$OUTPUT_ABS" \
  --name "$IMAGE_NAME"
