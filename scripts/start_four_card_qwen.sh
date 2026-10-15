#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
model_dir=${QWEN_MODEL_DIR:-$root/models/Qwen3.5-4B}
vllm_bin=${VLLM_BIN:-vllm}
[[ -f "$model_dir/config.json" ]] || {
  echo "Qwen3.5-4B files missing: $model_dir (set QWEN_MODEL_DIR)" >&2
  exit 1
}
command -v "$vllm_bin" >/dev/null || {
  echo "vLLM executable missing: $vllm_bin" >&2
  exit 1
}
mkdir -p outputs/qwen_replicas

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for gpu in 0 1 2; do
  port=$((8101 + gpu))
  CUDA_VISIBLE_DEVICES="$gpu" "$vllm_bin" serve "$model_dir" \
    --served-model-name Qwen3.5-4B --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 2 \
    --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION:-0.65}" \
    --limit-mm-per-prompt '{"image":3}' \
    > "outputs/qwen_replicas/gpu${gpu}_port${port}.log" 2>&1 &
  pids+=("$!")
done

for port in 8101 8102 8103; do
  ready=0
  for ((attempt=0; attempt<180; attempt++)); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ $ready -eq 1 ]] || {
    echo "Qwen on port $port failed to start; inspect outputs/qwen_replicas" >&2
    exit 1
  }
done
echo "Three Qwen replicas ready on GPU 0–2; keep this terminal open."
wait -n "${pids[@]}"
