#!/usr/bin/env bash
set -euo pipefail

ENGINE="${DIFFUSIONGEMMA_ENGINE:-transformers}"
PORT="${PORT:-8080}"

echo "========================================================================"
echo " Starting DiffusionGemma-as-Jev (OpenJev) Cloud Run Server"
echo " Engine : ${ENGINE}"
echo " Port   : ${PORT}"
echo " Steps  : ${DIFFUSIONGEMMA_DENOISING_STEPS:-1} (Single-Step Bubble-Sheet Canvas)"
echo "========================================================================"

if [[ "${ENGINE}" == "vllm" ]]; then
  MODEL_ID="${DIFFUSIONGEMMA_MODEL_ID:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
  VLLM_PORT="${VLLM_PORT:-8001}"
  echo "[vllm] Launching internal vLLM server on 127.0.0.1:${VLLM_PORT} for ${MODEL_ID}..."
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --model "${MODEL_ID}" \
    --quantization "${VLLM_QUANTIZATION:-modelopt_fp4}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --trust-remote-code \
    --disable-log-requests &

  echo "[vllm] Waiting for internal vLLM readiness..."
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "[vllm] Ready!"
      break
    fi
    sleep 2
  done
fi

exec uvicorn server:app --host 0.0.0.0 --port "${PORT}" --workers 1
