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
  # In vLLM mode we do NOT run our own server.py. The upstream PR ships
  # examples/features/diffusion_reads/structured_server.py, which already speaks
  # the Jev System One wire format AND drives the canvas correctly via
  # diffusion_seed_canvas / diffusion_read_only — one read for all questions.
  # Re-implementing that here would cost one HTTP round trip per question.
  #
  # Topology:  client -> structured_server.py (:$PORT) -> vllm serve (:$VLLM_PORT)

  # Preflight: the image only ships vLLM when built with --build-arg INSTALL_VLLM=true.
  # Fail loudly here rather than emitting a bare ModuleNotFoundError from the server.
  if ! python3 -c "import vllm" >/dev/null 2>&1; then
    echo "[vllm] FATAL: DIFFUSIONGEMMA_ENGINE=vllm but vLLM is not installed in this image." >&2
    echo "[vllm]   DiffusionGemma structured generation requires vllm-project/vllm#57250," >&2
    echo "[vllm]   which is not yet merged upstream. Rebuild with:" >&2
    echo "[vllm]     docker build --build-arg INSTALL_VLLM=true ..." >&2
    echo "[vllm]   Or set DIFFUSIONGEMMA_ENGINE=transformers to use the native engine." >&2
    exit 1
  fi

  STRUCTURED_SERVER="${DIFFUSIONGEMMA_STRUCTURED_SERVER:-/opt/vllm-pr/examples/features/diffusion_reads/structured_server.py}"
  if [[ ! -f "${STRUCTURED_SERVER}" ]]; then
    echo "[vllm] FATAL: structured_server.py not found at ${STRUCTURED_SERVER}." >&2
    echo "[vllm]   It ships in the PR's examples/ tree, which a wheel-only install discards." >&2
    echo "[vllm]   Rebuild the image, or point DIFFUSIONGEMMA_STRUCTURED_SERVER at a checkout." >&2
    exit 1
  fi

  # Verified 2026-09-20 on an NVIDIA L4 (23034 MiB): RedHatAI NVFP4 via the Marlin
  # FP4 fallback settles at ~20596 MiB with these flags.
  MODEL_ID="${DIFFUSIONGEMMA_MODEL_ID:-RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}"
  VLLM_PORT="${VLLM_PORT:-8001}"
  CANVAS_LENGTH="${DIFFUSIONGEMMA_CANVAS_LENGTH:-64}"
  echo "[vllm] Launching internal vLLM engine on 127.0.0.1:${VLLM_PORT} for ${MODEL_ID}..."
  echo "[vllm] Source: ${DIFFUSIONGEMMA_VLLM_SOURCE:-unknown}"
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --model "${MODEL_ID}" \
    --quantization "${VLLM_QUANTIZATION:-modelopt_fp4}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --max-logprobs "${VLLM_MAX_LOGPROBS:-32}" \
    --enable-prefix-caching \
    --attention-backend "${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" \
    --diffusion-config "{\"canvas_length\": ${CANVAS_LENGTH}}" \
    --trust-remote-code \
    --disable-log-requests &

  echo "[vllm] Waiting for internal vLLM readiness (up to 360s)..."
  VLLM_READY=0
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "[vllm] Ready after ~$((i * 2))s."
      VLLM_READY=1
      break
    fi
    sleep 2
  done
  if [[ "${VLLM_READY}" -ne 1 ]]; then
    echo "[vllm] FATAL: internal vLLM engine did not become ready within 360s." >&2
    exit 1
  fi

  # structured_server.py serves POST /v1/systemone. Agents reach it by setting
  # DIFFUSIONGEMMA_SYSTEM_ONE_PATH=/v1/systemone (see DiffusionGemmaBackend).
  echo "[vllm] Serving PR structured_server.py on 0.0.0.0:${PORT} -> 127.0.0.1:${VLLM_PORT}"
  exec python3 "${STRUCTURED_SERVER}" \
    --upstream "http://127.0.0.1:${VLLM_PORT}" \
    --model "${MODEL_ID}" \
    --tokenizer "${DIFFUSIONGEMMA_TOKENIZER:-${MODEL_ID}}" \
    --canvas "${CANVAS_LENGTH}" \
    --host 0.0.0.0 \
    --port "${PORT}"
fi

exec uvicorn server:app --host 0.0.0.0 --port "${PORT}" --workers 1
