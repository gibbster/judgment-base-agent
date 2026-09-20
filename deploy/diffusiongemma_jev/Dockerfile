FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/app/hf_cache \
    DIFFUSIONGEMMA_ENGINE=transformers \
    DIFFUSIONGEMMA_DENOISING_STEPS=1 \
    PRELOAD_MODEL_ON_STARTUP=true

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch, HuggingFace transformers>=5.8.0 (with native DiffusionGemmaForBlockDiffusion),
# accelerate, bitsandbytes (for 4-bit NF4 on NVIDIA L4 24GB GPU), and FastAPI/Uvicorn
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch \
    "transformers>=5.8.0" \
    accelerate \
    bitsandbytes \
    safetensors \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    pydantic

# ---------------------------------------------------------------------------
# Optional: vLLM with DiffusionGemma structured-generation support.
#
# Structured generation for DiffusionGemma is NOT in an upstream vLLM release.
# It lives in vllm-project/vllm#57250, which as of 2026-09-20 is OPEN, has
# merge conflicts, and has no formal review approvals. We therefore pin the
# author's fork at an exact commit rather than tracking a moving branch.
#
# This build is GPU-only and adds ~30-60 min of compile time, so it is opt-in:
#   docker build --build-arg INSTALL_VLLM=true ...
#
# Upgrade path: when #57250 merges, set VLLM_GIT_URL to the upstream repo and
# VLLM_GIT_REF to the merge commit (or drop this block for a PyPI release).
# ---------------------------------------------------------------------------
ARG INSTALL_VLLM=false
ARG VLLM_GIT_URL=https://github.com/mmastrac/vllm.git
ARG VLLM_GIT_REF=ceb8eebf3eedddb964a50180f33838a9a6b13ee2

ENV DIFFUSIONGEMMA_VLLM_SOURCE="${VLLM_GIT_URL}@${VLLM_GIT_REF}"
# The PR's Jev-compatible System One server. `pip install git+...` builds a wheel
# and discards examples/, so the source tree is cloned separately to get it.
ENV DIFFUSIONGEMMA_STRUCTURED_SERVER="/opt/vllm-pr/examples/features/diffusion_reads/structured_server.py"

RUN if [ "${INSTALL_VLLM}" = "true" ]; then \
      echo "[build] Installing vLLM from ${VLLM_GIT_URL}@${VLLM_GIT_REF}" && \
      apt-get update && apt-get install -y --no-install-recommends git build-essential && \
      git clone --filter=blob:none "${VLLM_GIT_URL}" /opt/vllm-pr && \
      git -C /opt/vllm-pr checkout "${VLLM_GIT_REF}" && \
      pip install --no-cache-dir "/opt/vllm-pr" && \
      rm -rf /var/lib/apt/lists/* && \
      test -f "${DIFFUSIONGEMMA_STRUCTURED_SERVER}" && \
      python3 -c "import vllm; print('[build] vLLM', vllm.__version__)" ; \
    else \
      echo "[build] Skipping vLLM (INSTALL_VLLM=false). Engine 'vllm' will be unavailable." ; \
    fi

# Pre-warm the DiffusionGemma tokenizer & base weights into /app/hf_cache for instant cold-start
RUN python3 -c "from transformers import DiffusionGemmaForBlockDiffusion, AutoTokenizer; m='trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion'; AutoTokenizer.from_pretrained(m); DiffusionGemmaForBlockDiffusion.from_pretrained(m)"

COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
