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

# Pre-warm the DiffusionGemma tokenizer & base weights into /app/hf_cache for instant cold-start
RUN python3 -c "from transformers import DiffusionGemmaForBlockDiffusion, AutoTokenizer; m='trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion'; AutoTokenizer.from_pretrained(m); DiffusionGemmaForBlockDiffusion.from_pretrained(m)"

COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
