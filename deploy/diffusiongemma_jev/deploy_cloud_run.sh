#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-medquad-assistant-capstone}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-diffusiongemma-jev}"
REPO_NAME="${REPO_NAME:-medquad-repo}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo " Deploying DiffusionGemma-as-Jev (Option A) to GCP Cloud Run"
echo " Project : ${PROJECT_ID}"
echo " Region  : ${REGION}"
echo " Service : ${SERVICE_NAME}"
echo " Image   : ${IMAGE_URI}"
echo "========================================================================"

# 1. Ensure Artifact Registry repository exists
if ! gcloud artifacts repositories describe "${REPO_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" >/dev/null 2>&1; then
  echo "[1/3] Creating Artifact Registry repository ${REPO_NAME}..."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --project="${PROJECT_ID}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="DiffusionGemma-as-Jev Container Repository"
else
  echo "[1/3] Using existing Artifact Registry repository ${REPO_NAME}."
fi

# 2. Build and push container image via Cloud Build
echo "[2/3] Building container image via Cloud Build..."
gcloud builds submit "${SCRIPT_DIR}" \
  --quiet \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_URI}"

# 3. Deploy to Cloud Run GPU (1x NVIDIA L4 24GB VRAM), with automatic fallback if L4 quota is 0
echo "[3/3] Deploying ${SERVICE_NAME} to Cloud Run (Option A: 1x NVIDIA L4 GPU)..."
if gcloud run deploy "${SERVICE_NAME}" \
  --quiet \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_URI}" \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu=8 \
  --memory=32Gi \
  --no-cpu-throttling \
  --concurrency=8 \
  --timeout=300 \
  --allow-unauthenticated \
  --set-env-vars="DIFFUSIONGEMMA_ENGINE=${DIFFUSIONGEMMA_ENGINE:-transformers},DIFFUSIONGEMMA_DENOISING_STEPS=1"; then
  echo "✅ Deployed with 1x NVIDIA L4 GPU!"
else
  echo "⚠️  NVIDIA L4 GPU quota unavailable in ${PROJECT_ID}/${REGION}; deploying with Cloud Run 4 vCPU / 16 GiB..."
  gcloud run deploy "${SERVICE_NAME}" \
    --quiet \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE_URI}" \
    --cpu=4 \
    --memory=16Gi \
    --no-cpu-throttling \
    --concurrency=8 \
    --timeout=300 \
    --allow-unauthenticated \
    --set-env-vars="DIFFUSIONGEMMA_ENGINE=transformers,DIFFUSIONGEMMA_DENOISING_STEPS=1"
fi

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo ""
echo "========================================================================"
echo " 🚀 DiffusionGemma-as-Jev Deployed Successfully!"
echo " URL: ${SERVICE_URL}"
echo " Export for ADK agents:"
echo "   export DIFFUSIONGEMMA_JEV_URL=\"${SERVICE_URL}\""
echo "========================================================================"
