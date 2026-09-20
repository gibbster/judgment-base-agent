#!/usr/bin/env bash
#
# Open an IAP tunnel to the shared DiffusionGemma GPU VM and health-check it.
#
#   ./scripts/connect_gpu.sh          # tunnel on localhost:8011, stays in foreground
#   ./scripts/connect_gpu.sh --start  # also start the VM if it is stopped
#
# Leave this running in its own terminal, then in a second terminal:
#
#   cp examples/.env.example examples/.env   # uncomment the DIFFUSIONGEMMA_* block
#   PYTHONPATH=. adk web examples --port 8008
#
# The VM has no external IP by design, so IAP is the only way in. You need
# roles/iap.tunnelResourceAccessor and roles/compute.viewer on the project.
set -euo pipefail

PROJECT="${DJEV_PROJECT:-medquad-assistant-capstone}"
ZONE="${DJEV_ZONE:-us-central1-a}"
INSTANCE="${DJEV_INSTANCE:-djev-vllm-l4}"
LOCAL_PORT="${DJEV_LOCAL_PORT:-8011}"
REMOTE_PORT="${DJEV_REMOTE_PORT:-8011}"

START_IF_STOPPED=false
[[ "${1:-}" == "--start" ]] && START_IF_STOPPED=true

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v gcloud >/dev/null || die "gcloud not found; install the Google Cloud SDK"

if lsof -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  die "localhost:${LOCAL_PORT} is already in use. A tunnel may already be running."
fi

log "Checking ${INSTANCE} in ${PROJECT}/${ZONE}"
STATUS="$(gcloud compute instances describe "${INSTANCE}" \
  --zone "${ZONE}" --project "${PROJECT}" \
  --format='value(status)' 2>/dev/null)" \
  || die "cannot read ${INSTANCE}. Do you have access to project ${PROJECT}?"

if [[ "${STATUS}" != "RUNNING" ]]; then
  if [[ "${START_IF_STOPPED}" != true ]]; then
    die "${INSTANCE} is ${STATUS}. Re-run with --start to boot it (~5 min, then it bills ~\$0.71/hr)."
  fi
  log "${INSTANCE} is ${STATUS}; starting it"
  gcloud compute instances start "${INSTANCE}" --zone "${ZONE}" --project "${PROJECT}"
  log "Waiting for the model to load (systemd brings both services up; ~5 min cold)"
fi

# Tunnel first, then poll through it: the Jev port is not otherwise reachable.
log "Opening tunnel localhost:${LOCAL_PORT} -> ${INSTANCE}:${REMOTE_PORT}"
gcloud compute ssh "${INSTANCE}" \
  --zone "${ZONE}" --project "${PROJECT}" --tunnel-through-iap \
  -- -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=1000 \
  -o ExitOnForwardFailure=yes &
TUNNEL_PID=$!
trap 'kill "${TUNNEL_PID}" 2>/dev/null || true' EXIT

log "Waiting for the Jev endpoint to answer"
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1; then
    # The first judgment after a cold start pays for CUDA graph capture and an
    # empty prefix cache: measured 30.6 s, versus 258 ms once warm. Burn that
    # cost here so nobody's first impression is a 30-second hang.
    log "Warming the model (one throwaway judgment, up to ~40 s)"
    curl -sf --max-time 90 "http://127.0.0.1:${LOCAL_PORT}/v1/systemone" \
      -H 'Content-Type: application/json' \
      -d '{"state":{"warmup":true},"questions":{"q0":{"type":"noul","instructions":"Is this a warmup?"}}}' \
      >/dev/null 2>&1 || log "warm-up call failed; the endpoint is up but the first real call will be slow"

    cat <<EOF

$(printf '\033[1;32m==> Ready.\033[0m') Jev System One is at http://127.0.0.1:${LOCAL_PORT}

  DIFFUSIONGEMMA_JEV_URL=http://127.0.0.1:${LOCAL_PORT}
  DIFFUSIONGEMMA_SYSTEM_ONE_PATH=/v1/systemone

Leave this terminal open. Ctrl-C closes the tunnel.
EOF
    wait "${TUNNEL_PID}"
    exit 0
  fi
  kill -0 "${TUNNEL_PID}" 2>/dev/null || die "tunnel died; check your IAP permissions"
  sleep 5
done

die "endpoint never came up. On the VM: systemctl status djev-vllm djev-jev"
