#!/usr/bin/env bash
#
# Install the DiffusionGemma-as-Jev systemd units on the GPU VM.
#
# Run this ON THE VM, as root:
#
#   gcloud compute scp --recurse deploy/diffusiongemma_jev/systemd \
#     djev-vllm-l4:~/systemd --zone us-central1-a \
#     --project medquad-assistant-capstone --tunnel-through-iap
#   gcloud compute ssh djev-vllm-l4 --zone us-central1-a \
#     --project medquad-assistant-capstone --tunnel-through-iap \
#     --command "sudo bash ~/systemd/install.sh"
#
# Without these units the servers are one-off nohup processes: they die on
# reboot and never come back after the VM is stopped to save money.
set -euo pipefail

UNIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR=/etc/systemd/system
UNITS=(djev-vllm.service djev-jev.service)

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: must run as root (use: sudo bash $0)" >&2
  exit 1
fi

# Fail loudly now rather than leaving a unit that silently never starts.
for required in \
  /opt/venv/vllm/bin/vllm \
  /opt/venv/vllm/bin/python \
  /opt/vllm-pr/examples/features/diffusion_reads/structured_server.py \
  /opt/hf
do
  if [[ ! -e "${required}" ]]; then
    echo "error: expected path missing on this VM: ${required}" >&2
    exit 1
  fi
done

echo "==> Stopping any hand-launched nohup servers"
pkill -f 'vllm serve RedHatAI' || true
pkill -f 'structured_server.py' || true
sleep 3

echo "==> Installing units into ${SYSTEMD_DIR}"
for unit in "${UNITS[@]}"; do
  install -m 0644 "${UNIT_DIR}/${unit}" "${SYSTEMD_DIR}/${unit}"
  echo "    ${unit}"
done

echo "==> Enabling and starting"
systemctl daemon-reload
systemctl enable "${UNITS[@]}"
# Starting djev-jev pulls in djev-vllm via Requires=.
systemctl restart djev-jev.service

cat <<'EOF'

==> Installed. The engine takes ~5 minutes to load weights on a cold start.

Watch progress:
  journalctl -u djev-vllm -f
  tail -f /var/log/djev-vllm.log

Verify when ready:
  curl -sf http://127.0.0.1:8011/health && echo OK

Both units are enabled, so `gcloud compute instances start djev-vllm-l4`
is now enough to bring the whole stack back.
EOF
