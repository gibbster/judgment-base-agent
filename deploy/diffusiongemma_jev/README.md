# Deploying `DiffusionGemma-as-Jev` on GCP Cloud Run

This directory packages Google's **`DiffusionGemma-26B-A4B`** (`google/diffusiongemma-26B-A4B-it`, Apache-2.0, ungated) as a self-hosted **System One / Jev** server on Google Cloud Run.

## How It Works ("Bubble-Sheet" Single-Step Denoising)

1. **`DiffusionGemma-26B-A4B`** is a sparse Mixture-of-Experts discrete diffusion model — 25.2B total parameters but only **3.8B active** per step, with a 256-token canvas.
2. Instead of generating free-form text over the recommended 48 denoising steps, [`server.py`](./server.py) lays your `JudgmentSchema` (`Choice`, `Score`, `Noul`) out as a **single-step bubble sheet** (`DIFFUSIONGEMMA_DENOISING_STEPS=1`). Every question gets its own answer slot in one canvas, so a single forward pass scores all of them at once.
3. From that pass it reads the exact softmax distribution (`probabilities`), expected value (`Score`), boolean probability (`Noul`), and normalized Shannon entropy (`confidence` / `_epistemic_clarity`).

Because all criteria share one canvas, latency is near-`O(1)` in the number of criteria — measured on a live deployment, 15 criteria cost only **+8 ms** over 1.

## Engines

| Engine | `DIFFUSIONGEMMA_ENGINE` | Notes |
| :-- | :-- | :-- |
| **`transformers`** (default) | `transformers` | [`server.py`](./server.py) runs a native `DiffusionGemmaForBlockDiffusion` encoder-prefill + bidirectional-decoder canvas pass. Requires `transformers >= 5.8.0`. Runs on CPU or GPU. No vLLM dependency. |
| **vLLM** (opt-in, GPU-only) | `vllm` | `entrypoint.sh` runs `vllm serve` on an internal port **and serves the PR's own `structured_server.py` on `$PORT`**. `server.py` is not started at all. Requires [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) — **not merged**. Must be built in explicitly; see below. |

### Why vLLM mode does not use `server.py`

PR #57250 ships `examples/features/diffusion_reads/structured_server.py`, which already
speaks this exact Jev System One wire format **and** drives the canvas properly through
the PR's `diffusion_seed_canvas` / `diffusion_read_only` sampling params — all questions
answered from one read.

Our earlier vLLM path in `server.py` posted plain `/v1/completions` once *per question*
and ignored those params, which is `O(N)` in criteria — the precise opposite of the
property that makes this model worth deploying. It has been deleted rather than ported.

One wire-level difference: the PR's server listens on `POST /v1/systemone` (no
underscore). Point agents at it with `DIFFUSIONGEMMA_SYSTEM_ONE_PATH=/v1/systemone`.

### Building the vLLM engine

> [!WARNING]
> DiffusionGemma structured generation is **not in any released vLLM**. As of 2026-09-20, PR #57250 is **open**, has **merge conflicts**, and has **no formal review approvals**. The author's own note: *"I need to take a pass at cleaning up LLM comments before this is ready to land."* Treat this engine as experimental.

The image does not ship vLLM by default. Opt in at build time — it is pinned to an exact fork commit, not a moving branch:

```bash
docker build \
  --build-arg INSTALL_VLLM=true \
  --build-arg VLLM_GIT_URL=https://github.com/mmastrac/vllm.git \
  --build-arg VLLM_GIT_REF=ceb8eebf3eedddb964a50180f33838a9a6b13ee2 \
  -t diffusiongemma-jev:vllm deploy/diffusiongemma_jev/
```

If you set `DIFFUSIONGEMMA_ENGINE=vllm` on an image built without that flag, `entrypoint.sh` exits with an explicit error rather than a bare `ModuleNotFoundError`.

The build clones the PR source tree to `/opt/vllm-pr` rather than `pip install git+...`, because a wheel-only install discards `examples/` — and that is where `structured_server.py` lives.

**Memory sizing.** Measured on an NVIDIA L4 (23034 MiB): `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` via the Marlin FP4 fallback settles at **20596 MiB**, ready in ~300 s. The entrypoint defaults to that checkpoint with `--max-model-len 16384`, `--max-logprobs 32`, `--enable-prefix-caching`, `--attention-backend TRITON_ATTN`, and `--diffusion-config '{"canvas_length": 64}'`.

**When #57250 merges**, point `VLLM_GIT_URL` at upstream and bump `VLLM_GIT_REF` to the merge commit — or drop the build arg entirely once it reaches a PyPI release.

## Endpoints

| Route | Purpose |
| :-- | :-- |
| `GET /health` | Liveness + reports active `engine` and resolved `model`. (`transformers` mode) |
| `POST /v1/system_one` | Batched judgment scoring. (`transformers` mode) |
| `POST /v1/judgment` | Alias of `/v1/system_one`. (`transformers` mode) |
| `POST /v1/systemone` | Batched judgment scoring served by the PR's `structured_server.py`. (`vllm` mode) |

---

## 1. One-Command Deployment

```bash
export PROJECT_ID="your-gcp-project-id"
chmod +x deploy/diffusiongemma_jev/deploy_cloud_run.sh
./deploy/diffusiongemma_jev/deploy_cloud_run.sh
```

The script builds in Cloud Build, auto-creates the Artifact Registry repo, deploys to Cloud Run, and prints your service URL. No HuggingFace token is needed — the model is ungated.

### GPU vs CPU

The script first tries `--gpu=1 --gpu-type=nvidia-l4`, then **falls back to 4 vCPU / 16 GiB CPU** if that fails.

> **Cloud Run GPU services require `--min-instances >= 1`**, so a GPU deployment does *not* scale to zero and bills continuously. Only the CPU fallback scales to `$0` when idle.

L4 quota is `0` on new projects; request it at [g.co/cloudrun/gpu-quota](https://g.co/cloudrun/gpu-quota). Among the public quantizations, `nvidia/...NVFP4` (17.53 GiB) fits an L4's 24 GB — `RedHatAI/...FP8-dynamic` (25.33 GiB) does not.

Without a GPU the server falls back to a small test checkpoint. That exercises the complete request path but produces near-uniform probabilities, so it validates **plumbing, not judgment quality**.

---

## 2. Switching Existing Agents & `adk web` to Your Container

**Zero code changes** are required in any existing `JudgmentAgent`, `JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`, or the five apps in `examples/`.

Set `DIFFUSIONGEMMA_JEV_URL` in your `examples/.env` (or shell):

```bash
export DIFFUSIONGEMMA_JEV_URL="https://diffusiongemma-jev-<hash>-uc.a.run.app"
```

When it is set, `TypeSafeBackend()` automatically delegates every `evaluate()` call to `DiffusionGemmaBackend`.

### Authenticating to a private service

Cloud Run services are private by default, and an org policy enforcing Domain Restricted Sharing will reject an `allUsers` binding outright. Supply an identity token via `DIFFUSIONGEMMA_API_KEY`, which the backend sends as `Authorization: Bearer`:

```bash
# A *user* account token has the gcloud OAuth client ID as its `aud` and will 401;
# `--audiences` is rejected for user credentials. Impersonate a service account
# granted roles/run.invoker instead:
export DIFFUSIONGEMMA_API_KEY="$(gcloud auth print-identity-token \
  --impersonate-service-account=YOUR_SA@PROJECT.iam.gserviceaccount.com \
  --audiences="$DIFFUSIONGEMMA_JEV_URL" --include-email)"
```

### Explicit Python usage

```python
from judgment_base_agent import DiffusionGemmaBackend, JudgmentAgent

agent = JudgmentAgent(
    name="triage_router",
    schema=SupportTriageSchema,
    backend=DiffusionGemmaBackend(
        base_url="https://diffusiongemma-jev-<hash>-uc.a.run.app",
    ),
)
```

---

## 3. Configuration Reference

| Variable | Default | Purpose |
| :-- | :-- | :-- |
| `DIFFUSIONGEMMA_ENGINE` | `transformers` | `transformers` or `vllm`. |
| `DIFFUSIONGEMMA_MODEL_ID` | auto | Overrides model selection. Defaults to the 26B model when CUDA with >= 15 GB VRAM is present, otherwise a small test checkpoint. |
| `DIFFUSIONGEMMA_DENOISING_STEPS` | `1` | Canvas denoising steps. |
| `DIFFUSIONGEMMA_TEMPERATURE` | — | Sampling temperature. |
| `PRELOAD_MODEL_ON_STARTUP` | `true` | Load weights at boot instead of on first request. |
