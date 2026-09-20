"""Tests pinning the container's engine topology.

The container ships two engines:

1. ``transformers`` — our own single-step ``DiffusionGemmaForBlockDiffusion``
   canvas forward pass, implemented in ``server.py``. Works on CPU or GPU and
   has no dependency on any unmerged vLLM work.
2. ``vllm`` — ``vllm serve`` plus the PR's own ``structured_server.py``, which
   already speaks Jev's ``POST /v1/systemone`` wire format and drives the
   canvas correctly via ``diffusion_seed_canvas`` / ``diffusion_read_only``.

``server.py`` must therefore contain **no** vLLM scoring path of its own. The
previous hand-rolled implementation issued one ``/v1/completions`` call per
question (O(N) instead of O(1)) and ignored the PR's canvas extra-args, making
it strictly worse than the upstream server it duplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "deploy" / "diffusiongemma_jev" / "entrypoint.sh"


def test_server_ships_no_hand_rolled_vllm_scoring_path() -> None:
    """server.py must not re-implement vLLM scoring; structured_server.py owns it."""
    from deploy.diffusiongemma_jev import server as container_server

    for removed in ("_vllm_logprobs", "_score_single_question_vllm", "VLLM_BASE_URL", "_match_logprob"):
        assert not hasattr(container_server, removed), (
            f"{removed} is redundant with the PR's structured_server.py and must be deleted"
        )


@pytest.mark.asyncio
async def test_server_uses_the_transformers_canvas_regardless_of_engine_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """server.py only ever runs the transformers canvas, even if ENGINE=vllm leaks in.

    In ``vllm`` mode the entrypoint does not start ``server.py`` at all, so the
    env var must not select a second, divergent code path inside it.
    """
    import httpx

    from deploy.diffusiongemma_jev import server as container_server

    calls: list[str] = []

    def fake_canvas(req: Any) -> dict[str, Any]:
        calls.append("transformers")
        return {
            "model": "stub",
            "engine": "transformers_diffusion_canvas_1step",
            "choices": {},
            "scores": {},
            "nouls": {"escalate": {"noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(container_server, "_evaluate_canvas_transformers_sync", fake_canvas)
    monkeypatch.setenv("DIFFUSIONGEMMA_ENGINE", "vllm")

    transport = httpx.ASGITransport(app=container_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/v1/system_one",
            json={
                "state": {"ticket": "Charged twice on my invoice!"},
                "questions": {"escalate": {"type": "noul", "instructions": "Needs a human?"}},
            },
        )

    assert resp.status_code == 200
    assert calls == ["transformers"]


def test_entrypoint_serves_the_prs_structured_server_in_vllm_mode() -> None:
    """ENGINE=vllm must expose the PR's structured_server.py on $PORT."""
    script = ENTRYPOINT.read_text(encoding="utf-8")

    assert "structured_server.py" in script, "vLLM mode must serve the PR's Jev-compatible server"
    assert "--upstream" in script, "structured_server.py needs the vLLM engine as its upstream"
    assert "--max-logprobs" in script, "diffusion_read_only returns logprobs at every canvas position"
    assert "--enable-prefix-caching" in script, "shared state prefix is re-read on every request"


def test_entrypoint_does_not_start_our_fastapi_server_in_vllm_mode() -> None:
    """The transformers-only FastAPI app must not be launched alongside the PR server."""
    script = ENTRYPOINT.read_text(encoding="utf-8")

    vllm_branch = script.split('if [[ "${ENGINE}" == "vllm" ]]; then', 1)[-1].split("\nfi\n", 1)[0]
    assert "uvicorn server:app" not in vllm_branch, (
        "vLLM mode must exec structured_server.py, not our redundant FastAPI app"
    )
