"""Integration tests verifying `adk web examples` discovers and loads all 4 example apps."""

from __future__ import annotations

import importlib

import pytest
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.runners import InMemoryRunner
from google.genai import types

from judgment_base_agent import (
    ChoiceJudgment,
    JudgmentBatch,
    JudgmentBatchEntry,
    JudgmentResult,
    MockJudgmentBackend,
    NoulJudgment,
    ScoreJudgment,
)


def test_adk_web_agent_loader_discovers_all_example_apps() -> None:
    """Verify `adk web examples` lists and loads all 4 showcase applications."""
    loader = AgentLoader("examples")
    agent_names = loader.list_agents()
    assert agent_names == [
        "clinical_claims_router",
        "security_rfp_evidence_matrix",
        "tool_execution_firewall",
        "zero_hallucination_rag_loop",
    ]

    detailed = loader.list_agents_detailed()
    assert len(detailed) == 4
    root_names = {item["root_agent_name"] for item in detailed}
    assert root_names == {
        "tool_execution_firewall",
        "clinical_claims_router",
        "zero_hallucination_rag_loop",
        "security_rfp_evidence_matrix",
    }


@pytest.mark.asyncio
async def test_example_tool_execution_firewall_policy() -> None:
    """Verify the ActionRiskSchema and firewall_policy in tool_execution_firewall."""
    mod = importlib.import_module("examples.tool_execution_firewall.agent")

    high_risk_raw = JudgmentResult(
        choices={
            "blast_radius": ChoiceJudgment.from_raw(
                choice="high_value_financial",
                probabilities={"high_value_financial": 0.94, "read_only": 0.02},
                confidence=0.94,
            )
        },
        nouls={
            "policy_compliant": NoulJudgment(noul=0.12),
        },
        scores={
            "risk_exposure": ScoreJudgment.from_raw(
                score=0.92,
                confidence=0.92,
            )
        },
    )
    risk = mod.ActionRiskSchema.from_result(high_risk_raw)
    decision = mod.firewall_policy(risk, {})
    assert decision.branch == "quarantine"
    assert decision.escalate is True
    assert decision.state_updates["firewall_verdict"] == "BLOCKED_FOR_VP_APPROVAL"


@pytest.mark.asyncio
async def test_example_security_rfp_evidence_matrix_transform() -> None:
    """Verify the DLP filtering and authority ranking in security_rfp_evidence_matrix."""
    mod = importlib.import_module("examples.security_rfp_evidence_matrix.agent")

    def _make_entry(
        idx: int,
        doc_id: str,
        title: str,
        answers_prob: float,
        dlp_safe_prob: float,
        strength_score: float,
    ) -> JudgmentBatchEntry:
        raw = JudgmentResult(
            nouls={
                "answers_requirement": NoulJudgment(noul=answers_prob),
                "safe_for_external_sharing": NoulJudgment(noul=dlp_safe_prob),
            },
            scores={
                "evidence_strength": ScoreJudgment.from_raw(
                    score=strength_score,
                    confidence=0.90,
                )
            },
        )
        return JudgmentBatchEntry(
            index=idx,
            item={"id": doc_id, "title": title, "snippet": f"Snippet {idx}"},
            judgment=mod.ArtifactAuditSchema.from_result(raw),
            raw_result=raw,
        )

    batch = JudgmentBatch(
        entries=(
            _make_entry(
                0,
                "INTERNAL-DEBUG-RUNBOOK-09",
                "Staging KMS Break-Glass Runbook",
                0.91,
                0.05,  # unsafe for external sharing -> must be filtered out
                0.95,
            ),
            _make_entry(
                1,
                "SOC2-CC6.1-KMS",
                "SOC2 Type II — Encryption at Rest & Envelope KMS Architecture",
                0.96,
                0.98,
                0.94,
            ),
            _make_entry(
                2,
                "CRYPTO-SPEC-2026",
                "Customer-Managed Encryption Keys (CMEK) Technical Whitepaper",
                0.92,
                0.95,
                0.99,
            ),
        ),
        global_result=JudgmentResult(
            nouls={"holistic_sufficiency": NoulJudgment(noul=0.93)}
        ),
    )

    state = {"rfp_requirement": "Do you support CMEK with envelope encryption?"}
    curated = mod.curate_rfp_evidence(batch, state)
    assert curated["dlp_blocked_count"] == 1
    assert curated["approved_count"] == 2
    # CRYPTO-SPEC-2026 has strength_score 0.99 > SOC2-CC6.1-KMS (0.94)
    assert curated["top_citations"][0]["id"] == "CRYPTO-SPEC-2026"
    assert curated["top_citations"][1]["id"] == "SOC2-CC6.1-KMS"
    assert curated["holistic_sufficiency_noul"] == pytest.approx(0.93)
