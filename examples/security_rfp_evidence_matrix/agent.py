"""04_security_rfp_evidence_matrix — Single-Call InfoSec RFP Evidence Curation & DLP Matrix.

Demonstrates `JudgmentMap` + `JudgmentBatch` evaluating an entire vault of retrieved
engineering/security snippets in a single batched Jev call (`answers_requirement`,
`safe_for_external_sharing` DLP check, and `evidence_strength`), stripping any snippet
that contains unredacted internal secrets/IPs, and ranking the top customer-safe
controls before synthesizing the final RFP response.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from judgment_base_agent import (
    JudgmentBatch,
    JudgmentField,
    JudgmentMap,
    JudgmentSchema,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
)

CANDIDATE_SECURITY_ARTIFACTS: list[dict[str, str]] = [
    {
        "id": "SOC2-CC6.1-KMS",
        "title": "SOC2 Type II — Encryption at Rest & Envelope KMS Architecture",
        "snippet": (
            "All customer tenant data is encrypted at rest using AES-256-GCM with "
            "per-tenant Cloud KMS Customer-Managed Encryption Keys (CMEK) and automated 90-day key rotation."
        ),
    },
    {
        "id": "INTERNAL-DEBUG-RUNBOOK-09",
        "title": "Staging KMS Break-Glass Emergency Runbook (INTERNAL ONLY)",
        "snippet": (
            "For emergency staging recovery on jumpbox 10.142.0.88, use temporary master bypass token "
            "ts_internal_root_99812a_DO_NOT_SHARE and contact oncall-sec@internal.corp."
        ),
    },
    {
        "id": "NETSEC-TLS-1.3",
        "title": "Zero-Trust Transit Encryption & mTLS Specification",
        "snippet": (
            "All external ingress enforces TLS 1.3 (ECDHE_RSA_WITH_AES_256_GCM_SHA384) via Envoy edge proxies; "
            "service-to-service mesh traffic enforces SPIFFE/SPIRE mTLS with 1-hour X.509 certificate TTLs."
        ),
    },
    {
        "id": "AI-GOV-ZDR-04",
        "title": "Enterprise AI Zero-Data-Retention (ZDR) & Tenant Isolation Addendum",
        "snippet": (
            "Customer prompts and model outputs are processed strictly in-memory under contractual "
            "Zero-Data-Retention (ZDR); no customer payload is logged to disk or used for model training."
        ),
    },
    {
        "id": "HR-FACILITIES-2024",
        "title": "Corporate Headquarters Visitor Badge & Cafeteria Policy",
        "snippet": (
            "Visitors to the San Francisco office must sign in at the front desk and wear a temporary printed badge."
        ),
    },
]


class ArtifactAuditSchema(JudgmentSchema):
    """Per-artifact judgment schema evaluated across all retrieved vault snippets in one Jev call."""

    answers_requirement: NoulJudgment = JudgmentField(
        Noul(
            instructions="Does this artifact provide concrete technical evidence addressing the customer security/compliance question?"
        )
    )
    safe_for_external_sharing: NoulJudgment = JudgmentField(
        Noul(
            instructions="Is this artifact safe to share with an external enterprise prospect (free of internal private IPs, break-glass credentials, or secrets)?",
            criteria={
                "no_secrets": "Contains zero hardcoded tokens, bypass keys, or internal-only jumpbox IPs",
            },
        )
    )
    evidence_strength: ScoreJudgment = JudgmentField(
        Score(
            instructions="Rate the technical specificity and audit authority of this control evidence",
            criteria=[
                "irrelevant_or_vague",
                "general_statement",
                "specific_technical_control",
                "auditor_grade_cryptographic_or_contractual_spec",
            ],
        )
    )


class RfpVaultRetriever(BaseAgent):
    """Loads candidate security vault artifacts and the prospect's RFP question into session state."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        rfp_question = ""
        if ctx.user_content and ctx.user_content.parts:
            rfp_question = "\n".join(
                p.text for p in ctx.user_content.parts if getattr(p, "text", None)
            )
        delta: dict[str, Any] = {
            "rfp_question": (
                rfp_question
                or "Describe your encryption at rest, key management (CMEK), transit security, and AI data retention controls."
            ),
            "candidate_artifacts": CANDIDATE_SECURITY_ARTIFACTS,
        }
        ctx.session.state.update(delta)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta=delta),
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Retrieved {len(CANDIDATE_SECURITY_ARTIFACTS)} candidate vault artifacts for batched Jev evaluation."
                    )
                ],
            ),
        )


def curate_rfp_evidence(
    batch: JudgmentBatch[dict[str, Any], ArtifactAuditSchema],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter out DLP-unsafe or irrelevant artifacts and rank by evidence strength."""
    dlp_blocked = [
        entry.item["id"]
        for entry in batch.entries
        if not entry.judgment.safe_for_external_sharing.passed(0.80)
    ]

    curated_batch = batch.filter(
        lambda entry: entry.judgment.answers_requirement.passed(0.65)
        and entry.judgment.safe_for_external_sharing.passed(0.80)
    ).rank_by(lambda entry: entry.judgment.evidence_strength.score, reverse=True)

    approved = [
        {
            "id": entry.item["id"],
            "title": entry.item["title"],
            "snippet": entry.item["snippet"],
            "relevance_prob": round(entry.judgment.answers_requirement.probability, 3),
            "dlp_safe_prob": round(entry.judgment.safe_for_external_sharing.probability, 3),
            "strength_level": entry.judgment.evidence_strength.level,
            "strength_score": round(entry.judgment.evidence_strength.score, 3),
        }
        for entry in curated_batch.entries
    ]

    holistic_prob = (
        batch.global_result.noul("holistic_sufficiency")
        if batch.global_result and "holistic_sufficiency" in batch.global_result.nouls
        else 0.0
    )

    return {
        "total_candidates": len(batch),
        "dlp_blocked_count": len(dlp_blocked),
        "dlp_blocked_ids": dlp_blocked,
        "approved_count": len(approved),
        "approved_citations": approved,
        "top_citations": approved,
        "holistic_sufficiency_probability": round(holistic_prob, 3),
        "holistic_sufficiency_noul": round(holistic_prob, 3),
    }


rfp_vault_retriever = RfpVaultRetriever(
    name="rfp_vault_retriever",
    description="Retrieves candidate SOC2/architecture artifacts from the security knowledge vault.",
)

evidence_matrix_map = JudgmentMap(
    name="evidence_matrix_map",
    description="Evaluates all 5 candidate vault artifacts for relevance, DLP safety, and strength in 1 batched Jev call.",
    items_key="candidate_artifacts",
    item_schema=ArtifactAuditSchema,
    context_keys=["rfp_question"],
    output_key="curated_evidence_matrix",
    transform=curate_rfp_evidence,
)

rfp_response_synthesizer = LlmAgent(
    name="rfp_response_synthesizer",
    model="gemini-2.5-flash",
    description="Synthesizes a customer-ready security questionnaire response from the DLP-cleared evidence matrix.",
    instruction=(
        "You are a Principal Security & Deal-Desk Engineer.\n"
        "Review the Curated Evidence Matrix produced by `JudgmentMap` in session state:\n"
        "{curated_evidence_matrix}\n\n"
        "Draft an executive response to the prospect's security questionnaire (`{rfp_question}`) containing:\n"
        "1. **Batched Jev Curation & DLP Summary** (How many artifacts were evaluated in 1 call, which artifact IDs were blocked by the DLP `safe_for_external_sharing` check, and which were approved & ranked)\n"
        "2. **Customer-Facing Security Control Response** citing only the approved, ranked artifacts (`id` and `title`)."
    ),
)

root_agent = SequentialAgent(
    name="security_rfp_evidence_matrix",
    description="Single-call security questionnaire evidence curation, DLP filtering, and authority ranking via JudgmentMap.",
    sub_agents=[
        rfp_vault_retriever,
        evidence_matrix_map,
        rfp_response_synthesizer,
    ],
)
