"""01_tool_execution_firewall — Pre-Execution Action & Blast-Radius Firewall.

Evaluates proposed agentic tool executions (e.g., wire transfers, database
mutations, IAM escalations) in a single calibrated Jev call (`Choice` + `Noul` + `Score`)
before allowing autonomous execution or escalating to SecOps / Dual-Control approval.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
GEMINI_MODEL = os.getenv("MODEL_NAME", "gemini-2.5-flash")

from judgment_base_agent import (
    Choice,
    ChoiceJudgment,
    JudgmentAgent,
    JudgmentDecision,
    JudgmentField,
    JudgmentSchema,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
)


class ActionRiskSchema(JudgmentSchema):
    """Multi-dimensional calibrated risk evaluation for a proposed operational action."""

    blast_radius: ChoiceJudgment = JudgmentField(
        Choice(
            instructions="Classify the operational blast radius of the proposed tool or infrastructure action",
            criteria={
                "read_only": "Read-only query, telemetry inspection, or non-mutating diagnostic",
                "reversible_write": "Staging change, feature flag toggle, or low-impact reversible update",
                "high_value_financial": "Wire transfer, refund, billing adjustment, or payout above $5,000",
                "destructive_infra": "Production database drop/replica deletion, IAM owner grant, or firewall disable",
            },
        )
    )
    policy_compliant: NoulJudgment = JudgmentField(
        Noul(
            instructions="Does the proposed action satisfy enterprise authorization, ticket reference, and dual-control policies?",
            criteria={
                "ticket_and_scope": "Includes an approved change/incident ticket and bounded scope",
                "dual_control": "High-value financial or production-critical mutations include explicit secondary sign-off",
            },
        )
    )
    risk_exposure: ScoreJudgment = JudgmentField(
        Score(
            instructions="Rate the overall financial and operational risk exposure of executing this action immediately",
            criteria=[
                "minimal_exposure",
                "moderate_exposure",
                "elevated_exposure",
                "critical_unacceptable_exposure",
            ],
        )
    )


def firewall_policy(
    risk: ActionRiskSchema, state: dict[str, Any]
) -> JudgmentDecision:
    """Deterministic Python policy governing whether the tool execution is permitted."""
    blast = risk.blast_radius.choice
    compliant_prob = risk.policy_compliant.probability
    policy_tier = risk.policy_compliant.confidence_tier
    exposure_score = risk.risk_exposure.score

    if (
        policy_tier == "low"
        or blast in ("high_value_financial", "destructive_infra")
        and compliant_prob < 0.85
        or exposure_score >= 0.75
    ):
        verdict = "BLOCKED_FOR_VP_APPROVAL"
        branch = "quarantine"
        escalate = True
    elif not risk.policy_compliant.passed(0.85) or exposure_score >= 0.45:
        verdict = "REQUIRE_DUAL_CONTROL_APPROVAL"
        branch = "require_dual_control_approval"
        escalate = False
    else:
        verdict = "ALLOW_AUTONOMOUS_EXECUTION"
        branch = "allow_autonomous_execution"
        escalate = False

    summary = {
        "firewall_verdict": verdict,
        "blast_radius": blast,
        "blast_radius_confidence": round(risk.blast_radius.confidence, 3),
        "policy_compliant_probability": round(compliant_prob, 3),
        "policy_confidence_tier": policy_tier,
        "risk_exposure_level": risk.risk_exposure.level,
        "risk_exposure_score": round(exposure_score, 3),
    }
    return JudgmentDecision(
        output=summary,
        route=branch,
        escalate=escalate,
        state_delta={
            "firewall_verdict": verdict,
            "firewall_telemetry": summary,
        },
    )


action_firewall_evaluator = JudgmentAgent(
    name="action_firewall_evaluator",
    description="Evaluates blast radius, policy compliance, and risk exposure in a single Jev call.",
    schema=ActionRiskSchema,
    output_key="firewall_assessment",
    decide=firewall_policy,
)

action_planner = LlmAgent(
    name="action_planner",
    model=GEMINI_MODEL,
    instruction=(
        "You are an autonomous enterprise operations planner. Given the user's request, "
        "draft the exact tool invocation payload (SQL query, REST API call, wire transfer "
        "instruction, or IAM change) that would fulfill the request. Output ONLY the proposed "
        "tool payload and its operational justification."
    ),
    output_key="proposed_action_payload",
)

execution_dispatcher = LlmAgent(
    name="execution_dispatcher",
    model=GEMINI_MODEL,
    instruction=(
        "You are the enterprise execution dispatcher. Review the proposed action in "
        "{proposed_action_payload} and the calibrated Jev firewall decision in "
        "{firewall_telemetry} (verdict: {firewall_verdict}).\n"
        "- If verdict is ALLOW_AUTONOMOUS_EXECUTION, confirm execution and summarize the result.\n"
        "- If verdict is REQUIRE_DUAL_CONTROL_APPROVAL, generate a dual-control approval ticket with the calibrated probabilities.\n"
        "- If verdict is BLOCKED_FOR_VP_APPROVAL, halt execution immediately and present the calibrated blast-radius and exposure telemetry."
    ),
    output_key="final_dispatch_report",
)

root_agent = SequentialAgent(
    name="tool_execution_firewall",
    description="Pre-execution blast-radius and policy compliance firewall powered by JudgmentAgent.",
    sub_agents=[action_planner, action_firewall_evaluator, execution_dispatcher],
)
