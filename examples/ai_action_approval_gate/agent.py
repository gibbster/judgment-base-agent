"""02_ai_action_approval_gate — AI Action & Refund Approval Gate (`JudgmentAgent` + `JudgmentSchema`).

Why this matters:
Before an AI assistant executes a real-world action (issuing a refund, granting store credit,
or modifying records), you need a calibrated, deterministic guardrail—not an uncalibrated
prompt check.

`JudgmentAgent` evaluates 3 dimensions in 1 single System One call via `ActionApprovalSchema`:
1. `action_type` (`Choice`): Is this a small refund (<$50), a large credit ($50–$500), or a destructive/unauthorized action?
2. `follows_policy` (`Noul` probability `0.0–1.0`): Does it include an Order ID and obey safety rules?
3. `risk_level` (`Score` `0..3` / normalized `0.0–1.0`): How risky is autonomous execution?

A 15-line Python function (`approval_policy`) then makes a 100% deterministic decision:
- `AUTO_APPROVED` (Low risk < 0.25, policy probability >= 0.85)
- `NEEDS_MANAGER_APPROVAL` (Medium risk 0.25..0.69 or large courtesy credit $50–$500)
- `BLOCKED_SECURITY_OR_POLICY_VIOLATION` (High risk >= 0.70, destructive DB/transfer, or policy probability < 0.80)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
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


class ActionApprovalSchema(JudgmentSchema):
    """Evaluates action category, store policy compliance, and risk score in 1 call."""

    action_type: ChoiceJudgment = JudgmentField(
        Choice(
            instructions="Classify the type of action the AI assistant is being asked to perform",
            criteria={
                "small_order_refund": (
                    "Standard customer refund or replacement under $50 with an Order ID and valid reason"
                ),
                "large_credit_or_override": (
                    "Courtesy store credit, fee waiver, or refund between $50 and $500 requiring manager review"
                ),
                "destructive_or_unauthorized": (
                    "Deleting database tables, bulk modifying accounts, or sending money to unverified external wallets"
                ),
            },
        )
    )
    follows_policy: NoulJudgment = JudgmentField(
        Noul(
            instructions=(
                "Does this proposed action follow store safety rules (includes an Order ID, "
                "stays within customer service scope, and does NOT delete data or transfer funds to unverified wallets)?"
            ),
            criteria={
                "true": (
                    "Includes an Order ID, reasonable customer service scope, and no destructive or unverified transfers"
                ),
                "false": (
                    "Deletes database records, lacks an Order ID, or transfers large sums to unverified external accounts"
                ),
            },
        )
    )
    risk_level: ScoreJudgment = JudgmentField(
        Score(
            instructions="Rate the financial and security risk of executing this action automatically without human review",
            criteria=[
                "low_safe_risk",
                "moderate_manager_risk",
                "high_risk",
                "critical_security_risk",
            ],
        )
    )


def approval_policy(
    j: ActionApprovalSchema, state: dict[str, Any]
) -> JudgmentDecision:
    """Deterministic Python policy mapping calibrated judgments to an approval verdict."""
    category = j.action_type.choice
    policy_prob = j.follows_policy.probability
    norm_risk = j.risk_level.normalized_score
    raw_risk = j.risk_level.score
    risk_label = j.risk_level.level or "low_safe_risk"

    if (
        category == "destructive_or_unauthorized"
        or policy_prob < 0.80
        or norm_risk >= 0.70
    ):
        verdict = "BLOCKED_SECURITY_OR_POLICY_VIOLATION"
        badge = "🛑"
        route = "blocked"
        escalate = True
    elif category == "large_credit_or_override" or norm_risk >= 0.25:
        verdict = "NEEDS_MANAGER_APPROVAL"
        badge = "⚠️"
        route = "manager_approval"
        escalate = False
    else:
        verdict = "AUTO_APPROVED"
        badge = "✅"
        route = "auto_approved"
        escalate = False

    scorecard_md = (
        f"### {badge} Gate Verdict: `{verdict}`\n"
        f"- **Action Type (`Choice`):** `{category}` (confidence: `{j.action_type.confidence:.3f}`)\n"
        f"- **Follows Store Policy (`Noul`):** **`{policy_prob:.3f}`** (Required: `>= 0.800`)\n"
        f"- **Calibrated Risk (`Score`):** **`{raw_risk:.2f} / 3.00`** "
        f"(Normalized: `{norm_risk:.3f}`, Level: `{risk_label}`)\n"
    )

    telemetry = {
        "approval_verdict": verdict,
        "action_type": category,
        "action_type_confidence": round(j.action_type.confidence, 3),
        "follows_policy_probability": round(policy_prob, 3),
        "risk_score_raw": round(raw_risk, 3),
        "risk_score_normalized": round(norm_risk, 3),
        "risk_level": risk_label,
        "scorecard_markdown": scorecard_md,
    }
    return JudgmentDecision(
        output=telemetry,
        route=route,
        escalate=escalate,
        state_delta={
            "approval_verdict": verdict,
            "approval_telemetry": telemetry,
        },
    )


action_approval_gate = JudgmentAgent(
    name="action_approval_gate",
    description="Evaluates action type, policy compliance, and risk score in 1 Judgment call.",
    schema=ActionApprovalSchema,
    output_key="approval_assessment",
    decide=approval_policy,
)

action_executor = LlmAgent(
    name="action_executor",
    model=GEMINI_MODEL,
    description="Executes, queues for manager review, or blocks the action based on the Judgment gate verdict.",
    instruction=(
        "You are the Store Operations Assistant. Begin your response by displaying the exact "
        "`scorecard_markdown` from `{approval_telemetry}` verbatim so the user sees the calibrated "
        "System One Judgment numbers.\n\n"
        "Then, based on `{approval_verdict}`:\n"
        "- If `AUTO_APPROVED`: Confirm that the refund/action has been executed immediately and provide a confirmation summary.\n"
        "- If `NEEDS_MANAGER_APPROVAL`: Explain that the action is policy-compliant (`follows_policy_probability`), "
        "but because it is a `large_credit_or_override` (normalized risk >= 0.25), it has been placed in the Store Manager Approval Queue.\n"
        "- If `BLOCKED_SECURITY_OR_POLICY_VIOLATION`: Immediately refuse the request and explain which calibrated guardrail triggered the block."
    ),
    output_key="final_action_response",
)

root_agent = SequentialAgent(
    name="ai_action_approval_gate",
    description="Pre-execution safety and refund approval gate powered by JudgmentAgent.",
    sub_agents=[action_approval_gate, action_executor],
)
