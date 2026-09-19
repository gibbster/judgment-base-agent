"""03_zero_hallucination_rag_loop — Self-Healing Financial & Legal Covenant Synthesizer.

Demonstrates `JudgmentGuard` (`threshold=0.88`, `escalate_on_pass=True`) inside an
ADK `LoopAgent(max_iterations=3)`. A Gemini analyst drafts an executive brief against
authoritative credit-facility covenants, and `JudgmentGuard` audits strict numerical
and citation grounding with Jev. If grounding probability < 0.88, the loop continues;
as soon as grounding probability >= 0.88, `JudgmentGuard` emits `escalate=True` to
terminate the `LoopAgent` and release the verified brief.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from judgment_base_agent import JudgmentGuard

AUTHORITATIVE_CREDIT_COVENANTS = """
=== CREDIT AGREEMENT SCHEDULE 7.1 (SENIOR SECURED FACILITY — ACME CORP) ===
1. Maximum Consolidated Net Leverage Ratio (Section 7.1(a)):
   - Shall not exceed 3.25x at the end of any fiscal quarter.
   - Temporary Acquisition Step-Up: Permitted up to 3.75x for two (2) consecutive quarters following a Material Permitted Acquisition (> $25.0M).
2. Minimum Unrestricted Liquidity Covenant (Section 7.1(b)):
   - Borrower must maintain at least $15,000,000 in unrestricted cash and cash equivalents at all times.
3. Equity Cure Right (Section 8.2):
   - Exercised within ten (10) business days after delivery of the Compliance Certificate.
   - Limited to at most two (2) cures in any four-quarter rolling period, and four (4) cures over the life of the Facility.
4. Change of Control Notification (Section 6.4):
   - Written notice to Administrative Agent required at least fifteen (15) business days prior to consummation.
===========================================================================
""".strip()


class CovenantVaultSeeder(BaseAgent):
    """Seeds authoritative financial covenant records into ADK session state."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_query = ""
        if ctx.user_content and ctx.user_content.parts:
            user_query = "\n".join(
                p.text for p in ctx.user_content.parts if getattr(p, "text", None)
            )
        delta: dict[str, Any] = {
            "source_covenant_vault": AUTHORITATIVE_CREDIT_COVENANTS,
            "user_covenant_query": user_query or "Summarize the leverage, liquidity, and equity cure covenants.",
            "grounding_audit": "Initial pass — no prior audit failures.",
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
                        text="Loaded Schedule 7.1 Senior Secured Credit Facility Covenants into session state."
                    )
                ],
            ),
        )


covenant_vault_seeder = CovenantVaultSeeder(
    name="covenant_vault_seeder",
    description="Loads the authoritative Schedule 7.1 Credit Agreement covenants into session state.",
)

memo_drafter = LlmAgent(
    name="memo_drafter",
    model="gemini-2.5-flash",
    description="Drafts or refines an executive covenant memo grounded strictly in Schedule 7.1.",
    output_key="draft_covenant_memo",
    instruction=(
        "You are a Senior Credit & Legal Analyst.\n"
        "Answer the user query strictly using the Authoritative Credit Agreement Schedule below.\n"
        "Do NOT invent any numbers, cure periods, or thresholds not present in the schedule.\n"
        "Cite exact section numbers (e.g., Section 7.1(a), Section 8.2).\n\n"
        "Authoritative Covenant Schedule:\n{source_covenant_vault}\n\n"
        "User Query:\n{user_covenant_query}\n\n"
        "Prior Calibrated Guard Audit (if revising):\n{grounding_audit}"
    ),
)

grounding_guard = JudgmentGuard(
    name="grounding_guard",
    description="Audits numerical fidelity and section grounding with Jev (threshold=0.88, escalate_on_pass=True).",
    instructions=(
        "Is every financial ratio, dollar threshold, business-day deadline, and section citation "
        "in `draft_covenant_memo` 100% supported by `source_covenant_vault` with zero hallucinated terms?"
    ),
    criteria={
        "numerical_fidelity": "Every ratio (3.25x / 3.75x), dollar figure ($15M / $25M), and day count (10 / 15 business days) matches Schedule 7.1",
        "zero_speculation": "Contains no unverified legal interpretations or external assumptions",
    },
    threshold=0.88,
    escalate_on_pass=True,
    state_keys=["source_covenant_vault", "draft_covenant_memo"],
    output_key="grounding_audit",
)

synthesis_loop = LoopAgent(
    name="self_healing_synthesis_loop",
    description="Iteratively drafts and verifies the covenant memo until JudgmentGuard passes (probability >= 0.88).",
    sub_agents=[memo_drafter, grounding_guard],
    max_iterations=3,
)

root_agent = SequentialAgent(
    name="zero_hallucination_rag_loop",
    description="Self-healing legal/financial synthesis pipeline gated by JudgmentGuard.",
    sub_agents=[covenant_vault_seeder, synthesis_loop],
)
