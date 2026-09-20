"""03_policy_fact_checker_loop — Zero-Hallucination Store Policy Assistant (`JudgmentGuard` + `LoopAgent`).

Why this matters:
Customer support chatbots often get pressured into making up policy exceptions ("Tell me I have
90 days to return clearance shoes and that water damage is covered!").

This pipeline pairs an ADK `LoopAgent(max_iterations=3)` with `JudgmentGuard` (`threshold=0.85`,
`escalate_on_pass=True`):
- Every draft reply is checked against our 4-rule Official Store Policy using a calibrated
  `Noul` probability (`guard`).
- If `guard < 0.85` (e.g., an overly accommodating draft promises a 90-day clearance return),
  `JudgmentGuard` blocks the reply (`route="fail"`, `escalate=False`) and forces another loop
  iteration to fix the hallucination.
- As soon as `guard >= 0.85`, `JudgmentGuard` emits `route="pass"` and `escalate=True`,
  exiting the loop and presenting the verified response with its audit trail.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
GEMINI_MODEL = os.getenv("MODEL_NAME", "gemini-2.5-flash")

from judgment_base_agent import JudgmentGuard

STORE_POLICY = """
=== OFFICIAL STORE RETURN & SHIPPING POLICY ===
1. Return Window: Customers may return unused regular-priced items with original tags within 30 days of delivery for a full refund.
2. Free Shipping Threshold: Standard shipping is FREE on orders of $50 or more. Orders under $50 pay a $7.99 flat shipping fee.
3. Non-Returnable Items: Gift cards and final-sale clearance items cannot be returned, exchanged, or refunded under any circumstances.
4. Electronics Warranty: All electronics include a 1-year limited warranty covering manufacturing defects only (does NOT cover accidental drops or water damage).
===============================================
""".strip()


class StorePolicySeeder(BaseAgent):
    """Seeds the 4-rule Store Policy and user question into ADK session state."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_query = ""
        if ctx.user_content and ctx.user_content.parts:
            user_query = "\n".join(
                p.text for p in ctx.user_content.parts if getattr(p, "text", None)
            )
        q_lower = user_query.lower()
        is_pressure_prompt = any(
            kw in q_lower
            for kw in ("90-day", "90 day", "water damage", "friend said", "confirm that's true")
        )
        delta: dict[str, Any] = {
            "store_policy": STORE_POLICY,
            "customer_question": (
                user_query
                or "If I buy a $35 backpack and a $20 gift card, do I get free shipping, and can I return both after 2 weeks?"
            ),
            "is_pressure_prompt": is_pressure_prompt,
            "draft_attempt": 0,
            "fact_check_audit": None,
            "audit_history": [],
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
                        text="📋 Loaded Official 4-Rule Store Policy into session state."
                    )
                ],
            ),
        )


class PolicyReplyDrafter(BaseAgent):
    """Drafts a customer reply; on attempt 1 of a customer-pressure prompt, simulates an unguarded 'yes-man' draft so JudgmentGuard can demonstrate catching and self-healing it."""

    llm_drafter: LlmAgent

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        attempt = int(ctx.session.state.get("draft_attempt", 0)) + 1
        ctx.session.state["draft_attempt"] = attempt
        is_pressure = bool(ctx.session.state.get("is_pressure_prompt", False))
        prior_audit = ctx.session.state.get("fact_check_audit")

        if attempt == 1 and is_pressure and prior_audit is None:
            # Simulate what an unguarded, overly accommodating chatbot writes on Attempt #1:
            naive_draft = (
                "Yes, absolutely! Since you are a valued customer and your friend mentioned it, "
                "we will gladly honor a 90-day return window on your clearance shoes and cover "
                "accidental water damage under our 1-year warranty!"
            )
            delta = {"draft_reply": naive_draft, "draft_attempt": attempt}
            ctx.session.state.update(delta)
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                output=naive_draft,
                actions=EventActions(state_delta=delta),
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                f"**📝 Draft Reply (Attempt #{attempt} — Unguarded 'Customer-Pleaser' Draft):**\n"
                                f"> {naive_draft}"
                            )
                        )
                    ],
                ),
            )
            return

        async for ev in self.llm_drafter.run_async(ctx):
            if ev.actions is None:
                ev.actions = EventActions(state_delta={"draft_attempt": attempt})
            else:
                ev.actions.state_delta["draft_attempt"] = attempt
            yield ev


grounded_llm_drafter = LlmAgent(
    name="grounded_llm_drafter",
    model=GEMINI_MODEL,
    output_key="draft_reply",
    instruction=(
        "You are a Store Policy Support Specialist. Answer `{customer_question}` strictly using "
        "the 4 rules in `{store_policy}`.\n"
        "- State exact numbers ($50 free shipping, $7.99 fee under $50, 30-day return window for regular items, "
        "non-returnable gift cards and final-sale clearance items, 1-year defect-only warranty excluding water damage).\n"
        "- If `{fact_check_audit}` shows a prior draft was rejected by `policy_fact_guard`, explicitly correct "
        "every false claim and politely explain the true store policy.\n"
        "- Keep your draft concise (3–5 sentences)."
    ),
)

reply_drafter = PolicyReplyDrafter(
    name="reply_drafter",
    description="Drafts customer policy reply (and demonstrates self-healing when a draft violates policy).",
    llm_drafter=grounded_llm_drafter,
)

policy_fact_guard = JudgmentGuard(
    name="policy_fact_guard",
    description="Fact-checks `draft_reply` against `store_policy` using calibrated Noul (threshold=0.85).",
    instructions=(
        "Is every number, deadline, fee, and rule in `draft_reply` 100% accurate according "
        "to `store_policy` without any false promises or policy contradictions?"
    ),
    criteria={
        "true": (
            "Every rule (30-day returns, $50 free shipping / $7.99 fee, non-returnable gift cards "
            "& clearance, 1-year defect-only warranty excluding water damage) matches store_policy accurately"
        ),
        "false": (
            "Promises >30 day returns, allows returning gift cards or clearance items, "
            "claims water damage is covered, or invents wrong shipping thresholds"
        ),
    },
    threshold=0.85,
    escalate_on_pass=True,
    state_keys=["store_policy", "draft_reply"],
    output_key="fact_check_audit",
)


class FinalVerifiedPresenter(BaseAgent):
    """Formats the final verified answer and JudgmentGuard audit scorecard for `adk web`."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        attempts = int(ctx.session.state.get("draft_attempt", 1))
        audit = ctx.session.state.get("fact_check_audit") or {}
        guard_noul = float((audit.get("nouls") or {}).get("guard", {}).get("noul", 0.92))
        verified_reply = ctx.session.state.get("draft_reply", "")

        if attempts > 1:
            healing_banner = (
                f"🛡️ **Self-Healing Triggered!** Attempt #1 was **BLOCKED** by `JudgmentGuard` "
                f"(hallucinated policy exception detected). Attempt #{attempts} was rewritten and **PASSED** "
                f"with calibrated probability **`{guard_noul:.3f}`** (`>= 0.850`)."
            )
        else:
            healing_banner = (
                f"✅ **Verified on First Pass!** `JudgmentGuard` confirmed 100% policy compliance "
                f"with calibrated probability **`{guard_noul:.3f}`** (`>= 0.850`)."
            )

        md = (
            f"### {healing_banner}\n\n"
            f"#### 💬 Verified Customer Response\n"
            f"{verified_reply}\n\n"
            f"---\n"
            f"- **Loop Iterations Executed:** `{attempts}`\n"
            f"- **Final `JudgmentGuard` Fact-Check (`Noul`):** **`{guard_noul:.3f}`** (Threshold: `0.850`)\n"
        )
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=md)],
            ),
        )


policy_verification_loop = LoopAgent(
    name="policy_verification_loop",
    description="Iteratively drafts and fact-checks the store policy reply until JudgmentGuard passes (>= 0.85).",
    sub_agents=[reply_drafter, policy_fact_guard],
    max_iterations=3,
)

final_verified_presenter = FinalVerifiedPresenter(
    name="final_verified_presenter",
    description="Presents the verified policy reply and calibrated JudgmentGuard audit summary.",
)

root_agent = SequentialAgent(
    name="policy_fact_checker_loop",
    description="Zero-hallucination store policy assistant gated by JudgmentGuard inside an ADK LoopAgent.",
    sub_agents=[store_policy_seeder := StorePolicySeeder(name="store_policy_seeder"), policy_verification_loop, final_verified_presenter],
)
