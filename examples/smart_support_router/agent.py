"""01_smart_support_router — Smart Customer Support & Refund Router (`JudgmentSwitch`).

Why this matters:
Standard LLM routers ALWAYS guess a department—even when a customer sends a vague greeting
or mixes a billing question with a technical bug. `JudgmentSwitch` evaluates both the
target department (`Choice`) and intent clarity (`Noul`) in 1 calibrated System One call.
When confidence is >= 0.75, it routes immediately (`instant_refund`, `tech_support`, or
`cancel_subscription`). When confidence drops below 0.75 (`confidence_floor=0.75`), it
automatically routes to `ask_clarifying_question` instead of sending the user to the wrong team!
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.workflow import START, Workflow
from google.genai import types

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from judgment_base_agent import (
    Choice,
    ChoiceJudgment,
    JudgmentField,
    JudgmentSchema,
    JudgmentSwitch,
    Noul,
    NoulJudgment,
)


class SupportRouterSchema(JudgmentSchema):
    """Evaluates both the target department (`Choice`) and request clarity (`Noul`) in 1 call."""

    route: ChoiceJudgment = JudgmentField(
        Choice(
            instructions="Classify which support department should handle this customer message",
            criteria={
                "instant_refund": (
                    "Customer reports a duplicate charge, accidental double billing, "
                    "or asks for a refund on a specific charge"
                ),
                "tech_support": (
                    "Customer reports a software bug, crash, error code, "
                    "or broken feature that needs engineering help"
                ),
                "cancel_subscription": (
                    "Customer explicitly asks to cancel their subscription "
                    "or close their account"
                ),
            },
        )
    )
    is_clear_and_specific: NoulJudgment = JudgmentField(
        Noul(
            instructions=(
                "Does the customer message clearly specify a single actionable issue "
                "(either a specific refund, a specific technical bug, or a cancellation) "
                "without being vague or mixing multiple conflicting requests?"
            ),
            criteria={
                "true": (
                    "Names one clear issue (e.g. a specific duplicate charge, "
                    "a specific app crash/error code, or a direct cancellation request)"
                ),
                "false": (
                    "Vague greeting/question without details, or mixes multiple "
                    "unrelated issues (e.g. both billing and app acting weird) at once"
                ),
            },
        )
    )


support_router_switch = JudgmentSwitch(
    name="support_router_switch",
    description="Routes customer requests with a strict 0.75 calibrated confidence floor.",
    schema=SupportRouterSchema,
    route_key="route",
    confidence_floor=0.75,
    uncertain_route="ask_clarifying_question",
    output_key="routing_judgment",
)


class SupportTrackHandler(BaseAgent):
    """Formats a clear Markdown response and Calibrated Judgment Scorecard for `adk web`."""

    track_id: str
    badge: str
    headline: str
    customer_reply: str

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        j_data: dict[str, Any] = ctx.session.state.get("routing_judgment") or {}
        route_info = j_data.get("route") or {}
        clear_info = j_data.get("is_clear_and_specific") or {}

        top_choice = route_info.get("choice", self.track_id)
        choice_conf = float(route_info.get("confidence", 0.0))
        probs = route_info.get("probabilities") or {}
        clear_prob = float(clear_info.get("noul", 1.0))
        effective_conf = min(choice_conf, clear_prob)

        prob_lines = "\n".join(
            f"  - `{k}`: **{float(v):.1%}**" for k, v in probs.items()
        )
        markdown = (
            f"### {self.badge} Routed to: `{self.track_id}`\n"
            f"**{self.headline}**\n\n"
            f"> {self.customer_reply}\n\n"
            f"---\n"
            f"#### 📊 Calibrated Judgment Scorecard (1 System One Call)\n"
            f"- **Top Department (`Choice`):** `{top_choice}` (choice confidence: `{choice_conf:.3f}`)\n"
            f"- **Single Clear Request (`Noul`):** `{clear_prob:.3f}`\n"
            f"- **Effective Routing Confidence:** **`{effective_conf:.3f}`** (Floor: `0.750`)\n"
            f"- **Class Probabilities:**\n{prob_lines}\n"
        )
        payload = {
            "selected_track": self.track_id,
            "top_choice": top_choice,
            "choice_confidence": round(choice_conf, 3),
            "clarity_noul": round(clear_prob, 3),
            "effective_confidence": round(effective_conf, 3),
            "customer_reply": self.customer_reply,
        }
        delta = {"active_support_track": payload}
        ctx.session.state.update(delta)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            output=payload,
            actions=EventActions(state_delta=delta),
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=markdown)],
            ),
        )


instant_refund = SupportTrackHandler(
    name="instant_refund",
    description="Automatically processes verified duplicate charge refunds.",
    track_id="instant_refund",
    badge="💸",
    headline="Instant Auto-Refund Approved",
    customer_reply=(
        "We verified the duplicate charge on your card and issued an instant refund. "
        "It will appear on your statement within 2–3 business days."
    ),
)

tech_support = SupportTrackHandler(
    name="tech_support",
    description="Routes reproducible software crashes and error codes to Tier-2 Engineering.",
    track_id="tech_support",
    badge="🛠️",
    headline="Escalated to Tier-2 Technical Support",
    customer_reply=(
        "We captured your error code and crash report and opened a priority engineering ticket. "
        "A specialist is reviewing the logs now."
    ),
)

cancel_subscription = SupportTrackHandler(
    name="cancel_subscription",
    description="Handles subscription cancellation and billing termination requests.",
    track_id="cancel_subscription",
    badge="👋",
    headline="Subscription Cancellation Processed",
    customer_reply=(
        "Your subscription has been scheduled for cancellation and auto-renewal is now turned off. "
        "You will retain access until the end of your current billing cycle."
    ),
)

ask_clarifying_question = SupportTrackHandler(
    name="ask_clarifying_question",
    description="Safe fallback when calibrated confidence < 0.75 (vague or mixed intent).",
    track_id="ask_clarifying_question",
    badge="🤔",
    headline="Safe Fallback Triggered — Asking Clarifying Question Instead of Guessing",
    customer_reply=(
        "I want to make sure I get you to the right specialist! Are you looking to "
        "**(1) review a charge/refund on your bill**, or **(2) troubleshoot a technical error in the app**?"
    ),
)

root_agent = Workflow(
    name="smart_support_router",
    description=(
        "ADK 2.0 Graph Workflow using JudgmentSwitch (confidence_floor=0.75) to route customer "
        "messages across instant refunds, tech support, cancellation, or a clarifying question."
    ),
    edges=[
        (START, support_router_switch),
        (
            support_router_switch,
            {
                "instant_refund": instant_refund,
                "tech_support": tech_support,
                "cancel_subscription": cancel_subscription,
                "ask_clarifying_question": ask_clarifying_question,
            },
        ),
    ],
)

app = App(name="smart_support_router", root_agent=root_agent)
