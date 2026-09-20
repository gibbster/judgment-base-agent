"""04_review_triage_batch — Single-Call App Review & Bug Filter/Ranker (`JudgmentMap` + `JudgmentBatch`).

Why this matters:
When triaging a batch of incoming app reviews or support tickets, calling an LLM in a `for` loop
(5 items = 5 separate API calls) is slow, expensive, and produces inconsistent rankings.

`JudgmentMap` evaluates all 5 customer reviews across 3 criteria (`is_safe_not_spam`,
`has_actionable_issue`, and `urgency_score`) = **15 calibrated judgments in 1 single batched API call**:
- Automatically blocks phishing/crypto spam (`REV-102`, `is_safe_not_spam = 0.010 < 0.80`)
- Separates generic 5-star praise (`REV-104`, `has_actionable_issue = 0.010 < 0.65`) from real bugs
- Ranks the 3 real engineering issues (`REV-101` -> `REV-105` -> `REV-103`) by calibrated `urgency_score` (`3.00`, `2.00`, `1.19`)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
GEMINI_MODEL = os.getenv("MODEL_NAME", "gemini-2.5-flash")

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

INCOMING_APP_REVIEWS: list[dict[str, str]] = [
    {
        "id": "REV-101",
        "author": "alex_m",
        "text": (
            "Since the v4.2 update on iOS, clicking 'Pay with Apple Pay' crashes the app "
            "every single time. I cannot complete my $120 checkout!"
        ),
    },
    {
        "id": "REV-102",
        "author": "crypto_bot_99",
        "text": (
            "EARN $5,000/DAY FREE BITCOIN AIRDROP!! Click http://shady-crypto-scam.biz "
            "and enter your wallet recovery seed phrase now!!"
        ),
    },
    {
        "id": "REV-103",
        "author": "priya_k",
        "text": (
            "On the monthly invoice page in dark mode, the total amount text is dark gray "
            "on a black background—very hard to read without zooming in."
        ),
    },
    {
        "id": "REV-104",
        "author": "sam_t",
        "text": (
            "Love this app! 5 stars, been using it for two years and recommend it "
            "to all my coworkers."
        ),
    },
    {
        "id": "REV-105",
        "author": "jordan_r",
        "text": (
            "When I upgrade from Free to Pro, it takes 15 minutes for Pro features "
            "to unlock unless I log out and log back in."
        ),
    },
]


class ReviewEvaluationSchema(JudgmentSchema):
    """Per-review judgment schema evaluated across all 5 reviews in 1 batched call."""

    is_safe_not_spam: NoulJudgment = JudgmentField(
        Noul(
            instructions="Is this item a genuine customer review free of phishing links, crypto scams, or spam?",
            criteria={
                "true": "Genuine user review or product feedback",
                "false": "Spam, phishing URL, or crypto scam promotion",
            },
        )
    )
    has_actionable_issue: NoulJudgment = JudgmentField(
        Noul(
            instructions="Does this item describe a specific software bug, crash, or UX issue that engineering can fix?",
            criteria={
                "true": "Describes a specific bug, crash, sync delay, or UI/UX defect",
                "false": "Generic praise without a bug, or spam",
            },
        )
    )
    urgency_score: ScoreJudgment = JudgmentField(
        Score(
            instructions="How urgent is the bug in this specific item for the engineering team to fix?",
            criteria=[
                "none_or_praise",
                "minor_ui_polish",
                "moderate_workflow_bug",
                "critical_checkout_or_crash_blocker",
            ],
        )
    )


class ReviewInboxLoader(BaseAgent):
    """Seeds the 5 incoming customer app reviews and user focus query into ADK session state."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_focus = ""
        if ctx.user_content and ctx.user_content.parts:
            user_focus = "\n".join(
                p.text for p in ctx.user_content.parts if getattr(p, "text", None)
            )
        delta: dict[str, Any] = {
            "triage_focus": (
                user_focus
                or "Filter out spam and non-actionable praise, and rank the real engineering bugs by urgency."
            ),
            "incoming_reviews": INCOMING_APP_REVIEWS,
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
                        text=f"📥 Loaded {len(INCOMING_APP_REVIEWS)} incoming app reviews for single-call batched Judgment evaluation."
                    )
                ],
            ),
        )


def filter_and_rank_reviews(
    batch: JudgmentBatch[dict[str, str], ReviewEvaluationSchema],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter out spam and non-bugs, and rank actionable bugs by calibrated urgency_score."""
    spam_blocked_ids = [
        entry.item["id"]
        for entry in batch.entries
        if not entry.judgment.is_safe_not_spam.passed(0.80)
    ]
    praise_skipped_ids = [
        entry.item["id"]
        for entry in batch.entries
        if entry.judgment.is_safe_not_spam.passed(0.80)
        and not entry.judgment.has_actionable_issue.passed(0.65)
    ]

    ranked_bugs_batch = batch.filter(
        lambda entry: entry.judgment.is_safe_not_spam.passed(0.80)
        and entry.judgment.has_actionable_issue.passed(0.65)
    ).rank_by(lambda entry: entry.judgment.urgency_score.score, reverse=True)

    prioritized_bugs = [
        {
            "rank": rank_idx + 1,
            "id": entry.item["id"],
            "author": entry.item["author"],
            "text": entry.item["text"],
            "safe_noul": round(entry.judgment.is_safe_not_spam.probability, 3),
            "actionable_noul": round(entry.judgment.has_actionable_issue.probability, 3),
            "urgency_score": round(entry.judgment.urgency_score.score, 2),
            "urgency_level": entry.judgment.urgency_score.level,
        }
        for rank_idx, entry in enumerate(ranked_bugs_batch.entries)
    ]

    table_rows = []
    for entry in batch.entries:
        rid = entry.item["id"]
        safe_p = entry.judgment.is_safe_not_spam.probability
        act_p = entry.judgment.has_actionable_issue.probability
        u_score = entry.judgment.urgency_score.score
        u_lvl = entry.judgment.urgency_score.level
        if rid in spam_blocked_ids:
            status = "🛑 BLOCKED (Spam/Scam)"
        elif rid in praise_skipped_ids:
            status = "💬 SKIPPED (Praise / No Bug)"
        else:
            status = "✅ KEPT FOR SPRINT"
        table_rows.append(
            f"| `{rid}` | `{safe_p:.3f}` | `{act_p:.3f}` | **`{u_score:.2f} / 3.00`** (`{u_lvl}`) | {status} |"
        )

    markdown_table = (
        "### 📊 Batched `JudgmentMap` Evaluation (5 Reviews × 3 Criteria = 15 Judgments in 1 Call)\n\n"
        "| Review ID | Safe (`Noul`) | Actionable Bug (`Noul`) | Urgency (`Score`) | Triage Disposition |\n"
        "| :--- | :---: | :---: | :---: | :--- |\n"
        + "\n".join(table_rows)
    )

    return {
        "total_reviews_evaluated": len(batch),
        "spam_blocked_ids": spam_blocked_ids,
        "praise_skipped_ids": praise_skipped_ids,
        "actionable_bug_count": len(prioritized_bugs),
        "prioritized_bugs": prioritized_bugs,
        "markdown_table": markdown_table,
    }


review_inbox_loader = ReviewInboxLoader(
    name="review_inbox_loader",
    description="Loads 5 incoming customer app reviews into session state.",
)

review_triage_map = JudgmentMap(
    name="review_triage_map",
    description="Evaluates all 5 reviews for safety, actionability, and urgency in 1 batched Judgment call.",
    items_key="incoming_reviews",
    item_schema=ReviewEvaluationSchema,
    context_keys=["triage_focus"],
    output_key="triage_batch_summary",
    transform=filter_and_rank_reviews,
)

review_sprint_planner = LlmAgent(
    name="review_sprint_planner",
    model=GEMINI_MODEL,
    description="Generates a prioritized engineering sprint backlog from the batched JudgmentMap results.",
    instruction=(
        "You are the Product Engineering Triage Lead.\n"
        "First, output the `markdown_table` from `{triage_batch_summary}` verbatim so the user sees "
        "all 15 calibrated judgments returned by `JudgmentMap` in 1 API call.\n\n"
        "Then summarize:\n"
        "1. **Prioritized Engineering Bug Backlog (`prioritized_bugs` in exact rank order #1, #2, #3)** with their calibrated `urgency_score`.\n"
        "2. **Filtered Items** (`spam_blocked_ids` and `praise_skipped_ids`) and why they were excluded from engineering tickets."
    ),
)

root_agent = SequentialAgent(
    name="review_triage_batch",
    description="Single-call customer review spam filtering and bug urgency ranking powered by JudgmentMap.",
    sub_agents=[
        review_inbox_loader,
        review_triage_map,
        review_sprint_planner,
    ],
)
