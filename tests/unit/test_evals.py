"""Unit tests for ADK Rubric-based LLM-as-a-Judge evaluation (judgment_base_agent.evals)."""

from __future__ import annotations

from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_metrics import (
    EvalMetric,
    EvalStatus,
    RubricsBasedCriterion,
)
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent
from google.genai import types as genai_types
import pytest

from judgment_base_agent.backends.mock import MockJudgmentBackend
from judgment_base_agent.evals import (
    CLARITY_KEY,
    JudgmentRubric,
    JudgmentRubricEvaluator,
    RubricItem,
    evaluate_rubric_metric,
    format_rubric_scorecard,
    register_judgment_eval_metrics,
)


def _make_invocation(
    user_text: str,
    response_text: str,
    tool_name: str | None = None,
    tool_output: dict | None = None,
) -> Invocation:
    intermediate = None
    if tool_name is not None:
        intermediate = IntermediateData(
            tool_uses=[genai_types.FunctionCall(name=tool_name, args={"query": user_text})],
            tool_responses=[
                genai_types.FunctionResponse(
                    name=tool_name,
                    response=tool_output or {"status": "ok"},
                )
            ],
        )
    return Invocation(
        invocation_id="inv-1",
        user_content=genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=user_text)]
        ),
        final_response=genai_types.Content(
            role="model", parts=[genai_types.Part.from_text(text=response_text)]
        ),
        intermediate_data=intermediate,
    )


@pytest.mark.asyncio
async def test_judgment_rubric_evaluator_weighted_pass() -> None:
    """JudgmentRubricEvaluator computes calibrated weighted scores and returns PASSED."""
    rubric = JudgmentRubric(
        question="Does the agent response satisfy the support rubric?",
        threshold=0.75,
        min_clarity=0.50,
        items=(
            RubricItem(
                rubric_id="factual_accuracy",
                description="States the 30-day return policy accurately",
                weight=2.0,
                min_score=0.80,
                veto=True,
            ),
            RubricItem(
                rubric_id="tone_and_clarity",
                description="Uses a polite and empathetic customer support tone",
                weight=1.0,
                min_score=0.60,
                veto=False,
            ),
        ),
    )

    backend = MockJudgmentBackend(
        {
            "factual_accuracy": 0.95,
            "tone_and_clarity": 0.80,
            CLARITY_KEY: 0.92,
        }
    )

    evaluator = JudgmentRubricEvaluator(rubric=rubric, backend=backend)
    inv = _make_invocation(
        "Can I return my headphones after 10 days?",
        "Yes! You are well within our 30-day return window for a full refund.",
        tool_name="lookup_policy",
        tool_output={"return_days": 30},
    )

    result = await evaluator.evaluate_invocations([inv])
    assert result.overall_eval_status == EvalStatus.PASSED
    # Weighted average: (2.0 * 0.95 + 1.0 * 0.80) / 3.0 = 2.70 / 3.0 = 0.90
    assert result.overall_score == pytest.approx(0.90, abs=1e-3)
    assert len(result.per_invocation_results) == 1
    per_inv = result.per_invocation_results[0]
    assert per_inv.eval_status == EvalStatus.PASSED
    assert per_inv.rubric_scores is not None
    scores_by_id = {r.rubric_id: r.score for r in per_inv.rubric_scores}
    assert scores_by_id["factual_accuracy"] == pytest.approx(0.95)
    assert scores_by_id["tone_and_clarity"] == pytest.approx(0.80)

    scorecard = format_rubric_scorecard(result, rubric)
    assert "PASSED" in scorecard
    assert "factual_accuracy" in scorecard


@pytest.mark.asyncio
async def test_judgment_rubric_evaluator_veto_fails_even_when_average_high() -> None:
    """A failed veto (hard-fail) criterion forces EvalStatus.FAILED even if weighted average >= threshold."""
    rubric = JudgmentRubric(
        question="Does the response meet quality and compliance standards?",
        threshold=0.70,
        items=(
            RubricItem(
                rubric_id="politeness",
                description="Extremely polite and friendly greeting",
                weight=3.0,
                veto=False,
            ),
            RubricItem(
                rubric_id="no_pii_leak",
                description="Never asks the user for their plaintext password or CVV",
                weight=1.0,
                min_score=0.85,
                veto=True,
            ),
        ),
    )

    backend = MockJudgmentBackend(
        {
            "politeness": 0.98,
            "no_pii_leak": 0.20,
            CLARITY_KEY: 0.95,
        }
    )

    evaluator = JudgmentRubricEvaluator(rubric=rubric, backend=backend)
    inv = _make_invocation(
        "Help me reset my account",
        "I'd love to help! Please reply with your current password so I can check.",
    )

    result = await evaluator.evaluate_invocations([inv])
    # Weighted average is (3.0 * 0.98 + 1.0 * 0.20) / 4.0 = 0.785 >= 0.70,
    # BUT no_pii_leak is a veto criterion (< 0.85), so overall status MUST be FAILED!
    assert result.overall_score == pytest.approx(0.785, abs=1e-3)
    assert result.overall_eval_status == EvalStatus.FAILED
    per_inv = result.per_invocation_results[0]
    assert per_inv.eval_status == EvalStatus.FAILED
    assert any("VETO FAILED" in (r.rationale or "") for r in (per_inv.rubric_scores or []))


@pytest.mark.asyncio
async def test_judgment_rubric_evaluator_epistemic_abstention_on_low_clarity() -> None:
    """When clarity < min_clarity, evaluator abstains with EvalStatus.NOT_EVALUATED instead of guessing."""
    rubric = JudgmentRubric(
        question="Is the response grounded in the database query result?",
        threshold=0.75,
        min_clarity=0.65,
        items=(
            RubricItem(
                rubric_id="sql_grounding",
                description="Matches exact numbers returned by the SQL tool",
                weight=1.0,
            ),
        ),
    )

    backend = MockJudgmentBackend(
        {
            "sql_grounding": 0.50,
            CLARITY_KEY: 0.30,
        }
    )

    evaluator = JudgmentRubricEvaluator(rubric=rubric, backend=backend)
    inv = _make_invocation("What was Q3 revenue?", "Q3 revenue was $4.2M.")

    result = await evaluator.evaluate_invocations([inv])
    assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
    assert result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED


@pytest.mark.asyncio
async def test_adk_eval_metric_and_custom_function_integration() -> None:
    """Standard ADK EvalMetric with RubricsBasedCriterion works via evaluate_rubric_metric & registry."""
    adk_metric = EvalMetric(
        metric_name="judgment_rubric_quality_v1",
        threshold=0.75,
        criterion=RubricsBasedCriterion(
            threshold=0.75,
            rubrics=[
                Rubric(
                    rubric_id="conciseness",
                    rubric_content=RubricContent(
                        text_property="Response is concise and directly answers the question"
                    ),
                ),
                Rubric(
                    rubric_id="correctness",
                    rubric_content=RubricContent(
                        text_property="Response states the correct refund timeline"
                    ),
                ),
            ],
        ),
    )

    backend = MockJudgmentBackend(
        {
            "conciseness": 0.92,
            "correctness": 0.88,
            CLARITY_KEY: 0.91,
        }
    )

    inv = _make_invocation(
        "How long do refunds take?",
        "Refunds take 3 to 5 business days.",
    )

    result = await evaluate_rubric_metric(
        adk_metric, [inv], expected_invocations=None, backend=backend
    )
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.overall_score == pytest.approx(0.90, abs=1e-3)

    registry = register_judgment_eval_metrics()
    registered_names = [m.metric_name for m in registry.get_registered_metrics()]
    assert "judgment_rubric_quality_v1" in registered_names
