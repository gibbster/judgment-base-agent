"""Model-agnostic System One / Judgment primitives for Google ADK workflows."""

from judgment_base_agent.agent import JudgmentAgent, JudgmentDecision, judgment_node
from judgment_base_agent.backends import (
    BaseJudgmentBackend,
    MockJudgmentBackend,
    TypeSafeBackend,
)
from judgment_base_agent.errors import (
    JudgmentConfigError,
    JudgmentError,
    JudgmentEvaluationError,
)
from judgment_base_agent.presets import (
    JudgmentBatch,
    JudgmentBatchEntry,
    JudgmentGuard,
    JudgmentMap,
    JudgmentSwitch,
)
from judgment_base_agent.primitives import (
    Choice,
    ChoiceJudgment,
    ConfidenceTier,
    JudgmentResult,
    JudgmentUsage,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
    classify_confidence_tier,
)
from judgment_base_agent.schema import JudgmentField, JudgmentSchema

# Model-agnostic aliases
SystemOneAgent = JudgmentAgent

JudgmentRouter = JudgmentSwitch
SystemOneRouter = JudgmentSwitch

JudgmentGate = JudgmentGuard
SystemOneGate = JudgmentGuard

__all__ = [
    "BaseJudgmentBackend",
    "Choice",
    "ChoiceJudgment",
    "ConfidenceTier",
    "JudgmentAgent",
    "JudgmentBatch",
    "JudgmentBatchEntry",
    "JudgmentConfigError",
    "JudgmentDecision",
    "JudgmentError",
    "JudgmentEvaluationError",
    "JudgmentField",
    "JudgmentGate",
    "JudgmentGuard",
    "JudgmentMap",
    "JudgmentResult",
    "JudgmentRouter",
    "JudgmentSchema",
    "JudgmentSwitch",
    "JudgmentUsage",
    "MockJudgmentBackend",
    "Noul",
    "NoulJudgment",
    "Score",
    "ScoreJudgment",
    "SystemOneAgent",
    "SystemOneGate",
    "SystemOneRouter",
    "TypeSafeBackend",
    "classify_confidence_tier",
    "judgment_node",
]
