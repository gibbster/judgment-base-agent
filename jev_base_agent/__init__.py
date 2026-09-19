"""Model-agnostic System One / Judgment primitives for Google ADK workflows."""

from jev_base_agent.agent import JudgmentAgent, JudgmentDecision, judgment_node
from jev_base_agent.backends import (
    BaseJudgmentBackend,
    MockJudgmentBackend,
    TypeSafeBackend,
)
from jev_base_agent.errors import (
    JudgmentConfigError,
    JudgmentError,
    JudgmentEvaluationError,
)
from jev_base_agent.presets import (
    JudgmentBatch,
    JudgmentBatchEntry,
    JudgmentGuard,
    JudgmentMap,
    JudgmentSwitch,
)
from jev_base_agent.primitives import (
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
from jev_base_agent.schema import JudgmentField, JudgmentSchema

# Model-agnostic & TypeSafe/Jev aliases
SystemOneAgent = JudgmentAgent
JevAgent = JudgmentAgent

JudgmentRouter = JudgmentSwitch
SystemOneRouter = JudgmentSwitch
JevRouter = JudgmentSwitch

JudgmentGate = JudgmentGuard
SystemOneGate = JudgmentGuard
JevGate = JudgmentGuard

__all__ = [
    "BaseJudgmentBackend",
    "Choice",
    "ChoiceJudgment",
    "ConfidenceTier",
    "JevAgent",
    "JevGate",
    "JevRouter",
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
