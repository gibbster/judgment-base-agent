# ADK Judgment Primitives (`judgment-base-agent`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-agnostic, high-ergonomics Google ADK library (`judgment_base_agent`) providing `JudgmentAgent`, `JudgmentSchema`, `JudgmentSwitch`, `JudgmentGuard`, and `JudgmentMap` (`JudgmentBatch`) to seamlessly integrate calibrated System One models (`model="judgment-latest"`) into both ADK 2.0 Graph `Workflow`s and ADK Composite Agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`).

**Architecture:** The package separates model-agnostic question/answer primitives (`Choice`, `Score`, `Noul`, `JudgmentResult`) and declarative Pydantic schemas (`JudgmentSchema`) from the evaluation backend protocol (`BaseJudgmentBackend`, implemented by `TypeSafeBackend` and `MockJudgmentBackend`). `JudgmentAgent` subclasses `google.adk.agents.BaseAgent` and emits unified ADK `Event`s containing `output`, `actions.route`, `actions.state_delta`, `actions.escalate`, and `actions.transfer_to_agent`, allowing the same class and its three universal workflow presets (`JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`) to work identically in `google.adk.workflow.Workflow` and `SequentialAgent` / `LoopAgent`.

**Tech Stack:** Python `>= 3.11`, `google-adk >= 2.7.0`, `typesafe-sdk >= 0.1.0`, `pydantic >= 2.0`, `pytest`, `pytest-asyncio`, `pytest-cov`.

**Spec:** [`docs/superpowers/specs/2026-09-19-judgment-agent-adk-design.md`](../specs/2026-09-19-judgment-agent-adk-design.md)

## Global Constraints

- Target Framework: `google-adk >= 2.7.0` and `typesafe-sdk >= 0.1.0`.
- Python runner in this environment: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest`.
- Minimum test coverage: `>= 85%` across `judgment_base_agent`.
- Immutability: All question primitives (`Choice`, `Score`, `Noul`), answer models (`ChoiceJudgment`, `ScoreJudgment`, `NoulJudgment`, `JudgmentResult`), `JudgmentBatch`, and `JudgmentDecision` must be frozen/immutable.
- Primary names (`JudgmentAgent`, `JudgmentSchema`, `JudgmentField`, `JudgmentDecision`, `JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`, `JudgmentBatch`, `judgment_node`) and aliases (`SystemOneAgent`, `SystemOneAgent`, `JudgmentRouter`, `SystemOneRouter`, `SystemOneRouter`, `JudgmentGate`, `SystemOneGate`, `SystemOneGate`) must all be exported from `judgment_base_agent`.
- No hardcoded secrets (`TYPESAFE_API_KEY` read from environment or explicit argument).

---

### Task 1: Project Scaffolding, Primitives (`primitives.py`) & Declarative Schema (`schema.py`)

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `judgment_base_agent/errors.py`
- Create: `judgment_base_agent/primitives.py`
- Create: `judgment_base_agent/schema.py`
- Test: `tests/unit/test_primitives_and_schema.py`

**Interfaces:**
- Consumes: `pydantic.BaseModel`, `pydantic.ConfigDict`, `pydantic.Field`
- Produces:
  - `judgment_base_agent.errors.JudgmentError`, `JudgmentConfigError`, `JudgmentEvaluationError`
  - `judgment_base_agent.primitives.Choice`, `Score`, `Noul`, `ConfidenceTier`, `classify_confidence_tier(confidence: float, floor: float = 0.50) -> ConfidenceTier`, `ChoiceJudgment`, `ScoreJudgment`, `NoulJudgment`, `JudgmentUsage`, `JudgmentResult`
  - `judgment_base_agent.schema.JudgmentField(question: Choice | Score | Noul | Any, *, key: str | None = None)`, `JudgmentSchema(BaseModel)` with `.build_questions() -> dict[str, Any]` and `.from_result(result: JudgmentResult) -> Self`

- [ ] **Step 1: Create `pyproject.toml`, `.gitignore`, and write the failing unit test `tests/unit/test_primitives_and_schema.py`**

Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "judgment-base-agent"
version = "0.1.0"
description = "Model-agnostic System One / Judgment primitives (JudgmentAgent, JudgmentSwitch, JudgmentGuard, JudgmentMap) for Google ADK workflows."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "google-adk>=2.7.0",
    "pydantic>=2.0.0",
    "typesafe-sdk>=0.1.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["."]
```

Create `tests/unit/test_primitives_and_schema.py`:
```python
"""Unit tests for primitives.py, errors.py, and schema.py."""

import pytest
from pydantic import ValidationError

from judgment_base_agent.errors import JudgmentConfigError
from judgment_base_agent.primitives import (
    Choice,
    ChoiceJudgment,
    JudgmentResult,
    JudgmentUsage,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
    classify_confidence_tier,
)
from judgment_base_agent.schema import JudgmentField, JudgmentSchema


def test_question_primitives_immutability_and_validation() -> None:
    c = Choice(instructions="Which dept?", criteria=["billing", "tech"])
    assert c.normalized_criteria() == {"billing": None, "tech": None}
    with pytest.raises(Exception):
        c.instructions = "mutated"  # type: ignore[misc]

    with pytest.raises(JudgmentConfigError):
        Choice(instructions="", criteria=["a"])
    with pytest.raises(JudgmentConfigError):
        Choice(instructions="Valid?", criteria=[])
    with pytest.raises(JudgmentConfigError):
        Score(instructions="Valid?", criteria=["only_one"])
    with pytest.raises(JudgmentConfigError):
        Noul(instructions="   ")


def test_confidence_tier_and_judgment_result_accessors() -> None:
    assert classify_confidence_tier(0.85) == "high"
    assert classify_confidence_tier(0.60, floor=0.50) == "medium"
    assert classify_confidence_tier(0.40, floor=0.50) == "low"

    res = JudgmentResult(
        choices={
            "tone": ChoiceJudgment.from_raw(
                choice="frustrated",
                probabilities={"calm": 0.1, "frustrated": 0.9},
                confidence=0.82,
            )
        },
        scores={
            "urgency": ScoreJudgment.from_raw(
                score=1.8,
                legend={"0": "low", "1": "med", "2": "high"},
                probabilities={"0": 0.0, "1": 0.2, "2": 0.8},
                confidence=0.75,
            )
        },
        nouls={"billing": NoulJudgment(noul=0.94)},
        model="judgment-latest",
        usage=JudgmentUsage(input_tokens=42, output_tokens=7),
    )

    assert res.choice("tone") == "frustrated"
    assert res.confidence("tone") == pytest.approx(0.82)
    assert res.choices["tone"].confidence_tier == "high"
    assert res.score("urgency") == pytest.approx(1.8)
    assert res.confidence("urgency") == pytest.approx(0.75)
    assert res.scores["urgency"].confidence_tier == "medium"
    assert res.noul("billing") == pytest.approx(0.94)
    assert res["billing"].noul == pytest.approx(0.94)

    with pytest.raises(KeyError):
        res.get("unknown_key")


def test_declarative_judgment_schema_roundtrip() -> None:
    class TicketSchema(JudgmentSchema):
        billing: NoulJudgment = JudgmentField(
            Noul(instructions="Is `ticket` about billing?")
        )
        tone: ChoiceJudgment = JudgmentField(
            Choice(
                instructions="What is the tone?",
                criteria={"calm": None, "frustrated": None},
            )
        )
        urgency: ScoreJudgment = JudgmentField(
            Score(
                instructions="How urgent?",
                criteria=["can wait", "today"],
            )
        )

    questions = TicketSchema.build_questions()
    assert set(questions.keys()) == {"billing", "tone", "urgency"}
    assert isinstance(questions["billing"], Noul)
    assert isinstance(questions["tone"], Choice)
    assert isinstance(questions["urgency"], Score)

    raw_result = JudgmentResult(
        choices={
            "tone": ChoiceJudgment.from_raw(
                choice="calm",
                probabilities={"calm": 0.9, "frustrated": 0.1},
                confidence=0.85,
            )
        },
        scores={
            "urgency": ScoreJudgment.from_raw(
                score=0.2,
                legend={"0": "can wait", "1": "today"},
                probabilities={"0": 0.8, "1": 0.2},
                confidence=0.78,
            )
        },
        nouls={"billing": NoulJudgment(noul=0.91)},
    )

    parsed = TicketSchema.from_result(raw_result)
    assert parsed.billing.noul == pytest.approx(0.91)
    assert parsed.tone.choice == "calm"
    assert parsed.urgency.score == pytest.approx(0.2)
    assert parsed.raw_result == raw_result

    # Ensure ADK 2.0 FunctionNode dict->Pydantic auto-conversion works
    dumped = parsed.model_dump()
    rehydrated = TicketSchema.model_validate(dumped)
    assert rehydrated.tone.choice == "calm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest tests/unit/test_primitives_and_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgment_base_agent'`

- [ ] **Step 3: Implement `judgment_base_agent/errors.py`, `judgment_base_agent/primitives.py`, and `judgment_base_agent/schema.py`**

Create `judgment_base_agent/errors.py`:
```python
"""Custom exception hierarchy for judgment_base_agent."""

from __future__ import annotations


class JudgmentError(Exception):
    """Base exception for all JudgmentAgent errors."""


class JudgmentConfigError(JudgmentError):
    """Raised when a question, schema, or agent configuration is invalid."""


class JudgmentEvaluationError(JudgmentError):
    """Raised when a judgment backend fails to evaluate a request."""
```

Create `judgment_base_agent/primitives.py`:
```python
"""Model-agnostic question and answer primitives for System One / Judgment workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from judgment_base_agent.errors import JudgmentConfigError

ConfidenceTier = Literal["high", "medium", "low"]


def classify_confidence_tier(
    confidence: float, floor: float = 0.50
) -> ConfidenceTier:
    """Classify a calibrated confidence value into high, medium, or low tiers."""
    if confidence >= 0.80:
        return "high"
    if confidence >= floor:
        return "medium"
    return "low"


@dataclass(frozen=True)
class Choice:
    """Unordered categorical judgment over a closed set of options."""

    instructions: str
    criteria: Mapping[str, str | None] | Sequence[str]

    def __post_init__(self) -> None:
        if not (self.instructions or "").strip():
            raise JudgmentConfigError("Choice.instructions must be a non-empty string.")
        if not self.criteria:
            raise JudgmentConfigError("Choice.criteria must contain at least one option.")

    def normalized_criteria(self) -> dict[str, str | None]:
        """Return criteria normalized as a dict mapping option key to description or None."""
        if isinstance(self.criteria, Mapping):
            return {str(k): (str(v) if v is not None else None) for k, v in self.criteria.items()}
        return {str(item): None for item in self.criteria}


@dataclass(frozen=True)
class Score:
    """Ordered spectrum judgment over 2 or more defined levels."""

    instructions: str
    criteria: Sequence[str]

    def __post_init__(self) -> None:
        if not (self.instructions or "").strip():
            raise JudgmentConfigError("Score.instructions must be a non-empty string.")
        if not self.criteria or len(self.criteria) < 2:
            raise JudgmentConfigError("Score.criteria must contain at least 2 ordered levels.")

    def normalized_criteria(self) -> list[str]:
        """Return criteria normalized as a list of level strings."""
        return [str(level) for level in self.criteria]


@dataclass(frozen=True)
class Noul:
    """Calibrated binary probability judgment in [0.0, 1.0]."""

    instructions: str
    criteria: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not (self.instructions or "").strip():
            raise JudgmentConfigError("Noul.instructions must be a non-empty string.")


class ChoiceJudgment(BaseModel):
    """Typed answer for a Choice question."""

    model_config = ConfigDict(frozen=True)

    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 1.0
    confidence_tier: ConfidenceTier = "high"

    @classmethod
    def from_raw(
        cls,
        choice: str,
        probabilities: Mapping[str, float] | None = None,
        confidence: float = 1.0,
        confidence_floor: float = 0.50,
    ) -> ChoiceJudgment:
        probs = {str(k): float(v) for k, v in (probabilities or {}).items()}
        conf = float(confidence)
        return cls(
            choice=str(choice),
            probabilities=probs,
            confidence=conf,
            confidence_tier=classify_confidence_tier(conf, floor=confidence_floor),
        )


class ScoreJudgment(BaseModel):
    """Typed answer for a Score question."""

    model_config = ConfigDict(frozen=True)

    score: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 1.0
    confidence_tier: ConfidenceTier = "high"

    @classmethod
    def from_raw(
        cls,
        score: float,
        legend: Mapping[str, str] | None = None,
        probabilities: Mapping[str, float] | None = None,
        confidence: float = 1.0,
        confidence_floor: float = 0.50,
    ) -> ScoreJudgment:
        conf = float(confidence)
        return cls(
            score=float(score),
            legend={str(k): str(v) for k, v in (legend or {}).items()},
            probabilities={str(k): float(v) for k, v in (probabilities or {}).items()},
            confidence=conf,
            confidence_tier=classify_confidence_tier(conf, floor=confidence_floor),
        )


class NoulJudgment(BaseModel):
    """Typed answer for a Noul question."""

    model_config = ConfigDict(frozen=True)

    noul: float


class JudgmentUsage(BaseModel):
    """Token usage telemetry for a judgment evaluation call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0


class JudgmentResult(BaseModel):
    """Immutable container of all Choice, Score, and Noul judgments from an evaluation."""

    model_config = ConfigDict(frozen=True)

    choices: dict[str, ChoiceJudgment] = Field(default_factory=dict)
    scores: dict[str, ScoreJudgment] = Field(default_factory=dict)
    nouls: dict[str, NoulJudgment] = Field(default_factory=dict)
    model: str = "judgment-latest"
    usage: JudgmentUsage | None = None

    def get(self, key: str) -> ChoiceJudgment | ScoreJudgment | NoulJudgment:
        """Retrieve any judgment answer by question key."""
        if key in self.choices:
            return self.choices[key]
        if key in self.scores:
            return self.scores[key]
        if key in self.nouls:
            return self.nouls[key]
        raise KeyError(f"Question key '{key}' not found in JudgmentResult.")

    def __getitem__(self, key: str) -> ChoiceJudgment | ScoreJudgment | NoulJudgment:
        return self.get(key)

    def choice(self, key: str) -> str:
        """Return the selected choice string for a Choice question."""
        if key not in self.choices:
            raise KeyError(f"Choice key '{key}' not found in JudgmentResult.")
        return self.choices[key].choice

    def score(self, key: str) -> float:
        """Return the numeric score for a Score question."""
        if key not in self.scores:
            raise KeyError(f"Score key '{key}' not found in JudgmentResult.")
        return self.scores[key].score

    def noul(self, key: str) -> float:
        """Return the probability in [0, 1] for a Noul question."""
        if key not in self.nouls:
            raise KeyError(f"Noul key '{key}' not found in JudgmentResult.")
        return self.nouls[key].noul

    def confidence(self, key: str) -> float:
        """Return the calibrated confidence for a Choice or Score question."""
        if key in self.choices:
            return self.choices[key].confidence
        if key in self.scores:
            return self.scores[key].confidence
        if key in self.nouls:
            # Distance from 0.5 normalized to [0.0, 1.0] for binary Noul
            return abs(self.nouls[key].noul - 0.5) * 2.0
        raise KeyError(f"Question key '{key}' not found in JudgmentResult.")
```

Create `judgment_base_agent/schema.py`:
```python
"""Declarative Pydantic schema binding fields to Choice, Score, and Noul primitives."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from judgment_base_agent.errors import JudgmentConfigError
from judgment_base_agent.primitives import JudgmentResult

_JUDGMENT_QUESTION_META = "judgment_question"
_JUDGMENT_KEY_META = "judgment_key"


def JudgmentField(
    question: Any,
    *,
    key: str | None = None,
    description: str | None = None,
) -> Any:
    """Bind a Pydantic schema field to a Choice, Score, or Noul question definition."""
    if question is None:
        raise JudgmentConfigError("JudgmentField requires a non-None question object.")
    return Field(
        default=...,
        description=description,
        json_schema_extra={
            _JUDGMENT_QUESTION_META: question,
            _JUDGMENT_KEY_META: key,
        },
    )


class JudgmentSchema(BaseModel):
    """Base class for declarative, strongly-typed Judgment schemas."""

    model_config = ConfigDict(frozen=True)
    _raw_result: JudgmentResult | None = PrivateAttr(default=None)

    @property
    def raw_result(self) -> JudgmentResult | None:
        """Return the underlying JudgmentResult (with usage and model metadata) if available."""
        return self._raw_result

    @classmethod
    def build_questions(cls) -> dict[str, Any]:
        """Extract the mapping of question keys to question primitives declared on this schema."""
        questions: dict[str, Any] = {}
        for field_name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and _JUDGMENT_QUESTION_META in extra:
                q_key = extra.get(_JUDGMENT_KEY_META) or field_name
                questions[str(q_key)] = extra[_JUDGMENT_QUESTION_META]
        if not questions:
            raise JudgmentConfigError(
                f"JudgmentSchema '{cls.__name__}' defines no fields using JudgmentField(...)."
            )
        return questions

    @classmethod
    def from_result(cls, result: JudgmentResult) -> Self:
        """Instantiate and validate this schema from a JudgmentResult."""
        payload: dict[str, Any] = {}
        for field_name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and _JUDGMENT_QUESTION_META in extra:
                q_key = str(extra.get(_JUDGMENT_KEY_META) or field_name)
                payload[field_name] = result.get(q_key)
        instance = cls.model_validate(payload)
        object.__setattr__(instance, "_raw_result", result)
        return instance
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest tests/unit/test_primitives_and_schema.py -v`
Expected: PASS (all 3 tests passing)

- [ ] **Step 5: Commit Task 1**

```bash
git add pyproject.toml .gitignore judgment_base_agent/errors.py judgment_base_agent/primitives.py judgment_base_agent/schema.py tests/unit/test_primitives_and_schema.py
git commit -m "feat: add model-agnostic judgment primitives and JudgmentSchema"
```

---

### Task 2: Model-Agnostic Backend Protocol (`backends/base.py`), `TypeSafeBackend` (`backends/typesafe.py`) & `MockJudgmentBackend` (`backends/mock.py`)

**Files:**
- Create: `judgment_base_agent/backends/__init__.py`
- Create: `judgment_base_agent/backends/base.py`
- Create: `judgment_base_agent/backends/typesafe.py`
- Create: `judgment_base_agent/backends/mock.py`
- Test: `tests/unit/test_backends.py`

**Interfaces:**
- Consumes: `Choice`, `Score`, `Noul`, `ChoiceJudgment`, `ScoreJudgment`, `NoulJudgment`, `JudgmentUsage`, `JudgmentResult`, `JudgmentConfigError`, `JudgmentEvaluationError`
- Produces:
  - `BaseJudgmentBackend(Protocol)` with `async def evaluate(self, state: Any, questions: Mapping[str, Any], model: str | None = None) -> JudgmentResult`
  - `TypeSafeBackend(api_key: str | None = None, default_model: str = "judgment-latest", confidence_floor: float = 0.50, client: Any | None = None)`
  - `MockJudgmentBackend(responses: Mapping[str, Any] | Callable[[Any, Mapping[str, Any], str], Mapping[str, Any]], default_model: str = "mock-judgment", confidence_floor: float = 0.50)`

- [ ] **Step 1: Write the failing unit test `tests/unit/test_backends.py`**

Create `tests/unit/test_backends.py`:
```python
"""Unit tests for TypeSafeBackend and MockJudgmentBackend."""

from types import SimpleNamespace
import pytest
import typesafe_sdk

from judgment_base_agent.backends.mock import MockJudgmentBackend
from judgment_base_agent.backends.typesafe import TypeSafeBackend
from judgment_base_agent.errors import JudgmentConfigError, JudgmentEvaluationError
from judgment_base_agent.primitives import Choice, Noul, Score


@pytest.mark.asyncio
async def test_mock_judgment_backend_static_and_callable() -> None:
    backend = MockJudgmentBackend(
        responses={
            "dept": {"choice": "billing", "confidence": 0.91, "probabilities": {"billing": 0.91, "tech": 0.09}},
            "urgent": 0.88,
            "severity": {"score": 1.5, "confidence": 0.72},
        }
    )
    questions = {
        "dept": Choice(instructions="Dept?", criteria=["billing", "tech"]),
        "urgent": Noul(instructions="Urgent?"),
        "severity": Score(instructions="Severity?", criteria=["low", "med", "high"]),
    }
    res = await backend.evaluate(state={"msg": "help"}, questions=questions)
    assert res.choice("dept") == "billing"
    assert res.noul("urgent") == pytest.approx(0.88)
    assert res.score("severity") == pytest.approx(1.5)
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_typesafe_backend_with_injected_client_and_sdk_primitives() -> None:
    captured: dict = {}

    class FakeClient:
        async def system_one(self, *, state, questions, model=None):
            captured["state"] = state
            captured["questions"] = questions
            captured["model"] = model
            return SimpleNamespace(
                model=model or "judgment-latest",
                usage=SimpleNamespace(input_tokens=15, output_tokens=3),
                choices={
                    "dept": SimpleNamespace(
                        choice="tech",
                        probabilities={"billing": 0.05, "tech": 0.95},
                        confidence=0.92,
                    )
                },
                scores={
                    "frustration": SimpleNamespace(
                        score=1.2,
                        legend={"0": "calm", "1": "annoyed", "2": "angry"},
                        probabilities={"0": 0.1, "1": 0.6, "2": 0.3},
                        confidence=0.65,
                    )
                },
                nouls={"refund": SimpleNamespace(noul=0.12)},
            )

    backend = TypeSafeBackend(client=FakeClient(), default_model="judgment-latest")
    questions = {
        "dept": Choice(instructions="Which dept?", criteria=["billing", "tech"]),
        "frustration": typesafe_sdk.Score(
            instructions="Frustration?", criteria=["calm", "annoyed", "angry"]
        ),
        "refund": Noul(
            instructions="Refund?",
            criteria={"true": "Asks for money back", "false": "Does not"},
        ),
    }

    result = await backend.evaluate(state={"text": "500 error"}, questions=questions)
    assert captured["model"] == "judgment-latest"
    assert isinstance(captured["questions"]["dept"], typesafe_sdk.Choice)
    assert isinstance(captured["questions"]["refund"], typesafe_sdk.Noul)
    assert result.choice("dept") == "tech"
    assert result.score("frustration") == pytest.approx(1.2)
    assert result.noul("refund") == pytest.approx(0.12)
    assert result.usage is not None
    assert result.usage.input_tokens == 15


@pytest.mark.asyncio
async def test_typesafe_backend_missing_api_key_raises_clean_error(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    backend = TypeSafeBackend(api_key=None)
    with pytest.raises(JudgmentConfigError, match="TYPESAFE_API_KEY"):
        await backend.evaluate(
            state="hello",
            questions={"q": Noul(instructions="Is greeting?")},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest tests/unit/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgment_base_agent.backends'`

- [ ] **Step 3: Implement `judgment_base_agent/backends/` (`base.py`, `typesafe.py`, `mock.py`, `__init__.py`)**

Create `judgment_base_agent/backends/base.py`:
```python
"""Protocol for model-agnostic Judgment evaluation backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from judgment_base_agent.primitives import JudgmentResult


@runtime_checkable
class BaseJudgmentBackend(Protocol):
    """Protocol implemented by System One / calibrated judgment backends."""

    async def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> JudgmentResult:
        """Evaluate all questions in parallel against the given state and return a JudgmentResult."""
        ...
```

Create `judgment_base_agent/backends/typesafe.py`:
```python
"""TypeSafe AI System One Judgment backend implementation."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import typesafe_sdk

from judgment_base_agent.errors import JudgmentConfigError, JudgmentEvaluationError
from judgment_base_agent.primitives import (
    Choice,
    ChoiceJudgment,
    JudgmentResult,
    JudgmentUsage,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
)


def _to_typesafe_question(q: Any) -> Any:
    """Convert model-agnostic Choice/Score/Noul or dict into typesafe_sdk question objects."""
    if isinstance(q, (typesafe_sdk.Choice, typesafe_sdk.Score, typesafe_sdk.Noul)):
        return q
    if isinstance(q, Choice):
        return typesafe_sdk.Choice(
            instructions=q.instructions,
            criteria=q.normalized_criteria(),
        )
    if isinstance(q, Score):
        return typesafe_sdk.Score(
            instructions=q.instructions,
            criteria=q.normalized_criteria(),
        )
    if isinstance(q, Noul):
        return typesafe_sdk.Noul(
            instructions=q.instructions,
            criteria=dict(q.criteria) if q.criteria else None,
        )
    if isinstance(q, Mapping):
        q_type = str(q.get("type", "")).lower()
        if q_type == "choice":
            return typesafe_sdk.Choice(
                instructions=str(q["instructions"]),
                criteria=q["criteria"],
            )
        if q_type == "score":
            return typesafe_sdk.Score(
                instructions=str(q["instructions"]),
                criteria=list(q["criteria"]),
            )
        if q_type == "noul":
            return typesafe_sdk.Noul(
                instructions=str(q["instructions"]),
                criteria=q.get("criteria"),
            )
    raise JudgmentConfigError(f"Unsupported question primitive type: {type(q)!r}")


def _question_kind(q: Any) -> str:
    if isinstance(q, (Choice, typesafe_sdk.Choice)):
        return "choice"
    if isinstance(q, (Score, typesafe_sdk.Score)):
        return "score"
    if isinstance(q, (Noul, typesafe_sdk.Noul)):
        return "noul"
    if isinstance(q, Mapping):
        return str(q.get("type", "")).lower()
    return "unknown"


class TypeSafeBackend:
    """Judgment backend powered by TypeSafe's AsyncTypeSafeClient (default model: 'judgment-latest')."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "judgment-latest",
        confidence_floor: float = 0.50,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.confidence_floor = confidence_floor
        self.client = client

    async def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> JudgmentResult:
        """Send state and batched questions to TypeSafe system_one and normalize the result."""
        if not questions:
            return JudgmentResult(model=model or self.default_model)

        sdk_questions = {
            str(k): _to_typesafe_question(v) for k, v in questions.items()
        }
        target_model = model or self.default_model

        try:
            if self.client is not None:
                try:
                    response = await self.client.system_one(
                        state=state, questions=sdk_questions, model=target_model
                    )
                except TypeError:
                    response = await self.client.system_one(
                        state=state, questions=sdk_questions
                    )
            else:
                resolved_key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
                if not resolved_key:
                    raise JudgmentConfigError(
                        "TYPESAFE_API_KEY environment variable or explicit api_key is required "
                        "to evaluate questions with TypeSafeBackend."
                    )
                async with typesafe_sdk.AsyncTypeSafeClient(
                    api_key=resolved_key
                ) as live_client:
                    response = await live_client.system_one(
                        state=state,
                        questions=sdk_questions,
                        model=target_model,
                    )
        except JudgmentConfigError:
            raise
        except Exception as exc:
            raise JudgmentEvaluationError(
                f"TypeSafe evaluation failed for model '{target_model}': {exc}"
            ) from exc

        answers_map = getattr(response, "answers", None) or {}
        choices_map = getattr(response, "choices", None) or {}
        scores_map = getattr(response, "scores", None) or {}
        nouls_map = getattr(response, "nouls", None) or {}

        parsed_choices: dict[str, ChoiceJudgment] = {}
        parsed_scores: dict[str, ScoreJudgment] = {}
        parsed_nouls: dict[str, NoulJudgment] = {}

        for key, orig_q in questions.items():
            kind = _question_kind(orig_q)
            if kind == "choice":
                raw_c = answers_map.get(key) or choices_map[key]
                parsed_choices[key] = ChoiceJudgment.from_raw(
                    choice=str(raw_c.choice),
                    probabilities=getattr(raw_c, "probabilities", {}) or {},
                    confidence=float(getattr(raw_c, "confidence", 1.0)),
                    confidence_floor=self.confidence_floor,
                )
            elif kind == "score":
                raw_s = answers_map.get(key) or scores_map[key]
                parsed_scores[key] = ScoreJudgment.from_raw(
                    score=float(raw_s.score),
                    legend=getattr(raw_s, "legend", {}) or {},
                    probabilities=getattr(raw_s, "probabilities", {}) or {},
                    confidence=float(getattr(raw_s, "confidence", 1.0)),
                    confidence_floor=self.confidence_floor,
                )
            elif kind == "noul":
                raw_n = answers_map.get(key) or nouls_map[key]
                parsed_nouls[key] = NoulJudgment(noul=float(raw_n.noul))

        raw_usage = getattr(response, "usage", None)
        usage = (
            JudgmentUsage(
                input_tokens=int(getattr(raw_usage, "input_tokens", 0)),
                output_tokens=int(getattr(raw_usage, "output_tokens", 0)),
            )
            if raw_usage is not None
            else None
        )

        return JudgmentResult(
            choices=parsed_choices,
            scores=parsed_scores,
            nouls=parsed_nouls,
            model=str(getattr(response, "model", target_model)),
            usage=usage,
        )
```

Create `judgment_base_agent/backends/mock.py`:
```python
"""Deterministic MockJudgmentBackend for offline unit and integration tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from judgment_base_agent.backends.typesafe import _question_kind
from judgment_base_agent.primitives import (
    ChoiceJudgment,
    JudgmentResult,
    JudgmentUsage,
    NoulJudgment,
    ScoreJudgment,
)


class MockJudgmentBackend:
    """Offline mock backend supporting static answer dicts or dynamic responder callables."""

    def __init__(
        self,
        responses: (
            Mapping[str, Any]
            | Callable[[Any, Mapping[str, Any], str], Mapping[str, Any]]
        ),
        default_model: str = "mock-judgment",
        confidence_floor: float = 0.50,
    ) -> None:
        self.responses = responses
        self.default_model = default_model
        self.confidence_floor = confidence_floor
        self.calls: list[dict[str, Any]] = []

    async def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> JudgmentResult:
        target_model = model or self.default_model
        self.calls.append(
            {"state": state, "questions": dict(questions), "model": target_model}
        )

        raw_map = (
            self.responses(state, questions, target_model)
            if callable(self.responses)
            else self.responses
        )

        parsed_choices: dict[str, ChoiceJudgment] = {}
        parsed_scores: dict[str, ScoreJudgment] = {}
        parsed_nouls: dict[str, NoulJudgment] = {}

        for key, q in questions.items():
            kind = _question_kind(q)
            val = raw_map.get(key)
            if kind == "choice":
                if isinstance(val, ChoiceJudgment):
                    parsed_choices[key] = val
                elif isinstance(val, Mapping):
                    parsed_choices[key] = ChoiceJudgment.from_raw(
                        choice=str(val.get("choice", "")),
                        probabilities=val.get("probabilities") or {},
                        confidence=float(val.get("confidence", 0.90)),
                        confidence_floor=self.confidence_floor,
                    )
                else:
                    choice_str = str(val) if val is not None else ""
                    parsed_choices[key] = ChoiceJudgment.from_raw(
                        choice=choice_str,
                        probabilities={choice_str: 0.90} if choice_str else {},
                        confidence=0.90,
                        confidence_floor=self.confidence_floor,
                    )
            elif kind == "score":
                if isinstance(val, ScoreJudgment):
                    parsed_scores[key] = val
                elif isinstance(val, Mapping):
                    parsed_scores[key] = ScoreJudgment.from_raw(
                        score=float(val.get("score", 0.0)),
                        legend=val.get("legend") or {},
                        probabilities=val.get("probabilities") or {},
                        confidence=float(val.get("confidence", 0.90)),
                        confidence_floor=self.confidence_floor,
                    )
                else:
                    parsed_scores[key] = ScoreJudgment.from_raw(
                        score=float(val if val is not None else 0.0),
                        confidence=0.90,
                        confidence_floor=self.confidence_floor,
                    )
            elif kind == "noul":
                if isinstance(val, NoulJudgment):
                    parsed_nouls[key] = val
                elif isinstance(val, Mapping):
                    parsed_nouls[key] = NoulJudgment(noul=float(val.get("noul", 0.0)))
                else:
                    parsed_nouls[key] = NoulJudgment(
                        noul=float(val if val is not None else 0.0)
                    )

        return JudgmentResult(
            choices=parsed_choices,
            scores=parsed_scores,
            nouls=parsed_nouls,
            model=target_model,
            usage=JudgmentUsage(input_tokens=10, output_tokens=len(questions)),
        )
```

Create `judgment_base_agent/backends/__init__.py`:
```python
"""Judgment evaluation backends."""

from judgment_base_agent.backends.base import BaseJudgmentBackend
from judgment_base_agent.backends.mock import MockJudgmentBackend
from judgment_base_agent.backends.typesafe import TypeSafeBackend

__all__ = [
    "BaseJudgmentBackend",
    "MockJudgmentBackend",
    "TypeSafeBackend",
]
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest tests/unit/test_backends.py -v`
Expected: PASS (all 3 tests passing)

- [ ] **Step 5: Commit Task 2**

```bash
git add judgment_base_agent/backends/ tests/unit/test_backends.py
git commit -m "feat: add BaseJudgmentBackend protocol, TypeSafeBackend, and MockJudgmentBackend"
```

---

### Task 3: Core `JudgmentAgent(BaseAgent)`, `JudgmentDecision` & `@judgment_node` Decorator (`agent.py`)

**Files:**
- Create: `judgment_base_agent/agent.py`
- Test: `tests/unit/test_judgment_agent.py`

**Interfaces:**
- Consumes: `google.adk.agents.BaseAgent`, `google.adk.agents.invocation_context.InvocationContext`, `google.adk.events.Event`, `google.adk.events.EventActions`, `google.adk.events.request_input.RequestInput`, `BaseJudgmentBackend`, `TypeSafeBackend`, `JudgmentSchema`, `JudgmentResult`
- Produces:
  - `JudgmentDecision` frozen dataclass (`output`, `route`, `escalate`, `transfer_to_agent`, `state_delta`, `request_input_id`, `request_input_prompt`)
  - `JudgmentAgent(BaseAgent)` supporting `schema` or `questions` (static mapping or callable), `state_keys` or `state_builder`, `output_key`, `decide` hook, `on_error` fallback, `model="judgment-latest"`, `backend`
  - `@judgment_node` decorator wrapping a Python `(typed_result, state, node_input) -> Any` function into a `JudgmentAgent`

- [ ] **Step 1: Write the failing unit test `tests/unit/test_judgment_agent.py`**

Create `tests/unit/test_judgment_agent.py`:
```python
"""Unit tests for JudgmentAgent, JudgmentDecision, and @judgment_node."""

import pytest
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from judgment_base_agent.agent import JudgmentAgent, JudgmentDecision, judgment_node
from judgment_base_agent.backends.mock import MockJudgmentBackend
from judgment_base_agent.errors import JudgmentConfigError, JudgmentEvaluationError
from judgment_base_agent.primitives import Choice, ChoiceJudgment, Noul, NoulJudgment
from judgment_base_agent.schema import JudgmentField, JudgmentSchema


class TriageSchema(JudgmentSchema):
    billing: NoulJudgment = JudgmentField(Noul(instructions="Is billing?"))
    department: ChoiceJudgment = JudgmentField(
        Choice(instructions="Department?", criteria=["billing", "support"])
    )


@pytest.mark.asyncio
async def test_judgment_agent_with_schema_and_decide_hook() -> None:
    mock_backend = MockJudgmentBackend(
        responses={
            "billing": 0.95,
            "department": {"choice": "billing", "confidence": 0.88},
        }
    )

    agent = JudgmentAgent(
        name="triage_step",
        schema=TriageSchema,
        state_keys=["ticket"],
        output_key="triage_output",
        backend=mock_backend,
        decide=lambda res, state: JudgmentDecision(
            route="billing_route" if res.billing.noul > 0.8 else "support_route",
            escalate=True,
            state_delta={"routed_to": res.department.choice},
        ),
    )

    app = App(name="test_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="test_app",
        user_id="u1",
        state={"ticket": "I was charged twice!"},
    )

    events = []
    async for ev in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="go")]),
    ):
        events.append(ev)

    assert len(events) == 1
    ev = events[0]
    assert ev.actions.route == "billing_route"
    assert ev.actions.escalate is True
    assert ev.actions.state_delta["routed_to"] == "billing"
    assert ev.actions.state_delta["triage_output"]["billing"]["noul"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_judgment_agent_on_error_fallback() -> None:
    class FailingBackend:
        async def evaluate(self, state, questions, model=None):
            raise JudgmentEvaluationError("API rate limit")

    agent = JudgmentAgent(
        name="resilient_step",
        questions={"ok": Noul(instructions="Is ok?")},
        backend=FailingBackend(),
        on_error=lambda exc, state: JudgmentDecision(
            route="fallback_edge",
            output={"error": str(exc)},
        ),
    )

    app = App(name="test_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="test_app", user_id="u1")

    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
        )
    ]
    assert events[-1].actions.route == "fallback_edge"
    assert events[-1].output == {"error": "API rate limit"}


@pytest.mark.asyncio
async def test_judgment_node_decorator() -> None:
    mock_backend = MockJudgmentBackend(responses={"billing": 0.92, "department": "billing"})

    @judgment_node(name="decorated_triage", schema=TriageSchema, backend=mock_backend)
    def handle_triage(res: TriageSchema, state: dict) -> str:
        return "fast_track" if res.billing.noul > 0.9 else "standard"

    assert isinstance(handle_triage, JudgmentAgent)
    app = App(name="test_app", root_agent=handle_triage)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="test_app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="check")]),
        )
    ]
    assert events[-1].actions.route == "fast_track"


def test_judgment_agent_requires_schema_or_questions() -> None:
    with pytest.raises(JudgmentConfigError):
        JudgmentAgent(name="invalid_agent")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest tests/unit/test_judgment_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgment_base_agent.agent'`

- [ ] **Step 3: Implement `judgment_base_agent/agent.py`**

Create `judgment_base_agent/agent.py`:
```python
"""Core JudgmentAgent(BaseAgent), JudgmentDecision, and @judgment_node decorator."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
import inspect
import json
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.events.request_input import RequestInput
from google.genai import types
from pydantic import BaseModel, ConfigDict

from judgment_base_agent.backends.base import BaseJudgmentBackend
from judgment_base_agent.backends.typesafe import TypeSafeBackend
from judgment_base_agent.errors import JudgmentConfigError
from judgment_base_agent.primitives import JudgmentResult
from judgment_base_agent.schema import JudgmentSchema


@dataclass(frozen=True)
class JudgmentDecision:
    """User-returned decision envelope controlling ADK Event output, routing, and escalation."""

    output: Any = None
    route: str | int | bool | list[str] | None = None
    escalate: bool | None = None
    transfer_to_agent: str | None = None
    state_delta: dict[str, Any] = field(default_factory=dict)
    request_input_id: str | None = None
    request_input_prompt: str | None = None


def normalize_decision(
    raw_decision: Any,
    default_output: Any,
    transfer_to_sub_agent: bool = False,
) -> JudgmentDecision:
    """Normalize user decide() return value (str, bool, dict, JudgmentDecision) into JudgmentDecision."""
    if raw_decision is None:
        return JudgmentDecision(output=default_output)
    if isinstance(raw_decision, JudgmentDecision):
        return JudgmentDecision(
            output=raw_decision.output if raw_decision.output is not None else default_output,
            route=raw_decision.route,
            escalate=raw_decision.escalate,
            transfer_to_agent=raw_decision.transfer_to_agent,
            state_delta=dict(raw_decision.state_delta),
            request_input_id=raw_decision.request_input_id,
            request_input_prompt=raw_decision.request_input_prompt,
        )
    if isinstance(raw_decision, bool):
        return JudgmentDecision(
            output=default_output,
            route="pass" if raw_decision else "fail",
            escalate=raw_decision,
        )
    if isinstance(raw_decision, str):
        return JudgmentDecision(
            output=default_output,
            route=raw_decision,
            transfer_to_agent=raw_decision if transfer_to_sub_agent else None,
        )
    if isinstance(raw_decision, list):
        return JudgmentDecision(output=default_output, route=raw_decision)
    return JudgmentDecision(output=raw_decision)


def _extract_node_input(ctx: InvocationContext) -> Any:
    """Extract predecessor output or latest user message text from ADK InvocationContext."""
    if ctx.session and ctx.session.events:
        for ev in reversed(ctx.session.events):
            if getattr(ev, "output", None) is not None:
                return ev.output
            if ev.content and ev.content.parts:
                texts = [p.text for p in ev.content.parts if getattr(p, "text", None)]
                if texts:
                    return "\n".join(texts)
    if ctx.user_content and ctx.user_content.parts:
        texts = [p.text for p in ctx.user_content.parts if getattr(p, "text", None)]
        if texts:
            return "\n".join(texts)
    return None


def _serialize_output(val: Any) -> Any:
    """Ensure output is JSON-serializable for ADK Event.output and state_delta."""
    if isinstance(val, BaseModel):
        return val.model_dump()
    return val


class JudgmentAgent(BaseAgent):
    """Model-agnostic System One / Judgment step for Google ADK 2.0 Workflows & Composite Agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "judgment-latest"
    schema_cls: type[JudgmentSchema] | None = None
    questions: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None
    state_keys: Sequence[str] | None = None
    state_builder: Callable[..., Any] | None = None
    output_key: str | None = None
    decide: Callable[..., Any] | None = None
    on_error: Callable[[Exception, dict[str, Any]], JudgmentDecision] | None = None
    transfer_to_sub_agent: bool = False
    backend: BaseJudgmentBackend | None = None

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        model: str = "judgment-latest",
        schema: type[JudgmentSchema] | None = None,
        questions: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
        state_keys: Sequence[str] | None = None,
        state_builder: Callable[..., Any] | None = None,
        output_key: str | None = None,
        decide: Callable[..., Any] | None = None,
        on_error: Callable[[Exception, dict[str, Any]], JudgmentDecision] | None = None,
        transfer_to_sub_agent: bool = False,
        backend: BaseJudgmentBackend | None = None,
        sub_agents: Sequence[BaseAgent] | None = None,
    ) -> None:
        if schema is None and questions is None and not hasattr(self, "_custom_evaluate"):
            raise JudgmentConfigError(
                f"JudgmentAgent '{name}' requires either `schema` (a JudgmentSchema subclass) "
                "or `questions` (a dict or callable)."
            )
        super().__init__(
            name=name,
            description=description or f"Evaluates calibrated judgments using {model}.",
            sub_agents=list(sub_agents or []),
            model=model,
            schema_cls=schema,
            questions=questions,
            state_keys=list(state_keys) if state_keys is not None else None,
            state_builder=state_builder,
            output_key=output_key,
            decide=decide,
            on_error=on_error,
            transfer_to_sub_agent=transfer_to_sub_agent,
            backend=backend or TypeSafeBackend(default_model=model),
        )

    def _resolve_state(
        self, session_state: dict[str, Any], node_input: Any
    ) -> Any:
        if self.state_builder is not None:
            sig = inspect.signature(self.state_builder)
            if len(sig.parameters) >= 2:
                return self.state_builder(session_state, node_input)
            return self.state_builder(session_state)
        if self.state_keys is not None:
            return {k: session_state.get(k) for k in self.state_keys}
        if node_input is not None and node_input != "":
            return node_input
        return session_state

    def _resolve_questions(
        self, state_payload: Any, session_state: dict[str, Any], node_input: Any
    ) -> Mapping[str, Any]:
        if self.schema_cls is not None:
            return self.schema_cls.build_questions()
        if callable(self.questions):
            sig = inspect.signature(self.questions)
            if len(sig.parameters) >= 2:
                return self.questions(state_payload, node_input)
            return self.questions(state_payload)
        return dict(self.questions or {})

    async def _evaluate_core(
        self, session_state: dict[str, Any], node_input: Any
    ) -> Any:
        state_payload = self._resolve_state(session_state, node_input)
        resolved_questions = self._resolve_questions(
            state_payload, session_state, node_input
        )
        active_backend = self.backend or TypeSafeBackend(default_model=self.model)
        raw_result: JudgmentResult = await active_backend.evaluate(
            state=state_payload,
            questions=resolved_questions,
            model=self.model,
        )
        if self.schema_cls is not None:
            return self.schema_cls.from_result(raw_result)
        return raw_result

    def _invoke_decide(
        self, typed_output: Any, session_state: dict[str, Any], node_input: Any
    ) -> JudgmentDecision:
        default_serialized = _serialize_output(typed_output)
        if self.decide is None:
            return JudgmentDecision(output=default_serialized)
        sig = inspect.signature(self.decide)
        if len(sig.parameters) >= 3:
            raw_decision = self.decide(typed_output, session_state, node_input)
        elif len(sig.parameters) == 2:
            raw_decision = self.decide(typed_output, session_state)
        else:
            raw_decision = self.decide(typed_output)
        return normalize_decision(
            raw_decision,
            default_output=default_serialized,
            transfer_to_sub_agent=self.transfer_to_sub_agent,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        session_state = dict(ctx.session.state)
        node_input = _extract_node_input(ctx)

        try:
            typed_output = await self._evaluate_core(session_state, node_input)
            decision = self._invoke_decide(typed_output, session_state, node_input)
        except Exception as exc:
            if self.on_error is not None:
                decision = self.on_error(exc, session_state)
            else:
                raise

        if decision.request_input_id:
            yield RequestInput(
                interrupt_id=decision.request_input_id,
                message=decision.request_input_prompt or "Additional input required.",
            )
            return

        final_output = _serialize_output(decision.output)
        state_delta = dict(decision.state_delta)
        if self.output_key:
            state_delta[self.output_key] = final_output

        for k, v in state_delta.items():
            ctx.session.state[k] = v

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            output=final_output,
            actions=EventActions(
                state_delta=state_delta,
                route=decision.route,
                escalate=decision.escalate,
                transfer_to_agent=decision.transfer_to_agent,
            ),
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            final_output
                            if isinstance(final_output, str)
                            else json.dumps(final_output, indent=2, default=str)
                        )
                    )
                ],
            ),
        )


def judgment_node(
    *,
    name: str | None = None,
    schema: type[JudgmentSchema] | None = None,
    questions: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
    model: str = "judgment-latest",
    state_keys: Sequence[str] | None = None,
    state_builder: Callable[..., Any] | None = None,
    output_key: str | None = None,
    backend: BaseJudgmentBackend | None = None,
) -> Callable[[Callable[..., Any]], JudgmentAgent]:
    """Decorator turning a Python decision function `(result, state) -> decision` into a JudgmentAgent."""

    def decorator(func: Callable[..., Any]) -> JudgmentAgent:
        return JudgmentAgent(
            name=name or func.__name__,
            description=func.__doc__ or "",
            model=model,
            schema=schema,
            questions=questions,
            state_keys=state_keys,
            state_builder=state_builder,
            output_key=output_key,
            decide=func,
            backend=backend,
        )

    return decorator
```

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest tests/unit/test_judgment_agent.py -v`
Expected: PASS (all 4 tests passing)

- [ ] **Step 5: Commit Task 3**

```bash
git add judgment_base_agent/agent.py tests/unit/test_judgment_agent.py
git commit -m "feat: add JudgmentAgent, JudgmentDecision, and @judgment_node"
```

---

### Task 4: Universal Workflow Presets (`JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap` / `JudgmentBatch`), Package Exports (`__init__.py`) & ADK 2.0 Integration Tests

**Files:**
- Create: `judgment_base_agent/presets.py`
- Create: `judgment_base_agent/__init__.py`
- Create: `README.md`
- Test: `tests/unit/test_presets.py`
- Test: `tests/integration/test_adk_workflows.py`

**Interfaces:**
- Consumes: `JudgmentAgent`, `JudgmentDecision`, `JudgmentSchema`, `Choice`, `Score`, `Noul`, `JudgmentResult`, `BaseJudgmentBackend`, `TypeSafeBackend`
- Produces:
  - `JudgmentSwitch` (aliases: `JudgmentRouter`, `SystemOneRouter`, `SystemOneRouter`)
  - `JudgmentGuard` (aliases: `JudgmentGate`, `SystemOneGate`, `SystemOneGate`)
  - `JudgmentBatchEntry[TItem, TJudgment]`, `JudgmentBatch[TItem, TJudgment]` (`.items()`, `.filter()`, `.rank_by()`, `.map()`, `.all()`, `.any()`, `.reduce()`, `.to_dict()`), `JudgmentMap`
  - `judgment_base_agent.__init__` exporting all primary symbols and aliases (`SystemOneAgent`, `SystemOneAgent`, etc.)

- [ ] **Step 1: Write the failing unit test `tests/unit/test_presets.py` and integration test `tests/integration/test_adk_workflows.py`**

Create `tests/unit/test_presets.py`:
```python
"""Unit tests for JudgmentSwitch, JudgmentGuard, JudgmentMap, and JudgmentBatch."""

import pytest
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from judgment_base_agent import (
    Choice,
    ChoiceJudgment,
    SystemOneAgent,
    SystemOneGate,
    SystemOneRouter,
    JudgmentBatch,
    JudgmentField,
    JudgmentGuard,
    JudgmentMap,
    JudgmentRouter,
    JudgmentSchema,
    JudgmentSwitch,
    MockJudgmentBackend,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
    SystemOneAgent,
)


def test_aliases_are_identical() -> None:
    assert SystemOneAgent is SystemOneAgent
    assert JudgmentRouter is JudgmentSwitch
    assert SystemOneRouter is JudgmentSwitch
    assert SystemOneGate is JudgmentGuard


@pytest.mark.asyncio
async def test_judgment_switch_confidence_floor_and_custom_policy() -> None:
    low_conf_backend = MockJudgmentBackend(
        responses={"route": {"choice": "billing", "confidence": 0.42}}
    )
    switch = JudgmentSwitch(
        name="ticket_router",
        instructions="Which team should handle `ticket`?",
        routes={"billing": "Charges", "tech": "Bugs"},
        confidence_floor=0.60,
        uncertain_route="human_triage",
        backend=low_conf_backend,
    )

    app = App(name="app", root_agent=switch)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="ambiguous")]),
        )
    ]
    assert events[-1].actions.route == "human_triage"


@pytest.mark.asyncio
async def test_judgment_guard_pass_and_fail_escalation() -> None:
    backend = MockJudgmentBackend(responses={"guard": 0.85})
    guard = JudgmentGuard(
        name="safety_guard",
        instructions="Is the draft completely grounded?",
        threshold=0.75,
        escalate_on_pass=True,
        backend=backend,
    )

    app = App(name="app", root_agent=guard)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="check")]),
        )
    ]
    assert events[-1].actions.route == "pass"
    assert events[-1].actions.escalate is True


class DocEvalSchema(JudgmentSchema):
    relevant: NoulJudgment = JudgmentField(Noul(instructions="Is doc relevant?"))
    quality: ScoreJudgment = JudgmentField(
        Score(instructions="Doc quality?", criteria=["poor", "good", "excellent"])
    )


@pytest.mark.asyncio
async def test_judgment_map_filter_rank_map_and_reduce() -> None:
    backend = MockJudgmentBackend(
        responses={
            "item_0__relevant": 0.92,
            "item_0__quality": 1.8,
            "item_1__relevant": 0.20,
            "item_1__quality": 0.5,
            "item_2__relevant": 0.85,
            "item_2__quality": 1.95,
            "holistic_ok": 0.90,
        }
    )

    mapper = JudgmentMap(
        name="doc_reranker",
        items_key="docs",
        output_key="reranked_docs",
        item_schema=DocEvalSchema,
        global_questions={"holistic_ok": Noul(instructions="Are docs sufficient?")},
        transform=lambda batch, state: {
            "top_docs": (
                batch.filter(lambda item, j: j.relevant.noul >= 0.50)
                .rank_by(lambda item, j: j.quality.score, top_k=2)
                .map(lambda item, j: {"id": item["id"], "score": j.quality.score})
            ),
            "all_relevant": batch.all(lambda item, j: j.relevant.noul >= 0.50),
            "any_relevant": batch.any(lambda item, j: j.relevant.noul >= 0.50),
            "total_score": batch.reduce(
                lambda acc, item, j: acc + j.quality.score, 0.0
            ),
            "holistic": batch.global_result.noul("holistic_ok"),
        },
        backend=backend,
    )

    app = App(name="app", root_agent=mapper)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="app",
        user_id="u1",
        state={
            "docs": [
                {"id": "d0", "text": "First relevant doc"},
                {"id": "d1", "text": "Irrelevant noise"},
                {"id": "d2", "text": "Best relevant doc"},
            ]
        },
    )

    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="run")]),
        )
    ]
    # Single batched backend call for all 3 items + global question!
    assert len(backend.calls) == 1
    out = events[-1].output
    assert [d["id"] for d in out["top_docs"]] == ["d2", "d0"]
    assert out["all_relevant"] is False
    assert out["any_relevant"] is True
    assert out["total_score"] == pytest.approx(1.8 + 0.5 + 1.95)
    assert out["holistic"] == pytest.approx(0.90)
```

Create `tests/integration/test_adk_workflows.py`:
```python
"""End-to-end integration tests for ADK 2.0 Graph Workflow and Composite Agents (SequentialAgent, LoopAgent)."""

import pytest
from google.adk.agents import BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow
from google.genai import types

from judgment_base_agent import (
    JudgmentField,
    JudgmentGuard,
    JudgmentMap,
    JudgmentSchema,
    JudgmentSwitch,
    MockJudgmentBackend,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
)


@pytest.mark.asyncio
async def test_adk2_graph_workflow_conditional_routing_with_judgment_switch() -> None:
    backend = MockJudgmentBackend(
        responses={"route": {"choice": "refund", "confidence": 0.94}}
    )

    router = JudgmentSwitch(
        name="intent_router",
        instructions="Classify customer request intent",
        routes={"refund": "Customer wants money back", "tech": "Technical issue"},
        backend=backend,
    )

    def handle_refund(node_input: dict) -> str:
        return f"REFUND_PROCESSED:{node_input['choices']['route']['choice']}"

    def handle_tech(node_input: dict) -> str:
        return "TECH_SUPPORT"

    wf = Workflow(
        name="customer_workflow",
        edges=[
            ("START", router),
            (router, handle_refund, "refund"),
            (router, handle_tech, "tech"),
        ],
    )

    app = App(name="wf_app", root_agent=wf)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="wf_app", user_id="u1")

    outputs = []
    async for ev in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text="Please refund my duplicate charge")]
        ),
    ):
        if ev.output is not None:
            outputs.append(ev.output)

    assert "REFUND_PROCESSED:refund" in outputs


class DraftProducer(BaseAgent):
    """Test helper agent that increments attempt counter in state."""

    async def _run_async_impl(self, ctx: InvocationContext):
        attempt = int(ctx.session.state.get("attempt", 0)) + 1
        ctx.session.state["attempt"] = attempt
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta={"attempt": attempt}),
        )


@pytest.mark.asyncio
async def test_adk_sequential_and_loop_agent_with_judgment_guard() -> None:
    # Passes on attempt 2
    backend = MockJudgmentBackend(
        responses=lambda state, q, m: {"guard": 0.92 if state.get("attempt", 0) >= 2 else 0.30}
    )

    guard = JudgmentGuard(
        name="quality_gate",
        instructions="Is the draft grounded?",
        threshold=0.80,
        output_key="gate_result",
        backend=backend,
    )

    loop = LoopAgent(
        name="refinement_loop",
        sub_agents=[DraftProducer(name="producer"), guard],
        max_iterations=5,
    )

    pipeline = SequentialAgent(
        name="main_pipeline",
        sub_agents=[loop],
    )

    app = App(name="loop_app", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="loop_app", user_id="u1")

    async for _ in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="start")]),
    ):
        pass

    updated_session = await runner.session_service.get_session(
        app_name="loop_app", user_id="u1", session_id=session.id
    )
    assert updated_session.state["attempt"] == 2
    assert updated_session.state["gate_result"]["nouls"]["guard"]["noul"] == pytest.approx(0.92)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest tests/unit/test_presets.py tests/integration/test_adk_workflows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgment_base_agent.presets'`

- [ ] **Step 3: Implement `judgment_base_agent/presets.py`, `judgment_base_agent/__init__.py`, and `README.md`**

Create `judgment_base_agent/presets.py`:
```python
"""Universal workflow presets: JudgmentSwitch, JudgmentGuard, and JudgmentMap (JudgmentBatch)."""

from __future__ import annotations

from collections.abc import Callable, Generic, Mapping, Sequence
from dataclasses import dataclass
import inspect
from typing import Any, TypeVar

from google.adk.agents import BaseAgent

from judgment_base_agent.agent import (
    JudgmentAgent,
    JudgmentDecision,
    _serialize_output,
    normalize_decision,
)
from judgment_base_agent.backends.base import BaseJudgmentBackend
from judgment_base_agent.backends.typesafe import TypeSafeBackend
from judgment_base_agent.errors import JudgmentConfigError
from judgment_base_agent.primitives import Choice, JudgmentResult, Noul
from judgment_base_agent.schema import JudgmentSchema

TItem = TypeVar("TItem")
TJudgment = TypeVar("TJudgment")
TOut = TypeVar("TOut")
TAcc = TypeVar("TAcc")


class JudgmentSwitch(JudgmentAgent):
    """Multi-way semantic router (`switch` / `match`) for ADK 2.0 Workflow edges and sub-agents."""

    def __init__(
        self,
        *,
        name: str,
        instructions: str | None = None,
        routes: Mapping[str, str | None] | Sequence[str] | None = None,
        schema: type[JudgmentSchema] | None = None,
        questions: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
        route_key: str = "route",
        confidence_floor: float | None = None,
        uncertain_route: str = "uncertain",
        route_policy: Callable[[Any, dict[str, Any]], Any] | None = None,
        transfer_to_sub_agent: bool = False,
        model: str = "judgment-latest",
        state_keys: Sequence[str] | None = None,
        state_builder: Callable[..., Any] | None = None,
        output_key: str | None = None,
        on_error: Callable[[Exception, dict[str, Any]], JudgmentDecision] | None = None,
        backend: BaseJudgmentBackend | None = None,
        sub_agents: Sequence[BaseAgent] | None = None,
    ) -> None:
        resolved_questions = questions
        if schema is None and resolved_questions is None:
            if not instructions or not routes:
                raise JudgmentConfigError(
                    f"JudgmentSwitch '{name}' requires either (`instructions` + `routes`) or (`schema` / `questions`)."
                )
            resolved_questions = {
                route_key: Choice(instructions=instructions, criteria=routes)
            }

        sub_agent_names = {sa.name for sa in (sub_agents or [])}

        def _default_decide(res: Any, state: dict[str, Any]) -> JudgmentDecision:
            if route_policy is not None:
                return normalize_decision(
                    route_policy(res, state),
                    default_output=_serialize_output(res),
                    transfer_to_sub_agent=transfer_to_sub_agent,
                )
            raw_res = res.raw_result if isinstance(res, JudgmentSchema) and res.raw_result else res
            selected = raw_res.choice(route_key)
            conf = raw_res.confidence(route_key)
            if confidence_floor is not None and conf < confidence_floor:
                chosen_route = uncertain_route
            else:
                chosen_route = selected

            target_agent = (
                chosen_route
                if (transfer_to_sub_agent or chosen_route in sub_agent_names)
                else None
            )
            return JudgmentDecision(
                output=_serialize_output(res),
                route=chosen_route,
                transfer_to_agent=target_agent,
            )

        super().__init__(
            name=name,
            model=model,
            schema=schema,
            questions=resolved_questions,
            state_keys=state_keys,
            state_builder=state_builder,
            output_key=output_key,
            decide=_default_decide,
            on_error=on_error,
            transfer_to_sub_agent=transfer_to_sub_agent,
            backend=backend,
            sub_agents=sub_agents,
        )


class JudgmentGuard(JudgmentAgent):
    """Semantic assertion / guardrail / loop-exit gate (`if` / `while`) for ADK workflows."""

    def __init__(
        self,
        *,
        name: str,
        instructions: str | None = None,
        criteria: Mapping[str, str] | None = None,
        threshold: float = 0.70,
        schema: type[JudgmentSchema] | None = None,
        questions: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
        guard_key: str = "guard",
        predicate: Callable[[Any, dict[str, Any]], bool | JudgmentDecision] | None = None,
        pass_route: str = "pass",
        fail_route: str = "fail",
        escalate_on_pass: bool = True,
        model: str = "judgment-latest",
        state_keys: Sequence[str] | None = None,
        state_builder: Callable[..., Any] | None = None,
        output_key: str | None = None,
        on_error: Callable[[Exception, dict[str, Any]], JudgmentDecision] | None = None,
        backend: BaseJudgmentBackend | None = None,
    ) -> None:
        resolved_questions = questions
        if schema is None and resolved_questions is None:
            if not instructions:
                raise JudgmentConfigError(
                    f"JudgmentGuard '{name}' requires `instructions`, `schema`, or `questions`."
                )
            resolved_questions = {
                guard_key: Noul(instructions=instructions, criteria=criteria)
            }

        def _guard_decide(res: Any, state: dict[str, Any]) -> JudgmentDecision:
            if predicate is not None:
                verdict = predicate(res, state)
                if isinstance(verdict, JudgmentDecision):
                    return verdict
                passed = bool(verdict)
            else:
                raw_res = (
                    res.raw_result
                    if isinstance(res, JudgmentSchema) and res.raw_result
                    else res
                )
                passed = raw_res.noul(guard_key) >= threshold

            return JudgmentDecision(
                output=_serialize_output(res),
                route=pass_route if passed else fail_route,
                escalate=escalate_on_pass if passed else False,
            )

        super().__init__(
            name=name,
            model=model,
            schema=schema,
            questions=resolved_questions,
            state_keys=state_keys,
            state_builder=state_builder,
            output_key=output_key,
            decide=_guard_decide,
            on_error=on_error,
            backend=backend,
        )


@dataclass(frozen=True)
class JudgmentBatchEntry(Generic[TItem, TJudgment]):
    """Immutable pairing of a single collection item with its evaluated judgment."""

    index: int
    item: TItem
    judgment: TJudgment
    raw_result: JudgmentResult


@dataclass(frozen=True)
class JudgmentBatch(Generic[TItem, TJudgment]):
    """Composable Map / Filter / Rank / Reduce result container for JudgmentMap."""

    entries: tuple[JudgmentBatchEntry[TItem, TJudgment], ...]
    global_result: JudgmentResult

    def items(self) -> list[TItem]:
        """Return the list of items currently in the batch."""
        return [e.item for e in self.entries]

    def judgments(self) -> list[TJudgment]:
        """Return the list of judgments corresponding to each item in the batch."""
        return [e.judgment for e in self.entries]

    def filter(
        self, predicate: Callable[[TItem, TJudgment], bool]
    ) -> JudgmentBatch[TItem, TJudgment]:
        """Return a new JudgmentBatch containing only entries satisfying `predicate(item, judgment)`."""
        filtered = tuple(
            e for e in self.entries if predicate(e.item, e.judgment)
        )
        return JudgmentBatch(entries=filtered, global_result=self.global_result)

    def rank_by(
        self,
        key: Callable[[TItem, TJudgment], float],
        *,
        reverse: bool = True,
        top_k: int | None = None,
    ) -> JudgmentBatch[TItem, TJudgment]:
        """Return a new JudgmentBatch sorted by `key(item, judgment)` and optionally truncated to `top_k`."""
        sorted_entries = sorted(
            self.entries,
            key=lambda e: float(key(e.item, e.judgment)),
            reverse=reverse,
        )
        if top_k is not None:
            sorted_entries = sorted_entries[:top_k]
        return JudgmentBatch(
            entries=tuple(sorted_entries), global_result=self.global_result
        )

    def map(self, mapper: Callable[[TItem, TJudgment], TOut]) -> list[TOut]:
        """Transform each `(item, judgment)` pair in the batch using `mapper`."""
        return [mapper(e.item, e.judgment) for e in self.entries]

    def all(self, predicate: Callable[[TItem, TJudgment], bool]) -> bool:
        """Return True if all entries in the batch satisfy `predicate(item, judgment)`."""
        return all(predicate(e.item, e.judgment) for e in self.entries)

    def any(self, predicate: Callable[[TItem, TJudgment], bool]) -> bool:
        """Return True if at least one entry in the batch satisfies `predicate(item, judgment)`."""
        return any(predicate(e.item, e.judgment) for e in self.entries)

    def reduce(
        self,
        reducer: Callable[[TAcc, TItem, TJudgment], TAcc],
        initial: TAcc,
    ) -> TAcc:
        """Fold over all entries in the batch starting from `initial`."""
        acc = initial
        for e in self.entries:
            acc = reducer(acc, e.item, e.judgment)
        return acc

    def to_dict(self) -> dict[str, Any]:
        """Serialize the batch and global judgments to a JSON-serializable dictionary."""
        return {
            "entries": [
                {
                    "index": e.index,
                    "item": _serialize_output(e.item),
                    "judgment": _serialize_output(e.judgment),
                }
                for e in self.entries
            ],
            "global_result": self.global_result.model_dump(),
        }


class JudgmentMap(JudgmentAgent):
    """Universal collection primitive (Map / Filter / Rank / Reduce) executing in one batched call."""

    _custom_evaluate = True

    items_key: str | None = None
    items_getter: Callable[..., Sequence[Any]] | None = None
    item_schema: type[JudgmentSchema] | None = None
    item_questions: Callable[[Any, int], Mapping[str, Any]] | None = None
    global_questions: (
        Mapping[str, Any]
        | Callable[[dict[str, Any], Sequence[Any]], Mapping[str, Any]]
        | None
    ) = None
    item_state_builder: Callable[[Any, int, dict[str, Any]], Any] | None = None
    transform: Callable[[JudgmentBatch[Any, Any], dict[str, Any]], Any] | None = None

    def __init__(
        self,
        *,
        name: str,
        items_key: str | None = None,
        items_getter: Callable[..., Sequence[Any]] | None = None,
        item_schema: type[JudgmentSchema] | None = None,
        item_questions: Callable[[Any, int], Mapping[str, Any]] | None = None,
        global_questions: (
            Mapping[str, Any]
            | Callable[[dict[str, Any], Sequence[Any]], Mapping[str, Any]]
            | None
        ) = None,
        item_state_builder: Callable[[Any, int, dict[str, Any]], Any] | None = None,
        transform: Callable[[JudgmentBatch[Any, Any], dict[str, Any]], Any] | None = None,
        decide: Callable[[JudgmentBatch[Any, Any], dict[str, Any]], Any] | None = None,
        model: str = "judgment-latest",
        output_key: str | None = None,
        on_error: Callable[[Exception, dict[str, Any]], JudgmentDecision] | None = None,
        backend: BaseJudgmentBackend | None = None,
    ) -> None:
        if item_schema is None and item_questions is None:
            raise JudgmentConfigError(
                f"JudgmentMap '{name}' requires either `item_schema` or `item_questions`."
            )
        super().__init__(
            name=name,
            model=model,
            output_key=output_key,
            decide=decide,
            on_error=on_error,
            backend=backend,
        )
        self.items_key = items_key
        self.items_getter = items_getter
        self.item_schema = item_schema
        self.item_questions = item_questions
        self.global_questions = global_questions
        self.item_state_builder = item_state_builder
        self.transform = transform

    def _extract_items(
        self, session_state: dict[str, Any], node_input: Any
    ) -> list[Any]:
        if self.items_getter is not None:
            sig = inspect.signature(self.items_getter)
            if len(sig.parameters) >= 2:
                return list(self.items_getter(session_state, node_input) or [])
            return list(self.items_getter(session_state) or [])
        if self.items_key is not None:
            return list(session_state.get(self.items_key) or [])
        if isinstance(node_input, Sequence) and not isinstance(node_input, (str, bytes)):
            return list(node_input)
        return []

    async def _evaluate_core(
        self, session_state: dict[str, Any], node_input: Any
    ) -> JudgmentBatch[Any, Any]:
        items = self._extract_items(session_state, node_input)
        if not items:
            return JudgmentBatch(
                entries=(),
                global_result=JudgmentResult(model=self.model),
            )

        batched_questions: dict[str, Any] = {}
        item_q_keys: list[dict[str, str]] = []
        serialized_items: list[Any] = []

        base_schema_questions = (
            self.item_schema.build_questions() if self.item_schema is not None else None
        )

        for idx, item in enumerate(items):
            per_item_qs = (
                base_schema_questions
                if base_schema_questions is not None
                else dict(self.item_questions(item, idx))  # type: ignore[misc]
            )
            key_map: dict[str, str] = {}
            for field_key, q_obj in per_item_qs.items():
                namespaced_key = f"item_{idx}__{field_key}"
                batched_questions[namespaced_key] = q_obj
                key_map[field_key] = namespaced_key
            item_q_keys.append(key_map)
            serialized_items.append(
                self.item_state_builder(item, idx, session_state)
                if self.item_state_builder is not None
                else _serialize_output(item)
            )

        resolved_global: dict[str, Any] = {}
        if self.global_questions is not None:
            resolved_global = dict(
                self.global_questions(session_state, items)
                if callable(self.global_questions)
                else self.global_questions
            )
            batched_questions.update(resolved_global)

        state_payload = {
            "items": serialized_items,
            "context": {
                k: v for k, v in session_state.items() if k != self.items_key
            },
        }

        active_backend = self.backend or TypeSafeBackend(default_model=self.model)
        combined_result = await active_backend.evaluate(
            state=state_payload,
            questions=batched_questions,
            model=self.model,
        )

        entries: list[JudgmentBatchEntry[Any, Any]] = []
        for idx, (item, key_map) in enumerate(zip(items, item_q_keys)):
            sub_choices = {
                fk: combined_result.choices[nk]
                for fk, nk in key_map.items()
                if nk in combined_result.choices
            }
            sub_scores = {
                fk: combined_result.scores[nk]
                for fk, nk in key_map.items()
                if nk in combined_result.scores
            }
            sub_nouls = {
                fk: combined_result.nouls[nk]
                for fk, nk in key_map.items()
                if nk in combined_result.nouls
            }
            item_res = JudgmentResult(
                choices=sub_choices,
                scores=sub_scores,
                nouls=sub_nouls,
                model=combined_result.model,
                usage=combined_result.usage,
            )
            typed_j: Any = (
                self.item_schema.from_result(item_res)
                if self.item_schema is not None
                else item_res
            )
            entries.append(
                JudgmentBatchEntry(
                    index=idx,
                    item=item,
                    judgment=typed_j,
                    raw_result=item_res,
                )
            )

        global_res = JudgmentResult(
            choices={
                k: v for k, v in combined_result.choices.items() if k in resolved_global
            },
            scores={
                k: v for k, v in combined_result.scores.items() if k in resolved_global
            },
            nouls={
                k: v for k, v in combined_result.nouls.items() if k in resolved_global
            },
            model=combined_result.model,
            usage=combined_result.usage,
        )
        return JudgmentBatch(entries=tuple(entries), global_result=global_res)

    def _invoke_decide(
        self,
        batch_output: JudgmentBatch[Any, Any],
        session_state: dict[str, Any],
        node_input: Any,
    ) -> JudgmentDecision:
        default_payload = (
            self.transform(batch_output, session_state)
            if self.transform is not None
            else batch_output.to_dict()
        )
        if self.decide is None:
            return JudgmentDecision(output=default_payload)
        raw_decision = self.decide(batch_output, session_state)
        return normalize_decision(
            raw_decision,
            default_output=default_payload,
            transfer_to_sub_agent=self.transfer_to_sub_agent,
        )
```

Create `judgment_base_agent/__init__.py`:
```python
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
SystemOneAgent = JudgmentAgent

JudgmentRouter = JudgmentSwitch
SystemOneRouter = JudgmentSwitch
SystemOneRouter = JudgmentSwitch

JudgmentGate = JudgmentGuard
SystemOneGate = JudgmentGuard
SystemOneGate = JudgmentGuard

__all__ = [
    "BaseJudgmentBackend",
    "Choice",
    "ChoiceJudgment",
    "ConfidenceTier",
    "SystemOneAgent",
    "SystemOneGate",
    "SystemOneRouter",
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
```

Create `README.md`:
```markdown
# `judgment-base-agent` — Calibrated Judgment Primitives for Google ADK

Model-agnostic **System One / Judgment** primitives (`JudgmentAgent`, `JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`) that make adding **System One (`model="judgment-latest"`)** or any calibrated judgment backend to **Google ADK (`google-adk >= 2.7.0`)** workflows effortless.

Works identically inside:
1. **ADK 2.0 Graph `Workflow` (`from google.adk.workflow import Workflow`)** — emitting `Event(output=..., actions=EventActions(route=..., state_delta=...))` and optional `RequestInput` HITL interrupts.
2. **ADK Composite Agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`)** — subclassing `BaseAgent` with automatic `state_delta` persistence, `LoopAgent` escalation (`escalate=True`), and sub-agent transfers.
```

- [ ] **Step 4: Run full test suite with coverage check**

Run: `/usr/local/google/home/mbonnardot/capstone/mbonnardot-medquad-assistant/venv/bin/pytest --cov=judgment_base_agent --cov-report=term-missing -v`
Expected: All unit and integration tests PASS with `>= 85%` coverage.

- [ ] **Step 5: Commit Task 4**

```bash
git add judgment_base_agent/presets.py judgment_base_agent/__init__.py README.md tests/unit/test_presets.py tests/integration/test_adk_workflows.py
git commit -m "feat: add JudgmentSwitch, JudgmentGuard, JudgmentMap, and ADK 2.0 workflow integration tests"
```
