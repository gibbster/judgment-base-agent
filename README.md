# `jev-base-agent`

**Calibrated System One / Judgment Primitives for Google ADK (`google-adk >= 2.7.0`)**

`jev-base-agent` provides drop-in, model-agnostic `BaseAgent` primitives that integrate calibrated judgment models—such as **Jev (`model="jev-latest"`)** via the TypeSafe SDK—into both **ADK 2.0 Graph `Workflow`** and **ADK Composite Agents** (`SequentialAgent`, `ParallelAgent`, `LoopAgent`).

---

## Why `jev-base-agent`?

Standard `LlmAgent` nodes rely on open-ended text generation (`Nom`) to make control-flow decisions, leading to brittle string parsing, uncalibrated confidence, and multiple round-trips.

`jev-base-agent` separates **fast, calibrated System One evaluation** from **user-owned control flow**:

- **Single Batched Call:** Evaluates multiple typed questions (`Noul`, `Kat`, `Skala`, `Nom`) or entire item collections in **one** inference call.
- **Calibrated Confidence Tiers:** Every answer carries a calibrated probability (`0.0–1.0`) and `ConfidenceTier` (`HIGH >= 0.85`, `MODERATE >= 0.70`, `LOW >= 0.55`, `UNCERTAIN < 0.55`).
- **Code Owns the Workflow:** Models return typed values and confidence scores; your Python `decide` policy owns routing, loop escalation, and human-in-the-loop interrupts.
- **Dual ADK Compatibility:** Works identically as a node in `google.adk.workflow.Workflow` (`Event(output=..., actions=EventActions(route=...))`) and inside `SequentialAgent` / `LoopAgent` (`state_delta`, `escalate`, `transfer_to_agent`).

---

## Installation

```bash
pip install -e .
export TYPESAFE_API_KEY="your-typesafe-api-key"
```

---

## Primitives Overview

| Primitive | Purpose | Aliases |
| :--- | :--- | :--- |
| **`JudgmentAgent`** | Base ADK agent evaluating a `JudgmentSchema` or question dict and running a `decide` policy. | `SystemOneAgent`, `JevAgent` |
| **`JudgmentSwitch`** | Conditional multi-route branching node with optional `confidence_floor` and `uncertain_route`. | `JudgmentRouter`, `SystemOneRouter`, `JevRouter` |
| **`JudgmentGuard`** | Binary gate (`Noul`) or custom predicate that routes, halts, or exits a `LoopAgent` (`escalate=True`). | `JudgmentGate`, `SystemOneGate`, `JevGate` |
| **`JudgmentMap`** | Evaluates `N` collection items in a **single batched call** and returns a composable `JudgmentBatch`. | — |
| **`@judgment_node`** | Decorator turning a typed Python policy function into a `JudgmentAgent`. | — |

### Question Types

| Type | Output | Description |
| :--- | :--- | :--- |
| `Noul(question)` | `bool` | Yes/No binary judgment with calibrated probability (`0.0–1.0`). |
| `Kat(question, options)` | `str` | Mutually exclusive classification across 2–26 options. |
| `Skala(question, low=0, high=10)` | `float` | Continuous bounded rating scale with normalized confidence. |
| `Nom(question)` | `str` | Short open-ended text extraction (returns value only; `confidence=None`). |

---

## Quick Start

### 1. Declarative `JudgmentSchema` + User-Owned Policy (`JudgmentAgent`)

Bundle binary, categorical, and continuous questions into one schema. All fields are evaluated in a single call, and your `decide` function controls routing:

```python
from jev_base_agent import (
    ConfidenceTier,
    JudgmentAgent,
    JudgmentDecision,
    JudgmentField,
    JudgmentSchema,
    Kat,
    Noul,
    Skala,
)


class TriageJudgment(JudgmentSchema):
    intent: str = JudgmentField(
        Kat("Classify user intent", ["billing", "technical", "account"])
    )
    is_urgent: bool = JudgmentField(
        Noul("Is the user blocked or reporting a production outage?")
    )
    severity: float = JudgmentField(
        Skala("Rate severity from 0 (minor) to 10 (critical outage)", low=0, high=10)
    )


def triage_policy(triage: TriageJudgment) -> JudgmentDecision:
    if triage.tier("intent") == ConfidenceTier.UNCERTAIN:
        return JudgmentDecision(route="clarify_with_user")
    if triage.is_urgent and triage.severity >= 7.0:
        return JudgmentDecision(route="pagerduty_escalation")
    return JudgmentDecision(route=f"handle_{triage.intent}")


triage_agent = JudgmentAgent(
    name="triage_judgment",
    schema=TriageJudgment,
    state_template="Customer Ticket:\n{ticket_text}",
    output_key="triage",
    decide=triage_policy,
)
```

---

### 2. ADK 2.0 Graph `Workflow` (`JudgmentSwitch`)

Use `JudgmentSwitch` inside `google.adk.workflow.Workflow` to route across conditional edges with built-in confidence floors:

```python
from google.adk.workflow import START, Workflow
from jev_base_agent import JudgmentSwitch

router = JudgmentSwitch(
    name="router",
    instructions="Classify the user request",
    routes={
        "search": "Requires retrieving external documentation",
        "code": "Requires writing or debugging source code",
        "direct": "Can be answered directly from context",
    },
    confidence_floor=0.70,
    uncertain_route="clarify",
    state_template="User request: {user_query}",
    output_key="routing_decision",
)

workflow = (
    Workflow(name="assistant_workflow")
    .add_node(router)
    .add_node(search_node)
    .add_node(code_node)
    .add_node(clarify_node)
    .add_edge(START, "router")
    .add_edge("router", "search_node", when="search")
    .add_edge("router", "code_node", when="code")
    .add_edge("router", "clarify_node", when="clarify")
)
```

---

### 3. ADK `LoopAgent` & `SequentialAgent` (`JudgmentGuard`)

Use `JudgmentGuard` with `escalate_on_pass=True` inside a `LoopAgent` to exit the refinement loop as soon as the quality check passes:

```python
from google.adk.agents import LoopAgent, SequentialAgent
from jev_base_agent import JudgmentGuard

quality_gate = JudgmentGuard(
    name="quality_gate",
    question="Is the draft factually grounded in the source context and free of PII?",
    threshold=0.85,
    state_template="Source:\n{source_docs}\n\nDraft:\n{draft_response}",
    output_key="quality_check",
    escalate_on_pass=True,
)

refinement_loop = LoopAgent(
    name="draft_refinement_loop",
    sub_agents=[draft_writer_agent, quality_gate],
    max_iterations=3,
)

pipeline = SequentialAgent(
    name="grounded_qa_pipeline",
    sub_agents=[retriever_agent, refinement_loop],
)
```

---

### 4. Universal Collection Batching (`JudgmentMap` & `JudgmentBatch`)

Evaluate `N` items in **one batched `system_one` call** and filter, rank, transform, or aggregate them with `JudgmentBatch`:

```python
from jev_base_agent import JudgmentField, JudgmentMap, JudgmentSchema, Noul, Skala


class CandidateEval(JudgmentSchema):
    is_relevant: bool = JudgmentField(Noul("Does this item help answer the query?"))
    quality: float = JudgmentField(Skala("Rate source quality 0-10", low=0, high=10))


rerank_candidates = JudgmentMap(
    name="rerank_candidates",
    items_key="raw_candidates",
    item_template="Query: {user_query}\nCandidate: {item}",
    item_schema=CandidateEval,
    output_key="top_candidates",
    transform=lambda batch: (
        batch.filter(
            lambda e: e.judgment.is_relevant and e.confidence("is_relevant") >= 0.70
        )
        .rank_by(lambda e: e.judgment.quality, reverse=True)
        .items[:3]
    ),
)
```

`JudgmentBatch` supports:
- `.filter(fn)` -> `JudgmentBatch[TItem, TJudgment]`
- `.rank_by(key_fn, reverse=True)` -> `JudgmentBatch[TItem, TJudgment]`
- `.map(fn)` -> `list[R]`
- `.all(fn)` / `.any(fn)` -> `bool`
- `.reduce(fn, initial)` -> `R`
- `.items` / `.judgments` / `.global_result`

---

### 5. Deterministic Testing with `MockJudgmentBackend`

Write fast, zero-network unit and integration tests for your ADK workflows by injecting `MockJudgmentBackend`:

```python
from jev_base_agent import JudgmentGuard, MockJudgmentBackend

mock_backend = MockJudgmentBackend(
    answers={"pass": (True, 0.94)}
)

guard = JudgmentGuard(
    name="quality_gate",
    question="Is the response safe?",
    threshold=0.85,
    backend=mock_backend,
)
```

---

## Development & Testing

```bash
pytest --cov=jev_base_agent --cov-report=term-missing -v
```
