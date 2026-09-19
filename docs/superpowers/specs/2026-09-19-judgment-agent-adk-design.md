# ADK Judgment Primitives (`judgment-base-agent`) Design Specification

**Date:** 2026-09-19  
**Status:** Draft (Awaiting User Review)  
**Target Framework:** Google ADK (`google-adk >= 2.7.0`) & TypeSafe SDK (`typesafe-sdk >= 0.1.0`)

---

## 1. Purpose & Design Philosophy

Modern AI workflows built with **Google ADK** combine two distinct cognitive modes:
- **System 2 (Slow / Generative Reasoning):** Handled by ADK's `LlmAgent` (`model="gemini-3.8-flash"`, etc.), which generates free-form text, reasons across turns, and invokes tools. However, using `output_schema` on `LlmAgent` disables tool calling and delegation, incurs full generative latency, and produces no calibrated probabilities.
- **System 1 (Fast / Calibrated Judgment):** Handled by System One models such as **System One** (`model="judgment-latest"`), which evaluate application state against constrained primitives (`Choice`, `Score`, `Noul`) in parallel and return typed answers with calibrated probability distributions and confidence scores.

Following TypeSafe's core principle — *"Code owns the workflow; the model supplies programmable common sense where ordinary code needs semantic understanding"* — this library provides **model-agnostic judgment primitives** (`JudgmentAgent`, `JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`) that work natively as first-class steps inside **both**:
1. **ADK 2.0 Graph `Workflow` (`from google.adk.workflow import Workflow`)** — emitting `Event(output=..., actions=EventActions(route=..., state_delta=...))` and optional `RequestInput` interrupts for Human-in-the-Loop (HITL).
2. **ADK Composite Workflow Agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) & Multi-Agent Trees** — subclassing `google.adk.agents.BaseAgent` to read `ctx.session.state`, persist `output_key` via `state_delta`, terminate loops via `escalate=True`, and hand off execution via `transfer_to_agent`.

---

## 2. Package Architecture & Module Structure

```text
judgment_base_agent/
├── __init__.py           # Public API exports + aliases (JudgmentAgent, SystemOneAgent, SystemOneAgent, ...)
├── errors.py             # Custom exceptions (JudgmentConfigError, JudgmentEvaluationError)
├── primitives.py         # Model-agnostic question & answer models (Choice, Score, Noul, JudgmentResult, ...)
├── schema.py             # Declarative JudgmentSchema & JudgmentField for typed IDE-autocompleted outputs
├── backends/
│   ├── __init__.py
│   ├── base.py           # BaseJudgmentBackend Protocol
│   ├── typesafe.py       # TypeSafeBackend (default model="judgment-latest" via AsyncTypeSafeClient)
│   └── mock.py           # MockJudgmentBackend for deterministic, zero-network unit testing
├── agent.py              # Core JudgmentAgent(BaseAgent), JudgmentDecision, and @judgment_node decorator
└── presets.py            # Universal workflow primitives: JudgmentSwitch, JudgmentGuard, JudgmentMap, JudgmentBatch
```

### Naming & Model-Agnostic Aliases
- **Primary Names:** `JudgmentAgent`, `JudgmentSchema`, `JudgmentField`, `JudgmentDecision`, `JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`, `JudgmentBatch`, `@judgment_node`.
- **Aliases Exported in `__init__.py`:**
  - `SystemOneAgent = JudgmentAgent`, `SystemOneAgent = JudgmentAgent`
  - `JudgmentRouter = JudgmentSwitch`, `SystemOneRouter = JudgmentSwitch`, `SystemOneRouter = JudgmentSwitch`
  - `JudgmentGate = JudgmentGuard`, `SystemOneGate = JudgmentGuard`, `SystemOneGate = JudgmentGuard`

---

## 3. Core Primitives & Model-Agnostic Backend Protocol (`primitives.py`, `backends/`)

### 3.1 Question Primitives
Users may define questions using either the library's model-agnostic frozen dataclasses or native `typesafe_sdk` classes (`typesafe_sdk.Choice`, `typesafe_sdk.Score`, `typesafe_sdk.Noul`):

```python
@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: Mapping[str, str | None] | Sequence[str]

@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: Sequence[str]

@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: Mapping[str, str] | None = None
```

### 3.2 Answer & Result Models (Immutable Pydantic Models)
```python
ConfidenceTier = Literal["high", "medium", "low"]

class ChoiceJudgment(BaseModel):
    model_config = ConfigDict(frozen=True)
    choice: str
    probabilities: dict[str, float]
    confidence: float
    confidence_tier: ConfidenceTier

class ScoreJudgment(BaseModel):
    model_config = ConfigDict(frozen=True)
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float
    confidence_tier: ConfidenceTier

class NoulJudgment(BaseModel):
    model_config = ConfigDict(frozen=True)
    noul: float

class JudgmentUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: int = 0
    output_tokens: int = 0

class JudgmentResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    choices: dict[str, ChoiceJudgment] = Field(default_factory=dict)
    scores: dict[str, ScoreJudgment] = Field(default_factory=dict)
    nouls: dict[str, NoulJudgment] = Field(default_factory=dict)
    model: str = "judgment-latest"
    usage: JudgmentUsage | None = None

    def get(self, key: str) -> ChoiceJudgment | ScoreJudgment | NoulJudgment: ...
    def choice(self, key: str) -> str: ...
    def score(self, key: str) -> float: ...
    def noul(self, key: str) -> float: ...
    def confidence(self, key: str) -> float: ...
```

### 3.3 `BaseJudgmentBackend` Protocol
```python
class BaseJudgmentBackend(Protocol):
    async def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> JudgmentResult:
        ...
```
- **`TypeSafeBackend`**: Uses `typesafe_sdk.AsyncTypeSafeClient` (reading `TYPESAFE_API_KEY` from env or constructor argument, defaulting `model="judgment-latest"`). Converts `Choice`, `Score`, and `Noul` objects to `typesafe_sdk` primitives, executes `client.system_one(state=state, questions=questions, model=model)`, and normalizes both `response.answers` and `response.choices`/`scores`/`nouls` into `JudgmentResult`.
- **`MockJudgmentBackend`**: Accepts either a static mapping of question keys to answers or a callable `(state, questions, model) -> dict[str, Any]`, enabling 100% offline deterministic testing of ADK workflows.

---

## 4. Declarative `JudgmentSchema` (`schema.py`)

For static question sets, developers can declare a strongly-typed schema class (similar to Pydantic `BaseModel`) that binds each field to a `Choice`, `Score`, or `Noul` question and populates the corresponding `ChoiceJudgment`, `ScoreJudgment`, or `NoulJudgment` on evaluation:

```python
class TicketJudgment(JudgmentSchema):
    billing: NoulJudgment = JudgmentField(
        Noul(instructions="Is `ticket` about a billing or charge issue?")
    )
    tone: ChoiceJudgment = JudgmentField(
        Choice(
            instructions="What is the customer's tone in `ticket`?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        )
    )
    urgency: ScoreJudgment = JudgmentField(
        Score(
            instructions="How urgent is `ticket`?",
            criteria=["can wait", "this week", "today"],
        )
    )
```
- `TicketJudgment.build_questions() -> dict[str, Choice | Score | Noul]` extracts the question definitions.
- `TicketJudgment.from_result(result: JudgmentResult) -> TicketJudgment` validates and populates all fields while attaching `instance.raw_result` (`JudgmentResult`) for token usage and metadata inspection.
- Because `JudgmentSchema` inherits from `pydantic.BaseModel`, ADK 2.0 `FunctionNode`s with `node_input: TicketJudgment` automatically deserialize dicts produced by upstream `JudgmentAgent` nodes.

---

## 5. Core `JudgmentAgent(BaseAgent)` & User-Owned `decide` Contract (`agent.py`)

### 5.1 `JudgmentDecision` Control Envelope
Because user code owns workflow policy (thresholds, confidence handling, escalation, routing), `decide` callbacks can return a `JudgmentDecision` (or a shorthand `str` for route, `bool` for pass/escalate, or raw ADK `Event`):

```python
@dataclass(frozen=True)
class JudgmentDecision:
    output: Any = None                           # Overrides Event.output & state[output_key]
    route: str | int | bool | list[str] | None = None  # Sets EventActions.route for ADK 2.0 Workflow edges
    escalate: bool | None = None                 # Sets EventActions.escalate for ADK LoopAgent
    transfer_to_agent: str | None = None         # Sets EventActions.transfer_to_agent
    state_delta: dict[str, Any] = field(default_factory=dict)
    request_input_id: str | None = None          # Optional ADK 2.0 HITL RequestInput interrupt_id
    request_input_prompt: str | None = None      # Optional ADK 2.0 HITL RequestInput message
```

### 5.2 `JudgmentAgent` Execution Lifecycle (`_run_async_impl`)
1. **Extract Input & State (`_resolve_state`):**
   - Inspects the latest event in `ctx.session.events` (including ADK 2.0 `Workflow` predecessor `Event.output` if present) alongside `ctx.session.state`.
   - Resolves the state payload via:
     - `state_builder(dict(ctx.session.state), node_input)` if `state_builder` is provided;
     - `{k: ctx.session.state.get(k) for k in state_keys}` if `state_keys` is provided;
     - `node_input` (if non-empty) or `dict(ctx.session.state)` otherwise.
2. **Resolve Questions (`_resolve_questions`):**
   - From `schema.build_questions()` if `schema` is set;
   - From `questions(state_payload, node_input)` if `questions` is callable;
   - From static `questions` mapping otherwise.
3. **Evaluate via Backend:**
   - Calls `await backend.evaluate(state=state_payload, questions=resolved_questions, model=self.model)`.
   - If `schema` is provided, converts `JudgmentResult` into `typed_output = schema.from_result(result)`. Otherwise `typed_output = result`.
4. **Apply User `decide` Hook:**
   - If `self.decide` is provided, calls `decision = self.decide(typed_output, dict(ctx.session.state))`.
   - Normalizes shorthand return values:
     - `str` -> `JudgmentDecision(route=decision)` (plus `transfer_to_agent=decision` if `transfer_to_sub_agent=True`)
     - `bool` -> `JudgmentDecision(route="pass" if decision else "fail", escalate=decision)`
   - If `decision.request_input_id` is set, yields `RequestInput(interrupt_id=decision.request_input_id, message=decision.request_input_prompt)` for ADK 2.0 HITL.
5. **Emit Unified ADK `Event`:**
   - Serializes `final_output` (`decision.output` if provided, else `typed_output.model_dump()`).
   - Builds `state_delta` combining `{self.output_key: final_output}` (when `output_key` is set) and `decision.state_delta`.
   - Updates `ctx.session.state` and yields:
     ```python
     Event(
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
             parts=[types.Part.from_text(text=json.dumps(final_output, indent=2))],
         ),
     )
     ```

---

## 6. Universal Workflow Presets (`presets.py`)

All presets inherit from `JudgmentAgent` (`BaseAgent`) so they work identically in ADK 2.0 `Workflow` graphs and `SequentialAgent` / `LoopAgent` pipelines.

### 6.1 `JudgmentSwitch` (Aliases: `JudgmentRouter`, `SystemOneRouter`, `SystemOneRouter`)
Multi-way semantic branching (`switch` / `match` or multi-branch fan-out):
- **Inputs:**
  - `instructions: str` and `routes: Mapping[str, str | None] | Sequence[str]` (constructs a primary `Choice` question under key `"route"`), OR a `schema` / `questions` dict for multi-factor routing.
  - `route_policy: Callable[[JudgmentResult, dict[str, Any]], str | list[str] | JudgmentDecision] | None = None`
  - Optional convenience parameters when `route_policy` is omitted: `confidence_floor: float | None = None` and `uncertain_route: str = "uncertain"`.
- **Behavior:**
  - Evaluates the routing question(s).
  - If `route_policy` is provided, delegates routing decision to `route_policy(result, state)`.
  - Otherwise, selects `choice = result.choice("route")`; if `confidence_floor` is set and `result.confidence("route") < confidence_floor`, routes to `uncertain_route`; else routes to `choice` (and sets `transfer_to_agent=choice` if matching sub-agent exists in `self.sub_agents`).

### 6.2 `JudgmentGuard` (Aliases: `JudgmentGate`, `SystemOneGate`, `SystemOneGate`)
Semantic assertion, guardrail, and loop-termination gate (`if` / `while`):
- **Inputs:**
  - `schema: type[JudgmentSchema] | None` or `questions: Mapping[str, Any] | Callable` (or single `instructions: str` shorthand creating a `Noul` question `"guard"`).
  - `predicate: Callable[[Any, dict[str, Any]], bool | JudgmentDecision]` (defaults to `lambda res, _: res.noul("guard") >= threshold` when `instructions` + `threshold: float = 0.7` shorthand is used).
  - `pass_route: str = "pass"`, `fail_route: str = "fail"`, `escalate_on_pass: bool = True`.
- **Behavior:**
  - Evaluates the guard question(s) and invokes `predicate(typed_output, state)`.
  - When `True`: emits `route=pass_route` and `escalate=escalate_on_pass` (automatically breaking out of an enclosing ADK `LoopAgent` when the quality/grounding condition is met!).
  - When `False`: emits `route=fail_route` and `escalate=False`.

### 6.3 `JudgmentMap` & `JudgmentBatch[TItem, TJudgment]`
Universal collection primitive (**Map / Filter / Rank / Reduce**) that evaluates `N` items in a **single batched `backend.evaluate()` request**:
- **Inputs:**
  - `items_key: str | None = None` or `items_getter: Callable[[dict[str, Any], Any], Sequence[TItem]] | None = None`
  - `item_schema: type[JudgmentSchema] | None = None` or `item_questions: Callable[[TItem, int], Mapping[str, Any]] | None = None`
  - `global_questions: Mapping[str, Any] | Callable[[dict[str, Any], Sequence[TItem]], Mapping[str, Any]] | None = None` (for holistic questions evaluated alongside the item questions in the same API call)
  - `item_state_builder: Callable[[TItem, int, dict[str, Any]], Any] | None = None`
  - `transform: Callable[[JudgmentBatch[Any, Any], dict[str, Any]], Any] | None = None`
  - `decide: Callable[[JudgmentBatch[Any, Any], dict[str, Any]], JudgmentDecision | str | bool | None] | None = None`
- **How Batching Works:**
  1. Extracts `items = [item_0, item_1, ..., item_{N-1}]`. If `items` is empty, returns an empty `JudgmentBatch(entries=[], global_result=...)` without making a network call.
  2. For each item `i`, generates its questions from `item_schema` or `item_questions(item, i)` and namespaces each question key as `item_{i}__{field_key}`.
  3. Merges all `N × M` item questions + any `global_questions` into a single `questions` dictionary and builds a structured `state` containing `{"items": [...], "context": ...}`.
  4. Executes **one** `await backend.evaluate(state, all_questions, model=self.model)` call.
  5. Reconstructs per-item `JudgmentResult` (and `item_schema` instance if `item_schema` is set) into `JudgmentBatchEntry(index=i, item=item_i, judgment=item_judgment)`.
- **`JudgmentBatch` Composable Operations:**
  - `batch.items() -> list[TItem]`: Returns the current list of items.
  - `batch.filter(predicate: Callable[[TItem, TJudgment], bool]) -> JudgmentBatch[TItem, TJudgment]`: Returns a new immutable `JudgmentBatch` containing only entries where `predicate(item, judgment)` is `True`.
  - `batch.rank_by(key: Callable[[TItem, TJudgment], float], reverse: bool = True, top_k: int | None = None) -> JudgmentBatch[TItem, TJudgment]`: Returns a new immutable `JudgmentBatch` sorted by `key(item, judgment)` (e.g., `Score` or `Noul`) and sliced to `top_k`.
  - `batch.map(mapper: Callable[[TItem, TJudgment], TOut]) -> list[TOut]`: Transforms each `(item, judgment)` pair into a new value (e.g., enriched dicts or verification records).
  - `batch.all(predicate: Callable[[TItem, TJudgment], bool]) -> bool` and `batch.any(predicate) -> bool`: Boolean quantifiers over the collection.
  - `batch.reduce(reducer: Callable[[TAcc, TItem, TJudgment], TAcc], initial: TAcc) -> TAcc`: Folds over all entries.

---

## 7. Error Handling & Security

1. **No Hardcoded Secrets:** `TypeSafeBackend` reads `TYPESAFE_API_KEY` from environment variables (`os.environ.get("TYPESAFE_API_KEY")`) or explicit injection, never logging or serializing API keys in ADK events or state.
2. **Fail-Fast Configuration Validation:** Invalid question types, missing `schema`/`questions`, or empty `Choice`/`Score` criteria raise `JudgmentConfigError` at agent construction time.
3. **Customizable Runtime Error Policy (`on_error`):** `JudgmentAgent` accepts `on_error: Callable[[Exception, dict[str, Any]], JudgmentDecision] | None = None`. If the backend raises an API/timeout error and `on_error` is provided, `JudgmentAgent` emits the fallback `JudgmentDecision` (e.g., `JudgmentDecision(route="fallback", output={"error": "unavailable"})`) instead of crashing the ADK workflow.

---

## 8. Testing & Verification Plan

- **Minimum Test Coverage:** `>= 85%` across all modules (`pytest --cov=judgment_base_agent --cov-report=term-missing`).
- **Unit Tests (`tests/unit/`):**
  - `test_primitives_and_schema.py`: Validation, immutability, confidence tier calculation, `JudgmentSchema.build_questions()` and `from_result()`.
  - `test_backends.py`: `TypeSafeBackend` conversion & response extraction (`answers` and `choices`/`scores`/`nouls` compatibility), `MockJudgmentBackend`, missing API key handling.
  - `test_judgment_agent.py`: State resolution (`state_keys`, `state_builder`, `node_input`), `decide` return normalization (`str`, `bool`, `JudgmentDecision`), `on_error` fallback.
  - `test_presets.py`: `JudgmentSwitch` routing & confidence policy, `JudgmentGuard` pass/fail + `escalate`, `JudgmentMap` / `JudgmentBatch` (`filter`, `rank_by`, `map`, `all`, `reduce`, empty list handling).
- **Integration Tests with Google ADK (`tests/integration/`):**
  - **ADK 2.0 Graph `Workflow` Integration:** End-to-end `InMemoryRunner` tests verifying `JudgmentSwitch` conditional edge branching, `JudgmentMap` RAG reranking/filtering node, and `JudgmentGuard` loop cycle in `google.adk.workflow.Workflow`.
  - **ADK Composite Agents Integration:** End-to-end `InMemoryRunner` tests verifying `SequentialAgent` state propagation and `LoopAgent` early exit via `JudgmentGuard` (`escalate=True`).
