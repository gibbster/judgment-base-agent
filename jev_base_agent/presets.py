"""Universal workflow presets: JudgmentSwitch, JudgmentGuard, and JudgmentMap (JudgmentBatch)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
from typing import Any, Generic, TypeVar

from google.adk.agents import BaseAgent

from jev_base_agent.agent import (
    JudgmentAgent,
    JudgmentDecision,
    _serialize_output,
    normalize_decision,
)
from jev_base_agent.backends.base import BaseJudgmentBackend
from jev_base_agent.backends.typesafe import TypeSafeBackend
from jev_base_agent.errors import JudgmentConfigError
from jev_base_agent.primitives import Choice, JudgmentResult, Noul
from jev_base_agent.schema import JudgmentSchema

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
        model: str = "jev-latest",
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
        model: str = "jev-latest",
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
        model: str = "jev-latest",
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
            questions={},
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
