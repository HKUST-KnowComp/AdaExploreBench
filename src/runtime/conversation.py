"""Provider-independent benchmark conversation state.

The environment owns the experiment semantics, while this module owns the
single append-only ledger that is sent to every provider adapter.  Keeping the
ledger small and explicit prevents a compact state card from becoming another
copy of the full episode history.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


CONVERSATION_SCHEMA_VERSION = "adaexplorebench_benchmark_conversation_v4"


class TaskSpec(dict[str, Any]):
    """JSON-shaped one-time task description supplied by the environment."""


class StateCard(dict[str, Any]):
    """JSON-shaped authoritative snapshot of only the current control state."""


@dataclass
class AssistantTurn:
    visible_content: str
    parsed_call: dict[str, Any] | None = None
    valid: bool | None = None
    provider_continuation: Any | None = None
    event_id: str = ""
    interaction: int | None = None
    stage: str = "control"
    batch_id: str | None = None
    batch_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "assistant_turn",
            "visible_content": self.visible_content,
            "parsed_call": _clone(self.parsed_call),
            "valid": self.valid,
            "provider_continuation": _clone(self.provider_continuation),
            "event_id": self.event_id,
            "interaction": self.interaction,
            "stage": self.stage,
            "batch_id": self.batch_id,
            "batch_number": self.batch_number,
        }


@dataclass
class EnvironmentTurn:
    result: dict[str, Any] | None
    state_card: StateCard
    error: dict[str, Any] | None = None
    event_id: str = ""
    interaction: int | None = None
    stage: str = "control"
    batch_id: str | None = None
    batch_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "environment_turn",
            "result": _clone(self.result),
            "state_card": _clone(self.state_card),
            "error": _clone(self.error),
            "event_id": self.event_id,
            "interaction": self.interaction,
            "stage": self.stage,
            "batch_id": self.batch_id,
            "batch_number": self.batch_number,
        }


ConversationEvent = AssistantTurn | EnvironmentTurn


@dataclass
class ConversationSession:
    system_instruction: str
    task_spec: TaskSpec
    events: list[ConversationEvent] = field(default_factory=list)
    history_policy: str = "eval_case_given"
    history_policy_version: str = "eval_case_visibility_v1"
    action_spans: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.history_policy not in {"eval_case_given", "eval_case_hidden"}:
            raise ValueError(
                "history policy must be eval_case_given or eval_case_hidden"
            )

    def _next_event_id(self) -> str:
        return f"event_{len(self.events) + 1:04d}"

    def append_initial_state(self, state_card: StateCard) -> EnvironmentTurn:
        if self.events:
            raise ValueError("initial state can only be appended to an empty session")
        return self.append_environment(None, state_card)

    def append_assistant(
        self,
        visible_content: str,
        *,
        parsed_call: dict[str, Any] | None = None,
        valid: bool | None = None,
        provider_continuation: Any | None = None,
        interaction: int | None = None,
        stage: str = "control",
        batch_id: str | None = None,
        batch_number: int | None = None,
    ) -> AssistantTurn:
        if self.events and not isinstance(self.events[-1], EnvironmentTurn):
            raise ValueError("assistant turns must follow an environment turn")
        if not self.events:
            raise ValueError("assistant turn requires an initial environment turn")
        turn = AssistantTurn(
            visible_content=str(visible_content),
            parsed_call=_clone(parsed_call),
            valid=valid,
            provider_continuation=_clone(provider_continuation),
            event_id=self._next_event_id(),
            interaction=interaction,
            stage=stage,
            batch_id=batch_id,
            batch_number=batch_number,
        )
        self.events.append(turn)
        if stage == "action_solving":
            self._append_span_event(batch_id, turn.event_id)
        return turn

    def append_environment(
        self,
        result: dict[str, Any] | None,
        state_card: StateCard,
        *,
        error: dict[str, Any] | None = None,
        interaction: int | None = None,
        stage: str = "control",
        batch_id: str | None = None,
        batch_number: int | None = None,
    ) -> EnvironmentTurn:
        if self.events and not isinstance(self.events[-1], AssistantTurn):
            raise ValueError("environment turns must follow an assistant turn")
        turn = EnvironmentTurn(
            result=_clone(result),
            state_card=StateCard(_clone(state_card)),
            error=_clone(error),
            event_id=self._next_event_id(),
            interaction=interaction,
            stage=stage,
            batch_id=batch_id,
            batch_number=batch_number,
        )
        self.events.append(turn)
        if stage == "reveal":
            self.action_spans.append({
                "batch_id": batch_id,
                "batch_number": batch_number,
                "reveal_event_id": turn.event_id,
                "completion_event_id": None,
                "solving_event_ids": [],
                "status": "pending",
            })
        elif stage == "action_solving":
            self._append_span_event(batch_id, turn.event_id)
        elif stage == "completion":
            span = next(
                (
                    item for item in reversed(self.action_spans)
                    if item.get("batch_id") == batch_id
                    and item.get("status") == "pending"
                ),
                None,
            )
            if span is None:
                raise ValueError("action completion requires a matching reveal event")
            span["completion_event_id"] = turn.event_id
            span["status"] = "completed"
        return turn

    def _append_span_event(self, batch_id: str | None, event_id: str) -> None:
        span = next(
            (
                item for item in reversed(self.action_spans)
                if item.get("batch_id") == batch_id and item.get("status") == "pending"
            ),
            None,
        )
        if span is None:
            raise ValueError("action-solving event requires a matching reveal event")
        span.setdefault("solving_event_ids", []).append(event_id)

    @property
    def latest_state_card(self) -> StateCard:
        for event in reversed(self.events):
            if isinstance(event, EnvironmentTurn):
                return StateCard(_clone(event.state_card))
        raise ValueError("conversation has no environment state")

    @property
    def interaction_count(self) -> int:
        return sum(isinstance(event, AssistantTurn) for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONVERSATION_SCHEMA_VERSION,
            "system_instruction": self.system_instruction,
            "system_prompt_hash": hashlib.sha256(
                self.system_instruction.encode("utf-8")
            ).hexdigest(),
            "task_spec": _clone(self.task_spec),
            "events": [event.to_dict() for event in self.events],
            "history_policy": self.history_policy,
            "history_policy_version": self.history_policy_version,
            "action_spans": _clone(self.action_spans),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationSession":
        session = cls(
            system_instruction=str(value["system_instruction"]),
            task_spec=TaskSpec(_clone(value["task_spec"])),
            history_policy=str(value.get("history_policy") or "eval_case_given"),
            history_policy_version=str(
                value.get("history_policy_version") or "legacy_eval_case_given_v1"
            ),
            action_spans=_clone(value.get("action_spans") or []),
        )
        for index, event in enumerate(value.get("events") or [], start=1):
            common = {
                "event_id": str(event.get("event_id") or f"event_{index:04d}"),
                "interaction": event.get("interaction"),
                "stage": str(event.get("stage") or "control"),
                "batch_id": event.get("batch_id"),
                "batch_number": event.get("batch_number"),
            }
            if event.get("type") == "assistant_turn":
                session.events.append(AssistantTurn(
                    visible_content=str(event.get("visible_content") or ""),
                    parsed_call=_clone(event.get("parsed_call")),
                    valid=event.get("valid"),
                    provider_continuation=_clone(event.get("provider_continuation")),
                    **common,
                ))
            elif event.get("type") == "environment_turn":
                session.events.append(EnvironmentTurn(
                    result=_clone(event.get("result")),
                    state_card=StateCard(_clone(event.get("state_card")) or {}),
                    error=_clone(event.get("error")),
                    **common,
                ))
            else:
                raise ValueError("conversation artifact contains an unknown event")
        return session

    def visible_history_characters(self) -> int:
        total = len(self.system_instruction)
        total += sum(
            len(event.visible_content)
            for event in self.events
            if isinstance(event, AssistantTurn)
        )
        total += sum(
            len(_json(event.result)) + len(_json(event.state_card))
            for event in self.events
            if isinstance(event, EnvironmentTurn)
        )
        return total


def _clone(value: Any) -> Any:
    """Clone JSON-shaped data without exposing mutable event internals."""
    return deepcopy(value) if value is not None else None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "AssistantTurn",
    "ConversationEvent",
    "ConversationSession",
    "EnvironmentTurn",
    "StateCard",
    "TaskSpec",
    "CONVERSATION_SCHEMA_VERSION",
]
