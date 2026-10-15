"""Deterministic projections of the append-only benchmark conversation."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from .conversation import AssistantTurn, ConversationEvent, ConversationSession, EnvironmentTurn, StateCard


HISTORY_POLICY_VERSION = "eval_case_visibility_v1"
HistoryPolicy = Literal["eval_case_given", "eval_case_hidden"]


def validate_history_policy(value: Any) -> HistoryPolicy:
    policy = str(value or "").strip()
    if policy not in {"eval_case_given", "eval_case_hidden"}:
        raise ValueError(
            "history_policy must be eval_case_given or eval_case_hidden"
        )
    return policy  # type: ignore[return-value]


@dataclass(frozen=True)
class VisibleConversationView:
    system_instruction: str
    task_spec: dict[str, Any]
    events: list[ConversationEvent]
    history_policy: str

    @property
    def latest_state_card(self) -> StateCard:
        for event in reversed(self.events):
            if isinstance(event, EnvironmentTurn):
                return StateCard(deepcopy(event.state_card))
        raise ValueError("visible conversation has no environment state")

    def visible_history_characters(self) -> int:
        total = len(self.system_instruction)
        total += len(_json(self.task_spec))
        for event in self.events:
            if isinstance(event, AssistantTurn):
                total += len(event.visible_content)
            else:
                total += len(_json(event.result)) + len(_json(event.state_card))
        return total


@dataclass(frozen=True)
class HistoryProjection:
    view: VisibleConversationView
    event_mapping: list[dict[str, Any]]


def project_conversation(
    session: ConversationSession,
    *,
    persistent: bool = False,
) -> HistoryProjection:
    """Project the full log without mutating it.

    Runtime projections retain the current incomplete action span. Persistent
    projections, used by the Viewer, never expose temporary action material.
    """

    if session.history_policy == "eval_case_given":
        events = deepcopy(session.events)
        mapping = [
            {"source_event_id": event.event_id, "visible_event_id": event.event_id, "disposition": "retained"}
            for event in session.events
        ]
        return HistoryProjection(
            VisibleConversationView(
                session.system_instruction,
                deepcopy(session.task_spec),
                events,
                session.history_policy,
            ),
            mapping,
        )

    completed = {
        str(span.get("batch_id"))
        for span in session.action_spans
        if span.get("status") == "completed"
    }
    events: list[ConversationEvent] = []
    mapping: list[dict[str, Any]] = []
    for event in session.events:
        batch_id = str(event.batch_id) if event.batch_id is not None else None
        completed_span = batch_id in completed
        if event.stage == "action_solving" and (completed_span or persistent):
            mapping.append({
                "source_event_id": event.event_id,
                "visible_event_id": None,
                "disposition": "removed_action_solving",
            })
            continue
        if event.stage == "reveal" and (completed_span or persistent):
            projected = _sanitize_reveal(event)
            if projected is not None:
                events.append(projected)
            mapping.append({
                "source_event_id": event.event_id,
                "visible_event_id": (
                    projected.event_id if projected is not None else None
                ),
                "disposition": (
                    "rewritten_reveal"
                    if projected is not None
                    else "removed_empty_reveal"
                ),
            })
            continue
        if event.stage == "completion":
            projected = _sanitize_completion(event)
            events.append(projected)
            mapping.append({
                "source_event_id": event.event_id,
                "visible_event_id": projected.event_id,
                "disposition": "rewritten_completion",
            })
            continue
        events.append(deepcopy(event))
        mapping.append({
            "source_event_id": event.event_id,
            "visible_event_id": event.event_id,
            "disposition": "retained",
        })
    return HistoryProjection(
        VisibleConversationView(
            session.system_instruction,
            deepcopy(session.task_spec),
            events,
            session.history_policy,
        ),
        mapping,
    )


def persistent_history_payload(session: ConversationSession) -> dict[str, Any] | None:
    if session.history_policy != "eval_case_hidden":
        return None
    projection = project_conversation(session, persistent=True)
    return {
        "schema_version": HISTORY_POLICY_VERSION,
        "history_policy": session.history_policy,
        "system_instruction": session.system_instruction,
        "task_spec": deepcopy(session.task_spec),
        "events": [event.to_dict() for event in projection.view.events],
        "event_mapping": deepcopy(projection.event_mapping),
    }


def wire_request_payload(request: Any) -> dict[str, Any]:
    return {
        "protocol": request.protocol,
        "system_instruction": request.system_instruction,
        "messages": deepcopy(request.messages),
        "contents": deepcopy(request.contents),
    }


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _sanitize_reveal(event: EnvironmentTurn) -> EnvironmentTurn | None:
    original = event.result if isinstance(event.result, dict) else {}
    batch_label = event.batch_number if event.batch_number is not None else "unknown"
    sensitive = {
        "pending_action_batch",
        "required_batch_id",
        "required_next_tool",
        "tool_usability",
    }
    retained = {
        key: deepcopy(value)
        for key, value in original.items()
        if key not in sensitive and key not in {"type", "message"}
    }
    if not retained:
        return None
    original_message = original.get("message")
    if isinstance(original_message, str) and original_message.strip():
        message = (
            f"{original_message.strip()} The revealed action cases and transient "
            "action state are hidden from later agent context."
        )
    elif isinstance(original.get("error"), dict):
        message = (
            f"The environment ended the probe opportunity for action batch "
            f"{batch_label} after the repair limit was exhausted. The probe "
            "requirement was waived for this batch. The rejected tool response "
            "and transient action state are hidden from later agent context."
        )
    else:
        message = (
            "The revealed action cases and transient state are hidden from later "
            "agent context."
        )
    result = {"message": message, **retained}
    return EnvironmentTurn(
        result=result,
        state_card=StateCard(),
        error=None,
        event_id=event.event_id,
        interaction=event.interaction,
        stage=event.stage,
        batch_id=event.batch_id,
        batch_number=event.batch_number,
    )


def _sanitize_completion(event: EnvironmentTurn) -> EnvironmentTurn:
    original = event.result if isinstance(event.result, dict) else {}
    rows = [
        {
            # Read legacy action_case_id traces, but keep the current public
            # protocol's single context_id identity.
            "context_id": row.get("context_id", row.get("action_case_id")),
            "utility": row.get("utility"),
        }
        for row in (original.get("results") or [])
        if isinstance(row, dict)
        and row.get("action_case_id", row.get("context_id")) is not None
    ]
    failed = "failed_action_batch" in original
    batch_utility = original.get("batch_utility")
    if batch_utility is None:
        batch_utility = round(sum(float(row.get("utility") or 0) for row in rows), 2)
    message = (
        f"The environment ended action batch {event.batch_number} after the "
        "repair limit was exhausted. No actions were accepted; every case was "
        "recorded as unanswered. Invalid action attempts and action-solving "
        "messages are hidden from later agent context."
        if failed
        else f"The environment accepted action batch {event.batch_number}. Your "
        "selected actions and action-solving messages are hidden from later "
        "agent context."
    )
    result = {
        "message": message,
        "batch_number": event.batch_number,
        "results": rows,
        "batch_utility": batch_utility,
        "cumulative_utility": event.state_card.get(
            "cumulative_utility",
            event.state_card.get(
                "action_utility",
                original.get("cumulative_utility", original.get("action_utility")),
            ),
        ),
        "cumulative_probes": event.state_card.get(
            "cumulative_probes",
            event.state_card.get(
                "probe_count",
                original.get("cumulative_probes", original.get("probe_count")),
            ),
        ),
    }
    return EnvironmentTurn(
        result=result,
        state_card=StateCard(deepcopy(event.state_card)),
        error=None,
        event_id=event.event_id,
        interaction=event.interaction,
        stage=event.stage,
        batch_id=event.batch_id,
        batch_number=event.batch_number,
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "HISTORY_POLICY_VERSION",
    "HistoryPolicy",
    "HistoryProjection",
    "VisibleConversationView",
    "payload_sha256",
    "persistent_history_payload",
    "project_conversation",
    "validate_history_policy",
    "wire_request_payload",
]
