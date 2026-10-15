"""Provider wire-format adapters for the canonical benchmark session."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from runtime.conversation import (
    AssistantTurn,
    ConversationSession,
    EnvironmentTurn,
)
from runtime.history import VisibleConversationView


@dataclass
class ProviderConversationRequest:
    """A structured request understood by the matching generator client."""

    protocol: str
    system_instruction: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    contents: list[dict[str, Any]] = field(default_factory=list)
    source_event_ids: list[list[str]] = field(default_factory=list)

    @property
    def wire_characters(self) -> int:
        payload = {
            "system_instruction": self.system_instruction,
            "messages": self.messages,
            "contents": self.contents,
        }
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


class ConversationAdapter(Protocol):
    protocol: str
    reasoning_history_policy: str

    def build_request(self, session: ConversationSession | VisibleConversationView) -> ProviderConversationRequest:
        ...

    def capture_response(self, response: Any) -> Any | None:
        ...


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def environment_result_for_wire(event: EnvironmentTurn) -> dict[str, Any] | None:
    """Remove wire-only metadata and reveal fields duplicated by current state.

    Environment results remain self-contained in runtime artifacts, while a
    Batch-reveal results have one authoritative copy of current control state.
    Other results, including non-pending calculator results, remain intact
    apart from the internal ``type`` marker.
    """

    if event.result is None:
        return None
    result = deepcopy(event.result)
    result.pop("type", None)
    if event.stage == "reveal":
        for key, value in event.state_card.items():
            if key in result and result[key] == value:
                del result[key]
    return result or None


def _render_field_descriptions(task_spec: dict) -> str:
    lines = [
        "Input fields:",
        "Every field below is a numeric probe input and is required in a full "
        "`values` probe. Use JSON numbers only: `integer` requires a whole number "
        "and `decimal` accepts an integer or decimal number. `null`, strings, and "
        "booleans are invalid, even when a field description discusses an unavailable "
        "real-world observation.",
    ]
    for field_id, meta in (task_spec.get("raw_field_schema") or {}).items():
        display = meta.get("display_name") or field_id
        ftype = meta["type"]
        unit = meta.get("unit") or ""
        desc = meta.get("description") or ""
        # Value bounds (minimum/maximum/typical_range) stay evaluator-private:
        # revealing them would leak the sampling design behind each field.
        unit_str = f", unit: {unit}" if unit else ""
        lines.append(
            f"- **{display}** (`{field_id}`, type: {ftype}{unit_str}): {desc}"
        )
    constraints = task_spec.get("feasibility_constraints") or []
    if constraints:
        lines.append("\nConstraints:")
        for c in constraints:
            lines.append(f"- {c.get('message') or c.get('constraint_id')}")
    return "\n".join(lines)


def environment_message_text(
    session: ConversationSession | VisibleConversationView,
    event: EnvironmentTurn,
    *,
    initial: bool,
) -> str:
    sections: list[str] = []
    if initial:
        sections.append(_render_field_descriptions(session.task_spec))
    else:
        wire_result = environment_result_for_wire(event)
        if wire_result is not None:
            label = (
                "Probe result"
                if (
                    event.stage == "reveal"
                    and isinstance(wire_result.get("results"), list)
                    and not event.state_card
                )
                else "Environment result"
            )
            sections.append(f"{label}:\n" + _dump(wire_result))
    if event.state_card:
        sections.append("Current state:\n" + _dump(event.state_card))
    return "\n\n".join(sections)


def _canonical_events(session: ConversationSession | VisibleConversationView):
    for index, event in enumerate(session.events):
        if isinstance(event, EnvironmentTurn):
            yield "environment", index, event
        elif isinstance(event, AssistantTurn):
            yield "assistant", index, event
        else:  # pragma: no cover - guarded by ConversationSession methods
            raise TypeError(f"unsupported conversation event: {type(event)!r}")


class DeepSeekChatAdapter:
    protocol = "deepseek_chat"
    reasoning_history_policy = "excluded_from_history"

    def build_request(self, session: ConversationSession | VisibleConversationView) -> ProviderConversationRequest:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": session.system_instruction}
        ]
        source_event_ids: list[list[str]] = [[]]
        for kind, index, event in _canonical_events(session):
            if kind == "assistant":
                messages.append({"role": "assistant", "content": event.visible_content})
                source_event_ids.append([event.event_id])
            else:
                content = environment_message_text(session, event, initial=index == 0)
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\n" + content
                    source_event_ids[-1].append(event.event_id)
                else:
                    messages.append({"role": "user", "content": content})
                    source_event_ids.append([event.event_id])
        return ProviderConversationRequest(
            protocol=self.protocol,
            system_instruction=session.system_instruction,
            messages=messages,
            source_event_ids=source_event_ids,
        )

    def capture_response(self, response: Any) -> None:
        # The benchmark deliberately uses ordinary visible JSON chat turns. A
        # provider's private reasoning field is retained on GeneratorResponse
        # for artifacts but is not fed back into this history.
        return None


class OpenAIChatAdapter(DeepSeekChatAdapter):
    """Generic ordinary Chat Completions adapter for future OpenAI configs."""

    protocol = "openai_chat"


class OpenAIResponsesAdapter(DeepSeekChatAdapter):
    """OpenAI Responses input adapter with reasoning excluded from history."""

    protocol = "openai_responses"


class GeminiGenerateContentAdapter:
    protocol = "gemini_generate_content"
    reasoning_history_policy = "provider_continuation"

    def build_request(self, session: ConversationSession | VisibleConversationView) -> ProviderConversationRequest:
        contents: list[dict[str, Any]] = []
        source_event_ids: list[list[str]] = []
        for kind, index, event in _canonical_events(session):
            if kind == "environment":
                text = environment_message_text(session, event, initial=index == 0)
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].append({"text": text})
                    source_event_ids[-1].append(event.event_id)
                else:
                    contents.append({"role": "user", "parts": [{"text": text}]})
                    source_event_ids.append([event.event_id])
                continue

            continuation = event.provider_continuation
            if isinstance(continuation, dict) and isinstance(
                continuation.get("parts"), list
            ):
                model_content = deepcopy(continuation)
                model_content["role"] = "model"
            else:
                model_content = {
                    "role": "model",
                    "parts": [{"text": event.visible_content}],
                }
            contents.append(model_content)
            source_event_ids.append([event.event_id])
        return ProviderConversationRequest(
            protocol=self.protocol,
            system_instruction=session.system_instruction,
            contents=contents,
            source_event_ids=source_event_ids,
        )

    def capture_response(self, response: Any) -> Any | None:
        return deepcopy(getattr(response, "assistant_content", None))


CONVERSATION_ADAPTERS: dict[str, type[ConversationAdapter]] = {
    "deepseek_chat": DeepSeekChatAdapter,
    "gemini_generate_content": GeminiGenerateContentAdapter,
    "openai_chat": OpenAIChatAdapter,
    "openai_responses": OpenAIResponsesAdapter,
}


def get_conversation_adapter(protocol: str | None) -> ConversationAdapter:
    name = str(protocol or "").strip()
    adapter_type = CONVERSATION_ADAPTERS.get(name)
    if adapter_type is None:
        available = ", ".join(sorted(CONVERSATION_ADAPTERS))
        if not name:
            raise ValueError(
                "model config must declare conversation_protocol; "
                f"expected one of: {available}"
            )
        raise ValueError(
            f"unknown conversation_protocol {name!r}; expected one of: {available}"
        )
    return adapter_type()


__all__ = [
    "CONVERSATION_ADAPTERS",
    "ConversationAdapter",
    "DeepSeekChatAdapter",
    "GeminiGenerateContentAdapter",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
    "ProviderConversationRequest",
    "environment_result_for_wire",
    "environment_message_text",
    "get_conversation_adapter",
]
