from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": [
                "calculator",
                "probe_batch",
                "request_next_batch",
                "submit_action_batch",
            ],
        },
        "arguments": {"type": "object"},
    },
    "required": ["tool", "arguments"],
    "additionalProperties": False,
}


def decision_schema(*, enable_probe_skip: bool = True) -> dict[str, Any]:
    """Return the model decision schema for one probe policy.

    ``request_next_batch`` is deliberately omitted from the advertised enum
    when a setting requires a probe before every action batch.  Keep the
    module-level schema as the backwards-compatible default for callers that
    do not have a setting-specific policy.
    """

    schema = deepcopy(DECISION_SCHEMA)
    if not enable_probe_skip:
        schema["properties"]["tool"]["enum"].remove("request_next_batch")
    return schema


class ProtocolError(ValueError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


def format_error_messages(
    messages: list[str], *, prefix: str | None = None
) -> str:
    """Format validation failures for a model-facing runtime response."""

    if not messages:
        raise ValueError("at least one error message is required")
    if len(messages) == 1:
        body = messages[0]
        return f"{prefix}: {body}" if prefix else body
    body = "\n".join(
        f"{index}. {message}" for index, message in enumerate(messages, start=1)
    )
    return f"{prefix}:\n{body}" if prefix else body


def _extract_first_json_object(text: str) -> dict[str, Any]:
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(text[start : index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("model response must decode to a JSON object")
                return payload
    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def parse_tool_payload(
    text: str, *, enable_probe_skip: bool = True
) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ProtocolError("empty_model_response", "empty model response")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as direct_error:
        try:
            payload = _extract_first_json_object(cleaned)
        except Exception:
            raise ProtocolError(
                "invalid_json",
                f"{direct_error.msg} at line {direct_error.lineno}, "
                f"column {direct_error.colno} (char {direct_error.pos})",
            ) from direct_error
    if not isinstance(payload, dict):
        raise ProtocolError("invalid_tool_schema", "top-level payload must be an object")
    missing = sorted({"tool", "arguments"} - set(payload))
    extra = sorted(set(payload) - {"tool", "arguments"})
    schema_errors = []
    if missing:
        schema_errors.append(f"missing top-level key(s): {', '.join(missing)}")
    if extra:
        schema_errors.append(f"unexpected top-level key(s): {', '.join(extra)}")
    if schema_errors:
        raise ProtocolError(
            "invalid_tool_schema", format_error_messages(schema_errors)
        )
    allowed_tools = {
        "calculator",
        "probe_batch",
        "submit_action_batch",
    }
    if enable_probe_skip:
        allowed_tools.add("request_next_batch")
    if payload["tool"] not in allowed_tools:
        raise ProtocolError(
            "invalid_tool_schema", f"tool does not exist: {payload['tool']}"
        )
    if not isinstance(payload["arguments"], dict):
        raise ProtocolError("invalid_tool_schema", "arguments must be an object")
    return payload
