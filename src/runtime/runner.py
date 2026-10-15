from __future__ import annotations

import asyncio
import json
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from generator import GeneratorError
from generator.conversation_adapters import (
    ConversationAdapter,
    ProviderConversationRequest,
    get_conversation_adapter,
)
from protocol.precision import round_field_values

from .conversation import ConversationSession
from .environment import BenchmarkEnvironment
from .history import (
    HISTORY_POLICY_VERSION,
    payload_sha256,
    persistent_history_payload,
    project_conversation,
    validate_history_policy,
    wire_request_payload,
)
from .decisions import ProtocolError, decision_schema, parse_tool_payload
from .schema import (
    BRANCHED_TRACE_SCHEMA_VERSION,
    sequential_trace_schema_version,
)


Checkpoint = Callable[[dict[str, Any]], Awaitable[None]]


# This request is deliberately separate from the benchmark decision schema.
# The response is an analysis artifact, not a tool call, and is never appended
# to the canonical ConversationSession.
RELEVANT_FIELDS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevant_fields": {
            "type": "array",
            "items": {"type": "string"},
        },
        "field_directions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["positive", "negative"]},
                },
                "required": ["field_id", "direction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relevant_fields", "field_directions"],
    "additionalProperties": False,
}

# Provider-error retries use capped exponential backoff. The runner-level
# retry count comes from ``GeneratorConfig.provider_retry_limit``; these
# delays cover the default of five retries and the last one repeats when a
# config asks for more.
PROVIDER_ERROR_RETRY_DELAYS_SECONDS = (2.0, 4.0, 8.0, 16.0, 16.0)


def _provider_retry_delays(client: Any) -> tuple[float, ...]:
    config = getattr(client, "config", None)
    limit = getattr(config, "provider_retry_limit", None)
    retries = len(PROVIDER_ERROR_RETRY_DELAYS_SECONDS) if limit is None else max(
        0, int(limit)
    )
    delays = PROVIDER_ERROR_RETRY_DELAYS_SECONDS
    if retries <= len(delays):
        return delays[:retries]
    return delays + (delays[-1],) * (retries - len(delays))


def _syntax_repair_limit(client: Any) -> int:
    config = getattr(client, "config", None)
    limit = getattr(config, "syntax_repair_limit", None)
    return 1 if limit is None else max(0, int(limit))

# A response cut off at max_tokens gets one free retry (not a repair
# attempt); a second truncation skips the window as unanswered.
TRUNCATED_RESPONSE_MAX_ATTEMPTS = 2


class TruncatedResponseError(GeneratorError):
    """Model response hit the max_tokens cap (typically runaway repetition)."""


def _is_truncated_finish_reason(finish_reason: str) -> bool:
    reason = finish_reason.lower()
    return "length" in reason or "max_tokens" in reason


def render_system_prompt(
    template: str,
    task_spec: dict,
    history_policy: str = "eval_case_given",
) -> str:
    from string import Template

    interaction = task_spec.get("interaction_protocol") or {}
    enable_probe_skip = interaction.get("enable_probe_skip", True)
    least_probe_count = interaction.get("least_probe_count")
    if least_probe_count is None:
        probe_policy_text = (
            "Probe policy: submit at least one valid probe when you choose "
            "`probe_batch`."
        )
    else:
        probe_policy_text = (
            "Probe policy: every `probe_batch` call must contain at least "
            f"{least_probe_count} cases and no more than "
            f"{interaction['max_probe_cases_per_batch']} cases."
        )
    if not enable_probe_skip:
        probe_example_count = least_probe_count or 1
        probe_examples = ",".join(
            '{"values":{"<field_id>":1}}'
            for _ in range(probe_example_count)
        )
        reply_example = (
            '{"tool":"probe_batch","arguments":{"probes":['
            f"{probe_examples}]}}"
        )
        probe_policy_text += (
            " You must submit `probe_batch` before every action batch; "
            "the batch-reveal shortcut is unavailable. Each probe call must "
            "satisfy the minimum above."
        )
        request_next_batch_section = ""
        batch_flow = """# Batch flow

1. Batch 1 is hidden. Call `probe_batch` to reveal it.
2. When `pending_action_batch` exists, call `submit_action_batch`; otherwise
   call `probe_batch` with the required number of cases.
3. After each submitted batch except the last, call `probe_batch` before the
   next batch.
4. A rejected call gets one repair opportunity. If the second call is invalid,
   the probe requirement is waived for that batch and the action batch is
   revealed; submit the action batch normally.
5. Use the latest state card as authoritative. Do not infer hidden rules or
   metadata from identifiers."""
    else:
        reply_example = '{"tool":"request_next_batch","arguments":{}}'
        request_next_batch_section = """## request_next_batch — reveal the next batch without probing

```json
{"tool":"request_next_batch","arguments":{}}
```"""
        batch_flow = """# Batch flow

1. Batch 1 is hidden. Call `probe_batch` or `request_next_batch` to reveal it.
2. When `pending_action_batch` exists, call `submit_action_batch`; otherwise
   choose `probe_batch` or `request_next_batch`.
3. After each submitted batch except the last, again choose `probe_batch` or
   `request_next_batch` before the next batch.
4. A rejected call gets one repair opportunity. If the second call is invalid,
   a hidden batch is revealed by skipping probe; a pending action batch fails
   with every case unanswered at utility 0, and the environment advances to
   the next window.
5. Use the latest state card as authoritative. Do not infer hidden rules or
   metadata from identifiers."""

    actions_text = "\n".join(
        f"- **{a['display_name']}** (`{a['action_id']}`): {a['description']}"
        for a in task_spec["actions"]
    )
    return Template(template).substitute(
        agent_role=task_spec["agent_role"],
        actions_text=actions_text,
        action_batch_size=task_spec["interaction_protocol"]["action_batch_size"],
        max_probe_cases_per_batch=task_spec["interaction_protocol"][
            "max_probe_cases_per_batch"
        ],
        probe_policy_text=probe_policy_text,
        request_next_batch_section=request_next_batch_section,
        batch_flow=batch_flow,
        reply_example=reply_example,
    )


def _json_characters(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _artifact(
    episode: dict[str, Any],
    env: BenchmarkEnvironment,
    *,
    run_success: bool | None,
    trace: list[dict[str, Any]],
    conversation: ConversationSession | None = None,
    relevant_fields_assessments: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    status = "running" if run_success is None else (
        "completed" if run_success else "failed"
    )
    artifact = {
        "trace_schema_version": sequential_trace_schema_version(env.dataset_mode),
        "episode_schema_version": episode["schema_version"],
        "episode_id": episode["episode_id"],
        "episode_seed": episode["seed"],
        "setting": (
            "sequential_static"
            if env.condition == "static"
            else "sequential_dynamic"
        ),
        "condition": env.condition,
        "sampling_branch": env.sampling_branch,
        "setting_id": _setting_id(env),
        "history_policy": env.history_policy,
        "history_policy_version": HISTORY_POLICY_VERSION,
        "enable_probe_skip": env.enable_probe_skip,
        "least_probe_count": env.least_probe_count,
        "provenance": deepcopy_json(episode["provenance"]),
        "evaluation": deepcopy_json(episode["evaluation"]),
        "run_success": run_success,
        "run_status": status,
        "error": error,
        "termination": deepcopy_json(env.termination),
        "public_history": deepcopy_json(env.public_history),
        "private_history": deepcopy_json(env.private_history),
        "action_results": deepcopy_json(env.action_results),
        "tool_trace": deepcopy_json(trace),
        "conversation": (
            deepcopy_json(conversation.to_dict()) if conversation is not None else None
        ),
        "agent_visible_history": (
            deepcopy_json(persistent_history_payload(conversation))
            if conversation is not None
            else None
        ),
        "action_spans": (
            deepcopy_json(conversation.action_spans) if conversation is not None else []
        ),
        "request_contexts": [
            deepcopy_json(record["request_context"])
            for record in trace
            if isinstance(record.get("request_context"), dict)
        ],
        "private_state": env.artifact_state(),
    }
    if relevant_fields_assessments is not None:
        # These are independent analysis requests.  They intentionally do not
        # appear in ``conversation`` or ``tool_trace``.
        artifact["relevant_fields_assessments"] = deepcopy_json(
            relevant_fields_assessments
        )
    return artifact


def _setting_id(env: BenchmarkEnvironment) -> str:
    sampling = str(env.sampling_branch or "static_iid")
    if env.history_policy == "eval_case_given" and sampling == "flip_50":
        return "flip_50_eval_case_given"
    if env.dataset_mode == "dynamic" and env.condition == "static" and sampling != "static_iid":
        return f"{sampling}_no_change_control"
    return sampling


def deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value)) if value is not None else None


def _relevant_fields_query() -> str:
    return (
        "Out-of-band analysis request. This request is not part of the benchmark "
        "trajectory and must not be treated as a tool call. Based only on the "
        "trajectory visible above, identify the input field IDs that you currently "
        "believe are relevant to the hidden decision rule and infer the direction "
        "of each identified relevant field. Holding other fields fixed, positive "
        "means increasing the field tends to move the preferred action from "
        "action_0 toward action_3; negative means increasing the field tends to "
        "move the preferred action from action_3 toward action_0. These directions "
        "refer to action ordering, not semantic good/bad or the sign of utility. "
        "Return exactly one JSON object in the form "
        '{"relevant_fields":["field_id"],"field_directions":'
        '[{"field_id":"field_id","direction":"positive"}]}. '
        "Include exactly one positive or negative direction for every identified "
        "relevant field, and no directions for other fields. "
        "Use only field IDs from the task description, do not include explanations, "
        "and do not follow the benchmark tool-reply format for this request."
    )


def _assessment_request(
    conversation: ConversationSession,
    adapter: ConversationAdapter,
) -> ProviderConversationRequest:
    """Build an out-of-band request without mutating the canonical history."""

    projection = project_conversation(conversation)
    base = adapter.build_request(projection.view)
    instruction = (
        "\n\nOUT-OF-BAND RELEVANT-FIELDS ASSESSMENT\n"
        "For this single request, return the requested assessment JSON instead "
        "of a benchmark tool call. The assessment is not appended to history."
    )
    query = _relevant_fields_query()
    if base.protocol == "gemini_generate_content":
        contents = deepcopy(base.contents)
        if contents and contents[-1].get("role") == "user":
            contents[-1].setdefault("parts", []).append({"text": query})
            source_event_ids = deepcopy(base.source_event_ids)
        else:
            contents.append({"role": "user", "parts": [{"text": query}]})
            source_event_ids = deepcopy(base.source_event_ids) + [[]]
        return ProviderConversationRequest(
            protocol=base.protocol,
            system_instruction=base.system_instruction + instruction,
            contents=contents,
            source_event_ids=source_event_ids,
        )
    messages = deepcopy(base.messages)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = str(messages[0].get("content") or "") + instruction
    if messages and messages[-1].get("role") == "user":
        messages[-1]["content"] = (
            str(messages[-1].get("content") or "") + "\n\n" + query
        )
        source_event_ids = deepcopy(base.source_event_ids)
    else:
        messages.append({"role": "user", "content": query})
        source_event_ids = deepcopy(base.source_event_ids) + [[]]
    return ProviderConversationRequest(
        protocol=base.protocol,
        system_instruction=base.system_instruction + instruction,
        messages=messages,
        source_event_ids=source_event_ids,
    )


def _assessment_repair_request(
    base: ProviderConversationRequest,
    invalid_content: str,
    error: Exception,
) -> ProviderConversationRequest:
    """Append the invalid response and its error reason for one repair round."""

    feedback = (
        "The previous response to the out-of-band relevant-fields assessment "
        f"was invalid: {error}. " + _relevant_fields_query()
    )
    if base.protocol == "gemini_generate_content":
        contents = deepcopy(base.contents)
        contents.append({"role": "model", "parts": [{"text": invalid_content}]})
        contents.append({"role": "user", "parts": [{"text": feedback}]})
        return ProviderConversationRequest(
            protocol=base.protocol,
            system_instruction=base.system_instruction,
            contents=contents,
            source_event_ids=deepcopy(base.source_event_ids) + [[], []],
        )
    messages = deepcopy(base.messages)
    messages.append({"role": "assistant", "content": invalid_content})
    messages.append({"role": "user", "content": feedback})
    return ProviderConversationRequest(
        protocol=base.protocol,
        system_instruction=base.system_instruction,
        messages=messages,
        source_event_ids=deepcopy(base.source_event_ids) + [[], []],
    )


def _parse_relevant_fields_response(
    content: str,
) -> dict[str, Any]:
    text = str(content or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # JSON-object output is requested from every provider, but retaining a
        # small recovery path makes the diagnostic artifact useful when a model
        # wraps the object in Markdown fences or a short sentence.
        start = text.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("assessment response must be a JSON object")
    fields = parsed.get("relevant_fields")
    if not isinstance(fields, list) or any(not isinstance(item, str) for item in fields):
        raise ValueError("assessment relevant_fields must be an array of strings")
    # Preserve the model's JSON shape while normalizing whitespace and duplicate
    # IDs for straightforward downstream comparison.
    normalized: list[str] = []
    for field in fields:
        value = field.strip()
        if value and value not in normalized:
            normalized.append(value)
    directions = parsed.get("field_directions")
    if not isinstance(directions, list):
        raise ValueError("assessment field_directions must be an array")
    direction_fields: list[str] = []
    for item in directions:
        if not isinstance(item, dict) or not isinstance(item.get("field_id"), str):
            raise ValueError("each field direction must contain a field_id")
        if item.get("direction") not in ("positive", "negative"):
            raise ValueError("assessment directions must be positive or negative")
        direction_fields.append(item["field_id"])
    if set(direction_fields) != set(normalized) or len(direction_fields) != len(normalized):
        raise ValueError("assessment requires exactly one direction per relevant field")
    return {"relevant_fields": normalized, "field_directions": directions}


# Assessment statuses that count as finished for resume purposes: the model
# either produced a valid answer or used up its repair allowance. A
# ``provider_error`` assessment never got a usable response and is
# re-collected on resume. Legacy assessments are incomplete for this contract.
ASSESSMENT_TERMINAL_STATUSES = frozenset({"completed", "syntax_exhausted"})


def assessment_is_terminal(record: Any) -> bool:
    if not isinstance(record, dict) or record.get("status") not in ASSESSMENT_TERMINAL_STATUSES:
        return False
    if record["status"] == "syntax_exhausted":
        return record.get("assessment_version") == 2
    try:
        _parse_relevant_fields_response(json.dumps(record.get("response")))
    except (TypeError, ValueError):
        return False
    return True


async def _collect_relevant_fields_assessment(
    episode: dict[str, Any],
    client: Any,
    *,
    conversation: ConversationSession,
    adapter: ConversationAdapter,
    phase: str,
    setting_id: str,
    condition: str,
    sampling_branch: str | None,
    history_policy: str,
) -> dict[str, Any]:
    """Ask for relevant fields and directions outside the trajectory log.

    Follows the same config-driven failure policy as the benchmark trajectory
    (``GeneratorConfig.provider_retry_limit`` / ``syntax_repair_limit``):
    provider errors are retried with backoff, and a malformed response gets
    repair rounds with the invalid response and its error reason fed back.
    The terminal status is ``completed`` (valid answer), ``syntax_exhausted``
    (the model answered but used up its repair allowance), or
    ``provider_error`` (no usable response; a resume re-collects it).
    """

    request_id = (
        f"{episode['episode_id']}:{setting_id}:relevant_fields:{phase}"
    )
    request = _assessment_request(conversation, adapter)
    trace_context = {
        "episode_id": episode["episode_id"],
        "condition": condition,
        "sampling_branch": sampling_branch,
        "history_policy": history_policy,
        "setting_id": setting_id,
        "analysis": "relevant_fields",
        "analysis_phase": phase,
        # Diagnostic calls do not increment the environment counter, but a
        # monotonic context value keeps provider traces and progress tooling
        # able to place them after the preceding trajectory interaction.
        "interaction": conversation.interaction_count + 1,
        "request_id": request_id,
    }
    record: dict[str, Any] = {
        "assessment_version": 2,
        "phase": phase,
        "request_id": request_id,
        "status": "pending",
        "prompt": _relevant_fields_query(),
        "trajectory_interaction_count": conversation.interaction_count,
        "trajectory_event_count": len(conversation.events),
    }
    provider_delays = _provider_retry_delays(client)
    repair_limit = _syntax_repair_limit(client)
    provider_retry_events: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for attempt in range(repair_limit + 1):
        attempt_context = trace_context if attempt == 0 else {
            **trace_context,
            "repair_attempt": attempt,
        }
        response = None
        for retry_index in range(len(provider_delays) + 1):
            try:
                response = await client.generate(
                    request=request,
                    schema=RELEVANT_FIELDS_OUTPUT_SCHEMA,
                    trace_context=attempt_context,
                )
                break
            except Exception as exc:
                if retry_index < len(provider_delays):
                    delay_seconds = provider_delays[retry_index]
                    provider_retry_events.append(
                        {
                            "delay_seconds": delay_seconds,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                        }
                    )
                    await asyncio.sleep(delay_seconds)
                    continue
                # Provider failures exhaust the same retry budget as the
                # benchmark trajectory; without a response there is nothing
                # to repair, and a resume re-collects the assessment.
                record.update(
                    {
                        "status": "provider_error",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    }
                )
                if repairs:
                    record["repairs"] = repairs
                if provider_retry_events:
                    record["provider_retry_events"] = provider_retry_events
                return record
        assert response is not None
        record["raw_response"] = response.content
        try:
            parsed = _parse_relevant_fields_response(
                response.content,
            )
        except ValueError as exc:
            repairs.append(
                {
                    "raw_response": response.content,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }
            )
            if attempt < repair_limit:
                # A malformed assessment gets a repair opportunity: the
                # invalid response and the error reason go back to the model.
                request = _assessment_repair_request(
                    request, response.content, exc
                )
                continue
            # The model answered but exhausted its repair allowance. This is
            # a terminal assessment outcome and counts as complete on resume.
            record.update(
                {
                    "status": "syntax_exhausted",
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }
            )
            record["repairs"] = repairs
            if provider_retry_events:
                record["provider_retry_events"] = provider_retry_events
            return record
        allowed_fields = set(conversation.task_spec.get("raw_field_schema") or {})
        unknown_fields = [
            field
            for field in parsed["relevant_fields"]
            if field not in allowed_fields
        ]
        record.update(
            {
                "status": "completed",
                "response": parsed,
                "model": response.model,
                "usage": dict(response.usage or {}),
                "finish_reason": response.finish_reason,
            }
        )
        if unknown_fields:
            record["unknown_fields"] = unknown_fields
        if repairs:
            record["repairs"] = repairs
        if provider_retry_events:
            record["provider_retry_events"] = provider_retry_events
        return record
    return record


async def _checkpoint(
    callback: Checkpoint | None,
    episode: dict[str, Any],
    env: BenchmarkEnvironment,
    trace: list[dict[str, Any]],
    conversation: ConversationSession | None = None,
    relevant_fields_assessments: dict[str, Any] | None = None,
) -> None:
    if callback is not None:
        await callback(
            _artifact(
                episode,
                env,
                run_success=None,
                trace=trace,
                conversation=conversation,
                relevant_fields_assessments=relevant_fields_assessments,
            )
        )


async def _drive_agent_episode(
    episode: dict[str, Any],
    client: Any,
    *,
    env: BenchmarkEnvironment,
    trace: list[dict[str, Any]],
    conversation: ConversationSession,
    adapter: ConversationAdapter,
    checkpoint: Checkpoint | None = None,
    stop_after_batch: int | None = None,
    relevant_fields_assessments: dict[str, Any] | None = None,
) -> tuple[BenchmarkEnvironment, list[dict[str, Any]], ConversationSession, dict[str, Any]]:
    decision_schema_for_run = decision_schema(
        enable_probe_skip=env.enable_probe_skip
    )
    reasoning_history_policy = getattr(
        adapter, "reasoning_history_policy", "provider_continuation"
    )
    while not env.done and (
        stop_after_batch is None or env.next_batch_index < stop_after_batch
    ):
        request_id = (
            f"{episode['episode_id']}:{env.condition}:{env.history_policy}:"
            f"request_{len(trace) + 1:03d}"
        )
        projection = project_conversation(conversation)
        provider_request = adapter.build_request(projection.view)
        request_payload = wire_request_payload(provider_request)
        wire_items = provider_request.messages or provider_request.contents
        last_event_id = conversation.events[-1].event_id if conversation.events else None
        record: dict[str, Any] = {
            "request_id": request_id,
            "interaction": env.interactions_used + 1,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "status": "awaiting_response",
            "request_message_count": len(wire_items),
            "request_prompt_characters": sum(
                len(str(message.get("content") or ""))
                + sum(
                    len(str(part.get("text") or ""))
                    for part in (message.get("parts") or [])
                    if isinstance(part, dict)
                )
                for message in wire_items
            ),
            "provider_wire_request_characters": provider_request.wire_characters,
            "canonical_visible_history_characters": projection.view.visible_history_characters(),
            "static_prompt_characters": len(conversation.system_instruction),
            "task_spec_characters": _json_characters(conversation.task_spec),
            "latest_state_card_characters": _json_characters(
                projection.view.latest_state_card
            ),
            "history_event_count": len(conversation.events),
            "visible_history_event_count": len(projection.view.events),
            "request_context": {
                "request_id": request_id,
                "history_policy": conversation.history_policy,
                "history_policy_version": conversation.history_policy_version,
                "full_log_through_event_id": last_event_id,
                "full_log_event_count": len(conversation.events),
                "visible_event_count": len(projection.view.events),
                "event_mapping": deepcopy_json(projection.event_mapping),
                "wire_event_groups": deepcopy_json(provider_request.source_event_ids),
                "wire_request_sha256": payload_sha256(request_payload),
                "wire_request": request_payload,
            },
            "request_schema": decision_schema_for_run,
            "provider_attempt_count": 0,
            "provider_retry_events": [],
            "valid": None,
        }
        trace.append(record)
        await _checkpoint(
            checkpoint,
            episode,
            env,
            trace,
            conversation,
            relevant_fields_assessments,
        )
        started = time.monotonic()
        raw = None
        truncated_attempts = 0
        repeatedly_truncated = False
        provider_delays = _provider_retry_delays(client)
        for provider_attempt in range(1, len(provider_delays) + 2):
            record["provider_attempt_count"] = provider_attempt
            try:
                raw = await client.generate(
                    request=provider_request,
                    schema=decision_schema_for_run,
                    trace_context={
                        "episode_id": episode["episode_id"],
                        "condition": env.condition,
                        "sampling_branch": env.sampling_branch,
                        "history_policy": env.history_policy,
                        "enable_probe_skip": env.enable_probe_skip,
                        "least_probe_count": env.least_probe_count,
                        "interaction": env.interactions_used + 1,
                        "request_id": request_id,
                    },
                )
                if _is_truncated_finish_reason(raw.finish_reason):
                    raise TruncatedResponseError(
                        "response exceeded the maximum response length "
                        f"(finish_reason={raw.finish_reason!r})"
                    )
                break
            except TruncatedResponseError as exc:
                truncated_attempts += 1
                if truncated_attempts < TRUNCATED_RESPONSE_MAX_ATTEMPTS:
                    record["provider_retry_events"].append(
                        {
                            "failed_attempt": provider_attempt,
                            "delay_seconds": 0.0,
                            "error_type": "truncated_response",
                            "error": str(exc),
                        }
                    )
                    # A truncated response (typically runaway repetition)
                    # never enters the conversation and consumes no
                    # interaction/repair attempt; it gets one free retry
                    # before the window is recorded as unanswered.
                    await _checkpoint(
                        checkpoint,
                        episode,
                        env,
                        trace,
                        conversation,
                        relevant_fields_assessments,
                    )
                    continue
                repeatedly_truncated = True
                break
            except Exception as exc:
                retry_index = provider_attempt - 1
                if retry_index < len(provider_delays):
                    delay_seconds = provider_delays[retry_index]
                    record["provider_retry_events"].append(
                        {
                            "failed_attempt": provider_attempt,
                            "delay_seconds": delay_seconds,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                        }
                    )
                    # Provider failures do not enter the conversation or consume
                    # an environment interaction/repair attempt. Persist the
                    # still-unapplied request before backing off so a checkpoint
                    # remains safely resumable.
                    await _checkpoint(
                        checkpoint,
                        episode,
                        env,
                        trace,
                        conversation,
                        relevant_fields_assessments,
                    )
                    await asyncio.sleep(delay_seconds)
                    continue

                record.update(
                    {
                        "status": "provider_error",
                        "error": str(exc),
                        "error_type": "provider_error",
                        "latency_seconds": time.monotonic() - started,
                        "valid": False,
                    }
                )
                # A provider failure does not change the environment. Keep the
                # successfully applied trajectory resumable instead of converting
                # all remaining cases into unanswered terminal results. Resume
                # drops this unapplied request record and retries the same
                # interaction from the preceding checkpointed state.
                artifact = _artifact(
                    episode,
                    env,
                    run_success=False,
                    trace=trace,
                    conversation=conversation,
                    relevant_fields_assessments=relevant_fields_assessments,
                    error=f"provider error: {exc}",
                )
                if checkpoint:
                    await checkpoint(artifact)
                return env, trace, conversation, artifact

        assert raw is not None

        pending_before = deepcopy(env.pending_batch)
        counted = False
        tool: str | None = None
        payload: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        feedback: dict[str, Any] | None = None
        valid = False
        if repeatedly_truncated:
            feedback = env.record_truncated_response(
                "response exceeded the maximum response length on two "
                "consecutive attempts"
            )
            counted = True
            record.update(
                {
                    "interaction": env.interactions_used,
                    "status": "invalid_response",
                    "response": None,
                    "raw_response": raw.content,
                    "reasoning_content": raw.reasoning_content,
                    "reasoning_signature_present": (
                        raw.reasoning_signature_present
                    ),
                    "provider_continuation_present": bool(
                        getattr(raw, "assistant_content", None)
                    ),
                    "reasoning_history_policy": reasoning_history_policy,
                    "provider_raw_response": raw.raw,
                    "result": feedback,
                    "error": feedback["error"]["message"],
                    "error_type": "response_truncated",
                    "usage": dict(raw.usage or {}),
                    "model": raw.model or getattr(getattr(client, "config", None), "model", ""),
                    "latency_seconds": time.monotonic() - started,
                    "finish_reason": raw.finish_reason,
                    "valid": False,
                }
            )
        else:
            try:
                payload = parse_tool_payload(
                    raw.content,
                    enable_probe_skip=env.enable_probe_skip,
                )
                tool = payload["tool"]
                try:
                    result = env.execute(tool, payload["arguments"])
                finally:
                    counted = True
                valid = True
                record.update(
                    {
                        "interaction": env.interactions_used,
                        "status": "completed",
                        "response": payload,
                        "raw_response": raw.content,
                        "reasoning_content": raw.reasoning_content,
                        "reasoning_signature_present": (
                            raw.reasoning_signature_present
                        ),
                        "provider_continuation_present": bool(
                            getattr(raw, "assistant_content", None)
                        ),
                        "reasoning_history_policy": reasoning_history_policy,
                        "provider_raw_response": raw.raw,
                        "result": result,
                        "usage": dict(raw.usage or {}),
                        "model": raw.model or getattr(getattr(client, "config", None), "model", ""),
                        "latency_seconds": time.monotonic() - started,
                        "finish_reason": raw.finish_reason,
                        "valid": True,
                    }
                )
            except Exception as exc:
                error_type = (
                    exc.error_type
                    if isinstance(exc, ProtocolError)
                    else "environment_rejection"
                )
                feedback = env.record_invalid_response(
                    str(exc),
                    error_type,
                    tool=tool,
                    interaction_already_counted=counted,
                )
                record.update(
                    {
                        "interaction": env.interactions_used,
                        "status": "invalid_response",
                        "response": payload,
                        "raw_response": raw.content,
                        "reasoning_content": raw.reasoning_content,
                        "reasoning_signature_present": (
                            raw.reasoning_signature_present
                        ),
                        "provider_continuation_present": bool(
                            getattr(raw, "assistant_content", None)
                        ),
                        "reasoning_history_policy": reasoning_history_policy,
                        "provider_raw_response": raw.raw,
                        "result": feedback,
                        "error": str(exc),
                        "error_type": error_type,
                        "usage": dict(raw.usage or {}),
                        "model": raw.model or getattr(getattr(client, "config", None), "model", ""),
                        "latency_seconds": time.monotonic() - started,
                        "finish_reason": raw.finish_reason,
                        "valid": False,
                    }
                )
        conversation.append_assistant(
            raw.content,
            parsed_call=payload,
            valid=valid,
            provider_continuation=adapter.capture_response(raw),
            interaction=env.interactions_used,
            stage="action_solving" if pending_before is not None else "control",
            batch_id=(pending_before or {}).get("batch_id"),
            batch_number=(pending_before or {}).get("batch_number"),
        )
        pending_after = deepcopy(env.pending_batch)
        if pending_before is not None and pending_after is None:
            environment_stage = "completion"
            event_batch = pending_before
        elif pending_before is None and pending_after is not None:
            environment_stage = "reveal"
            event_batch = pending_after
        elif pending_before is not None:
            environment_stage = "action_solving"
            event_batch = pending_before
        else:
            environment_stage = "control"
            event_batch = None
        conversation.append_environment(
            result if valid else feedback,
            env.state_card(),
            error=(feedback or {}).get("error") if not valid else None,
            interaction=env.interactions_used,
            stage=environment_stage,
            batch_id=(event_batch or {}).get("batch_id"),
            batch_number=(event_batch or {}).get("batch_number"),
        )
        record["history_event_count"] = len(conversation.events)
        post_projection = project_conversation(conversation)
        record["canonical_visible_history_characters"] = post_projection.view.visible_history_characters()
        record["visible_history_event_count"] = len(post_projection.view.events)
        await _checkpoint(
            checkpoint,
            episode,
            env,
            trace,
            conversation,
            relevant_fields_assessments,
        )
    stopped = not env.done and stop_after_batch is not None
    succeeded = None if stopped else bool(
        env.termination and env.termination["status"] == "completed"
    )
    artifact = _artifact(
        episode,
        env,
        run_success=succeeded,
        trace=trace,
        conversation=conversation,
        relevant_fields_assessments=relevant_fields_assessments,
        error=None if succeeded is not False else str(env.termination),
    )
    if checkpoint:
        await checkpoint(artifact)
    return env, trace, conversation, artifact


def _agent_state(
    episode: dict[str, Any],
    client: Any,
    adapter: ConversationAdapter | None,
    *,
    interaction_policy: dict[str, Any],
    condition: str,
    sampling_branch: str | None = None,
    history_policy: str = "eval_case_given",
    enable_probe_skip: bool = True,
    least_probe_count: int | None = None,
) -> tuple[BenchmarkEnvironment, list[dict[str, Any]], ConversationSession, ConversationAdapter]:
    history_policy = validate_history_policy(history_policy)
    env = BenchmarkEnvironment(
        episode,
        interaction_policy=interaction_policy,
        condition=condition,
        sampling_branch=sampling_branch,
        history_policy=history_policy,
        enable_probe_skip=enable_probe_skip,
        least_probe_count=least_probe_count,
        syntax_repair_limit=_syntax_repair_limit(client),
    )
    if adapter is None:
        protocol = getattr(
            getattr(client, "config", None), "conversation_protocol", None
        )
        adapter = get_conversation_adapter(protocol)
    _task_spec = env.task_spec()
    template_name = (
        "agent_system_eval_case_given.txt"
        if history_policy == "eval_case_given"
        else "agent_system.txt"
    )
    _template = Path("prompts/runtime").joinpath(template_name).read_text()
    conversation = ConversationSession(
        system_instruction=render_system_prompt(
            _template, _task_spec, history_policy=history_policy
        ),
        task_spec=_task_spec,
        history_policy=history_policy,
        history_policy_version=HISTORY_POLICY_VERSION,
    )
    conversation.append_initial_state(env.state_card())
    return env, [], conversation, adapter


def _replayable_trace(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only interactions whose outputs were applied to the environment."""

    return [
        deepcopy(record)
        for record in artifact.get("tool_trace") or []
        if record.get("status") not in {"awaiting_response", "provider_error"}
        and record.get("valid") is not None
    ]


def _replay_records(
    env: BenchmarkEnvironment,
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        response = record.get("response") or {}
        if record.get("valid") is True:
            env.execute(
                str(response["tool"]),
                deepcopy(response.get("arguments") or {}),
            )
            continue
        if response.get("tool"):
            try:
                env.execute(
                    str(response["tool"]),
                    deepcopy(response.get("arguments") or {}),
                )
            except Exception as exc:
                env.record_invalid_response(
                    str(exc),
                    str(record.get("error_type") or "environment_rejection"),
                    tool=str(response["tool"]),
                    interaction_already_counted=True,
                )
            else:
                raise ValueError(
                    "resume artifact marks a successful tool call as invalid"
                )
            continue
        error_type = str(record.get("error_type") or "invalid_payload")
        if error_type == "response_truncated":
            # Repeatedly truncated responses skip the window outright at
            # runtime (no repair attempt is consumed); replay must advance
            # the environment the same way or a following valid submit finds
            # no pending batch.
            env.record_truncated_response(
                str(record.get("error") or "invalid response")
            )
            continue
        env.record_invalid_response(
            str(record.get("error") or "invalid response"),
            error_type,
            interaction_already_counted=False,
        )


def _validate_restored_state(
    artifact: dict[str, Any],
    env: BenchmarkEnvironment,
    trace: list[dict[str, Any]],
    conversation: ConversationSession,
) -> None:
    private_state = artifact.get("private_state") or {}
    expected_batches = private_state.get("completed_batch_count")
    expected_interactions = private_state.get("interactions_used")
    if expected_batches is not None and env.next_batch_index != int(expected_batches):
        raise ValueError("resume artifact completed-batch state is inconsistent")
    if expected_interactions is not None and env.interactions_used != int(
        expected_interactions
    ):
        raise ValueError("resume artifact interaction state is inconsistent")
    if conversation.interaction_count != len(trace):
        raise ValueError("resume artifact conversation and trace are inconsistent")
    if conversation.history_policy != env.history_policy:
        raise ValueError("resume artifact history policy is inconsistent")
    if conversation.history_policy_version != HISTORY_POLICY_VERSION:
        raise ValueError("resume artifact history policy version is incompatible")


def _restore_prefix_trajectory(
    episode: dict[str, Any],
    client: Any,
    adapter: ConversationAdapter | None,
    *,
    interaction_policy: dict[str, Any],
    sampling_branch: str,
    artifact: dict[str, Any],
    history_policy: str,
    enable_probe_skip: bool = True,
    least_probe_count: int | None = None,
) -> tuple[
    BenchmarkEnvironment,
    list[dict[str, Any]],
    ConversationSession,
    ConversationAdapter,
]:
    env, _, _, adapter = _agent_state(
        episode,
        client,
        adapter,
        interaction_policy=interaction_policy,
        condition="dynamic",
        sampling_branch=sampling_branch,
        history_policy=history_policy,
        enable_probe_skip=enable_probe_skip,
        least_probe_count=least_probe_count,
    )
    trace = _replayable_trace(artifact)
    _replay_records(env, trace)
    conversation = _restore_visible_conversation(
        artifact["conversation"], env
    )
    _validate_restored_state(artifact, env, trace, conversation)
    return env, trace, conversation, adapter


def _restore_branch_trajectory(
    artifact: dict[str, Any],
    *,
    fork_env: BenchmarkEnvironment,
    prefix_trace: list[dict[str, Any]],
    sampling_branch: str,
    condition: str,
    history_policy: str,
) -> tuple[BenchmarkEnvironment, list[dict[str, Any]], ConversationSession]:
    trace = _replayable_trace(artifact)
    if trace[: len(prefix_trace)] != prefix_trace:
        raise ValueError("resume branch does not share the recorded prefix")
    env = deepcopy(fork_env)
    if env.history_policy != history_policy:
        raise ValueError("resume branch has the wrong history policy")
    env.select_runtime_branch(sampling_branch, condition)
    _replay_records(env, trace[len(prefix_trace) :])
    conversation = _restore_visible_conversation(artifact["conversation"], env)
    _validate_restored_state(artifact, env, trace, conversation)
    return env, trace, conversation


def _restore_visible_conversation(
    serialized: dict[str, Any], env: BenchmarkEnvironment
) -> ConversationSession:
    """Load a conversation while migrating source ids out of model history.

    Checkpoints created before runtime id anonymization can contain release
    ids in visible environment turns or assistant payloads.  Remap those ids
    using the selected branch's alias table before the conversation is sent to
    a provider again.  Runtime ``case_###`` ids are left untouched.
    """

    payload = deepcopy(serialized)
    aliases = {
        source: target
        for source, target in getattr(env, "_context_id_aliases", {}).items()
        if source != target and not source.startswith("case_")
    }
    if aliases:
        for source, target in sorted(aliases.items(), key=lambda item: -len(item[0])):
            payload = _replace_text(payload, source, target)
    return ConversationSession.from_dict(payload)


def _replace_text(value: Any, source: str, target: str) -> Any:
    if isinstance(value, str):
        return value.replace(source, target)
    if isinstance(value, list):
        return [_replace_text(item, source, target) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_text(item, source, target)
            for key, item in value.items()
        }
    return value


async def _refresh_completed_assessments(
    episode: dict[str, Any],
    client: Any,
    artifact: dict[str, Any],
    adapter: ConversationAdapter,
    conversations: dict[str, ConversationSession],
) -> dict[str, Any]:
    """Update only diagnostics, using saved history without driving an environment."""
    artifact = deepcopy(artifact)
    assessments = artifact.get("relevant_fields_assessments")
    if not isinstance(assessments, dict):
        assessments = {}
    for phase, conversation in conversations.items():
        if not assessment_is_terminal(assessments.get(phase)):
            assessments[phase] = await _collect_relevant_fields_assessment(
                episode,
                client,
                conversation=conversation,
                adapter=adapter,
                phase=phase,
                setting_id=artifact["setting_id"],
                condition=artifact["condition"],
                sampling_branch=artifact.get("sampling_branch"),
                history_policy=artifact["history_policy"],
            )
    artifact["relevant_fields_assessments"] = assessments
    return artifact


async def run_agent_episode(
    episode: dict[str, Any],
    client: Any,
    *,
    interaction_policy: dict[str, Any],
    checkpoint: Checkpoint | None = None,
    adapter: ConversationAdapter | None = None,
    condition: str = "dynamic",
    sampling_branch: str | None = None,
    history_policy: str = "eval_case_given",
    enable_probe_skip: bool = True,
    least_probe_count: int | None = None,
    collect_relevant_fields: bool = False,
    rule_change: bool | None = None,
    resume_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resume_artifact and resume_artifact.get("run_status") == "completed":
        if not collect_relevant_fields:
            return deepcopy(resume_artifact)
        if condition != "static" or rule_change:
            raise ValueError("completed rule-change runs must resume through the branched runner")
        artifact = await _refresh_completed_assessments(
            episode,
            client,
            resume_artifact,
            adapter or get_conversation_adapter(
                getattr(getattr(client, "config", None), "conversation_protocol", None)
            ),
            {"final": ConversationSession.from_dict(resume_artifact["conversation"])},
        )
        if checkpoint:
            await checkpoint(artifact)
        return artifact
    env, trace, conversation, adapter = _agent_state(
        episode,
        client,
        adapter,
        interaction_policy=interaction_policy,
        condition=condition,
        sampling_branch=sampling_branch,
        history_policy=history_policy,
        enable_probe_skip=enable_probe_skip,
        least_probe_count=least_probe_count,
    )
    assessments: dict[str, Any] | None = {} if collect_relevant_fields else None
    is_rule_change = (
        bool(rule_change)
        if rule_change is not None
        else (env.dataset_mode == "dynamic" and env.condition == "dynamic")
    )
    if collect_relevant_fields and is_rule_change:
        fork_after_batch = int(episode["evaluation"]["transition"]["after_action_batch"])
        _, _, _, artifact = await _drive_agent_episode(
            episode,
            client,
            env=env,
            trace=trace,
            conversation=conversation,
            adapter=adapter,
            checkpoint=checkpoint,
            stop_after_batch=fork_after_batch,
            relevant_fields_assessments=assessments,
        )
        if artifact.get("run_status") == "failed":
            return artifact
        assert assessments is not None
        assessments["pre_change"] = await _collect_relevant_fields_assessment(
            episode,
            client,
            conversation=conversation,
            adapter=adapter,
            phase="pre_change",
            setting_id=str(artifact.get("setting_id") or _setting_id(env)),
            condition=env.condition,
            sampling_branch=env.sampling_branch,
            history_policy=env.history_policy,
        )
        artifact["relevant_fields_assessments"] = deepcopy_json(assessments)
        if checkpoint:
            await checkpoint(artifact)
        _, _, _, artifact = await _drive_agent_episode(
            episode,
            client,
            env=env,
            trace=trace,
            conversation=conversation,
            adapter=adapter,
            checkpoint=checkpoint,
            relevant_fields_assessments=assessments,
        )
        final_phase = "post_change"
    else:
        _, _, _, artifact = await _drive_agent_episode(
            episode,
            client,
            env=env,
            trace=trace,
            conversation=conversation,
            adapter=adapter,
            checkpoint=checkpoint,
            relevant_fields_assessments=assessments,
        )
        final_phase = "final"
    if collect_relevant_fields:
        assert assessments is not None
        assessments[final_phase] = await _collect_relevant_fields_assessment(
            episode,
            client,
            conversation=conversation,
            adapter=adapter,
            phase=final_phase,
            setting_id=str(artifact.get("setting_id") or _setting_id(env)),
            condition=env.condition,
            sampling_branch=env.sampling_branch,
            history_policy=env.history_policy,
        )
        artifact["relevant_fields_assessments"] = deepcopy_json(assessments)
        if checkpoint:
            await checkpoint(artifact)
    return artifact


def _prefix_hash(artifact: dict[str, Any]) -> str:
    payload = {
        "public_history": artifact.get("public_history"),
        "private_history": artifact.get("private_history"),
        "tool_trace": artifact.get("tool_trace"),
        "conversation": artifact.get("conversation"),
    }
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_runtime_branches(
    configured: Any,
) -> dict[str, dict[str, Any]]:
    """Validate and normalize first-class runtime branch definitions."""

    if not isinstance(configured, dict) or not configured:
        raise ValueError("runtime branches must be a non-empty mapping")
    required_keys = {"sampling", "rule_change", "history_policy"}
    optional_keys = {"enable_probe_skip", "least_probe_count"}
    branches: dict[str, dict[str, Any]] = {}
    for name, raw in configured.items():
        if not isinstance(name, str) or not name:
            raise ValueError("runtime branch names must be non-empty strings")
        if not isinstance(raw, dict):
            raise ValueError(f"runtime branch {name!r} must be a mapping")
        unexpected = set(raw) - required_keys - optional_keys
        missing = required_keys - set(raw)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing key(s): {', '.join(sorted(missing))}")
            if unexpected:
                details.append(
                    f"unexpected key(s): {', '.join(sorted(unexpected))}"
                )
            raise ValueError(
                f"runtime branch {name!r} has invalid keys ({'; '.join(details)})"
            )
        sampling = raw["sampling"]
        if not isinstance(sampling, str) or not sampling:
            raise ValueError(
                f"runtime branch {name!r} sampling must be a non-empty string"
            )
        rule_change = raw["rule_change"]
        if not isinstance(rule_change, bool):
            raise ValueError(
                f"runtime branch {name!r} rule_change must be a boolean"
            )
        normalized = {
            "sampling": sampling,
            "rule_change": rule_change,
            "history_policy": validate_history_policy(raw["history_policy"]),
        }
        if "enable_probe_skip" in raw:
            if not isinstance(raw["enable_probe_skip"], bool):
                raise ValueError(
                    f"runtime branch {name!r} enable_probe_skip must be a boolean"
                )
            normalized["enable_probe_skip"] = raw["enable_probe_skip"]
        if "least_probe_count" in raw:
            value = raw["least_probe_count"]
            if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
                value = None
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"runtime branch {name!r} least_probe_count must be a "
                    "positive integer or none"
                )
            normalized["least_probe_count"] = value
        branches[name] = normalized
    return branches


def _branch_probe_policy(spec: dict[str, Any]) -> tuple[bool, int | None]:
    """Return effective probe controls while preserving legacy config shape."""

    enable_probe_skip = spec.get("enable_probe_skip", True)
    least_probe_count = spec.get("least_probe_count")
    return bool(enable_probe_skip), least_probe_count


def _branch_prefix_key(spec: dict[str, Any]) -> tuple[str, bool, int | None]:
    enable_probe_skip, least_probe_count = _branch_probe_policy(spec)
    return spec["history_policy"], enable_probe_skip, least_probe_count


def _prefix_group_labels(
    groups: dict[tuple[str, bool, int | None], Any]
) -> dict[tuple[str, bool, int | None], str]:
    """Choose artifact keys for prefix groups without changing legacy names."""

    history_counts: dict[str, int] = {}
    for history_policy, _, _ in groups:
        history_counts[history_policy] = history_counts.get(history_policy, 0) + 1
    labels: dict[tuple[str, bool, int | None], str] = {}
    for key in groups:
        history_policy, enable_probe_skip, least_probe_count = key
        if history_counts[history_policy] == 1:
            labels[key] = history_policy
            continue
        minimum = "none" if least_probe_count is None else str(least_probe_count)
        labels[key] = (
            f"{history_policy}::enable_probe_skip={str(enable_probe_skip).lower()}"
            f"::least_probe_count={minimum}"
        )
    return labels


def _runtime_branch_plan(
    branches: dict[str, Any],
) -> list[tuple[str, str, str]]:
    return [
        (
            name,
            spec["sampling"],
            "dynamic" if spec["rule_change"] else "static",
        )
        for name, spec in resolve_runtime_branches(branches).items()
    ]


async def _run_branched_agent_episode_single_policy(
    episode: dict[str, Any],
    client: Any,
    *,
    interaction_policy: dict[str, Any],
    branches: dict[str, Any],
    checkpoint: Checkpoint | None = None,
    adapter: ConversationAdapter | None = None,
    resume_artifact: dict[str, Any] | None = None,
    collect_relevant_fields: bool = False,
) -> dict[str, Any]:
    """Run one shared natural-IID prefix and configurable isolated post branches."""

    fork_after_batch = int(
        episode["evaluation"]["transition"]["after_action_batch"]
    )
    normalized_branches = resolve_runtime_branches(branches)
    history_policies = {
        spec["history_policy"] for spec in normalized_branches.values()
    }
    if len(history_policies) != 1:
        raise ValueError("internal branch group must use one history_policy")
    history_policy = next(iter(history_policies))
    branch_plan = _runtime_branch_plan(normalized_branches)
    first_sampling_branch = branch_plan[0][1]
    first_spec = next(iter(normalized_branches.values()))
    enable_probe_skip, least_probe_count = _branch_probe_policy(first_spec)
    branch_snapshots = {
        name: deepcopy(artifact)
        for name, artifact in ((resume_artifact or {}).get("branches") or {}).items()
    }
    checkpoint_lock = asyncio.Lock()

    def parent_artifact(
        prefix_artifact: dict[str, Any],
        *,
        run_status: str = "running",
        run_success: bool | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "trace_schema_version": BRANCHED_TRACE_SCHEMA_VERSION,
            "episode_schema_version": episode["schema_version"],
            "episode_id": episode["episode_id"],
            "episode_seed": episode["seed"],
            "run_success": run_success,
            "run_status": run_status,
            "fork_after_batch": fork_after_batch,
            "history_policy": history_policy,
            "history_policy_version": HISTORY_POLICY_VERSION,
            "enable_probe_skip": enable_probe_skip,
            "least_probe_count": least_probe_count,
            "shared_prefix_hash": _prefix_hash(prefix_artifact),
            "shared_prefix": deepcopy(prefix_artifact),
            "branches": deepcopy(branch_snapshots),
            "error": error,
        }

    prefix: dict[str, Any]
    if resume_artifact and resume_artifact.get("shared_prefix"):
        prefix = deepcopy(resume_artifact["shared_prefix"])
        if resume_artifact.get("shared_prefix_hash") != _prefix_hash(prefix):
            raise ValueError("resume artifact shared prefix hash is invalid")
        if all(
            branch_snapshots.get(name, {}).get("run_status") == "completed"
            for name in normalized_branches
        ):
            # Completed trajectories need only their saved conversations for
            # diagnostics; do not replay any environment transitions.
            refreshed = deepcopy(resume_artifact)
            if refreshed.get("run_status") != "completed":
                refreshed.update(run_status="completed", run_success=True, error=None)
            adapter = adapter or get_conversation_adapter(
                getattr(getattr(client, "config", None), "conversation_protocol", None)
            )
            if collect_relevant_fields:
                for name, spec in normalized_branches.items():
                    existing = branch_snapshots[name]
                    phase_conversations = {}
                    if spec["rule_change"]:
                        phase_conversations["pre_change"] = ConversationSession.from_dict(
                            prefix["conversation"]
                        )
                    final_phase = "post_change" if spec["rule_change"] else "final"
                    phase_conversations[final_phase] = ConversationSession.from_dict(
                        existing["conversation"]
                    )
                    refreshed["branches"][name] = await _refresh_completed_assessments(
                        episode, client, existing, adapter, phase_conversations
                    )
                    if checkpoint:
                        await checkpoint(deepcopy(refreshed))
            return refreshed
        env, trace, conversation, adapter = _restore_prefix_trajectory(
            episode,
            client,
            adapter,
            interaction_policy=interaction_policy,
            sampling_branch=first_sampling_branch,
            artifact=prefix,
            history_policy=history_policy,
            enable_probe_skip=enable_probe_skip,
            least_probe_count=least_probe_count,
        )
    else:
        env, trace, conversation, adapter = _agent_state(
            episode,
            client,
            adapter,
            interaction_policy=interaction_policy,
            condition="dynamic",
            sampling_branch=first_sampling_branch,
            history_policy=history_policy,
            enable_probe_skip=enable_probe_skip,
            least_probe_count=least_probe_count,
        )

    async def save_prefix(partial: dict[str, Any]) -> None:
        nonlocal prefix
        prefix = deepcopy(partial)
        if checkpoint:
            async with checkpoint_lock:
                await checkpoint(parent_artifact(prefix))

    if env.next_batch_index > fork_after_batch:
        raise ValueError("resume prefix has advanced beyond the fork")
    if env.next_batch_index < fork_after_batch and not env.done:
        env, trace, conversation, prefix = await _drive_agent_episode(
            episode,
            client,
            env=env,
            trace=trace,
            conversation=conversation,
            adapter=adapter,
            checkpoint=save_prefix,
            stop_after_batch=fork_after_batch,
        )
    if prefix.get("run_status") == "failed":
        return parent_artifact(
            prefix,
            run_status="failed",
            run_success=False,
            error=prefix.get("error"),
        )
    if env.done:
        return parent_artifact(
            prefix,
            run_status="completed",
            run_success=True,
            error=prefix.get("error"),
        )
    if env.next_batch_index != fork_after_batch:
        raise ValueError("prefix trajectory did not reach the configured fork")

    trace = _replayable_trace(prefix)
    prefix = {
        **prefix,
        "tool_trace": deepcopy(trace),
        "run_success": None,
        "run_status": "running",
        "error": None,
        "termination": None,
    }
    prefix_hash = _prefix_hash(prefix)
    if checkpoint:
        async with checkpoint_lock:
            await checkpoint(parent_artifact(prefix))

    async def run_branch(
        run_name: str, sampling_branch: str, condition: str
    ) -> tuple[str, dict[str, Any]]:
        existing = branch_snapshots.get(run_name)
        branch_rule_change = bool(normalized_branches[run_name].get("rule_change"))
        required_assessments = (
            {"pre_change", "post_change"}
            if branch_rule_change
            else {"final"}
        )
        existing_assessments = (
            existing.get("relevant_fields_assessments")
            if isinstance(existing, dict)
            else None
        )
        if (
            existing
            and existing.get("run_status") == "completed"
            and (
                not collect_relevant_fields
                or all(
                    isinstance(existing_assessments, dict)
                    and assessment_is_terminal(existing_assessments.get(phase))
                    for phase in required_assessments
                )
            )
        ):
            return run_name, existing
        if existing and existing.get("run_status") == "completed":
            phase_conversations = {
                "post_change" if branch_rule_change else "final":
                    ConversationSession.from_dict(existing["conversation"]),
            }
            if branch_rule_change:
                phase_conversations = {"pre_change": deepcopy(conversation), **phase_conversations}
            artifact = await _refresh_completed_assessments(
                episode, client, existing, adapter, phase_conversations
            )
            async with checkpoint_lock:
                branch_snapshots[run_name] = deepcopy(artifact)
                if checkpoint:
                    await checkpoint(parent_artifact(prefix))
            return run_name, artifact
        if existing:
            branch_env, branch_trace, branch_conversation = (
                _restore_branch_trajectory(
                    existing,
                    fork_env=env,
                    prefix_trace=trace,
                    sampling_branch=sampling_branch,
                    condition=condition,
                    history_policy=history_policy,
                )
            )
        else:
            branch_env = deepcopy(env)
            branch_env.select_runtime_branch(sampling_branch, condition)
            branch_trace = deepcopy(trace)
            branch_conversation = deepcopy(conversation)

        assessments: dict[str, Any] | None = (
            deepcopy(existing_assessments)
            if collect_relevant_fields and isinstance(existing_assessments, dict)
            else ({} if collect_relevant_fields else None)
        )

        async def save_branch(partial: dict[str, Any]) -> None:
            snapshot = {
                **partial,
                "branch": run_name,
                "setting_id": run_name,
                "sampling_branch": sampling_branch,
                "fork_after_batch": fork_after_batch,
                "shared_prefix_hash": prefix_hash,
            }
            if assessments is not None:
                snapshot["relevant_fields_assessments"] = deepcopy_json(assessments)
            async with checkpoint_lock:
                branch_snapshots[run_name] = deepcopy(snapshot)
                if checkpoint:
                    await checkpoint(parent_artifact(prefix))

        if (
            collect_relevant_fields
            and branch_rule_change
            and not assessment_is_terminal(assessments.get("pre_change"))
        ):
            assessments["pre_change"] = await _collect_relevant_fields_assessment(
                episode,
                client,
                conversation=deepcopy(conversation),
                adapter=adapter,
                phase="pre_change",
                setting_id=run_name,
                condition=condition,
                sampling_branch=sampling_branch,
                history_policy=history_policy,
            )
            # Persist the forked diagnostic before post-change generation starts.
            await save_branch(
                _artifact(
                    episode,
                    branch_env,
                    run_success=None,
                    trace=branch_trace,
                    conversation=branch_conversation,
                    relevant_fields_assessments=assessments,
                )
            )

        result = await _drive_agent_episode(
            episode,
            client,
            env=branch_env,
            trace=branch_trace,
            conversation=branch_conversation,
            adapter=adapter,
            checkpoint=save_branch,
            relevant_fields_assessments=assessments,
        )
        artifact = result[3]
        if collect_relevant_fields:
            final_phase = "post_change" if branch_rule_change else "final"
            if not assessment_is_terminal(assessments.get(final_phase)):
                assessments[final_phase] = await _collect_relevant_fields_assessment(
                    episode,
                    client,
                    conversation=branch_conversation,
                    adapter=adapter,
                    phase=final_phase,
                    setting_id=run_name,
                    condition=condition,
                    sampling_branch=sampling_branch,
                    history_policy=history_policy,
                )
            artifact["relevant_fields_assessments"] = deepcopy_json(assessments)
        artifact["branch"] = run_name
        artifact["setting_id"] = run_name
        artifact["sampling_branch"] = sampling_branch
        artifact["fork_after_batch"] = fork_after_batch
        artifact["shared_prefix_hash"] = prefix_hash
        async with checkpoint_lock:
            branch_snapshots[run_name] = deepcopy(artifact)
            if checkpoint:
                await checkpoint(parent_artifact(prefix))
        return run_name, artifact

    branch_results = await asyncio.gather(
        *(run_branch(*branch) for branch in branch_plan)
    )
    branch_artifacts = dict(branch_results)
    for artifact in branch_artifacts.values():
        artifact["episode_id"] = episode["episode_id"]
        artifact["shared_prefix_hash"] = prefix_hash
    succeeded = all(
        artifact.get("run_status") == "completed"
        for artifact in branch_artifacts.values()
    )
    return {
        "trace_schema_version": BRANCHED_TRACE_SCHEMA_VERSION,
        "episode_schema_version": episode["schema_version"],
        "episode_id": episode["episode_id"],
        "episode_seed": episode["seed"],
        "run_success": succeeded,
        "run_status": "completed" if succeeded else "completed_with_failures",
        "fork_after_batch": fork_after_batch,
        "history_policy": history_policy,
        "history_policy_version": HISTORY_POLICY_VERSION,
        "history_policies": [history_policy],
        "shared_prefix_hash": prefix_hash,
        "shared_prefix": prefix,
        "branches": branch_artifacts,
    }


def _history_group_resume_artifact(
    resume_artifact: dict[str, Any] | None,
    *,
    history_policy: str,
    prefix_key: str | None = None,
    probe_policy: tuple[bool, int | None] | None = None,
    branch_names: set[str],
    primary_policy: str,
) -> dict[str, Any] | None:
    if not resume_artifact:
        return None
    prefixes = resume_artifact.get("shared_prefixes") or {}
    hashes = resume_artifact.get("shared_prefix_hashes") or {}
    lookup_key = prefix_key or history_policy
    prefix = prefixes.get(lookup_key)
    prefix_hash = hashes.get(lookup_key)
    # Single-policy artifacts (including outputs created before the
    # per-history ``shared_prefixes`` shape) store their prefix only in the
    # primary ``shared_prefix`` field.  When a resume adds a policy, the new
    # config may list that policy first, so use the legacy prefix according to
    # the artifact's recorded policy rather than the current config order.
    artifact_policy = resume_artifact.get("history_policy")
    legacy_branch_match = bool(
        branch_names & set((resume_artifact.get("branches") or {}).keys())
    )
    legacy_policy_match = True
    if probe_policy is not None and legacy_branch_match:
        expected_skip, expected_count = probe_policy
        legacy_policy_match = all(
            (
                bool(branch.get("enable_probe_skip", True)),
                branch.get("least_probe_count"),
            )
            == (expected_skip, expected_count)
            for name, branch in (resume_artifact.get("branches") or {}).items()
            if name in branch_names
        )
    if lookup_key == history_policy and (
        artifact_policy == history_policy
        or (artifact_policy is None and history_policy == primary_policy)
    ):
        prefix = prefix or resume_artifact.get("shared_prefix")
        prefix_hash = prefix_hash or resume_artifact.get("shared_prefix_hash")
    elif legacy_branch_match and legacy_policy_match and artifact_policy == history_policy:
        prefix = prefix or resume_artifact.get("shared_prefix")
        prefix_hash = prefix_hash or resume_artifact.get("shared_prefix_hash")
    branches = {
        name: deepcopy(branch)
        for name, branch in (resume_artifact.get("branches") or {}).items()
        if name in branch_names
    }
    if prefix is None and not branches:
        return None
    return {
        "trace_schema_version": BRANCHED_TRACE_SCHEMA_VERSION,
        "episode_schema_version": resume_artifact.get("episode_schema_version"),
        "episode_id": resume_artifact.get("episode_id"),
        "episode_seed": resume_artifact.get("episode_seed"),
        "run_success": resume_artifact.get("run_success"),
        "run_status": resume_artifact.get("run_status"),
        "fork_after_batch": resume_artifact.get("fork_after_batch"),
        "history_policy": history_policy,
        "history_policy_version": HISTORY_POLICY_VERSION,
        "shared_prefix_hash": prefix_hash,
        "shared_prefix": deepcopy(prefix),
        "branches": branches,
    }


async def run_branched_agent_episode(
    episode: dict[str, Any],
    client: Any,
    *,
    interaction_policy: dict[str, Any],
    branches: dict[str, Any],
    checkpoint: Checkpoint | None = None,
    adapter: ConversationAdapter | None = None,
    resume_artifact: dict[str, Any] | None = None,
    collect_relevant_fields: bool = False,
) -> dict[str, Any]:
    """Run first-class branches, sharing one prefix per history policy."""

    normalized_branches = resolve_runtime_branches(branches)
    groups: dict[
        tuple[str, bool, int | None], dict[str, dict[str, Any]]
    ] = {}
    for name, spec in normalized_branches.items():
        groups.setdefault(_branch_prefix_key(spec), {})[name] = spec
    group_labels = _prefix_group_labels(groups)
    primary_group = next(iter(groups))
    primary_policy = primary_group[0]
    if len(groups) == 1:
        return await _run_branched_agent_episode_single_policy(
            episode,
            client,
            interaction_policy=interaction_policy,
            branches=normalized_branches,
            checkpoint=checkpoint,
            adapter=adapter,
            resume_artifact=resume_artifact,
            collect_relevant_fields=collect_relevant_fields,
        )

    group_artifacts: dict[
        tuple[str, bool, int | None], dict[str, Any]
    ] = {}
    merge_lock = asyncio.Lock()

    def merged_artifact() -> dict[str, Any]:
        prefixes = {
            group_labels[key]: deepcopy(group["shared_prefix"])
            for key, group in group_artifacts.items()
            if group.get("shared_prefix") is not None
        }
        hashes = {
            group_labels[key]: group.get("shared_prefix_hash")
            for key, group in group_artifacts.items()
            if group.get("shared_prefix") is not None
        }
        branch_artifacts = {
            name: deepcopy(group_artifacts[_branch_prefix_key(spec)]["branches"][name])
            for name, spec in normalized_branches.items()
            if name
            in (
                group_artifacts.get(_branch_prefix_key(spec), {}).get("branches")
                or {}
            )
        }
        statuses = [
            group.get("run_status")
            for group in group_artifacts.values()
            if group
        ]
        all_finished = len(statuses) == len(groups) and all(
            status in {"completed", "completed_with_failures", "failed"}
            for status in statuses
        )
        succeeded = all_finished and all(status == "completed" for status in statuses)
        if not all_finished:
            run_status = "running"
            run_success = None
        elif succeeded:
            run_status = "completed"
            run_success = True
        else:
            run_status = "completed_with_failures"
            run_success = False
        primary = group_artifacts.get(primary_group) or {}
        return {
            "trace_schema_version": BRANCHED_TRACE_SCHEMA_VERSION,
            "episode_schema_version": episode["schema_version"],
            "episode_id": episode["episode_id"],
            "episode_seed": episode["seed"],
            "run_success": run_success,
            "run_status": run_status,
            "fork_after_batch": int(
                episode["evaluation"]["transition"]["after_action_batch"]
            ),
            "history_policy": primary_policy,
            "history_policy_version": HISTORY_POLICY_VERSION,
            "enable_probe_skip": primary_group[1],
            "least_probe_count": primary_group[2],
            "history_policies": sorted({key[0] for key in groups}),
            "shared_prefix_hash": primary.get("shared_prefix_hash"),
            "shared_prefix": deepcopy(primary.get("shared_prefix")),
            "shared_prefix_hashes": hashes,
            "shared_prefixes": prefixes,
            "branches": branch_artifacts,
        }

    group_resumes = {
        group_key: _history_group_resume_artifact(
            resume_artifact,
            history_policy=group_key[0],
            prefix_key=group_labels[group_key],
            probe_policy=(group_key[1], group_key[2]),
            branch_names=set(group),
            primary_policy=primary_policy,
        )
        for group_key, group in groups.items()
    }
    group_artifacts.update({
        group_key: deepcopy(artifact)
        for group_key, artifact in group_resumes.items()
        if artifact is not None
    })

    async def run_group(
        group_key: tuple[str, bool, int | None],
        group_branches: dict[str, dict[str, Any]],
    ) -> None:
        async def save_group(partial: dict[str, Any]) -> None:
            async with merge_lock:
                group_artifacts[group_key] = deepcopy(partial)
                if checkpoint:
                    await checkpoint(merged_artifact())

        result = await _run_branched_agent_episode_single_policy(
            episode,
            client,
            interaction_policy=interaction_policy,
            branches=group_branches,
            checkpoint=save_group,
            adapter=adapter,
            resume_artifact=group_resumes[group_key],
            collect_relevant_fields=collect_relevant_fields,
        )
        async with merge_lock:
            group_artifacts[group_key] = deepcopy(result)
            if checkpoint:
                await checkpoint(merged_artifact())

    await asyncio.gather(
        *(run_group(group_key, group) for group_key, group in groups.items())
    )
    return merged_artifact()


def run_baseline_episode(
    episode: dict[str, Any],
    policy: str,
    *,
    interaction_policy: dict[str, Any],
    seed: int = 0,
    condition: str = "dynamic",
    sampling_branch: str | None = None,
    history_policy: str = "eval_case_given",
    enable_probe_skip: bool = True,
    least_probe_count: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    env = BenchmarkEnvironment(
        episode,
        interaction_policy=interaction_policy,
        condition=condition,
        sampling_branch=sampling_branch,
        history_policy=history_policy,
        enable_probe_skip=enable_probe_skip,
        least_probe_count=least_probe_count,
    )
    trace = []
    while not env.done:
        if env.pending_batch is None and not env.probe_used_this_window:
            probe_args = {
                "probes": [
                    {
                        "values": round_field_values(
                            next(iter(env._pool.values()))["raw_values"],
                            env.raw_fields,
                        )
                    }
                    for _ in range(env.least_probe_count or 1)
                ]
            }
            probe = env.execute("probe_batch", probe_args)
            trace.append(
                {
                    "interaction": env.interactions_used,
                    "response": {
                        "tool": "probe_batch",
                        "arguments": probe_args,
                    },
                    "result": probe,
                }
            )
        elif env.pending_batch is None:
            if not env.enable_probe_skip:
                raise RuntimeError(
                    "strict probe baseline cannot reveal a batch without probing"
                )
            skip = env.execute("request_next_batch", {})
            trace.append(
                {
                    "interaction": env.interactions_used,
                    "response": {
                        "tool": "request_next_batch",
                        "arguments": {},
                    },
                    "result": skip,
                }
            )
        assert env.pending_batch is not None
        actions = {}
        for context_id in env.pending_batch["context_ids"]:
            values = env._pool[context_id]["raw_values"]
            vector = env._raw_utility_vector(values)
            if policy == "oracle":
                action_id = max(vector, key=vector.__getitem__)
            elif policy == "always_zero":
                action_id = "action_0"
            elif policy == "random":
                action_id = rng.choice([action["action_id"] for action in env.actions])
            else:
                raise ValueError(f"unsupported baseline policy: {policy}")
            actions[context_id] = action_id
        batch_id = env.pending_batch["batch_id"]
        submitted = env.execute(
            "submit_action_batch",
            {"batch_id": batch_id, "actions": actions},
        )
        trace.append(
            {
                "interaction": env.interactions_used,
                "response": {
                    "tool": "submit_action_batch",
                    "arguments": {"batch_id": batch_id},
                },
                "result": submitted,
            }
        )
    return _artifact(
        episode, env, run_success=True, trace=trace
    )


def run_branched_baseline_episode(
    episode: dict[str, Any],
    policy: str,
    *,
    interaction_policy: dict[str, Any],
    branches: dict[str, Any],
    seed: int = 0,
) -> dict[str, Any]:
    fork_after_batch = int(
        episode["evaluation"]["transition"]["after_action_batch"]
    )
    normalized_branches = resolve_runtime_branches(branches)
    branch_artifacts: dict[str, dict[str, Any]] = {}
    for index, (run_name, spec) in enumerate(normalized_branches.items()):
        artifact = run_baseline_episode(
            episode,
            policy,
            interaction_policy=interaction_policy,
            seed=seed + index,
            condition="dynamic" if spec["rule_change"] else "static",
            sampling_branch=spec["sampling"],
            history_policy=spec["history_policy"],
            enable_probe_skip=_branch_probe_policy(spec)[0],
            least_probe_count=_branch_probe_policy(spec)[1],
        )
        artifact["branch"] = run_name
        artifact["setting_id"] = run_name
        branch_artifacts[run_name] = artifact

    groups = {
        _branch_prefix_key(spec): spec
        for spec in normalized_branches.values()
    }
    group_labels = _prefix_group_labels({key: {} for key in groups})
    prefix_hashes: dict[str, str] = {}
    for branch_name, artifact in branch_artifacts.items():
        group_key = _branch_prefix_key(normalized_branches[branch_name])
        prefix_key = group_labels[group_key]
        if prefix_key not in prefix_hashes:
            prefix_hashes[prefix_key] = _prefix_hash({
                **artifact,
                "public_history": artifact["public_history"][: 2 * fork_after_batch],
                "private_history": artifact["private_history"][: 2 * fork_after_batch],
                "tool_trace": [
                    row for row in artifact["tool_trace"]
                    if int(row.get("interaction") or 0) <= 2 * fork_after_batch
                ],
            })
        artifact.update({
            "fork_after_batch": fork_after_batch,
            "shared_prefix_hash": prefix_hashes[prefix_key],
        })
    primary_group = _branch_prefix_key(next(iter(normalized_branches.values())))
    primary_prefix_key = group_labels[primary_group]
    primary_history_policy = primary_group[0]
    return {
        "trace_schema_version": BRANCHED_TRACE_SCHEMA_VERSION,
        "episode_schema_version": episode["schema_version"],
        "episode_id": episode["episode_id"],
        "episode_seed": episode["seed"],
        "run_success": True,
        "run_status": "completed",
        "fork_after_batch": fork_after_batch,
        "history_policy": primary_history_policy,
        "history_policy_version": HISTORY_POLICY_VERSION,
        "history_policies": sorted(
            {spec["history_policy"] for spec in normalized_branches.values()}
        ),
        "shared_prefix_hash": prefix_hashes[primary_prefix_key],
        "shared_prefix_hashes": prefix_hashes,
        "branches": branch_artifacts,
    }
