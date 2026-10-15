from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from generator import GeneratorResponse
from runtime.runner import run_agent_episode


class FlakyProviderClient:
    config = SimpleNamespace(
        model="flaky-provider",
        conversation_protocol="deepseek_chat",
    )

    def __init__(
        self,
        *,
        failures: int,
        first_response: str | None = None,
        truncated: int = 0,
        truncated_after: int = 0,
    ) -> None:
        self.failures_remaining = failures
        self.truncated_remaining = truncated
        self.truncated_after = truncated_after
        self.first_response = first_response
        self.calls: list[dict] = []
        self.successes = 0

    async def generate(self, *, request, schema, trace_context):
        del schema
        self.calls.append(dict(trace_context))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary provider outage")

        if self.truncated_remaining and self.successes >= self.truncated_after:
            self.truncated_remaining -= 1
            return GeneratorResponse(
                content="",
                model=self.config.model,
                usage={"completion_tokens": 8192},
                raw={"fake": True},
                finish_reason="length",
            )
        self.successes += 1
        if self.successes == 1 and self.first_response is not None:
            content = self.first_response
        else:
            state_text = request.messages[-1]["content"].rsplit(
                "Current state:\n", 1
            )[1]
            state = json.loads(state_text)
            pending = state["pending_action_batch"]
            if pending is None:
                payload = {"tool": "request_next_batch", "arguments": {}}
            else:
                payload = {
                    "tool": "submit_action_batch",
                    "arguments": {
                        "batch_id": pending["batch_id"],
                        "actions": {
                            case["context_id"]: "action_0"
                            for case in pending["cases"]
                        },
                    },
                }
            content = json.dumps(payload)
        return GeneratorResponse(
            content=content,
            model=self.config.model,
            usage={},
            raw={"fake": True},
        )


def _run(dynamic_episode, runtime_interaction_policy, client):
    return asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            sampling_branch="flip_50",
        )
    )


def test_provider_errors_retry_five_times_with_capped_exponential_backoff(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("runtime.runner.asyncio.sleep", record_delay)
    client = FlakyProviderClient(failures=6)

    artifact = _run(dynamic_episode, runtime_interaction_policy, client)

    assert artifact["run_status"] == "failed"
    assert len(client.calls) == 6
    assert {call["interaction"] for call in client.calls} == {1}
    assert delays == [2.0, 4.0, 8.0, 16.0, 16.0]
    record = artifact["tool_trace"][-1]
    assert record["status"] == "provider_error"
    assert record["provider_attempt_count"] == 6
    assert [
        event["delay_seconds"] for event in record["provider_retry_events"]
    ] == delays
    assert artifact["private_state"]["interactions_used"] == 0
    assert artifact["private_state"]["invalid_calls"] == []


def test_provider_retry_does_not_consume_syntax_repair_allowance(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("runtime.runner.asyncio.sleep", record_delay)
    client = FlakyProviderClient(failures=1, first_response="not valid json")

    artifact = _run(dynamic_episode, runtime_interaction_policy, client)

    assert artifact["run_status"] == "completed"
    assert delays == [2.0]
    first, repaired = artifact["tool_trace"][:2]
    assert first["status"] == "invalid_response"
    assert first["provider_attempt_count"] == 2
    assert first["result"]["repair_attempts_remaining"] == 1
    assert first["result"]["error"]["retryable"] is True
    assert repaired["status"] == "completed"
    assert repaired["interaction"] == 2


def test_truncated_response_gets_one_free_retry(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    async def record_delay(delay: float) -> None:
        raise AssertionError("truncated retries must not back off")

    monkeypatch.setattr("runtime.runner.asyncio.sleep", record_delay)
    client = FlakyProviderClient(failures=0, truncated=1)

    artifact = _run(dynamic_episode, runtime_interaction_policy, client)

    assert artifact["run_status"] == "completed"
    first = artifact["tool_trace"][0]
    assert first["status"] == "completed"
    assert first["provider_attempt_count"] == 2
    assert len(first["provider_retry_events"]) == 1
    event = first["provider_retry_events"][0]
    assert event["error_type"] == "truncated_response"
    assert event["delay_seconds"] == 0.0
    assert artifact["private_state"]["invalid_calls"] == []


def test_double_truncation_in_probe_window_skips_probe_step(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    client = FlakyProviderClient(failures=0, truncated=2)

    artifact = _run(dynamic_episode, runtime_interaction_policy, client)

    assert artifact["run_status"] == "completed"
    first, second = artifact["tool_trace"][:2]
    assert first["status"] == "invalid_response"
    assert first["error_type"] == "response_truncated"
    assert first["provider_attempt_count"] == 2
    assert first["finish_reason"] == "length"
    assert first["result"]["error"]["retryable"] is False
    assert "skipped the probe step" in first["result"]["message"]
    invalid_calls = artifact["private_state"]["invalid_calls"]
    assert [entry["error_type"] for entry in invalid_calls] == [
        "response_truncated"
    ]
    assert second["status"] == "completed"
    assert second["response"]["tool"] == "submit_action_batch"


def test_double_truncation_in_action_window_records_unanswered(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    client = FlakyProviderClient(failures=0, truncated=2, truncated_after=1)

    artifact = _run(dynamic_episode, runtime_interaction_policy, client)

    assert artifact["run_status"] == "completed"
    first, second, third = artifact["tool_trace"][:3]
    assert first["response"]["tool"] == "request_next_batch"
    assert second["status"] == "invalid_response"
    assert second["error_type"] == "response_truncated"
    assert second["provider_attempt_count"] == 2
    assert "recorded as unanswered" in second["result"]["message"]
    skipped_results = [
        item
        for item in artifact["action_results"]
        if item["batch_id"] == first["result"]["pending_action_batch"]["batch_id"]
    ]
    assert skipped_results
    assert all(item["action_id"] is None for item in skipped_results)
    assert all(item["utility"] == 0.0 for item in skipped_results)
    assert third["status"] == "completed"


def test_double_truncation_behaves_the_same_under_static_condition(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    client = FlakyProviderClient(failures=0, truncated=2, truncated_after=1)

    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            condition="static",
            sampling_branch="static_iid",
        )
    )

    assert artifact["run_status"] == "completed"
    second = artifact["tool_trace"][1]
    assert second["status"] == "invalid_response"
    assert second["error_type"] == "response_truncated"
    assert "recorded as unanswered" in second["result"]["message"]
    invalid_calls = artifact["private_state"]["invalid_calls"]
    assert [entry["error_type"] for entry in invalid_calls] == [
        "response_truncated"
    ]
