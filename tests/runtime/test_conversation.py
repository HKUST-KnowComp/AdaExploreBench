from __future__ import annotations

import asyncio
from types import SimpleNamespace

from generator import GeneratorResponse
from generator.conversation_adapters import (
    DeepSeekChatAdapter,
    GeminiGenerateContentAdapter,
    get_conversation_adapter,
)
from protocol.precision import round_field_values
from runtime.conversation import ConversationSession
from runtime.runner import run_agent_episode


def _session(dynamic_episode, interaction_policy):
    from pathlib import Path

    from runtime.environment import BenchmarkEnvironment
    from runtime.runner import render_system_prompt

    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=interaction_policy,
        sampling_branch="flip_50",
    )
    task_spec = env.task_spec()
    template = Path("prompts/runtime/agent_system.txt").read_text()
    session = ConversationSession(
        system_instruction=render_system_prompt(template, task_spec),
        task_spec=task_spec,
    )
    session.append_initial_state(env.state_card())
    return env, session


def test_initial_message_exposes_numeric_types_but_hides_field_value_bounds(
    dynamic_episode, runtime_interaction_policy
):
    env, session = _session(dynamic_episode, runtime_interaction_policy)
    text = DeepSeekChatAdapter().build_request(session).messages[1]["content"]
    fields_text = text.split("Current state:")[0]
    assert "Input fields:" in fields_text
    assert "Every field below is a numeric probe input" in fields_text
    assert "required in a full `values` probe" in fields_text
    assert "`null`, strings, and booleans are invalid" in fields_text
    assert "typical" not in fields_text
    for field_id, meta in dynamic_episode["agent_task"]["fields"].items():
        assert f"`{field_id}`, type: {meta['type']}" in fields_text
        assert f"{meta['minimum']}–{meta['maximum']}" not in fields_text


def test_system_prompt_explains_the_actual_utility_scale(
    dynamic_episode, runtime_interaction_policy
):
    _, session = _session(dynamic_episode, runtime_interaction_policy)
    prompt = session.system_instruction
    assert "not normalized to `[-1, 1]`" in prompt
    assert "approximately between `-0.33` and `0.98`" in prompt
    assert "theoretical lower and upper limits" in prompt


def test_system_prompt_explains_field_relevance(
    dynamic_episode, runtime_interaction_policy
):
    _, session = _session(dynamic_episode, runtime_interaction_policy)
    prompt = session.system_instruction
    assert "Some listed fields are distractors" in prompt
    assert "Only a subset affects the hidden rule" in prompt
    assert "utility; changing the other fields cannot change utility" in prompt
    assert "relevant subset" in prompt
    assert "infer it from probe results" in prompt


def test_system_prompt_explains_rule_families(
    dynamic_episode, runtime_interaction_policy
):
    _, session = _session(dynamic_episode, runtime_interaction_policy)
    prompt = session.system_instruction
    assert "exactly one of these four families" in prompt
    assert "`additive_aggregation`: combine weighted contributions" in prompt
    assert "`conjunctive_gating`: all conditions matter" in prompt
    assert "`disjunctive_sufficiency`: any sufficient condition" in prompt
    assert "`lexicographic_priority`: inspect fields in priority order" in prompt
    assert "family name and" in prompt
    assert "field roles" in prompt
    assert "not given directly" in prompt


def test_eval_case_given_prompt_explains_field_relevance(
    dynamic_episode, runtime_interaction_policy
):
    from pathlib import Path

    from runtime.environment import BenchmarkEnvironment
    from runtime.runner import render_system_prompt

    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    template = Path("prompts/runtime/agent_system_eval_case_given.txt").read_text()
    prompt = render_system_prompt(template, env.task_spec(), history_policy="eval_case_given")
    assert "Some listed fields are distractors" in prompt
    assert "Only a subset affects the hidden rule" in prompt


def test_system_prompt_states_minimum_probe_count(
    dynamic_episode, runtime_interaction_policy
):
    from pathlib import Path

    from runtime.environment import BenchmarkEnvironment
    from runtime.runner import render_system_prompt

    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        enable_probe_skip=False,
        least_probe_count=4,
    )
    template = Path("prompts/runtime/agent_system.txt").read_text()
    prompt = render_system_prompt(template, env.task_spec())
    assert (
        "every `probe_batch` call must contain at least 4 cases and no more than "
        "16 cases"
    ) in prompt
    assert "You must submit `probe_batch` before every action batch" in prompt
    assert "request_next_batch" not in prompt


def test_gemini_model_content_round_trips_opaque_parts(
    dynamic_episode, runtime_interaction_policy
):
    env, session = _session(dynamic_episode, runtime_interaction_policy)
    continuation = {
        "parts": [
            {"thought": True, "text": "opaque thought"},
            {"text": '{"tool":"probe_batch"}'},
            {"thought_signature": "opaque-signature"},
        ]
    }
    session.append_assistant(
        '{"tool":"probe_batch","arguments":{}}',
        parsed_call={"tool": "probe_batch", "arguments": {}},
        valid=True,
        provider_continuation=continuation,
    )
    session.append_environment(
        {"results": []},
        env.state_card(),
    )

    request = GeminiGenerateContentAdapter().build_request(session)
    assert [item["role"] for item in request.contents] == ["user", "model", "user"]
    assert request.contents[1] == {"role": "model", **continuation}
    assert "opaque thought" not in request.contents[2]["parts"][0]["text"]
    assert "Input fields:" in request.contents[0]["parts"][0]["text"]


def test_deepseek_history_keeps_visible_json_but_excludes_continuation(
    dynamic_episode,
    runtime_interaction_policy,
):
    env, session = _session(dynamic_episode, runtime_interaction_policy)
    session.append_assistant(
        '{"tool":"probe_batch","arguments":{}}',
        parsed_call={"tool": "probe_batch", "arguments": {}},
        valid=True,
        provider_continuation={"reasoning": "must not be sent"},
    )
    session.append_environment({"results": []}, env.state_card())
    messages = DeepSeekChatAdapter().build_request(session).messages
    assert [item["role"] for item in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == '{"tool":"probe_batch","arguments":{}}'
    assert "must not be sent" not in str(messages)


def test_environment_wire_deduplicates_state_after_reveal_and_probe(
    dynamic_episode,
    runtime_interaction_policy,
):
    env, session = _session(dynamic_episode, runtime_interaction_policy)
    adapter = DeepSeekChatAdapter()

    session.append_assistant(
        '{"tool":"request_next_batch","arguments":{}}',
        parsed_call={"tool": "request_next_batch", "arguments": {}},
        valid=True,
    )
    result = env.execute("request_next_batch", {})
    session.append_environment(
        result,
        env.state_card(),
        stage="reveal",
        batch_id="batch_01",
        batch_number=1,
    )
    reveal_message = adapter.build_request(session).messages[-1]["content"]
    assert reveal_message.count('"pending_action_batch"') == 1
    assert reveal_message.count('"tool_usability"') == 1
    assert "Environment result:" not in reveal_message

    probe_env, probe_session = _session(dynamic_episode, runtime_interaction_policy)
    values = round_field_values(
        dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
        dynamic_episode["agent_task"]["fields"],
    )
    probe_session.append_assistant(
        '{"tool":"probe_batch","arguments":{"probes":[]}}',
        parsed_call={"tool": "probe_batch", "arguments": {"probes": []}},
        valid=True,
    )
    probe_result = probe_env.execute(
        "probe_batch", {"probes": [{"values": values}]}
    )
    probe_session.append_environment(
        probe_result,
        probe_env.state_card(),
        stage="reveal",
        batch_id="batch_01",
        batch_number=1,
    )
    probe_message = adapter.build_request(probe_session).messages[-1]["content"]
    assert probe_message.count('"pending_action_batch"') == 1
    assert probe_message.count('"tool_usability"') == 1
    assert probe_message.count('"cumulative_utility"') == 1
    assert probe_message.count('"cumulative_probes"') == 1
    assert '"action_utility"' not in probe_message
    assert '"probe_count"' not in probe_message
    assert "Environment result:" in probe_message
    assert '"results"' in probe_message


def test_given_history_prompt_keeps_cases_visible_without_hidden_history_instruction(
    dynamic_episode, runtime_interaction_policy
):
    from pathlib import Path
    from runtime.environment import BenchmarkEnvironment
    from runtime.runner import render_system_prompt

    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    prompt = Path("prompts/runtime/agent_system_eval_case_given.txt").read_text()
    rendered = render_system_prompt(prompt, env.task_spec(), history_policy="eval_case_given")
    assert "rewrites retained history" not in rendered
    assert "hidden from later agent context" not in rendered


def test_non_pending_calculator_environment_result_is_not_compacted(
    dynamic_episode, runtime_interaction_policy
):
    env, session = _session(dynamic_episode, runtime_interaction_policy)
    session.append_assistant(
        '{"tool":"calculator","arguments":{"expressions":["1 + 1"]}}',
        parsed_call={"tool": "calculator", "arguments": {"expressions": ["1 + 1"]}},
        valid=True,
    )
    result = env.execute("calculator", {"expressions": ["1 + 1"]})
    session.append_environment(result, env.state_card())
    message = DeepSeekChatAdapter().build_request(session).messages[-1]["content"]
    assert '"results"' in message
    assert '"variables"' in message
    assert '"tool_usability"' in message
    assert '"cumulative_utility"' in message
    assert '"cumulative_probes"' in message
    assert '"type"' not in message


def test_reveal_error_is_environment_result_not_probe_result(
    dynamic_episode, runtime_interaction_policy
):
    env, session = _session(dynamic_episode, runtime_interaction_policy)
    reveal = env.execute("request_next_batch", {})
    session.append_assistant(
        "not valid json",
        parsed_call=None,
        valid=False,
        stage="control",
    )
    feedback = {
        **reveal,
        "error": {
            "type": "invalid_json",
            "message": "Expecting value at line 1, column 1 (char 0)",
            "retryable": False,
        },
        "message": (
            "You have used up your repair chance. The episode skipped the probe "
            "step for this batch and advanced to the action batch."
        ),
    }
    session.append_environment(
        feedback,
        env.state_card(),
        error=feedback["error"],
        stage="reveal",
        batch_id=reveal["pending_action_batch"]["batch_id"],
        batch_number=reveal["pending_action_batch"]["batch_number"],
    )
    message = DeepSeekChatAdapter().build_request(session).messages[-1]["content"]
    assert "Environment result:" in message
    assert "Probe result:" not in message


def test_unknown_or_missing_conversation_protocol_fails_fast():
    for protocol in (None, "unknown_protocol"):
        try:
            get_conversation_adapter(protocol)
        except ValueError as exc:
            assert "conversation_protocol" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected explicit protocol validation failure")


def test_runner_persists_one_canonical_conversation(
    dynamic_episode, runtime_interaction_policy
):
    class FakeClient:
        config = SimpleNamespace(model="fake", conversation_protocol="deepseek_chat")

        def __init__(self):
            self.requests = []
            self.calls = 0
            self.batch = 0

        async def generate(self, *, request, schema, trace_context):
            del schema, trace_context
            self.requests.append(request)
            if self.calls == 0:
                self.calls += 1
                return GeneratorResponse(
                    content=(
                        '{"tool":"calculator","arguments":'
                        '{"expressions":["(12.5+3)*4","sqrt(81)"]}}'
                    ),
                    model="fake",
                    usage={},
                    raw={"fake": True},
                )
            operational_call = self.calls - 1
            phase = operational_call % 2
            self.calls += 1
            if phase == 0:
                values = round_field_values(
                    dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
                    dynamic_episode["agent_task"]["fields"],
                )
                content = (
                    '{"tool":"probe_batch","arguments":{"probes":[{"values":'
                    + __import__("json").dumps(
                            values
                    )
                    + "}]}}"
                )
            else:
                timeline = [
                    *dynamic_episode["shared_prefix"]["batches"],
                    *dynamic_episode["post_branches"]["flip_50"]["batches"],
                ]
                batch_data = timeline[self.batch]
                self.batch += 1
                actions = {context_id: "action_0" for context_id in batch_data["context_ids"]}
                content = __import__("json").dumps(
                    {
                        "tool": "submit_action_batch",
                        "arguments": {
                            "batch_id": batch_data["batch_id"],
                            "actions": actions,
                        },
                    }
                )
            return GeneratorResponse(
                content=content,
                model="fake",
                usage={},
                raw={"fake": True},
            )

    client = FakeClient()
    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            sampling_branch="flip_50",
        )
    )
    conversation = artifact["conversation"]
    assert artifact["run_success"] is True
    assert conversation["schema_version"] == "adaexplorebench_benchmark_conversation_v4"
    assert len(conversation["events"]) == 51
    assert conversation["events"][0]["type"] == "environment_turn"
    assert all(
        "evidence_history" not in str(event)
        and "probe_contexts" not in str(event)
        for event in conversation["events"]
    )
    assert client.requests[-1].messages[2]["role"] == "assistant"
    assert artifact["tool_trace"][0]["result"]["results"] == [
        {"index": 0, "ok": True, "value": 62.0},
        {"index": 1, "ok": True, "value": 9.0},
    ]
    assert "calculator_result" not in str(client.requests[1].messages)
    assert "Environment result:" in str(client.requests[1].messages)
    assert artifact["private_state"]["calculator_calls"] == 1
    assert artifact["tool_trace"][-1]["history_event_count"] == 51


def test_eval_case_hidden_requests_drop_completed_action_context_but_keep_full_log(
    dynamic_episode, runtime_interaction_policy
):
    class FakeClient:
        config = SimpleNamespace(model="fake", conversation_protocol="deepseek_chat")

        def __init__(self):
            self.requests = []

        async def generate(self, *, request, schema, trace_context):
            del schema, trace_context
            self.requests.append(request)
            latest = request.messages[-1]["content"]
            state = __import__("json").loads(latest.rsplit("Current state:\n", 1)[1])
            pending = state["pending_action_batch"]
            payload = (
                {"tool": "request_next_batch", "arguments": {}}
                if pending is None
                else {
                    "tool": "submit_action_batch",
                    "arguments": {
                        "batch_id": pending["batch_id"],
                        "actions": {
                                case["context_id"]: "action_0"
                            for case in pending["cases"]
                        },
                    },
                }
            )
            return GeneratorResponse(
                content=__import__("json").dumps(payload),
                model="fake", usage={}, raw={"fake": True},
            )

    client = FakeClient()
    artifact = asyncio.run(run_agent_episode(
        dynamic_episode,
        client,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_hidden",
    ))
    first_batch = artifact["conversation"]["events"][3]["parsed_call"]
    assert first_batch["tool"] == "submit_action_batch"
    assert first_batch["arguments"]["actions"]
    next_request = client.requests[2].messages
    visible = __import__("json").dumps(next_request)
    latest_environment_message = next_request[-1]["content"]
    assert '"actions"' not in visible
    assert '"raw_values"' not in visible
    assert "The environment accepted action batch 1." in visible
    assert "batch_utility" in visible and "cumulative_utility" in visible
    assert "cumulative_probes" in visible
    assert latest_environment_message.count("Current state:") == 1
    assert "Probe result:" not in latest_environment_message
    assert "The environment accepted action batch 1." in latest_environment_message
    assert "selected actions and action-solving messages are hidden" in latest_environment_message
    assert "After you complete an action batch, the environment rewrites retained history" in next_request[0]["content"]
    assert artifact["history_policy"] == "eval_case_hidden"
    assert artifact["history_policy_version"] == "eval_case_visibility_v1"
    assert len(artifact["action_spans"]) == 12
    assert len(artifact["request_contexts"]) == len(artifact["tool_trace"])
    assert artifact["agent_visible_history"] is not None
    assert '"action_id": "action_0"' not in __import__("json").dumps(
        artifact["agent_visible_history"]["events"]
    )
