from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from generator import GeneratorResponse
from runtime.runner import (
    _parse_relevant_fields_response,
    assessment_is_terminal,
    run_agent_episode,
    run_branched_agent_episode,
)


def assessment_response(fields):
    return {
        "relevant_fields": fields,
        "field_directions": [
            {"field_id": field, "direction": "positive" if index % 2 == 0 else "negative"}
            for index, field in enumerate(fields)
        ],
    }


class AssessmentClient:
    config = SimpleNamespace(model="fake", conversation_protocol="deepseek_chat")

    def __init__(self, episode):
        self.episode = episode
        self.calls: list[dict] = []

    async def generate(self, *, request, schema, trace_context):
        self.calls.append(deepcopy(trace_context))
        if trace_context.get("analysis") == "relevant_fields":
            assert schema["required"] == ["relevant_fields", "field_directions"]
            prompt = request.messages[-1]["content"]
            assert "action_0 toward action_3" in prompt
            assert "action_3 toward action_0" in prompt
            assert "not semantic good/bad or the sign of utility" in prompt
            phase = trace_context["analysis_phase"]
            fields = list(self.episode["evaluation"]["transition"][
                "pre_rule"
                if phase == "pre_change" or trace_context.get("condition") == "static"
                else "post_rule"
            ]["relevant_fields"])
            return GeneratorResponse(
                content=json.dumps(assessment_response(fields)),
                model="fake",
                usage={},
                raw={"fake": True},
            )

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
        return GeneratorResponse(
            content=json.dumps(payload),
            model="fake",
            usage={},
            raw={"fake": True},
        )


def test_dynamic_assessments_are_out_of_band_and_phase_specific(
    dynamic_episode, runtime_interaction_policy
):
    client = AssessmentClient(dynamic_episode)
    artifact = asyncio.run(
        run_branched_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            branches={
                "static_iid": {
                    "sampling": "static_iid",
                    "rule_change": False,
                    "history_policy": "eval_case_hidden",
                },
                "flip_50": {
                    "sampling": "flip_50",
                    "rule_change": True,
                    "history_policy": "eval_case_hidden",
                },
            },
            collect_relevant_fields=True,
        )
    )

    dynamic = artifact["branches"]["flip_50"]
    static = artifact["branches"]["static_iid"]
    assert set(dynamic["relevant_fields_assessments"]) == {
        "pre_change",
        "post_change",
    }
    assert set(static["relevant_fields_assessments"]) == {"final"}
    assert dynamic["relevant_fields_assessments"]["pre_change"]["response"][
        "relevant_fields"
    ] == dynamic_episode["evaluation"]["transition"]["pre_rule"]["relevant_fields"]
    assert dynamic["relevant_fields_assessments"]["post_change"]["response"][
        "relevant_fields"
    ] == dynamic_episode["evaluation"]["transition"]["post_rule"]["relevant_fields"]
    assert (
        dynamic["relevant_fields_assessments"]["pre_change"][
            "trajectory_interaction_count"
        ]
        < dynamic["relevant_fields_assessments"]["post_change"][
            "trajectory_interaction_count"
        ]
    )
    assert static["relevant_fields_assessments"]["final"]["response"][
        "relevant_fields"
    ] == dynamic_episode["evaluation"]["transition"]["pre_rule"]["relevant_fields"]

    # The diagnostic requests are not benchmark interactions and do not create
    # assistant/environment events in the persisted canonical history.
    for branch in (dynamic, static):
        assert len(branch["tool_trace"]) == len(branch["conversation"]["events"]) // 2
        assert all(
            event.get("stage") != "relevant_fields"
            for event in branch["conversation"]["events"]
        )
    diagnostics = [call for call in client.calls if call.get("analysis")]
    assert len(diagnostics) == 3
    for branch in (dynamic, static):
        for record in branch["relevant_fields_assessments"].values():
            assert record["response"] == assessment_response(record["response"]["relevant_fields"])
    assert {call["analysis_phase"] for call in diagnostics} == {
        "pre_change",
        "post_change",
        "final",
    }
    assert all(call["interaction"] > 0 for call in diagnostics)


def test_malformed_assessment_gets_one_repair_round(
    dynamic_episode, runtime_interaction_policy
):
    class RepairingClient(AssessmentClient):
        async def generate(self, *, request, schema, trace_context):
            if trace_context.get("analysis") == "relevant_fields":
                self.calls.append(deepcopy(trace_context))
                repair_calls = [
                    call
                    for call in self.calls
                    if call.get("analysis") == "relevant_fields"
                ]
                if len(repair_calls) == 1:
                    # First diagnostic call answers with a benchmark tool call
                    # instead of the assessment JSON.
                    return GeneratorResponse(
                        content=json.dumps(
                            {"tool": "request_next_batch", "arguments": {}}
                        ),
                        model="fake",
                        usage={},
                        raw={"fake": True},
                    )
                assert trace_context.get("repair_attempt") == 1
                assert "was invalid" in request.messages[-1]["content"]
                assert (
                    request.messages[-2]["content"]
                    == '{"tool": "request_next_batch", "arguments": {}}'
                )
                return GeneratorResponse(
                    content=json.dumps(assessment_response(["field_0"])),
                    model="fake",
                    usage={},
                    raw={"fake": True},
                )
            return await super().generate(
                request=request, schema=schema, trace_context=trace_context
            )

    client = RepairingClient(dynamic_episode)
    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            condition="static",
            sampling_branch="flip_50",
            collect_relevant_fields=True,
        )
    )
    final = artifact["relevant_fields_assessments"]["final"]
    assert final["status"] == "completed"
    assert final["response"] == assessment_response(["field_0"])
    assert final["repairs"][0]["error_type"] == "ValueError"
    assert "request_next_batch" in final["repairs"][0]["raw_response"]


def test_assessment_syntax_exhaustion_is_terminal(
    dynamic_episode, runtime_interaction_policy
):
    class AlwaysInvalidClient(AssessmentClient):
        async def generate(self, *, request, schema, trace_context):
            if trace_context.get("analysis") == "relevant_fields":
                self.calls.append(deepcopy(trace_context))
                return GeneratorResponse(
                    content=json.dumps({"tool": "request_next_batch", "arguments": {}}),
                    model="fake",
                    usage={},
                    raw={"fake": True},
                )
            return await super().generate(
                request=request, schema=schema, trace_context=trace_context
            )

    client = AlwaysInvalidClient(dynamic_episode)
    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            condition="static",
            sampling_branch="flip_50",
            collect_relevant_fields=True,
        )
    )
    final = artifact["relevant_fields_assessments"]["final"]
    assert final["status"] == "syntax_exhausted"
    assert final["error_type"] == "ValueError"
    assert len(final["repairs"]) == 2
    diagnostics = [call for call in client.calls if call.get("analysis")]
    assert len(diagnostics) == 2
    assert diagnostics[1].get("repair_attempt") == 1
    # The exhausted diagnostic does not fail the episode itself.
    assert artifact["run_status"] == "completed"


def test_assessment_provider_errors_follow_retry_policy(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("runtime.runner.asyncio.sleep", record_delay)

    class FlakyAssessmentClient(AssessmentClient):
        def __init__(self, episode, failures: int) -> None:
            super().__init__(episode)
            self.failures_remaining = failures

        async def generate(self, *, request, schema, trace_context):
            if trace_context.get("analysis") == "relevant_fields":
                self.calls.append(deepcopy(trace_context))
                if self.failures_remaining:
                    self.failures_remaining -= 1
                    raise RuntimeError("temporary provider outage")
                return GeneratorResponse(
                    content=json.dumps(assessment_response(["field_0"])),
                    model="fake",
                    usage={},
                    raw={"fake": True},
                )
            return await super().generate(
                request=request, schema=schema, trace_context=trace_context
            )

    client = FlakyAssessmentClient(dynamic_episode, failures=2)
    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            condition="static",
            sampling_branch="flip_50",
            collect_relevant_fields=True,
        )
    )
    final = artifact["relevant_fields_assessments"]["final"]
    assert final["status"] == "completed"
    assert final["response"] == assessment_response(["field_0"])
    assert delays == [2.0, 4.0]
    assert [
        event["delay_seconds"] for event in final["provider_retry_events"]
    ] == [2.0, 4.0]

    failing = FlakyAssessmentClient(dynamic_episode, failures=99)
    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            failing,
            interaction_policy=runtime_interaction_policy,
            condition="static",
            sampling_branch="flip_50",
            collect_relevant_fields=True,
        )
    )
    final = artifact["relevant_fields_assessments"]["final"]
    assert final["status"] == "provider_error"
    assert final["error_type"] == "RuntimeError"
    diagnostics = [call for call in failing.calls if call.get("analysis")]
    assert len(diagnostics) == 6
    assert artifact["run_status"] == "completed"


def test_assessment_provider_error_is_recollected_on_resume(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("runtime.runner.asyncio.sleep", no_sleep)

    class ProviderDownClient(AssessmentClient):
        async def generate(self, *, request, schema, trace_context):
            if trace_context.get("analysis") == "relevant_fields":
                self.calls.append(deepcopy(trace_context))
                raise RuntimeError("temporary provider outage")
            return await super().generate(
                request=request, schema=schema, trace_context=trace_context
            )

    branches = {
        "static_iid": {
            "sampling": "static_iid",
            "rule_change": False,
            "history_policy": "eval_case_hidden",
        },
        "flip_50": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
        },
    }
    partial = asyncio.run(
        run_branched_agent_episode(
            dynamic_episode,
            ProviderDownClient(dynamic_episode),
            interaction_policy=runtime_interaction_policy,
            branches=branches,
            collect_relevant_fields=True,
        )
    )
    assert partial["run_status"] == "completed"
    assert {
        rec["status"]
        for branch in partial["branches"].values()
        for rec in branch["relevant_fields_assessments"].values()
    } == {"provider_error"}

    resumed_client = AssessmentClient(dynamic_episode)
    completed = asyncio.run(
        run_branched_agent_episode(
            dynamic_episode,
            resumed_client,
            interaction_policy=runtime_interaction_policy,
            branches=branches,
            collect_relevant_fields=True,
            resume_artifact=partial,
        )
    )
    # The completed trajectories are reused; only the failed diagnostics run.
    assert resumed_client.calls
    assert all(call.get("analysis") for call in resumed_client.calls)
    assert {
        rec["status"]
        for branch in completed["branches"].values()
        for rec in branch["relevant_fields_assessments"].values()
    } == {"completed"}


def test_non_rule_change_episode_gets_only_a_final_assessment(
    dynamic_episode, runtime_interaction_policy
):
    client = AssessmentClient(dynamic_episode)
    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            condition="static",
            sampling_branch="flip_50",
            collect_relevant_fields=True,
        )
    )
    assert set(artifact["relevant_fields_assessments"]) == {"final"}
    assert artifact["relevant_fields_assessments"]["final"]["response"] == assessment_response(
        dynamic_episode["evaluation"]["transition"][
            "pre_rule"
        ]["relevant_fields"]
    )
    assert [call["analysis_phase"] for call in client.calls if call.get("analysis")] == [
        "final"
    ]


def test_direct_dynamic_episode_forks_before_the_rule_change(
    dynamic_episode, runtime_interaction_policy
):
    client = AssessmentClient(dynamic_episode)
    artifact = asyncio.run(
        run_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=runtime_interaction_policy,
            condition="dynamic",
            sampling_branch="flip_50",
            collect_relevant_fields=True,
        )
    )
    assert set(artifact["relevant_fields_assessments"]) == {
        "pre_change",
        "post_change",
    }
    assert [call["analysis_phase"] for call in client.calls if call.get("analysis")] == [
        "pre_change",
        "post_change",
    ]


@pytest.mark.parametrize("missing", ["all", "legacy", "pre_change", "post_change", "none"])
def test_resume_refreshes_only_incomplete_assessments(
    dynamic_episode, runtime_interaction_policy, missing, monkeypatch
):
    from scripts.run_benchmark import _artifact_covers_requested_branches

    branches = {
        "flip_50": {
            "sampling": "flip_50", "rule_change": True,
            "history_policy": "eval_case_hidden",
        },
        "static_iid": {
            "sampling": "static_iid", "rule_change": False,
            "history_policy": "eval_case_hidden",
        },
    }
    kwargs = dict(
        interaction_policy=runtime_interaction_policy,
        branches=branches, collect_relevant_fields=True,
    )
    original = asyncio.run(run_branched_agent_episode(
        dynamic_episode, AssessmentClient(dynamic_episode), **kwargs
    ))
    saved = deepcopy(original)
    expected_calls = []
    for name, branch in saved["branches"].items():
        records = branch["relevant_fields_assessments"]
        for phase in list(records):
            if missing in ("all", "legacy", phase):
                expected_calls.append((name, phase))
                if missing == "legacy":
                    records[phase].pop("assessment_version")
                    records[phase]["response"].pop("field_directions")
                else:
                    del records[phase]
        if missing == "all":
            del branch["relevant_fields_assessments"]
    assert _artifact_covers_requested_branches(
        saved, set(branches), dataset_mode="dynamic", branch_specs=branches,
        collect_relevant_fields=True,
    ) == (missing == "none")

    def forbid_branch_restore(*args, **kwargs):
        raise AssertionError("completed branch must not replay transitions")

    monkeypatch.setattr("runtime.runner._restore_branch_trajectory", forbid_branch_restore)
    monkeypatch.setattr("runtime.runner._restore_prefix_trajectory", forbid_branch_restore)
    client = AssessmentClient(dynamic_episode)
    result = asyncio.run(run_branched_agent_episode(
        dynamic_episode, client, resume_artifact=saved, **kwargs
    ))
    assert all(call.get("analysis") == "relevant_fields" for call in client.calls)
    assert sorted((call["setting_id"], call["analysis_phase"]) for call in client.calls) == sorted(expected_calls)
    for name, branch in result["branches"].items():
        for key, value in original["branches"][name].items():
            if key != "relevant_fields_assessments":
                assert branch[key] == value, key
        for phase, record in branch["relevant_fields_assessments"].items():
            assert assessment_is_terminal(record)
            assert record["trajectory_event_count"] == original["branches"][name]["relevant_fields_assessments"][phase]["trajectory_event_count"]
    client.calls.clear()
    assert asyncio.run(run_branched_agent_episode(
        dynamic_episode, client, resume_artifact=result, **kwargs
    )) == result
    assert not client.calls


@pytest.mark.parametrize("missing", ["all", "legacy", "none"])
def test_completed_static_resume_preserves_trajectory(
    dynamic_episode, runtime_interaction_policy, missing, monkeypatch
):
    kwargs = dict(
        interaction_policy=runtime_interaction_policy, condition="static",
        sampling_branch="flip_50", collect_relevant_fields=True,
    )
    saved = asyncio.run(run_agent_episode(
        dynamic_episode, AssessmentClient(dynamic_episode), **kwargs
    ))
    if missing == "all":
        del saved["relevant_fields_assessments"]
    elif missing == "legacy":
        saved["relevant_fields_assessments"]["final"]["response"].pop("field_directions")

    async def forbid_trajectory(*args, **kwargs):
        raise AssertionError("completed trajectory must not run")

    monkeypatch.setattr("runtime.runner._drive_agent_episode", forbid_trajectory)
    client = AssessmentClient(dynamic_episode)
    result = asyncio.run(run_agent_episode(
        dynamic_episode, client, resume_artifact=saved, **kwargs
    ))
    assert len(client.calls) == (0 if missing == "none" else 1)
    assert all(call.get("analysis") == "relevant_fields" for call in client.calls)
    for key, value in saved.items():
        if key != "relevant_fields_assessments":
            assert result[key] == value


@pytest.mark.parametrize("directions", [
    None, {}, [],
    [{"field_id": "a", "direction": "up"}],
    [{"field_id": "b", "direction": "positive"}],
    [{"field_id": "a", "direction": "positive"}] * 2,
    [{"field_id": "a", "direction": "positive"}, {"field_id": "b", "direction": "negative"}],
])
def test_missing_or_invalid_directions_are_incomplete(directions):
    response = {"relevant_fields": ["a"], "field_directions": directions}
    with pytest.raises(ValueError):
        _parse_relevant_fields_response(json.dumps(response))
    assert not assessment_is_terminal({"status": "completed", "response": response})


def test_empty_field_set_and_versioned_syntax_exhaustion():
    response = assessment_response([])
    assert _parse_relevant_fields_response(json.dumps(response)) == response
    assert assessment_is_terminal({"status": "completed", "response": response})
    assert not assessment_is_terminal({"status": "syntax_exhausted"})
    assert assessment_is_terminal({"status": "syntax_exhausted", "assessment_version": 2})
