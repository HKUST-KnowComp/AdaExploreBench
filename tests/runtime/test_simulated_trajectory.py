from __future__ import annotations

import asyncio
import json

import pytest

from scripts.simulate_trajectory import (
    ENABLED_BRANCHES,
    MATCHED_NO_CHANGE_CONTROL_BRANCHES,
    SCENARIOS,
    PROBE_POLICY_SCENARIOS,
    EVAL_CASE_GIVEN_SETTING,
    PROBE_POLICY_SETTING,
    simulate_episode,
    write_run,
)
from runtime.metrics import episode_metrics
from runtime.schema import (
    BRANCHED_TRACE_SCHEMA_VERSION,
    DYNAMIC_TRACE_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
)


EXPECTED_RUN_BRANCHES = set(ENABLED_BRANCHES) | {
    f"{name}_no_change_control"
    for name in MATCHED_NO_CHANGE_CONTROL_BRANCHES
} | {EVAL_CASE_GIVEN_SETTING}


def _wire_contains(model, message: str) -> bool:
    encoded = json.dumps(message, ensure_ascii=False)[1:-1]
    return any(
        message in request or encoded in request
        for request in model.wire_requests
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_simulated_model_responses_are_applied_by_environment(
    monkeypatch, dynamic_episode, runtime_interaction_policy, scenario
):
    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("runtime.runner.asyncio.sleep", no_sleep)
    artifact, model = asyncio.run(
        simulate_episode(
            dynamic_episode,
            scenario,
            policy=runtime_interaction_policy,
            include_no_change_control=True,
        )
    )
    assert set(artifact["branches"]) == EXPECTED_RUN_BRANCHES
    assert model.calls
    assert model.wire_requests

    if scenario == "provider_error":
        for branch in artifact["branches"].values():
            statuses = [record.get("status") for record in branch["tool_trace"]]
            if branch["condition"] == "dynamic":
                assert branch["run_status"] == "failed"
                assert branch["termination"] is None
                assert len(branch["action_results"]) == 48
                assert "provider_error" in statuses
            else:
                assert branch["run_status"] == "completed"
                assert branch["termination"]["status"] == "completed"
                assert len(branch["action_results"]) == 96
                assert "provider_error" not in statuses
        return

    for branch in artifact["branches"].values():
        statuses = [record.get("status") for record in branch["tool_trace"]]
        if scenario == "action_repair_exhausted":
            assert branch["run_status"] == "completed"
            assert branch["run_success"] is True
            assert branch["termination"]["status"] == "completed"
            assert branch["termination"]["reason"] == "invalid_action_batch_after_repair"
            assert branch["termination"]["completed_count"] == 88
            assert branch["termination"]["unanswered_count"] == 8
            assert episode_metrics(branch)["completion_rate"] == pytest.approx(88 / 96)
        else:
            assert branch["run_status"] == "completed"
            assert branch["run_success"] is True
            assert branch["termination"]["status"] == "completed"

        error_scenarios = {
            "invalid_json",
            "schema_error",
            "calculator_trajectory",
            "combined_tool_schema",
            "combined_environment_rejection",
            "boundary_probe",
            "multiple_boundary_probe",
            "wrong_batch",
            "invalid_action",
            "repeated_environment_rejection",
            "action_repair_exhausted",
}


        if scenario in error_scenarios:
            assert "invalid_response" in statuses
            assert branch["private_state"]["invalid_calls"]
        if scenario in {"boundary_probe", "multiple_boundary_probe"}:
            errors = [record.get("error", "") for record in branch["tool_trace"]]
            assert any("exceeds its allowed maximum" in error for error in errors)
        if scenario == "multiple_boundary_probe":
            errors = [record.get("error", "") for record in branch["tool_trace"]]
            assert any("is below its allowed minimum" in error for error in errors)
        if scenario in {"valid_probe", "boundary_probe", "multiple_boundary_probe"}:
            batch_count = dynamic_episode["agent_task"]["interaction_protocol"][
                "action_batch_count"
            ]
            probe_cases_per_batch = 2 if scenario == "multiple_boundary_probe" else 1
            probe_events = [
                event
                for event in branch["public_history"]
                if event.get("type") == "probe_batch"
            ]
            metrics = episode_metrics(branch)
            assert len(probe_events) == batch_count
            assert all(
                event["probe_cases_used"] == probe_cases_per_batch
                for event in probe_events
            )
            assert metrics["probe_count"] == batch_count * probe_cases_per_batch
            assert metrics["action_utility"] == metrics["cumulative_utility"]
            if scenario == "valid_probe":
                for record in branch["tool_trace"]:
                    if (record.get("response") or {}).get("tool") != "probe_batch":
                        continue
                    for probe in (record["response"].get("arguments") or {}).get(
                        "probes", []
                    ):
                        values = probe.get("values") or {}
                        for field_id, value in values.items():
                            if (
                                dynamic_episode["agent_task"]["fields"][field_id]["type"]
                                == "decimal"
                            ):
                                assert value == round(float(value), 2)
        if scenario == "partial_actions":
            assert any(row["abstained"] for row in branch["action_results"])
        assert len(branch["action_results"]) == (
            dynamic_episode["agent_task"]["interaction_protocol"]["action_batch_count"]
            * dynamic_episode["agent_task"]["interaction_protocol"]["action_batch_size"]
        )

        if branch["history_policy"] == "eval_case_hidden":
            retained = branch["agent_visible_history"]
            retained_events = retained["events"]
            retained_environment_events = [
                event
                for event in retained_events
                if event.get("type") == "environment_turn"
            ]
            retained_results = [
                event.get("result") or {} for event in retained_environment_events
            ]
            outcome_messages = [
                (event.get("result") or {}).get("message", "")
                for event in retained_environment_events
                if event.get("stage") == "completion"
            ]
            assert outcome_messages
            assert all("hidden from later agent context" in message for message in outcome_messages)
            if scenario == "valid":
                assert all(
                    item["disposition"] != "rewritten_reveal"
                    for item in retained["event_mapping"]
                )
                assert sum(
                    item["disposition"] == "removed_empty_reveal"
                    for item in retained["event_mapping"]
                ) == dynamic_episode["agent_task"]["interaction_protocol"][
                    "action_batch_count"
                ]
                post_completion_message = branch["tool_trace"][2][
                    "request_context"
                ]["wire_request"]["messages"][-1]["content"]
                assert post_completion_message.count("Current state:") == 1
                assert "Probe result:" not in post_completion_message
            if scenario == "valid_probe":
                probe_history = [
                    event.get("result") or {}
                    for event in retained_environment_events
                    if event.get("stage") == "reveal"
                ]
                assert len(probe_history) == dynamic_episode["agent_task"][
                    "interaction_protocol"
                ]["action_batch_count"]
                assert all(
                    "revealed action cases and transient state are hidden"
                    in result["message"]
                    for result in probe_history
                )
                post_completion_message = branch["tool_trace"][2][
                    "request_context"
                ]["wire_request"]["messages"][-1]["content"]
                assert post_completion_message.count("Current state:") == 1
                assert "Probe result:" in post_completion_message
            if scenario == "schema_error":
                assert any(
                    event.get("stage") == "reveal"
                    and "used up your repair chance"
                    in (event.get("result") or {}).get("message", "")
                    and "hidden from later agent context"
                    in (event.get("result") or {}).get("message", "")
                    for event in retained_environment_events
                )
            if scenario == "action_repair_exhausted":
                assert any(
                    "No actions were accepted; every case was recorded as unanswered"
                    in message
                    for message in outcome_messages
                )

        invalid_records = [
            record
            for record in branch["tool_trace"]
            if record.get("status") == "invalid_response"
        ]
        result_records = [
            record
            for record in branch["tool_trace"]
            if record.get("status") in {"completed", "invalid_response"}
        ]
        for record in result_records:
            assert record["result"]["tool_usability"]
        visible_error_turns = [
            event
            for event in branch["conversation"]["events"]
            if event.get("type") == "environment_turn" and event.get("error")
        ]
        assert len(visible_error_turns) == len(invalid_records)
        result_turns = [
            event
            for event in branch["conversation"]["events"]
            if event.get("type") == "environment_turn" and event.get("result")
        ]
        assert len(result_turns) == len(result_records)
        for event in result_turns:
            assert event["result"]["tool_usability"] == event["state_card"][
                "tool_usability"
            ]
        if scenario == "combined_tool_schema":
            assert invalid_records
            for record, turn in zip(invalid_records, visible_error_turns, strict=True):
                assert record["error_type"] == "invalid_tool_schema"
                assert record["error"].splitlines() == [
                    "1. missing top-level key(s): arguments, tool",
                    "2. unexpected top-level key(s): unexpected",
                ]
                assert record["result"]["error"]["message"] == record["error"]
                assert set(turn["error"]) == {"type", "message", "retryable"}
                assert turn["error"]["message"] == record["error"]
                assert _wire_contains(model, record["error"])
        if scenario == "combined_environment_rejection":
            assert len(invalid_records) == 1
            record = invalid_records[0]
            assert record["error_type"] == "environment_rejection"
            assert record["error"].splitlines() == [
                "submit_action_batch arguments has invalid keys:",
                "1. missing key(s): actions",
                "2. unexpected key(s): unexpected",
            ]
            assert record["result"]["error"]["message"] == record["error"]
            assert set(visible_error_turns[0]["error"]) == {
                "type",
                "message",
                "retryable",
            }
            assert visible_error_turns[0]["error"]["message"] == record["error"]
            assert _wire_contains(model, record["error"])
        if scenario == "repeated_environment_rejection":
            batch_count = dynamic_episode["agent_task"]["interaction_protocol"][
                "action_batch_count"
            ]
            assert len(invalid_records) == batch_count
            for record in invalid_records:
                assert record["error_type"] == "environment_rejection"
                assert record["result"]["error"]["retryable"] is True
                assert record["result"]["repair_attempts_remaining"] == 1
                assert "message" not in record["result"]
                trace_index = branch["tool_trace"].index(record)
                repaired = branch["tool_trace"][trace_index + 1]
                assert repaired["status"] == "completed"
                assert repaired["response"]["tool"] == "submit_action_batch"
                assert repaired["response"]["arguments"]["batch_id"] == record[
                    "result"
                ]["required_batch_id"]
        if scenario == "action_repair_exhausted":
            assert len(invalid_records) == 3
            exhausted = [
                record
                for record in invalid_records
                if record["result"]["error"]["retryable"] is False
            ]
            assert len(exhausted) == 1
            assert "skipped the current action batch" in exhausted[0]["result"]["message"]
            assert "recorded as unanswered" in exhausted[0]["result"]["message"]
            assert any(
                turn["result"].get("message") == exhausted[0]["result"]["message"]
                for turn in visible_error_turns
            )
            repaired_later = invalid_records[-1]["result"]
            assert repaired_later["error"]["retryable"] is True
            assert repaired_later["repair_attempts_remaining"] == 1
            failed_rows = [
                row
                for row in branch["action_results"]
                if row.get("unanswered") and not row.get("abstained")
            ]
            assert len(failed_rows) == 8
        if scenario == "calculator_trajectory":
            calculator_records = [
                record
                for record in branch["tool_trace"]
                if record.get("status") == "completed"
                and (record.get("response") or {}).get("tool") == "calculator"
            ]
            calculator_rejections = [
                record
                for record in invalid_records
                if (record.get("response") or {}).get("tool") == "calculator"
            ]
            assert len(calculator_records) == 4
            assert len(calculator_rejections) == 3
            assert branch["private_state"]["calculator_calls"] == 4
            assert branch["private_state"]["calculator_expressions"] == 16
            assert branch["private_state"]["calculator_expression_errors"] == 1

            if branch["history_policy"] == "eval_case_hidden":
                events = branch["conversation"]["events"]
                mapping = {
                    item["source_event_id"]: item["disposition"]
                    for item in branch["agent_visible_history"]["event_mapping"]
                }
                for record in calculator_records:
                    matching = [
                        event
                        for event in events
                        if event.get("interaction") == record["interaction"]
                    ]
                    assistant = next(
                        event for event in matching if event["type"] == "assistant_turn"
                    )
                    environment = next(
                        event for event in matching if event["type"] == "environment_turn"
                    )
                    if assistant["stage"] == "action_solving":
                        assert mapping[assistant["event_id"]] == "removed_action_solving"
                        assert mapping[environment["event_id"]] == "removed_action_solving"
                    else:
                        assert mapping[assistant["event_id"]] == "retained"
                        assert mapping[environment["event_id"]] == "retained"

            initial, pending, forward, post = calculator_records
            assert [row["value"] for row in initial["result"]["results"]] == [
                5,
                20,
                15,
                16,
            ]
            assert initial["result"]["variables"] == {"X": 5, "Y": 20, "Z": 15}
            assert [row["value"] for row in pending["result"]["results"]] == [
                6,
                5,
                6,
                6,
            ]
            assert pending["result"]["results"][1]["variable"] == "X"
            assert pending["result"]["variables"] == {"X": 5}
            assert [row["value"] for row in forward["result"]["results"]] == [
                5,
                4,
                8,
            ]
            assert forward["result"]["variables"] == {"later": 4}
            assert post["result"]["results"][0]["ok"] is False
            assert "unknown variable: X" in post["result"]["results"][0][
                "error"
            ]["message"]
            assert [row["value"] for row in post["result"]["results"][1:]] == [
                6,
                36,
                12.0,
                21.0,
            ]
            assert post["result"]["variables"] == {"A": 6, "B": 36}
            for record in calculator_rejections:
                assert "calculator was already used" in record["error"]
                assert record["result"]["error"]["retryable"] is True
                assert record["result"]["repair_attempts_remaining"] == 1
                assert record["result"]["tool_usability"]
                trace_index = branch["tool_trace"].index(record)
                repaired = branch["tool_trace"][trace_index + 1]
                assert repaired["status"] == "completed"
                assert repaired["response"]["tool"] == "request_next_batch"
        if scenario == "boundary_probe":
            batch_count = dynamic_episode["agent_task"]["interaction_protocol"][
                "action_batch_count"
            ]
            assert len(invalid_records) == batch_count
            for record in invalid_records:
                assert record["result"]["error"]["retryable"] is True
                assert record["result"]["repair_attempts_remaining"] == 1
                trace_index = branch["tool_trace"].index(record)
                repaired = branch["tool_trace"][trace_index + 1]
                assert repaired["status"] == "completed"
                assert repaired["response"]["tool"] == "probe_batch"
        if scenario == "multiple_boundary_probe":
            batch_count = dynamic_episode["agent_task"]["interaction_protocol"][
                "action_batch_count"
            ]
            assert len(invalid_records) == batch_count
            for record in invalid_records:
                lines = record["error"].splitlines()
                assert lines[0] == "probe_batch rejected for multiple reasons:"
                assert len(lines) == 5
                assert lines[1].startswith("1. probe 1: ")
                assert "exceeds its allowed maximum" in lines[1]
                assert lines[2].startswith("2. probe 1: ")
                assert "is below its allowed minimum" in lines[2]
                assert lines[3].startswith("3. probe 2: ")
                assert "exceeds its allowed maximum" in lines[3]
                assert lines[4].startswith("4. probe 2: ")
                assert "is below its allowed minimum" in lines[4]
                assert record["result"]["error"]["message"] == record["error"]
                assert record["result"]["error"]["retryable"] is True
                assert record["result"]["repair_attempts_remaining"] == 1
                assert _wire_contains(model, record["error"])
                trace_index = branch["tool_trace"].index(record)
                repaired = branch["tool_trace"][trace_index + 1]
                assert repaired["status"] == "completed"
                assert repaired["response"]["tool"] == "probe_batch"
                assert len(repaired["result"]["results"]) == 2


@pytest.mark.parametrize("scenario", PROBE_POLICY_SCENARIOS)
def test_simulated_probe_policy_checks(
    dynamic_episode, runtime_interaction_policy, scenario
):
    artifact, model = asyncio.run(
        simulate_episode(
            dynamic_episode,
            scenario,
            policy=runtime_interaction_policy,
            include_no_change_control=True,
            include_eval_case_given=True,
        )
    )
    assert artifact["run_status"] == "completed"
    assert set(artifact["branches"]) == EXPECTED_RUN_BRANCHES | {
        PROBE_POLICY_SETTING
    }
    for name, branch in artifact["branches"].items():
        if name != PROBE_POLICY_SETTING:
            assert branch["enable_probe_skip"] is True
            assert branch["least_probe_count"] is None
            assert "request_next_batch" in branch["conversation"]["system_instruction"]
            assert "request_next_batch" in branch["tool_trace"][0]["request_schema"][
                "properties"
            ]["tool"]["enum"]
            continue
        assert branch["enable_probe_skip"] is False
        assert branch["least_probe_count"] == 4
        assert "request_next_batch" not in branch["conversation"]["system_instruction"]
        assert "request_next_batch" not in branch["tool_trace"][0]["request_schema"][
            "properties"
        ]["tool"]["enum"]
        invalid = [
            record
            for record in branch["tool_trace"]
            if record.get("status") == "invalid_response"
        ]
        assert invalid
        if scenario == "probe_skip_disabled":
            assert invalid[0]["error"] == "tool does not exist: request_next_batch"
        else:
            assert invalid[0]["error"].startswith(
                "You must submit at least 4 probe cases"
            )
            probe_events = [
                event
                for event in branch["public_history"]
                if event.get("type") == "probe_batch"
            ]
            assert probe_events
            assert all(event["probe_cases_used"] == 4 for event in probe_events)
    assert model.calls


def test_simulated_branch_keeps_release_ids_private_and_joinable(
    dynamic_episode, runtime_interaction_policy
):
    artifact, model = asyncio.run(
        simulate_episode(
            dynamic_episode,
            "valid",
            policy=runtime_interaction_policy,
            include_no_change_control=True,
        )
    )
    assert set(artifact["branches"]) == EXPECTED_RUN_BRANCHES
    for branch in artifact["branches"].values():
        sampling_branch = branch["sampling_branch"]
        release_ids = {
            item["context_id"]
            for item in dynamic_episode["shared_prefix"]["cases"]
        } | {
            item["context_id"]
            for item in dynamic_episode["post_branches"][sampling_branch]["cases"]
        }

        # Runtime ids are the only ids emitted through model-facing data.
        public_ids = {
            result["context_id"]
            for event in branch["public_history"]
            if event.get("type") == "submit_action_batch"
            for result in event.get("results") or []
        }
        assert public_ids.isdisjoint(release_ids)
        assert public_ids == {f"case_{index:03d}" for index in range(1, 97)}
        model_visible = json.dumps(
            {
                "public_history": branch["public_history"],
                "conversation": branch["conversation"],
                "tool_trace": branch["tool_trace"],
            }
        )
        assert all(release_id not in model_visible for release_id in release_ids)
        assert all(
            release_id not in request
            for release_id in release_ids
            for request in model.wire_requests
        )

        # Private artifact rows retain release ids so publication can join scores.
        private_ids = {row["context_id"] for row in branch["action_results"]}
        assert private_ids & release_ids
        assert all(
            row.get("runtime_context_id", "").startswith("case_")
            for row in branch["action_results"]
        )


def test_simulation_run_has_api_artifact_and_manifest_shape(
    tmp_path, dynamic_episode, runtime_interaction_policy
):
    artifact, _ = asyncio.run(
        simulate_episode(
            dynamic_episode,
            "wrong_batch",
            policy=runtime_interaction_policy,
            include_no_change_control=True,
        )
    )
    dataset = tmp_path / "episodes.jsonl"
    dataset.write_text(json.dumps(dynamic_episode) + "\n")
    run_dir = write_run(
        tmp_path / "simulated_wrong_batch",
        dataset,
        dynamic_episode,
        "wrong_batch",
        artifact,
        runtime_interaction_policy,
    )
    saved = json.loads((run_dir / f"{dynamic_episode['episode_id']}.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert saved["trace_schema_version"] == BRANCHED_TRACE_SCHEMA_VERSION
    assert saved["episode_schema_version"] == dynamic_episode["schema_version"]
    assert {
        branch["trace_schema_version"]
        for branch in saved["branches"].values()
    } == {DYNAMIC_TRACE_SCHEMA_VERSION}
    assert set(saved) >= {"shared_prefix", "branches", "run_status", "run_success"}
    assert manifest["manifest_schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["run_identity"]["episode_schema_versions"] == [
        dynamic_episode["schema_version"]
    ]
    assert manifest["run_identity"]["dataset_sha256"] == manifest["dataset"]["sha256"]
    assert manifest["evaluation_mode"] == "branched"
    assert manifest["simulation_scenario"] == "wrong_batch"
    assert set(manifest["branches"]) == EXPECTED_RUN_BRANCHES
    assert manifest["branches"]["flip_50_no_change_control"] == {
        "sampling": "flip_50",
        "rule_change": False,
        "history_policy": "eval_case_hidden",
    }
    assert manifest["branches"][EVAL_CASE_GIVEN_SETTING]["history_policy"] == (
        "eval_case_given"
    )
    assert manifest["history"]["policies"] == [
        "eval_case_given",
        "eval_case_hidden",
    ]


def test_simulation_emits_five_hidden_settings_and_independent_given_setting(
    dynamic_episode, runtime_interaction_policy
):
    artifact, _ = asyncio.run(simulate_episode(
        dynamic_episode,
        "valid",
        policy=runtime_interaction_policy,
    ))
    outcome = artifact["branches"]["flip_50"]
    given = artifact["branches"][EVAL_CASE_GIVEN_SETTING]
    assert outcome["history_policy"] == "eval_case_hidden"
    assert given["history_policy"] == "eval_case_given"
    assert outcome["agent_visible_history"] is not None
    assert given["agent_visible_history"] is None
    visible = json.dumps(outcome["agent_visible_history"]["events"])
    assert '"raw_values"' not in visible
    assert '"actions"' not in visible
    assert "The environment accepted action batch" in visible
    assert given["conversation"]["events"][2]["stage"] == "reveal"
