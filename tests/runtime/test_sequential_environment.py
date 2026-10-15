from __future__ import annotations

import json
from copy import deepcopy

import pytest

from runtime.environment import BenchmarkEnvironment
from protocol.mechanism import (
    hydrate_rule,
    intervention_score,
    utilities_from_score,
)
from protocol.precision import round_field_values


def test_dynamic_episode_survives_sorted_json_round_trip(
    dynamic_episode, runtime_interaction_policy
):
    serialized = json.dumps(dynamic_episode, sort_keys=True)
    restored = json.loads(serialized)
    BenchmarkEnvironment(
        restored,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )


def test_dynamic_episode_loads_and_completes_one_batch(
    dynamic_episode, runtime_interaction_policy
):
    assert set(dynamic_episode) == {
        "schema_version",
        "dataset_mode",
            "episode_id",
        "seed",
        "provenance",
        "agent_task",
        "evaluation",
        "shared_prefix",
        "post_branches",
    }
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    task = env.task_spec()
    assert len(task["raw_field_schema"]) == 8
    assert "evaluation" not in task
    env.execute("request_next_batch", {})
    result = env.execute(
        "submit_action_batch",
        {
            "batch_id": env.pending_batch["batch_id"],
            "actions": {
                context_id: "action_0"
                for context_id in env.pending_batch["context_ids"]
            },
        },
    )
    assert len(result["results"]) == 8
    assert result["tool_usability"] == env.state_card()["tool_usability"]


def test_model_facing_fields_use_seeded_shuffled_order(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    dataset_order = list(dynamic_episode["agent_task"]["fields"])
    schema_order = list(env.task_spec()["raw_field_schema"])
    assert sorted(schema_order) == sorted(dataset_order)
    assert schema_order != dataset_order
    replay = BenchmarkEnvironment(
        deepcopy(dynamic_episode),
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    assert list(replay.task_spec()["raw_field_schema"]) == schema_order
    env.execute("request_next_batch", {})
    case = env.state_card()["pending_action_batch"]["cases"][0]
    assert list(case["raw_values"]) == schema_order


def test_model_facing_decimal_values_use_two_decimal_places(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    env.execute("request_next_batch", {})
    visible = env.state_card()["pending_action_batch"]["cases"][0]["raw_values"]
    source = env._pool[env.pending_batch["context_ids"][0]]["raw_values"]
    fields = dynamic_episode["agent_task"]["fields"]
    for field_id, meta in fields.items():
        if meta["type"] == "decimal":
            assert visible[field_id] == round(float(source[field_id]), 2)


def test_range_rejection_reveals_only_the_violated_bound(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    field_id = next(iter(dynamic_episode["agent_task"]["fields"]))
    meta = dynamic_episode["agent_task"]["fields"][field_id]
    base = dict(dynamic_episode["shared_prefix"]["cases"][0]["raw_values"])

    too_high = {**base, field_id: meta["maximum"] + 1}
    with pytest.raises(ValueError) as high_error:
        env.execute("probe_batch", {"probes": [{"values": too_high}]})
    assert f"max: {meta['maximum']}" in str(high_error.value)
    assert "min:" not in str(high_error.value)

    too_low = {**base, field_id: meta["minimum"] - 1}
    with pytest.raises(ValueError) as low_error:
        env.execute("probe_batch", {"probes": [{"values": too_low}]})
    assert f"min: {meta['minimum']}" in str(low_error.value)
    assert "max:" not in str(low_error.value)


def test_probe_range_validation_matches_rule_normalization_bounds(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    for field_id in env.rule["relevant_fields"]:
        field = env.raw_fields[field_id]
        rule_range = env.rule["field_ranges"][field_id]
        assert rule_range == {
            "minimum": field["minimum"],
            "maximum": field["maximum"],
        }


def test_integer_probe_field_rejects_decimal_json_number(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    field_id = next(iter(dynamic_episode["agent_task"]["fields"]))
    env.episode["agent_task"]["fields"][field_id]["type"] = "integer"
    values = dict(dynamic_episode["shared_prefix"]["cases"][0]["raw_values"])
    values[field_id] = float(round(values[field_id]))

    with pytest.raises(
        ValueError,
        match=rf"field {field_id} must be a JSON integer",
    ):
        env.execute("probe_batch", {"probes": [{"values": values}]})


def test_decimal_probe_field_rejects_more_than_two_decimal_places(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    field_id = next(
        field_id
        for field_id, meta in env.raw_fields.items()
        if meta["type"] == "decimal"
    )
    meta = env.raw_fields[field_id]
    values = dict(env._pool[env.episode["batches"][0]["context_ids"][0]]["raw_values"])
    candidate = float(values[field_id]) + 0.001
    if candidate > float(meta["maximum"]):
        candidate = float(meta["maximum"]) - 0.001
    values[field_id] = candidate

    with pytest.raises(
        ValueError,
        match=rf"field {field_id} must use at most two decimal places",
    ):
        env.execute("probe_batch", {"probes": [{"values": values}]})


def test_probe_rejection_reports_all_invalid_fields(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    field_ids = list(dynamic_episode["agent_task"]["fields"])[:2]
    base = dict(dynamic_episode["shared_prefix"]["cases"][0]["raw_values"])
    for field_id in field_ids:
        maximum = dynamic_episode["agent_task"]["fields"][field_id]["maximum"]
        base[field_id] = maximum + 1

    with pytest.raises(ValueError) as error:
        env.execute("probe_batch", {"probes": [{"values": base}]})

    message = str(error.value)
    assert all(field_id in message for field_id in field_ids)
    assert message.count("exceeds its allowed maximum") == len(field_ids)


def test_probe_rejection_reports_errors_from_all_probes(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    field_ids = list(dynamic_episode["agent_task"]["fields"])[:2]
    base = dict(dynamic_episode["shared_prefix"]["cases"][0]["raw_values"])
    first_max = dynamic_episode["agent_task"]["fields"][field_ids[0]]["maximum"]
    second_min = dynamic_episode["agent_task"]["fields"][field_ids[1]]["minimum"]
    first = {**base, field_ids[0]: first_max + 1}
    second = {**base, field_ids[1]: second_min - 1}

    with pytest.raises(ValueError) as error:
        env.execute(
            "probe_batch",
            {"probes": [{"values": first}, {"values": second}]},
        )

    message = str(error.value)
    assert field_ids[0] in message and "exceeds its allowed maximum" in message
    assert field_ids[1] in message and "is below its allowed minimum" in message


def test_probe_schema_rejection_reports_all_missing_fields(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    field_ids = list(dynamic_episode["agent_task"]["fields"])[:2]
    values = dict(dynamic_episode["shared_prefix"]["cases"][0]["raw_values"])
    for field_id in field_ids:
        values.pop(field_id)

    with pytest.raises(ValueError) as error:
        env.execute("probe_batch", {"probes": [{"values": values}]})

    message = str(error.value)
    assert "missing key(s):" in message
    assert all(field_id in message for field_id in field_ids)


def test_new_context_probe_null_value_suggests_existing_context_mode(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    field_id = list(dynamic_episode["agent_task"]["fields"])[0]
    values = dict(dynamic_episode["shared_prefix"]["cases"][0]["raw_values"])
    values[field_id] = None

    with pytest.raises(ValueError) as error:
        env.execute("probe_batch", {"probes": [{"values": values}]})

    message = str(error.value)
    assert f"field {field_id} must be numeric; received None" in message
    assert "existing-context probe" in message
    assert "changes" in message


def test_repeated_invalid_action_batch_fails_unanswered_at_zero_utility(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    env.execute("request_next_batch", {})
    batch_id = env.pending_batch["batch_id"]
    feedbacks = []
    for _ in range(2):
        with pytest.raises(ValueError):
            env.execute(
                "submit_action_batch",
                {"batch_id": batch_id, "actions": {"case_999": "action_0"}},
            )
        feedbacks.append(
            env.record_invalid_response(
                "unexpected key",
                "environment_rejection",
                tool="submit_action_batch",
                interaction_already_counted=True,
            )
        )
    assert feedbacks[0]["repair_attempts_remaining"] == 1
    assert feedbacks[0]["error"]["retryable"] is True
    assert "message" not in feedbacks[0]
    feedback = feedbacks[1]
    assert feedback["error"]["retryable"] is False
    assert "repair_attempts_remaining" not in feedback
    assert "used up your repair chance" in feedback["message"]
    assert "skipped the current action batch" in feedback["message"]
    assert "every action in that batch was recorded as unanswered" in feedback["message"]
    assert feedback["tool_usability"] == env.state_card()["tool_usability"]
    assert env.next_batch_index == 1
    rows = [row for row in env.action_results if row["batch_id"] == batch_id]
    assert len(rows) == env.action_batch_size
    for row in rows:
        assert row["unanswered"] is True
        assert row["correct"] is False
        assert row["utility"] == 0.0
        oracle = row["oracle"]
        assert row["regret"] == round(
            oracle["utilities"][oracle["optimal_action_id"]], 2
        )


def test_repair_allowance_resets_for_each_batch_window(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )

    first = env.record_invalid_response("invalid JSON", "invalid_json")
    assert first["repair_attempts_remaining"] == 1
    assert first["error"]["retryable"] is True

    # A valid interaction repairs the response and resets the allowance.
    env.execute("request_next_batch", {})
    with pytest.raises(ValueError):
        env.execute(
            "submit_action_batch",
            {
                "batch_id": env.pending_batch["batch_id"],
                "actions": {"case_999": "action_0"},
            },
        )
    next_window = env.record_invalid_response(
        "unexpected key",
        "environment_rejection",
        tool="submit_action_batch",
        interaction_already_counted=True,
    )
    assert next_window["repair_attempts_remaining"] == 1
    assert "message" not in next_window


def test_repeated_invalid_probe_window_explains_action_batch_advance(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )

    env.record_invalid_response("invalid JSON", "invalid_json")
    feedback = env.record_invalid_response("invalid JSON", "invalid_json")

    assert feedback["error"]["retryable"] is False
    assert "repair_attempts_remaining" not in feedback
    assert "used up your repair chance" in feedback["message"]
    assert "skipped the probe step for this batch" in feedback["message"]
    assert "advanced to the action batch" in feedback["message"]
    assert feedback["required_next_tool"] == "submit_action_batch"
    assert feedback["required_batch_id"] == env.pending_batch["batch_id"]


def test_repeated_invalid_required_probe_waives_probe_without_unanswering_actions(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        enable_probe_skip=False,
        least_probe_count=3,
    )

    env.record_invalid_response("invalid JSON", "invalid_json")
    feedback = env.record_invalid_response("invalid JSON", "invalid_json")

    assert feedback["error"]["retryable"] is False
    assert "probe requirement is waived" in feedback["message"]
    assert feedback["required_next_tool"] == "submit_action_batch"
    assert env.pending_batch is not None
    assert env.next_batch_index == 0
    result = env.execute(
        "submit_action_batch",
        {
            "batch_id": env.pending_batch["batch_id"],
            "actions": {
                context_id: "action_0"
                for context_id in env.pending_batch["context_ids"]
            },
        },
    )
    assert len(result["results"]) == env.action_batch_size
    assert all(not row["unanswered"] for row in env.action_results)
    assert env.next_batch_index == 1
    assert not env.done


def test_dynamic_episode_supports_probe_from_release_cases(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    values = round_field_values(
        dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
        dynamic_episode["agent_task"]["fields"],
    )
    result = env.execute(
        "probe_batch",
        {"probes": [{"values": values}]},
    )
    assert len(result["results"]) == 1
    assert set(result["results"][0]["utilities"]) == {
        "action_0",
        "action_1",
        "action_2",
        "action_3",
    }


def _submit_pending_batch(env):
    env.execute("request_next_batch", {})
    return env.execute(
        "submit_action_batch",
        {
            "batch_id": env.pending_batch["batch_id"],
            "actions": {
                context_id: "action_0"
                for context_id in env.pending_batch["context_ids"]
            },
        },
    )


def test_eval_case_given_action_feedback_exposes_full_utility_vector(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    result = _submit_pending_batch(env)
    assert len(result["results"]) == 8
    for row in result["results"]:
        assert set(row["utilities"]) == {
            "action_0",
            "action_1",
            "action_2",
            "action_3",
        }
        assert row["utilities"][row["action_id"]] == row["utility"]


def test_eval_case_hidden_action_feedback_omits_full_utility_vector(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_hidden",
    )
    result = _submit_pending_batch(env)
    assert len(result["results"]) == 8
    for row in result["results"]:
        assert "utilities" not in row


def test_eval_case_hidden_rejects_completed_action_probe_base_without_incrementing_demand(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_hidden",
    )
    env.execute("request_next_batch", {})
    context_id = env.pending_batch["context_ids"][0]
    env.execute("submit_action_batch", {
        "batch_id": env.pending_batch["batch_id"],
        "actions": {},
    })
    before = env.probe_count
    with pytest.raises(ValueError, match="forbids using a completed action"):
        env.execute("probe_batch", {
            "probes": [{"base_context_id": context_id, "changes": {}}]
        })
    assert env.probe_count == before


def test_eval_case_given_allows_completed_action_probe_base(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    env.execute("request_next_batch", {})
    context_id = env.pending_batch["context_ids"][0]
    env.execute("submit_action_batch", {
        "batch_id": env.pending_batch["batch_id"],
        "actions": {},
    })
    result = env.execute("probe_batch", {
        "probes": [{"base_context_id": context_id, "changes": {}}]
    })
    assert result["results"][0]["base_context_id"] == context_id


def test_eval_case_given_feedback_includes_all_action_utilities(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    env.execute("request_next_batch", {})
    result = env.execute("submit_action_batch", {
        "batch_id": env.pending_batch["batch_id"],
        "actions": {
            context_id: "action_1"
            for context_id in env.pending_batch["context_ids"]
        },
    })
    for row in result["results"]:
        assert set(row["utilities"]) == {
            "action_0",
            "action_1",
            "action_2",
            "action_3",
        }
        assert row["utility"] == row["utilities"][row["action_id"]]


def test_eval_case_hidden_feedback_omits_all_action_utilities(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_hidden",
    )
    env.execute("request_next_batch", {})
    result = env.execute("submit_action_batch", {
        "batch_id": env.pending_batch["batch_id"],
        "actions": {
            context_id: "action_1"
            for context_id in env.pending_batch["context_ids"]
        },
    })
    for row in result["results"]:
        assert "utilities" not in row


def test_eval_case_given_failed_batch_reports_all_action_utilities(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    env.execute("request_next_batch", {})
    env.record_invalid_response("first", tool="submit_action_batch")
    feedback = env.record_invalid_response("second", tool="submit_action_batch")
    for row in feedback["results"]:
        assert row["action_id"] is None
        assert row["utility"] == 0.0
        assert set(row["utilities"]) == {
            "action_0",
            "action_1",
            "action_2",
            "action_3",
        }


def test_dynamic_episode_can_run_as_static_control(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        condition="static",
        sampling_branch="flip_50",
    )
    assert env.active_rule_phase == "pre_dynamic"
    env.execute("request_next_batch", {})
    assert env.pending_batch["batch_number"] == 1


def test_hidden_history_hides_completed_cases_but_keeps_context_ids(
    dynamic_episode, runtime_interaction_policy
):
    hidden = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_hidden",
    )
    hidden.execute("request_next_batch", {})
    hidden_case = hidden.state_card()["pending_action_batch"]["cases"][0]
    assert hidden_case["context_id"].startswith("case_")
    hidden_result = hidden.execute(
        "submit_action_batch",
        {
            "batch_id": hidden.pending_batch["batch_id"],
            "actions": {hidden_case["context_id"]: "action_0"},
        },
    )
    assert "context_id" in hidden_result["results"][0]
    assert "action_case_id" not in hidden_result["results"][0]

    given = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    given.execute("request_next_batch", {})
    given_case = given.state_card()["pending_action_batch"]["cases"][0]
    assert given_case["context_id"].startswith("case_")


def test_new_probe_uses_one_context_id_and_incremental_probe_links_base(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        history_policy="eval_case_given",
    )
    values = round_field_values(
        dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
        dynamic_episode["agent_task"]["fields"],
    )
    first = env.execute(
        "probe_batch",
        {"probes": [{"values": values}]},
    )["results"][0]
    assert set(first) == {"context_id", "probe_mode", "utilities"}
    second = env.execute(
        "submit_action_batch",
        {"batch_id": env.pending_batch["batch_id"], "actions": {}},
    )
    del second
    incremental = env.execute(
        "probe_batch",
        {"probes": [{"base_context_id": first["context_id"], "changes": {}}]},
    )["results"][0]
    assert set(incremental) == {"context_id", "probe_mode", "utilities", "base_context_id"}


def test_probe_increments_cumulative_probes_without_changing_cumulative_utility(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    values = round_field_values(
        dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
        dynamic_episode["agent_task"]["fields"],
    )
    assert env.action_utility == 0.0
    assert env.probe_count == 0
    result = env.execute(
        "probe_batch",
        {"probes": [{"values": values}]},
    )
    assert "probe_utility_cost" not in result
    assert "net_utility" not in result
    assert "action_utility" not in result
    assert "probe_count" not in result
    assert result["cumulative_utility"] == 0.0
    assert result["cumulative_probes"] == 1
    assert env.action_utility == result["cumulative_utility"]
    assert env.probe_count == result["cumulative_probes"]
    card = env.state_card()
    assert "available_tools" not in card
    assert card["cumulative_utility"] == env.action_utility
    assert card["cumulative_probes"] == env.probe_count


def test_maximum_probe_batch_only_changes_cumulative_probes(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    maximum = runtime_interaction_policy["max_probe_cases_per_batch"]
    values = round_field_values(
        dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
        dynamic_episode["agent_task"]["fields"],
    )
    result = env.execute(
        "probe_batch", {"probes": [{"values": values} for _ in range(maximum)]}
    )
    assert result["cumulative_probes"] == maximum
    assert result["cumulative_utility"] == 0.0
    assert env.probe_count == maximum
    assert env.action_utility == 0.0


def test_state_card_reports_tool_usability_with_reasons(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )

    def usability():
        return {
            entry["tool"]: entry for entry in env.state_card()["tool_usability"]
        }

    tools = usability()
    assert tools["probe_batch"] == {"tool": "probe_batch", "available": True, "reason": None}
    assert tools["request_next_batch"]["available"] is True
    assert tools["calculator"]["available"] is True
    assert tools["submit_action_batch"]["available"] is False
    assert "no action batch is pending" in tools["submit_action_batch"]["reason"]
    env.execute("request_next_batch", {})
    tools = usability()
    assert tools["probe_batch"]["available"] is False
    assert "pending" in tools["probe_batch"]["reason"]
    assert tools["request_next_batch"]["available"] is False
    assert tools["submit_action_batch"]["available"] is True
    env.execute(
        "submit_action_batch",
        {
            "batch_id": env.pending_batch["batch_id"],
            "actions": {
                context_id: "action_0"
                for context_id in env.pending_batch["context_ids"]
            },
        },
    )
    tools = usability()
    assert tools["probe_batch"]["available"] is True
    assert tools["request_next_batch"]["available"] is True


def test_probe_skip_policy_removes_batch_skip_tool_and_probe_reveals_batch(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        enable_probe_skip=False,
    )

    assert "request_next_batch" not in {
        item["tool"] for item in env.state_card()["tool_usability"]
    }
    assert "request_next_batch" not in env.task_spec()["interaction_protocol"]

    with pytest.raises(ValueError, match="tool does not exist"):
        env.execute("request_next_batch", {})

    values = round_field_values(
        dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
        dynamic_episode["agent_task"]["fields"],
    )
    result = env.execute("probe_batch", {"probes": [{"values": values}]})
    assert result["pending_action_batch"]["batch_number"] == 1


def test_least_probe_count_rejects_short_batch_and_accepts_at_least_count(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode,
        interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
        enable_probe_skip=False,
        least_probe_count=8,
    )
    values = round_field_values(
        dynamic_episode["shared_prefix"]["cases"][0]["raw_values"],
        dynamic_episode["agent_task"]["fields"],
    )
    with pytest.raises(
        ValueError, match="You must submit at least 8 probe cases; received 4"
    ):
        env.execute(
            "probe_batch",
            {"probes": [{"values": values} for _ in range(4)]},
        )
    assert env.pending_batch is None
    result = env.execute(
        "probe_batch",
        {"probes": [{"values": values} for _ in range(10)]},
    )
    assert result["cumulative_probes"] == 10
    assert result["pending_action_batch"]["batch_number"] == 1


def test_least_probe_count_above_probe_limit_is_invalid_config(
    dynamic_episode, runtime_interaction_policy
):
    with pytest.raises(
        ValueError,
        match=(
            "invalid config: least_probe_count cannot exceed "
            "max_probe_cases_per_batch"
        ),
    ):
        BenchmarkEnvironment(
            dynamic_episode,
            interaction_policy=runtime_interaction_policy,
            sampling_branch="flip_50",
            least_probe_count=(
                runtime_interaction_policy["max_probe_cases_per_batch"] + 1
            ),
        )


def test_action_batch_supports_explicit_abstention(
    dynamic_episode, runtime_interaction_policy
):
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    env.execute("request_next_batch", {})
    result = env.execute(
        "submit_action_batch",
        {"batch_id": env.pending_batch["batch_id"], "actions": {}},
    )
    assert "abstained_count" not in result
    assert result["batch_utility"] == 0.0
    assert result["cumulative_utility"] == 0.0
    assert result["cumulative_probes"] == 0
    assert all(
        row["unanswered"] and row["abstained"]
        for row in env.action_results[-8:]
    )


def test_abstention_disabled_requires_complete_actions(
    dynamic_episode, runtime_interaction_policy
):
    policy = deepcopy(runtime_interaction_policy)
    policy["allow_abstention"] = False
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=policy, sampling_branch="flip_50"
    )
    env.execute("request_next_batch", {})
    context_id = env.pending_batch["context_ids"][0]
    with pytest.raises(ValueError, match="missing key"):
        env.execute(
            "submit_action_batch",
            {
                "batch_id": env.pending_batch["batch_id"],
                "actions": {context_id: "action_0"},
            },
        )


def test_action_3_is_accepted_scored_and_reported(
    dynamic_episode, runtime_interaction_policy
):
    transition = dynamic_episode["evaluation"]["transition"]
    fields = dynamic_episode["agent_task"]["fields"]
    rules = {
        phase: hydrate_rule(transition[f"{phase.split('_')[0]}_rule"], fields)
        for phase in ("pre_dynamic", "post_dynamic")
    }
    # Results carry runtime-anonymized ids (``case_001`` …) assigned in
    # timeline order; key the expected values the same way.
    cases = {
        f"case_{index:03d}": round_field_values(case["raw_values"], fields)
        for index, case in enumerate(
            [
                *dynamic_episode["shared_prefix"]["cases"],
                *dynamic_episode["post_branches"]["flip_50"]["cases"],
            ],
            start=1,
        )
    }
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    while not env.done:
        env.execute("request_next_batch", {})
        phase = env.active_rule_phase
        result = env.execute(
            "submit_action_batch",
            {
                "batch_id": env.pending_batch["batch_id"],
                "actions": {
                    context_id: "action_3"
                    for context_id in env.pending_batch["context_ids"]
                },
            },
        )
        for row in result["results"]:
            assert row["action_id"] == "action_3"
            expected = round(
                utilities_from_score(
                    intervention_score(rules[phase], cases[row["context_id"]])
                )[3],
                2,
            )
            assert row["utility"] == expected
    assert len(env.action_results) == 96
    for row in env.action_results:
        assert row["action_id"] == "action_3"
        assert set(row["oracle"]["utilities"]) == {
            "action_0",
            "action_1",
            "action_2",
            "action_3",
        }
        assert row["correct"] == (row["oracle"]["optimal_action_id"] == "action_3")
