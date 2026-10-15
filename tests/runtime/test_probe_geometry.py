from __future__ import annotations

import pytest

from runtime.metrics import (
    aggregate_branched_metrics,
    aggregate_metrics,
    episode_metrics,
)
from runtime.probe_geometry import (
    compute_geometry,
    materialize_probe_points,
    probe_phase_point_sets,
)
from runtime.runner import run_baseline_episode


def _fields() -> dict:
    fields = {
        f"f{index}": {"type": "decimal", "minimum": 0.0, "maximum": 100.0}
        for index in range(1, 8)
    }
    fields["f8"] = {"type": "integer", "minimum": 0, "maximum": 10}
    return fields


def _values(updates: dict | None = None) -> dict:
    values = {f"f{index}": 0.0 for index in range(1, 8)}
    values["f8"] = 0
    values.update(updates or {})
    return values


def _threshold_rule(
    family: str, relevant: tuple[str, ...], threshold: float
) -> dict:
    return {
        "family": family,
        "relevant_fields": list(relevant),
        "field_directions": {field: 1 for field in relevant},
        "thresholds": {field: threshold for field in relevant},
    }


def _episode(
    *,
    rule_family: str = "conjunctive_gating",
    pre_rule: dict | None = None,
    post_rule: dict | None = None,
    after_action_batch: int = 2,
    shared_cases: list | None = None,
    branch_cases: list | None = None,
) -> dict:
    return {
        "episode_id": "geometry-fixture",
        "agent_task": {"fields": _fields()},
        "evaluation": {
            "rule_family": rule_family,
            "transition": {
                "after_action_batch": after_action_batch,
                "pre_rule": pre_rule
                if pre_rule is not None
                else _threshold_rule("conjunctive_gating", ("f1", "f2", "f3"), 50.0),
                "post_rule": post_rule
                if post_rule is not None
                else _threshold_rule("conjunctive_gating", ("f1", "f2", "f3"), 50.0),
            },
        },
        "shared_prefix": {"cases": shared_cases or []},
        "post_branches": {"flip_50": {"cases": branch_cases or []}},
    }


def _results(*utilities: dict, context_ids: list[str] | None = None) -> list:
    return [
        {
            "context_id": (
                context_ids[index]
                if context_ids
                else f"probe_{index + 1:03d}"
            ),
            "utilities": item,
        }
        for index, item in enumerate(utilities)
    ]


def _probe_event(
    probes: list,
    results: list,
    *,
    interaction: int = 1,
    valid: bool = True,
) -> dict:
    return {
        "interaction": interaction,
        "valid": valid,
        "response": {"tool": "probe_batch", "arguments": {"probes": probes}},
        "result": {"results": results},
    }


def _artifact(
    *events: dict,
    condition: str = "dynamic",
    after_action_batch: int = 2,
    private_windows: list[tuple[int, int]] | None = None,
) -> dict:
    return {
        "episode_id": "geometry-fixture",
        "condition": condition,
        "evaluation": {"transition": {"after_action_batch": after_action_batch}},
        "tool_trace": list(events),
        "private_history": [
            {"type": "probe_batch", "interaction": interaction, "batch_number": window}
            for interaction, window in (private_windows or [])
        ],
    }


def _geometry(artifact: dict, episode: dict) -> dict:
    condition = str(artifact.get("condition") or "dynamic")
    change_after_batch = artifact["evaluation"]["transition"]["after_action_batch"]
    points, unresolved = materialize_probe_points(artifact, episode)
    block = compute_geometry(
        points,
        episode,
        condition=condition,
        change_after_batch=change_after_batch,
    )
    block["unresolved_probes"] = unresolved
    return block


def test_materialize_normalizes_min_max_midpoint_and_integer_fields():
    episode = _episode()
    artifact = _artifact(
        _probe_event(
            [{"values": _values()}, {"values": _values({**{f"f{i}": 100.0 for i in range(1, 8)}, "f8": 10})}, {"values": _values({**{f"f{i}": 50.0 for i in range(1, 8)}, "f8": 5})}],
            _results({}, {}, {}),
        ),
        private_windows=[(1, 0)],
    )
    points, unresolved = materialize_probe_points(artifact, episode)
    assert unresolved == 0
    assert [point["sequence"] for point in points] == [0, 1, 2]
    assert all(point["window"] == 0 for point in points)
    assert points[0]["normalized_values"] == {field: 0.0 for field in _fields()}
    assert points[1]["normalized_values"] == {
        **{f"f{i}": 1.0 for i in range(1, 8)},
        "f8": 1.0,
    }
    midpoint = points[2]["normalized_values"]
    assert midpoint["f1"] == pytest.approx(0.5)
    assert midpoint["f8"] == pytest.approx(0.5)


def test_materialize_resolves_action_case_base_and_probe_chain():
    episode = _episode(
        shared_cases=[{"context_id": "case_1", "raw_values": _values({"f1": 100.0})}],
        branch_cases=[{"context_id": "case_2", "raw_values": _values({"f2": 100.0})}],
    )
    artifact = _artifact(
        _probe_event(
            [{"values": _values()}],
            _results({}, context_ids=["probe_001"]),
            interaction=1,
        ),
        _probe_event(
            [{"base_context_id": "case_1", "changes": {"f2": 100.0}}],
            _results({}, context_ids=["probe_002"]),
            interaction=2,
        ),
        _probe_event(
            [{"base_context_id": "case_2", "changes": {"f3": 50.0}}],
            _results({}, context_ids=["probe_003"]),
            interaction=3,
        ),
        _probe_event(
            [{"base_context_id": "probe_001", "changes": {"f3": 100.0}}],
            _results({}, context_ids=["probe_004"]),
            interaction=4,
        ),
    )
    points, unresolved = materialize_probe_points(artifact, episode)
    assert unresolved == 0
    assert points[1]["normalized_values"]["f1"] == pytest.approx(1.0)
    assert points[1]["normalized_values"]["f2"] == pytest.approx(1.0)
    assert points[2]["normalized_values"]["f2"] == pytest.approx(1.0)
    assert points[2]["normalized_values"]["f3"] == pytest.approx(0.5)
    assert points[3]["normalized_values"]["f3"] == pytest.approx(1.0)
    assert points[3]["normalized_values"]["f1"] == pytest.approx(0.0)


def test_materialize_resolves_runtime_case_aliases_positionally():
    episode = _episode(
        shared_cases=[
            {"context_id": "pre_case_01", "raw_values": _values({"f1": 100.0})},
            {"context_id": "pre_case_02", "raw_values": _values({"f2": 100.0})},
        ],
        branch_cases=[
            {"context_id": "flip_50_case_01", "raw_values": _values({"f3": 100.0})},
        ],
    )
    artifact = _artifact(
        _probe_event(
            [
                {"base_context_id": "case_002", "changes": {}},
                {"base_context_id": "case_003", "changes": {"f4": 50.0}},
            ],
            _results({}, {}),
        ),
    )
    artifact["sampling_branch"] = "flip_50"
    points, unresolved = materialize_probe_points(artifact, episode)
    assert unresolved == 0
    assert points[0]["normalized_values"]["f2"] == pytest.approx(1.0)
    assert points[0]["normalized_values"]["f1"] == pytest.approx(0.0)
    assert points[1]["normalized_values"]["f3"] == pytest.approx(1.0)
    assert points[1]["normalized_values"]["f4"] == pytest.approx(0.5)


def test_materialize_runtime_alias_resolves_prefix_without_branch():
    episode = _episode(
        shared_cases=[
            {"context_id": "pre_case_01", "raw_values": _values({"f1": 100.0})},
        ],
        branch_cases=[
            {"context_id": "flip_50_case_01", "raw_values": _values({"f3": 100.0})},
        ],
    )
    artifact = _artifact(
        _probe_event(
            [
                {"base_context_id": "case_001", "changes": {}},
                {"base_context_id": "case_002", "changes": {}},
            ],
            _results({}, {}),
        ),
    )
    points, unresolved = materialize_probe_points(artifact, episode)
    assert unresolved == 1
    assert points[0]["normalized_values"]["f1"] == pytest.approx(1.0)


def test_materialize_counts_unknown_base_and_skips_invalid_events():
    episode = _episode()
    artifact = _artifact(
        _probe_event(
            [{"base_context_id": "missing", "changes": {"f1": 1.0}}],
            _results({}),
        ),
        _probe_event([{"values": _values()}], _results({}), valid=False),
        _probe_event(
            [{"values": _values()}],
            [],
        ),
    )
    artifact["tool_trace"][2]["result"] = {"error": "environment_rejection"}
    points, unresolved = materialize_probe_points(artifact, episode)
    assert points == []
    assert unresolved == 1


def test_total_variance_extremes_and_empty():
    episode = _episode()
    low_high = _artifact(
        _probe_event(
            [{"values": _values()}, {"values": _values({**{f"f{i}": 100.0 for i in range(1, 8)}, "f8": 10})}],
            _results({}, {}),
        )
    )
    block = _geometry(low_high, episode)
    assert block["spread"]["total_variance"] == pytest.approx(4.0)
    assert block["probe_points"] == 2

    single = _artifact(_probe_event([{"values": _values()}], _results({})))
    block = _geometry(single, episode)
    assert block["spread"]["total_variance"] is None

    empty = _artifact()
    block = _geometry(empty, episode)
    assert block["probe_points"] == 0
    assert block["spread"]["total_variance"] is None
    assert block["pre"]["incoming_field_share"] is None
    assert block["pre"]["outgoing_field_share"] is None
    assert block["post"]["focus_variance_share"] is None


def test_focus_variance_share_directional_mass():
    episode = _episode()
    relevant_only = _artifact(
        _probe_event(
            [
                {"values": _values({"f1": 100.0, "f2": 100.0, "f3": 100.0})},
                {"values": _values()},
            ],
            _results({}, {}),
        )
    )
    block = _geometry(relevant_only, episode)
    assert block["pre"]["focus_variance_share"] == pytest.approx(1.0)

    distractor_only = _artifact(
        _probe_event(
            [
                {"values": _values({"f4": 100.0, "f5": 100.0, "f6": 100.0, "f7": 100.0, "f8": 10})},
                {"values": _values()},
            ],
            _results({}, {}),
        )
    )
    block = _geometry(distractor_only, episode)
    assert block["pre"]["focus_variance_share"] == pytest.approx(0.0)

    isotropic = _artifact(
        _probe_event(
            [
                {"values": _values({**{f"f{i}": 100.0 for i in range(1, 8)}, "f8": 10})},
                {"values": _values()},
            ],
            _results({}, {}),
        )
    )
    block = _geometry(isotropic, episode)
    assert block["pre"]["focus_variance_share"] == pytest.approx(3 / 8)


def test_windows_and_phase_rule_selection_dynamic_and_static():
    episode = _episode(
        pre_rule=_threshold_rule("conjunctive_gating", ("f1", "f2", "f3"), 50.0),
        post_rule=_threshold_rule("conjunctive_gating", ("f4", "f5", "f6"), 50.0),
    )
    pre_event = _probe_event(
        [
            {"values": _values({"f1": 100.0, "f2": 100.0, "f3": 100.0})},
            {"values": _values()},
        ],
        _results({}, {}),
        interaction=1,
    )
    post_event = _probe_event(
        [
            {"values": _values({"f4": 100.0, "f5": 100.0, "f6": 100.0})},
            {"values": _values()},
        ],
        _results({}, {}),
        interaction=2,
    )
    dynamic = _artifact(
        pre_event,
        post_event,
        condition="dynamic",
        private_windows=[(1, 1), (2, 2)],
    )
    block = _geometry(dynamic, episode)
    assert block["pre"]["focus_variance_share"] == pytest.approx(1.0)
    assert block["post"]["focus_variance_share"] == pytest.approx(1.0)
    # The swap fully replaces the relevant set, so each phase's swapped
    # fields carry all of its variance.
    assert block["pre"]["outgoing_field_share"] == pytest.approx(1.0)
    assert block["post"]["incoming_field_share"] == pytest.approx(1.0)

    static = _artifact(
        pre_event,
        post_event,
        condition="static",
        private_windows=[(1, 1), (2, 2)],
    )
    block = _geometry(static, episode)
    # Static branches keep the pre-rule relevant set after the fork.
    assert block["post"]["focus_variance_share"] == pytest.approx(0.0)
    assert block["pre"]["incoming_field_share"] is None
    assert block["pre"]["outgoing_field_share"] is None
    assert block["post"]["incoming_field_share"] is None
    assert block["post"]["outgoing_field_share"] is None

    fallback = _artifact(pre_event, post_event, condition="dynamic")
    points, _ = materialize_probe_points(fallback, episode)
    assert [point["window"] for point in points] == [0, 0, 1, 1]


def test_single_field_swap_incoming_outgoing_shares():
    episode = _episode(
        pre_rule=_threshold_rule("conjunctive_gating", ("f1", "f2", "f3"), 50.0),
        post_rule=_threshold_rule("conjunctive_gating", ("f2", "f3", "f4"), 50.0),
    )
    pre_event = _probe_event(
        [
            {"values": _values({"f1": 100.0})},
            {"values": _values()},
        ],
        _results({}, {}),
        interaction=1,
    )
    post_event = _probe_event(
        [
            {"values": _values({"f4": 100.0})},
            {"values": _values()},
        ],
        _results({}, {}),
        interaction=2,
    )
    artifact = _artifact(
        pre_event,
        post_event,
        condition="dynamic",
        private_windows=[(1, 1), (2, 2)],
    )
    block = _geometry(artifact, episode)
    # Only f1 leaves and only f4 enters: outgoing={f1}, incoming={f4}.
    assert block["pre"]["outgoing_field_share"] == pytest.approx(1.0)
    assert block["pre"]["incoming_field_share"] == pytest.approx(0.0)
    assert block["post"]["incoming_field_share"] == pytest.approx(1.0)
    assert block["post"]["outgoing_field_share"] == pytest.approx(0.0)


def test_phase_point_sets_split_dedup_and_count_reprobes():
    episode = _episode(after_action_batch=2)
    pre_event = _probe_event(
        [
            {"values": _values()},
            {"values": _values({"f1": 100.0})},
        ],
        _results({}, {}),
        interaction=1,
    )
    post_event = _probe_event(
        [
            {"values": _values()},
            {"values": _values()},
            {"values": _values({"f2": 100.0})},
        ],
        _results({}, {}, {}),
        interaction=2,
    )
    artifact = _artifact(
        pre_event, post_event, private_windows=[(1, 1), (2, 2)]
    )
    sets = probe_phase_point_sets(artifact, episode)
    assert sets["has_change"] is True
    assert len(sets["pre"]) == 2
    # The duplicated zero point collapses; the f2 point is new post-change.
    assert len(sets["post"]) == 2
    # Only the zero point re-probes a pre-change point, and duplicates of it
    # within the post phase count once.
    assert len(sets["post"] & sets["pre"]) == 1


def test_phase_point_sets_static_branch_keeps_single_phase():
    episode = _episode(after_action_batch=2)
    artifact = _artifact(
        _probe_event([{"values": _values()}], _results({}), interaction=1),
        condition="static",
        private_windows=[(1, 5)],
    )
    sets = probe_phase_point_sets(artifact, episode)
    assert sets["has_change"] is False
    assert len(sets["pre"]) == 1
    assert sets["post"] == set()


def test_phase_point_sets_without_transition_are_all_pre():
    episode = _episode()
    artifact = _artifact(
        _probe_event([{"values": _values()}], _results({})),
    )
    artifact.pop("evaluation")
    sets = probe_phase_point_sets(artifact, episode)
    assert sets["has_change"] is False
    assert len(sets["pre"]) == 1
    assert sets["post"] == set()


def test_phase_point_sets_without_episode_return_empty_sets():
    artifact = _artifact(
        _probe_event([{"values": _values()}], _results({})),
    )
    sets = probe_phase_point_sets(artifact)
    assert sets == {"pre": set(), "post": set(), "has_change": True}


def test_episode_metrics_phase_point_counts_and_aggregation_skip_none():
    episode = _episode()
    probed = _artifact(
        _probe_event([{"values": _values()}, {"values": _values({**{f"f{i}": 100.0 for i in range(1, 8)}, "f8": 10})}], _results({}, {}))
    )
    probed["run_success"] = True
    unprobed = _artifact(
        _probe_event([{"values": _values()}], _results({}))
    )
    unprobed["run_success"] = True

    metrics = episode_metrics(probed, episode)
    # Both probes sit in the pre-change window and are distinct points.
    assert metrics["unique_probe_count"] == 2
    assert metrics["reprobes_after_change"] == 0

    legacy = episode_metrics(probed)
    assert legacy["unique_probe_count"] is None
    assert legacy["reprobes_after_change"] is None

    aggregated = aggregate_metrics(
        [probed, unprobed], episodes={"geometry-fixture": episode}
    )
    assert aggregated["overall"]["mean_unique_probes"] == pytest.approx(1.5)
    assert aggregated["overall"]["mean_reprobes_after_change"] == pytest.approx(0.0)

    without_episodes = aggregate_metrics([probed])
    assert without_episodes["overall"]["mean_unique_probes"] is None
    assert without_episodes["overall"]["mean_reprobes_after_change"] is None


def test_episode_metrics_reprobes_count_distinct_pre_points_once():
    episode = _episode(after_action_batch=2)
    artifact = _artifact(
        _probe_event(
            [
                {"values": _values()},
                {"values": _values({"f1": 100.0})},
            ],
            _results({}, {}),
            interaction=1,
        ),
        _probe_event(
            [
                {"values": _values()},
                {"values": _values()},
                {"values": _values({"f1": 100.0})},
                {"values": _values({"f2": 100.0})},
            ],
            _results({}, {}, {}, {}),
            interaction=2,
        ),
        private_windows=[(1, 1), (2, 2)],
    )
    metrics = episode_metrics(artifact, episode)
    # Cross-phase duplicates are not deducted from the unique count:
    # |S_pre| = 2 and |S_post| = 3 even though two post probes revisit
    # pre-change points.
    assert metrics["unique_probe_count"] == 5
    # The zero point is re-probed twice and the f1 point once; each distinct
    # re-probed pre-change point counts once.
    assert metrics["reprobes_after_change"] == 2


def test_aggregate_branched_metrics_passes_episodes_through():
    episode = _episode()
    artifact = {
        "episode_id": "geometry-fixture",
        "fork_after_batch": 2,
        "branches": {
            "flip_50": _artifact(
                _probe_event([{"values": _values()}, {"values": _values({**{f"f{i}": 100.0 for i in range(1, 8)}, "f8": 10})}], _results({}, {}))
            ),
        },
    }
    metrics = aggregate_branched_metrics([artifact], episodes={"geometry-fixture": episode})
    overall = metrics["by_branch"]["flip_50"]["overall"]
    assert overall["mean_unique_probes"] == pytest.approx(2.0)
    assert overall["mean_reprobes_after_change"] == pytest.approx(0.0)
    assert metrics["metrics_schema_version"] == "adaexplorebench_branched_metrics_v4"


def test_oracle_baseline_phase_point_metrics(dynamic_episode, runtime_interaction_policy):
    artifact = run_baseline_episode(
        dynamic_episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        seed=9,
        condition="dynamic",
        sampling_branch="flip_50",
    )
    metrics = episode_metrics(artifact, dynamic_episode)
    assert metrics["unique_probe_count"] is not None
    assert 1 <= metrics["unique_probe_count"] <= metrics["probe_count"]
    assert metrics["reprobes_after_change"] is not None
    assert 0 <= metrics["reprobes_after_change"] <= metrics["unique_probe_count"]
    legacy = episode_metrics(artifact)
    assert legacy["unique_probe_count"] is None
    assert legacy["reprobes_after_change"] is None
