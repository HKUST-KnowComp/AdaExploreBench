from __future__ import annotations

from copy import deepcopy

import pytest

from protocol.mechanism import (
    ACTION_ANCHORS,
    ACTION_BOUNDARIES,
    ACTION_COUNT,
    ACTION_IDS,
    FAMILIES,
    explain_control_state,
    raw_evidence,
    utilities_from_score,
    validate_mechanism,
)


def _base(family: str) -> dict:
    return {
        "family": family,
        "relevant_fields": ["a", "b", "c"],
        "field_ranges": {
            field: {"minimum": 0, "maximum": 10} for field in ("a", "b", "c")
        },
        "field_directions": {"a": 1, "b": 1, "c": 1},
        "action_levels": list(ACTION_ANCHORS),
        "thresholds": {"a": 5, "b": 5, "c": 5},
    }


def test_mechanism_has_exactly_the_four_supported_families():
    assert FAMILIES == (
        "additive_aggregation",
        "conjunctive_gating",
        "disjunctive_sufficiency",
        "lexicographic_priority",
    )


def test_additive_aggregation_uses_all_contributions():
    mechanism = _base("additive_aggregation")
    mechanism.pop("thresholds")
    mechanism.update(weights={"a": 0.3, "b": 0.3, "c": 0.4}, bias=0.1)
    state = explain_control_state(mechanism, {"a": 10, "b": 0, "c": 5})
    assert state["contributions"] == {"a": 0.3, "b": -0.3, "c": 0.0}
    assert raw_evidence(mechanism, {"a": 10, "b": 0, "c": 5}) == pytest.approx(0.1)


def test_conjunctive_uses_minimum_margin():
    mechanism = _base("conjunctive_gating")
    state = explain_control_state(mechanism, {"a": 7, "b": 2, "c": 9})
    assert state["binding_gate"] == "b"
    assert state["raw_evidence"] == min(state["margins"].values())


def test_disjunctive_uses_maximum_margin_and_lists_triggers():
    mechanism = _base("disjunctive_sufficiency")
    state = explain_control_state(mechanism, {"a": 7, "b": 2, "c": 9})
    assert state["triggering_cue"] == "c"
    assert state["satisfied_triggers"] == ["a", "c"]
    assert state["raw_evidence"] == max(state["margins"].values())


def test_lexicographic_respects_priority_and_indifference_band():
    mechanism = _base("lexicographic_priority")
    mechanism.update(priority_order=["a", "b", "c"], indifference_band=0.25)
    state = explain_control_state(mechanism, {"a": 5.5, "b": 8, "c": 0})
    assert state["decisive_cue"] == "b"
    assert state["stopping_depth"] == 2


def test_additive_validator_rejects_single_field_dominance():
    mechanism = _base("additive_aggregation")
    mechanism.pop("thresholds")
    mechanism.update(weights={"a": 0.81, "b": 0.1, "c": 0.09}, bias=0.0)
    with pytest.raises(ValueError, match="dominated"):
        validate_mechanism(mechanism)


def test_lexicographic_validator_rejects_incomplete_order():
    mechanism = _base("lexicographic_priority")
    mechanism.update(priority_order=["a", "b"], indifference_band=0.2)
    with pytest.raises(ValueError, match="complete distinct priority order"):
        validate_mechanism(mechanism)


def test_shared_utility_mapping_is_zero_sum_and_action_ordered():
    samples = [-1.0 + 2.0 * index / 200 for index in range(201)]
    for score in samples:
        utilities = utilities_from_score(score)
        assert len(utilities) == ACTION_COUNT == 4
        assert sum(utilities) == pytest.approx(0.0)
    for score, expected in zip(ACTION_ANCHORS, ACTION_IDS):
        utilities = utilities_from_score(score)
        assert max(range(ACTION_COUNT), key=utilities.__getitem__) == ACTION_IDS.index(expected)


def test_utility_mapping_ties_at_boundaries_prefer_lower_index():
    assert ACTION_BOUNDARIES == (-0.5, 0.0, 0.5)
    for boundary, expected in zip(ACTION_BOUNDARIES, (0, 1, 2)):
        utilities = utilities_from_score(boundary)
        assert utilities[expected] == pytest.approx(utilities[expected + 1])
        others = set(range(ACTION_COUNT)) - {expected, expected + 1}
        assert all(utilities[expected] > utilities[other] for other in others)
        assert max(range(ACTION_COUNT), key=utilities.__getitem__) == expected


def test_utility_curves_outer_monotone_inner_unimodal():
    samples = [-1.0 + 2.0 * index / 2000 for index in range(2001)]
    curves = [utilities_from_score(score) for score in samples]
    # outer actions are monotone over the whole score range
    assert all(left[0] > right[0] for left, right in zip(curves, curves[1:]))
    assert all(left[3] < right[3] for left, right in zip(curves, curves[1:]))
    # inner actions are unimodal with peaks at their anchors (-0.25, +0.25)
    for action, anchor in ((1, -0.25), (2, 0.25)):
        split = samples.index(anchor)
        lower = [curve[action] for curve in curves[: split + 1]]
        upper = [curve[action] for curve in curves[split:]]
        assert all(left < right for left, right in zip(lower, lower[1:]))
        assert all(left > right for left, right in zip(upper, upper[1:]))


def test_utility_mapping_quarter_optimal_regions_and_constant_policy_bias():
    count = 200001
    samples = [-1.0 + 2.0 * index / (count - 1) for index in range(count)]
    regions = [0] * ACTION_COUNT
    means = [0.0] * ACTION_COUNT
    for score in samples:
        utilities = utilities_from_score(score)
        regions[max(range(ACTION_COUNT), key=utilities.__getitem__)] += 1
        for action in range(ACTION_COUNT):
            means[action] += utilities[action]
    # a uniform score makes every action optimal on exactly 25% of [-1, 1]
    assert [region / count for region in regions] == pytest.approx(
        [0.25] * ACTION_COUNT, abs=1e-4
    )
    # accepted constant-policy bias under a uniform score (regression value)
    assert [value / count for value in means] == pytest.approx(
        [-0.003342, 0.003342, 0.003342, -0.003342], abs=1e-4
    )


def test_validator_rejects_removed_family():
    mechanism = _base("non_compensatory")
    with pytest.raises(ValueError, match="invalid family"):
        validate_mechanism(deepcopy(mechanism))
