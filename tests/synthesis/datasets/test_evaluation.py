from __future__ import annotations

from copy import deepcopy

import pytest

from synthesis.datasets.contracts import validate_episode
from synthesis.datasets.dynamic import certification
from synthesis.datasets.evaluation import (
    _population_stats_from_scores,
    _scores_from_evidence,
    dynamic_population_rank,
    fit_frozen_pre_calibration,
    measure_rule_pair,
)


def _additive_rule() -> dict:
    return {
        "family": "additive_aggregation",
        "relevant_fields": ["x"],
        "weights": {"x": 1.0},
        "bias": 0.0,
    }


def test_quartile_calibration_maps_knots_to_action_boundaries():
    arrays = {"x": [-1.0 + 2.0 * index / 100 for index in range(101)]}
    calibrated = fit_frozen_pre_calibration(_additive_rule(), arrays)
    calibration = calibrated["score_calibration"]
    assert calibration["method"] == "frozen_pre_quartile_piecewise"
    assert calibration["evidence_knots"] == [-0.5, 0.0, 0.5]
    assert calibration["score_knots"] == [-0.5, 0.0, 0.5]
    assert calibration["segment_gains"] == [1.0, 1.0]
    assert calibration["clipped_mass_share"] == 0.0
    assert _scores_from_evidence(calibrated, [-0.5, 0.0, 0.5]) == pytest.approx(
        [-0.5, 0.0, 0.5]
    )


def test_quartile_calibration_rejects_degenerate_evidence():
    with pytest.raises(ValueError, match="cannot support four distinct"):
        fit_frozen_pre_calibration(_additive_rule(), {"x": [0.5] * 100})
    with pytest.raises(ValueError, match="cannot support four distinct"):
        fit_frozen_pre_calibration(_additive_rule(), {"x": [0.0] * 50 + [1.0] * 50})


def test_quartile_calibration_rejects_knots_collapsed_by_storage_precision():
    evidence = [1.0e-7, 1.1e-7, 1.2e-7, 1.3e-7, 1.4e-7]
    with pytest.raises(ValueError, match="six-decimal storage precision"):
        fit_frozen_pre_calibration(_additive_rule(), {"x": evidence})


def test_piecewise_scoring_rejects_non_increasing_serialized_knots():
    rule = _additive_rule()
    rule["score_calibration"] = {
        "method": "frozen_pre_quartile_piecewise",
        "evidence_knots": [0.0, 0.0, 0.5],
        "score_knots": [-0.5, 0.0, 0.5],
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        _scores_from_evidence(rule, [0.0])


def test_quartile_calibration_serializes_clipping_and_segment_gains():
    evidence = [-3.0] + [index / 100 for index in range(-49, 50)] + [3.0]
    calibrated = fit_frozen_pre_calibration(_additive_rule(), {"x": evidence})
    calibration = calibrated["score_calibration"]
    assert calibration["evidence_knots"] == [-0.25, 0.0, 0.25]
    assert calibration["score_knots"] == [-0.5, 0.0, 0.5]
    assert calibration["segment_gains"] == [2.0, 2.0]
    # Only the two tail draws map outside [-1, 1] before clipping.
    assert calibration["clipped_mass_share"] == round(2 / 101, 6)
    scores = _scores_from_evidence(calibrated, [-3.0, -0.25, 0.0, 0.25, 3.0])
    assert scores == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])


def test_post_rule_inherits_identical_frozen_calibration():
    arrays = {"x": [-1.0 + 2.0 * index / 100 for index in range(101)]}
    old = fit_frozen_pre_calibration(_additive_rule(), arrays)
    new = deepcopy(old)
    new["weights"] = {"x": 0.9}
    stats = measure_rule_pair(old, new, {"x": [-1.0, -0.5, 0.0, 0.5, 1.0]})
    assert stats["sample_count"] == 5
    tampered = deepcopy(new)
    tampered["score_calibration"]["segment_gains"] = [1.5, 1.0]
    with pytest.raises(ValueError, match="must share frozen calibration"):
        measure_rule_pair(old, tampered, {"x": [-1.0, -0.5, 0.0, 0.5, 1.0]})


def test_population_stats_use_four_action_shapes():
    stats = _population_stats_from_scores(
        [-0.75, -0.25, 0.25, 0.75],
        [-0.75, -0.25, 0.75, 0.25],
    )
    assert stats["action_transition_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    assert stats["old_action_marginal"] == [0.25] * 4
    assert stats["new_action_marginal"] == [0.25] * 4
    assert stats["maximum_uniform_deviation"] == 0.0
    assert len(stats["old_mean_utility_by_action"]) == 4
    assert len(stats["new_mean_utility_by_action"]) == 4
    assert "old_maximum_constant_action_expected_utility" in stats
    assert "new_maximum_constant_action_expected_utility" in stats


def test_population_stats_resolve_boundary_ties_to_lower_action():
    stats = _population_stats_from_scores(
        [-0.5, 0.0, 0.5, 1.0],
        [-0.5, 0.0, 0.5, 1.0],
    )
    assert stats["action_transition_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_dynamic_rank_uses_worst_of_nine_population_margins():
    centered = {
        "flip_rate": 0.5,
        "old_action_marginal": [0.25] * 4,
        "new_action_marginal": [0.25] * 4,
    }
    edge = {**centered, "new_action_marginal": [0.15, 0.25, 0.25, 0.35]}
    centered_rank = dynamic_population_rank(
        centered, (0.35, 0.65), (0.15, 0.35), 0.5, 1.0, 0.5, 1.0, 4, 4
    )
    edge_rank = dynamic_population_rank(
        edge, (0.35, 0.65), (0.15, 0.35), 0.5, 1.0, 0.5, 1.0, 4, 4
    )
    assert centered_rank > edge_rank


def test_dynamic_rank_tie_breaks_by_baseline_then_grid_order():
    stats = {
        "flip_rate": 0.5,
        "old_action_marginal": [0.25] * 4,
        "new_action_marginal": [0.25] * 4,
    }
    baseline = dynamic_population_rank(
        stats, (0.35, 0.65), (0.15, 0.35), 0.5, 1.0, 0.5, 1.0, 4, 4
    )
    distant = dynamic_population_rank(
        stats, (0.35, 0.65), (0.15, 0.35), 0.6, 1.0, 0.5, 1.0, 5, 4
    )
    later = dynamic_population_rank(
        stats, (0.35, 0.65), (0.15, 0.35), 0.5, 1.0, 0.5, 1.0, 5, 4
    )
    assert baseline > distant
    assert baseline > later


def test_episode_contract_accepts_four_actions_and_rejects_three(dynamic_episode):
    validate_episode(dynamic_episode, "dynamic")
    broken = deepcopy(dynamic_episode)
    broken["agent_task"]["actions"] = broken["agent_task"]["actions"][:3]
    with pytest.raises(ValueError, match="exactly four actions and 8 fields"):
        validate_episode(broken, "dynamic")


def test_episode_contract_rejects_non_numeric_field_type(dynamic_episode):
    broken = deepcopy(dynamic_episode)
    field_id = next(iter(broken["agent_task"]["fields"]))
    broken["agent_task"]["fields"][field_id]["type"] = "string"

    with pytest.raises(
        ValueError,
        match=rf"agent field {field_id} type must be integer or decimal",
    ):
        validate_episode(broken, "dynamic")


@pytest.mark.parametrize(
    ("constraints", "expected_fits", "selected_fit"),
    [
        ([], 2, 2),
        ([{"expr": {"field": "shared"}}], 2, 2),
        ([{"expr": {"field": "outgoing"}}], 4, 3),
    ],
)
def test_parameter_pool_recalibrates_each_stage_and_only_for_replacement_constraints(
    monkeypatch, dynamic_selection, constraints, expected_fits, selected_fit
):
    config = __import__("synthesis.datasets.config", fromlist=["load_config"]).load_config(
        __import__("pathlib").Path("configs/synthesis/dynamic.yaml")
    )
    config = __import__("dataclasses").replace(
        config,
        search=__import__("dataclasses").replace(
            config.search,
            coupling_grid=(0.1, 0.2),
            reversal_additive_grid=(1.0, 1.1),
            parameter_screening_sample_count=2,
            evaluation_sample_count=4,
        ),
    )
    core = {
        "group_id": "group",
        "shared_fields": ["shared", "reversed"],
        "outgoing_field": "outgoing",
        "incoming_field": "incoming",
        "reversal_target": "reversed",
    }
    sample_calls = []
    fit_calls = []
    score_calls = []

    monkeypatch.setattr(certification, "prepare_base_proposals", lambda *args: object())

    def sample(*args, effective_coupling, **kwargs):
        sample_calls.append((effective_coupling, kwargs["count"], len(kwargs["pool"].records)))
        kwargs["pool"].records.extend(
            [({"coupling": effective_coupling}, "profile")]
            * (kwargs["count"] - len(kwargs["pool"].records))
        )
        return [{"coupling": effective_coupling}] * kwargs["count"]

    def fit(rule, arrays):
        fit_calls.append(arrays)
        return {**rule, "score_calibration": {"fit": len(fit_calls)}}

    monkeypatch.setattr(certification, "sample_states", sample)
    monkeypatch.setattr(certification, "normalized_state_arrays", lambda s, f, x: x)
    monkeypatch.setattr(certification, "fit_frozen_pre_calibration", fit)
    def scores(rule, arrays):
        score_calls.append(arrays)
        return [0.0] * len(arrays)

    monkeypatch.setattr(
        certification,
        "precompute_rule_population",
        lambda rule, arrays: {
            "scores": scores(rule, arrays),
            "actions": [0] * len(arrays),
            "action_marginal": [1.0, 0.0, 0.0, 0.0],
        },
    )
    monkeypatch.setattr(
        certification,
        "_rule_pair",
        lambda *args, **kwargs: ({"rule": "old"}, {"rule": "new"}, {}),
    )
    monkeypatch.setattr(certification, "replacement_pair_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(certification, "measure_rule_pair", lambda *a, **k: {})
    monkeypatch.setattr(certification, "_optimistically_possible", lambda *a, **k: True)
    monkeypatch.setattr(certification, "_optimistic_transition_mask", lambda *a, **k: (True,) * 16)
    monkeypatch.setattr(
        certification,
        "dynamic_population_rank",
        lambda stats, *args: (float(args[2]), 0.0, 0, 0),
    )

    selected = certification._evaluate_structure(
        dynamic_selection,
        core,
        "additive_aggregation",
        "compound",
        17,
        config,
        {"constraints": constraints},
    )
    assert sample_calls == [
        (0.1, 2, 0),
        (0.2, 2, 0),
        (0.2, 4, 2),
        (0.1, 4, 2),
    ]
    assert len(fit_calls) == expected_fits
    # Calibration may be shared when replacement fields are unconstrained, but
    # pre actions always belong to one coupling-specific population pool.
    assert len(score_calls) == 4
    assert [rows[0]["coupling"] for rows in score_calls] == [0.1, 0.2, 0.2, 0.1]
    assert selected is not None
    assert selected["old"]["score_calibration"]["fit"] == selected_fit


def test_refinement_selects_two_best_settings_for_each_of_two_couplings(
    monkeypatch, dynamic_selection
):
    config = __import__("synthesis.datasets.config", fromlist=["load_config"]).load_config(
        __import__("pathlib").Path("configs/synthesis/dynamic.yaml")
    )
    config = __import__("dataclasses").replace(
        config,
        search=__import__("dataclasses").replace(
            config.search,
            coupling_grid=(0.1, 0.2, 0.3),
            reversal_additive_grid=(1.0, 1.1, 1.2),
            parameter_screening_sample_count=2,
            evaluation_sample_count=4,
        ),
    )
    core = {
        "group_id": "group",
        "shared_fields": ["shared", "reversed"],
        "outgoing_field": "outgoing",
        "incoming_field": "incoming",
        "reversal_target": "reversed",
    }
    refined_pairs = []
    q_calls = []

    monkeypatch.setattr(certification, "prepare_base_proposals", lambda *args: object())

    def sample(*args, effective_coupling, **kwargs):
        pool = kwargs["pool"]
        pool.records.extend(
            [({"coupling": effective_coupling}, "profile")]
            * (kwargs["count"] - len(pool.records))
        )
        return [{"coupling": effective_coupling}] * kwargs["count"]

    monkeypatch.setattr(certification, "sample_states", sample)
    monkeypatch.setattr(certification, "normalized_state_arrays", lambda s, f, x: x)
    monkeypatch.setattr(
        certification,
        "precompute_rule_population",
        lambda rule, arrays: {
            "scores": [0.0] * len(arrays),
            "actions": [0] * len(arrays),
            "action_marginal": [1.0, 0.0, 0.0, 0.0],
        },
    )
    monkeypatch.setattr(
        certification,
        "fit_frozen_pre_calibration",
        lambda rule, arrays: {**rule, "score_calibration": {}},
    )
    monkeypatch.setattr(
        certification,
        "_rule_pair",
        lambda *args, **kwargs: (
            {"rule": "old", "reversal": args[5]},
            {"rule": "new"},
            {},
        ),
    )
    monkeypatch.setattr(certification, "replacement_pair_diagnostics", lambda *a, **k: {})

    def measure(old, new, arrays, **kwargs):
        if len(arrays) == 4 and not kwargs.get("full_report", False):
            refined_pairs.append((arrays[0]["coupling"], old["reversal"]))
        return {"coupling": arrays[0]["coupling"], "reversal": old["reversal"]}

    monkeypatch.setattr(certification, "measure_rule_pair", measure)
    monkeypatch.setattr(certification, "_optimistically_possible", lambda *a, **k: True)
    monkeypatch.setattr(certification, "_optimistic_transition_mask", lambda *a, **k: (True,) * 16)
    def feasible(mask, *args):
        q_calls.append(mask)
        return True

    monkeypatch.setattr(certification, "target_mask_is_feasible", feasible)
    ranks = {
        (0.1, 1.0): 9.0,
        (0.1, 1.1): 8.0,
        (0.2, 1.2): 7.0,
        (0.2, 1.1): 6.0,
    }
    monkeypatch.setattr(
        certification,
        "dynamic_population_rank",
        lambda stats, *args: (ranks.get((stats["coupling"], stats["reversal"]), 0.0), 0.0, 0, 0),
    )

    selected = certification._evaluate_structure(
        dynamic_selection, core, "additive_aggregation", "compound", 17, config, {"constraints": []}
    )

    assert refined_pairs == [(0.1, 1.0), (0.1, 1.1), (0.2, 1.2), (0.2, 1.1)]
    assert {coupling for coupling, _ in refined_pairs} == {0.1, 0.2}
    assert {reversal for _, reversal in refined_pairs} == {1.0, 1.1, 1.2}
    assert len(q_calls) == 5
    assert selected is not None


def test_heldout_replacement_realism_is_a_final_gate(
    monkeypatch, dynamic_selection
):
    config = __import__("synthesis.datasets.config", fromlist=["load_config"]).load_config(
        __import__("pathlib").Path("configs/synthesis/dynamic.yaml")
    )
    core = {
        "group_id": "group",
        "shared_fields": ["shared", "reversed"],
        "outgoing_field": "outgoing",
        "incoming_field": "incoming",
        "reversal_target": "reversed",
    }
    selected = {
        "old": {"rule": "old"},
        "new": {"rule": "new"},
        "effective_coupling": 0.5,
        "_evaluation_pool": certification.PopulationPool(),
        "_evaluation_base": object(),
    }
    monkeypatch.setattr(
        certification,
        "sample_states",
        lambda *args, **kwargs: [{"row": index} for index in range(4)],
    )
    monkeypatch.setattr(certification, "normalized_state_arrays", lambda *args: {})
    monkeypatch.setattr(
        certification,
        "measure_rule_pair",
        lambda *args, **kwargs: {
            "flip_rate": 0.5,
            "old_action_marginal": [0.25] * 4,
            "new_action_marginal": [0.25] * 4,
        },
    )
    monkeypatch.setattr(
        certification,
        "replacement_pair_diagnostics",
        lambda *args, **kwargs: {"joint_realism_pass": False},
    )
    monkeypatch.setattr(
        certification,
        "solve_target_matrices",
        lambda *args, **kwargs: pytest.fail("Q must not run after realism rejection"),
    )
    rejected = {}
    assert certification.certify_candidate(
        dynamic_selection,
        core,
        "additive_aggregation",
        "compound",
        config,
        {},
        selected,
        rejected,
    ) is None
    assert rejected == {"heldout_replacement_realism_gate": 1}


def test_core_rejection_records_exact_terminal_stage(monkeypatch, dynamic_selection):
    config = __import__("synthesis.datasets.config", fromlist=["load_config"]).load_config(
        __import__("pathlib").Path("configs/synthesis/dynamic.yaml")
    )
    core = {"shared_fields": [], "outgoing_field": "a", "incoming_field": "b"}
    monkeypatch.setattr(certification, "distractor_sources", lambda *args: ())
    rejected = {}
    assert certification.evaluate_core(
        dynamic_selection,
        core,
        "additive_aggregation",
        "compound",
        config,
        seed=17,
        rejection_counts=rejected,
    ) is None
    assert rejected == {"distractor_capacity": 1}
