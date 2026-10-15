from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from statistics import mean, pstdev
from typing import Any

from .dynamic.rules import (
    COMPOUND_PARAMETERIZATION,
    REPLACEMENT_PARAMETERIZATION,
)
from protocol.mechanism import (
    ACTION_BOUNDARIES,
    ACTION_COUNT,
    ACTION_INDEXES,
    explain_control_state,
    intervention_score,
    utilities_from_score,
)



def _action(score: float) -> int:
    # A score exactly on a boundary resolves to the lower action index,
    # matching best_action's lowest-index tie rule.
    return sum(1 for boundary in ACTION_BOUNDARIES if score > boundary)



def normalized_state_arrays(
    bundle: dict[str, Any],
    fields: list[str],
    states: list[dict[str, Any]],
) -> dict[str, list[float]]:
    arrays = {}
    for field in fields:
        meta = bundle.fields[field]
        low = float(meta["minimum"])
        span = float(meta["maximum"]) - low
        arrays[field] = [
            max(-1.0, min(1.0, 2.0 * (float(state[field]) - low) / span - 1.0))
            for state in states
        ]
    return arrays



def _raw_evidence_array(
    rule: dict[str, Any],
    normalized: dict[str, list[float]],
) -> list[float]:
    fields = list(rule["relevant_fields"])
    count = len(normalized[fields[0]])
    family = rule["family"]
    if family == "additive_aggregation":
        return [
            float(rule["bias"])
            + sum(
                float(rule["weights"][field]) * normalized[field][index]
                for field in fields
            )
            for index in range(count)
        ]
    margins = {}
    for field in fields:
        bounds = rule["field_ranges"][field]
        low = float(bounds["minimum"])
        span = float(bounds["maximum"]) - low
        threshold = max(
            -1.0,
            min(
                1.0,
                2.0 * (float(rule["thresholds"][field]) - low) / span - 1.0,
            ),
        )
        direction = float(rule["field_directions"][field])
        margins[field] = [
            direction * (value - threshold)
            for value in normalized[field]
        ]
    if family == "conjunctive_gating":
        return [
            min(margins[field][index] for field in fields)
            for index in range(count)
        ]
    if family == "disjunctive_sufficiency":
        return [
            max(margins[field][index] for field in fields)
            for index in range(count)
        ]
    if family == "lexicographic_priority":
        order = list(rule["priority_order"])
        band = float(rule["indifference_band"])
        evidence = []
        for index in range(count):
            decisive = order[-1]
            for field in order:
                if abs(margins[field][index]) > band:
                    decisive = field
                    break
            evidence.append(margins[decisive][index])
        return evidence
    raise ValueError(f"unsupported mechanism family: {family}")



def _empirical_quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight



def _scores_from_evidence(
    rule: dict[str, Any],
    evidence: list[float],
) -> list[float]:
    calibration = rule.get("score_calibration") or {}
    if calibration.get("method") == "frozen_pre_quartile_piecewise":
        knots = [float(knot) for knot in calibration["evidence_knots"]]
        targets = [float(target) for target in calibration["score_knots"]]
        if (
            len(knots) != 3
            or len(targets) != 3
            or not knots[0] < knots[1] < knots[2]
            or not targets[0] < targets[1] < targets[2]
        ):
            raise ValueError(
                "frozen pre calibration requires three strictly increasing "
                "evidence and score knots"
            )
        lower_slope = (targets[1] - targets[0]) / (knots[1] - knots[0])
        upper_slope = (targets[2] - targets[1]) / (knots[2] - knots[1])
        return [
            max(
                -1.0,
                min(
                    1.0,
                    targets[0] + (value - knots[0]) * lower_slope
                    if value <= knots[1]
                    else targets[1] + (value - knots[1]) * upper_slope,
                ),
            )
            for value in evidence
        ]
    negative_anchor = max(
        abs(float(calibration.get("negative_anchor", -1.0))),
        1e-9,
    )
    positive_anchor = max(
        float(calibration.get("positive_anchor", 1.0)),
        1e-9,
    )
    return [
        (
            max(-1.0, value / negative_anchor)
            if value < 0
            else min(1.0, value / positive_anchor)
            if value > 0
            else 0.0
        )
        for value in evidence
    ]



def _normalized_threshold(rule: dict[str, Any], field: str) -> float:
    bounds = rule["field_ranges"][field]
    low = float(bounds["minimum"])
    span = float(bounds["maximum"]) - low
    return 2.0 * (float(rule["thresholds"][field]) - low) / span - 1.0


def _pressure_quantile(rule: dict[str, Any], field: str) -> float:
    bounds = rule["field_ranges"][field]
    low = float(bounds["minimum"])
    raw = (float(rule["thresholds"][field]) - low) / (
        float(bounds["maximum"]) - low
    )
    return raw if int(rule["field_directions"][field]) > 0 else 1.0 - raw



def replacement_invariant_diagnostics(
    bundle: dict[str, Any],
    old: dict[str, Any],
    new: dict[str, Any],
    transition: dict[str, Any],
    parameters: dict[str, Any],
    role_jitter_range: tuple[float, float],
    threshold_jitter_range: tuple[float, float],
) -> dict[str, Any]:
    """Validate a replacement with one explicit, bounded role jitter."""

    shared = list(transition["shared_fields"])
    outgoing = str(transition["outgoing_field"])
    incoming = str(transition["incoming_field"])
    checks: dict[str, bool] = {
        "family_equal": old.get("family") == new.get("family"),
        "relevant_identity_only": old.get("relevant_fields")
        == [*shared, outgoing]
        and new.get("relevant_fields") == [*shared, incoming],
        "action_levels_equal": old.get("action_levels") == new.get("action_levels"),
        "domain_template_equal": old.get("domain_template_id")
        == new.get("domain_template_id"),
        "domain_rationale_equal": old.get("domain_rationale")
        == new.get("domain_rationale"),
        "endpoint_direction_equal": old.get("field_directions", {}).get(outgoing)
        == new.get("field_directions", {}).get(incoming),
        "shared_directions_equal": all(
            old.get("field_directions", {}).get(field)
            == new.get("field_directions", {}).get(field)
            for field in shared
        ),
        "shared_ranges_equal": all(
            old.get("field_ranges", {}).get(field)
            == new.get("field_ranges", {}).get(field)
            for field in shared
        ),
        "score_calibration_equal": old.get("score_calibration")
        == new.get("score_calibration"),
    }
    family = str(old.get("family"))
    endpoint_parameter: dict[str, float] = {}
    if family == "additive_aggregation":
        reference = shared[0]
        pre_ratio = abs(float(old["weights"][outgoing])) / abs(
            float(old["weights"][reference])
        )
        post_ratio = abs(float(new["weights"][incoming])) / abs(
            float(new["weights"][reference])
        )
        observed_jitter = post_ratio / pre_ratio - 1.0
        recorded_jitter = float(
            parameters["replacement_role_strength_relative_jitter"]
        )
        endpoint_parameter = {
            "outgoing_weight": float(old["weights"][outgoing]),
            "incoming_weight": float(new["weights"][incoming]),
            "observed_relative_jitter": observed_jitter,
            "recorded_relative_jitter": recorded_jitter,
        }
        checks.update(
            {
                "replacement_role_jitter_recorded": math.isclose(
                    observed_jitter, recorded_jitter, rel_tol=1e-5, abs_tol=1e-5
                ),
                "replacement_role_jitter_bounded": role_jitter_range[0]
                <= recorded_jitter
                <= role_jitter_range[1],
                "bias_equal": old.get("bias") == new.get("bias"),
            }
        )
    else:
        outgoing_pressure = _pressure_quantile(old, outgoing)
        incoming_pressure = _pressure_quantile(new, incoming)
        observed_jitter = incoming_pressure - outgoing_pressure
        recorded_jitter = float(
            parameters["replacement_threshold_quantile_jitter"]
        )
        endpoint_parameter = {
            "outgoing_normalized_threshold": _normalized_threshold(old, outgoing),
            "incoming_normalized_threshold": _normalized_threshold(new, incoming),
            "observed_pressure_quantile_jitter": observed_jitter,
            "recorded_pressure_quantile_jitter": recorded_jitter,
        }
        checks.update(
            {
                "shared_thresholds_equal": all(
                    old["thresholds"][field] == new["thresholds"][field]
                    for field in shared
                ),
                "replacement_threshold_jitter_recorded": math.isclose(
                    observed_jitter, recorded_jitter, rel_tol=1e-5, abs_tol=1e-5
                ),
                "replacement_threshold_jitter_bounded": threshold_jitter_range[0]
                <= recorded_jitter
                <= threshold_jitter_range[1],
            }
        )
        if family == "lexicographic_priority":
            expected_order = [
                incoming if field == outgoing else field
                for field in old["priority_order"]
            ]
            checks.update(
                {
                    "priority_slot_equal": new.get("priority_order")
                    == expected_order,
                    "indifference_band_equal": old.get("indifference_band")
                    == new.get("indifference_band"),
                }
            )

    return {
        "parameterization": REPLACEMENT_PARAMETERIZATION,
        "checks": checks,
        "endpoint_role_parameter": endpoint_parameter,
        "pure_replacement_pass": all(checks.values()),
        "state_distribution_shared": True,
    }



def compound_invariant_diagnostics(
    bundle: dict[str, Any],
    old: dict[str, Any],
    joint_post: dict[str, Any],
    transition: dict[str, Any],
    parameters: dict[str, Any],
    role_jitter_range: tuple[float, float],
    threshold_jitter_range: tuple[float, float],
) -> dict[str, Any]:
    """Validate compound as one joint replacement-and-reversal mechanism."""

    target = str(transition["reversal_target"])
    shared = list(transition["shared_fields"])
    stable = str(transition["stable_relevant"])
    outgoing = str(transition["outgoing_field"])
    incoming = str(transition["incoming_field"])
    changed_directions = [
        field
        for field in shared
        if old["field_directions"].get(field)
        != joint_post["field_directions"].get(field)
    ]
    checks = {
        "distinct_visible_fields": len(
            {
                target,
                stable,
                str(transition["outgoing_field"]),
                str(transition["incoming_field"]),
                *map(str, transition["distractor_fields"]),
            }
        ) == 4 + len(transition["distractor_fields"]),
        "reversal_target_is_shared": target in shared,
        "stable_relevant_is_other_shared": stable in shared and stable != target,
        "relevant_identity_only": old.get("relevant_fields")
        == [*shared, outgoing]
        and joint_post.get("relevant_fields") == [*shared, incoming],
        "single_shared_direction_reversed": changed_directions == [target],
        "target_direction_reversed": (
            int(old["field_directions"][target])
            == -int(joint_post["field_directions"][target])
        ),
        "stable_direction_equal": old["field_directions"][stable]
        == joint_post["field_directions"][stable],
        "replacement_direction_equal": old["field_directions"][outgoing]
        == joint_post["field_directions"][incoming],
        "score_calibration_equal": old.get("score_calibration")
        == joint_post.get("score_calibration"),
    }
    role_parameter: dict[str, float] = {}
    if old["family"] == "additive_aggregation":
        old_stable = abs(float(old["weights"][stable]))
        post_stable = abs(float(joint_post["weights"][stable]))
        pre_role_ratio = abs(float(old["weights"][outgoing])) / old_stable
        post_role_ratio = (
            abs(float(joint_post["weights"][incoming])) / post_stable
        )
        reversal_multiplier = (
            abs(float(joint_post["weights"][target])) / post_stable
        ) / (abs(float(old["weights"][target])) / old_stable)
        observed_jitter = post_role_ratio / pre_role_ratio - 1.0
        recorded_jitter = float(
            parameters["replacement_role_strength_relative_jitter"]
        )
        role_parameter = {
            "pre_replacement_to_stable_ratio": pre_role_ratio,
            "post_replacement_to_stable_ratio": post_role_ratio,
            "reversal_magnitude_multiplier": reversal_multiplier,
            "observed_replacement_relative_jitter": observed_jitter,
            "recorded_replacement_relative_jitter": recorded_jitter,
        }
        checks.update(
            {
                "replacement_role_jitter_recorded": math.isclose(
                    observed_jitter, recorded_jitter, rel_tol=1e-5, abs_tol=1e-5
                ),
                "replacement_role_jitter_bounded": role_jitter_range[0]
                <= recorded_jitter
                <= role_jitter_range[1],
                "reversal_magnitude_positive": reversal_multiplier > 0.0,
                "weights_normalized": abs(
                    sum(abs(float(value)) for value in old["weights"].values())
                    - 1.0
                ) <= 1e-5
                and abs(
                    sum(
                        abs(float(value))
                        for value in joint_post["weights"].values()
                    )
                    - 1.0
                ) <= 1e-5,
                "bias_equal": old.get("bias") == joint_post.get("bias"),
            }
        )
    else:
        outgoing_pressure = _pressure_quantile(old, outgoing)
        incoming_pressure = _pressure_quantile(joint_post, incoming)
        observed_jitter = incoming_pressure - outgoing_pressure
        recorded_jitter = float(
            parameters["replacement_threshold_quantile_jitter"]
        )
        role_parameter = {
            "outgoing_normalized_threshold": _normalized_threshold(
                old, outgoing
            ),
            "incoming_normalized_threshold": _normalized_threshold(
                joint_post, incoming
            ),
            "observed_pressure_quantile_jitter": observed_jitter,
            "recorded_pressure_quantile_jitter": recorded_jitter,
        }
        checks.update(
            {
                "stable_threshold_equal": old["thresholds"][stable]
                == joint_post["thresholds"][stable],
                "replacement_threshold_jitter_recorded": math.isclose(
                    observed_jitter, recorded_jitter, rel_tol=1e-5, abs_tol=1e-5
                ),
                "replacement_threshold_jitter_bounded": threshold_jitter_range[0]
                <= recorded_jitter
                <= threshold_jitter_range[1],
            }
        )
        if old["family"] == "lexicographic_priority":
            expected_order = [
                incoming if field == outgoing else field
                for field in old["priority_order"]
            ]
            checks.update(
                {
                    "priority_slot_equal": joint_post.get("priority_order")
                    == expected_order,
                    "indifference_band_equal": old.get("indifference_band")
                    == joint_post.get("indifference_band"),
                }
            )
    return {
        "parameterization": COMPOUND_PARAMETERIZATION,
        "reversal_target": target,
        "stable_relevant": stable,
        "replacement_role_parameter": role_parameter,
        "checks": checks,
        "compound_structural_pass": all(checks.values()),
    }



def replacement_pair_diagnostics(
    bundle: dict[str, Any],
    outgoing: str,
    incoming: str,
    states: list[dict[str, Any]],
    *,
    minimum_absolute_correlation: float,
    maximum_absolute_correlation: float,
    minimum_mean_divergence: float,
) -> dict[str, Any]:
    def normalized(field: str, value: float) -> float:
        meta = bundle.fields[field]
        return (float(value) - float(meta["minimum"])) / (
            float(meta["maximum"]) - float(meta["minimum"])
        )

    left = [normalized(outgoing, state[outgoing]) for state in states]
    right = [normalized(incoming, state[incoming]) for state in states]
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_scale = sum((y - right_mean) ** 2 for y in right) ** 0.5
    correlation = (
        numerator / (left_scale * right_scale)
        if left_scale > 1e-12 and right_scale > 1e-12
        else 0.0
    )
    divergence = [abs(x - y) for x, y in zip(left, right)]
    return {
        "sample_count": len(states),
        "correlation": correlation,
        "mean_normalized_divergence": mean(divergence),
        "p95_normalized_divergence": sorted(divergence)[
            int(0.95 * (len(divergence) - 1))
        ],
        "joint_realism_pass": minimum_absolute_correlation
        <= abs(correlation)
        <= maximum_absolute_correlation
        and mean(divergence) >= minimum_mean_divergence,
    }



def population_stats(
    old: dict[str, Any],
    new: dict[str, Any],
    states: list[dict[str, Any]],
    *,
    full_report: bool = True,
) -> dict[str, Any]:
    old_scores = [intervention_score(old, state) for state in states]
    new_scores = [intervention_score(new, state) for state in states]
    return _population_stats_from_scores(
        old_scores,
        new_scores,
        full_report=full_report,
    )



def _population_stats_from_scores(
    old_scores: list[float],
    new_scores: list[float],
    *,
    full_report: bool = True,
    old_actions: list[int] | None = None,
    old_marginal: list[float] | None = None,
) -> dict[str, Any]:
    if old_actions is None:
        old_actions = [_action(score) for score in old_scores]
    new_actions = [_action(score) for score in new_scores]
    count = len(old_scores)
    matrix = [[0] * ACTION_COUNT for _ in range(ACTION_COUNT)]
    for before, after in zip(old_actions, new_actions):
        matrix[before][after] += 1
    new_counts = Counter(new_actions)
    if old_marginal is None:
        old_counts = Counter(old_actions)
        old_marginal = [old_counts[index] / count for index in ACTION_INDEXES]
    new_marginal = [new_counts[index] / count for index in ACTION_INDEXES]
    report = {
        "sample_count": count,
        "flip_rate": sum(a != b for a, b in zip(old_actions, new_actions)) / count,
        "old_action_marginal": old_marginal,
        "new_action_marginal": new_marginal,
        "maximum_uniform_deviation": max(
            abs(value - 1.0 / ACTION_COUNT)
            for value in [*old_marginal, *new_marginal]
        ),
        "action_marginal_tv": 0.5
        * sum(abs(a - b) for a, b in zip(old_marginal, new_marginal)),
        "action_transition_matrix": matrix,
        "mean_score_shift": mean(
            after - before for before, after in zip(old_scores, new_scores)
        ),
        "mean_absolute_score_shift": mean(
            abs(after - before) for before, after in zip(old_scores, new_scores)
        ),
        "uniform_random_expected_utility": 0.0,
    }
    if full_report:
        old_utilities = [utilities_from_score(score) for score in old_scores]
        new_utilities = [utilities_from_score(score) for score in new_scores]

        def distribution(values: list[float]) -> dict[str, Any]:
            ordered = sorted(values)
            return {
                "mean": mean(values),
                "standard_deviation": pstdev(values),
                "p05": ordered[int(0.05 * (len(ordered) - 1))],
                "p50": ordered[int(0.50 * (len(ordered) - 1))],
                "p95": ordered[int(0.95 * (len(ordered) - 1))],
            }

        old_mean_utility = [
            mean(row[action] for row in old_utilities) for action in ACTION_INDEXES
        ]
        new_mean_utility = [
            mean(row[action] for row in new_utilities) for action in ACTION_INDEXES
        ]
        report.update(
            {
                "old_score_distribution": distribution(old_scores),
                "new_score_distribution": distribution(new_scores),
                "old_mean_utility_by_action": old_mean_utility,
                "new_mean_utility_by_action": new_mean_utility,
                "old_maximum_constant_action_expected_utility": max(
                    abs(value) for value in old_mean_utility
                ),
                "new_maximum_constant_action_expected_utility": max(
                    abs(value) for value in new_mean_utility
                ),
            }
        )
    return report



class CalibrationDegeneracyError(ValueError):
    """The sampled pre-rule evidence cannot define four calibrated regions."""


def fit_frozen_pre_calibration(
    pre_rule: dict[str, Any],
    calibration_arrays: dict[str, list[float]],
) -> dict[str, Any]:
    """Fit frozen quartile calibration knots from pre evidence only."""

    evidence = _raw_evidence_array(pre_rule, calibration_arrays)
    q25 = _empirical_quantile(evidence, 0.25)
    q50 = _empirical_quantile(evidence, 0.5)
    q75 = _empirical_quantile(evidence, 0.75)
    region_mass = (
        sum(value <= q25 for value in evidence),
        sum(q25 < value <= q50 for value in evidence),
        sum(q50 < value <= q75 for value in evidence),
        sum(value > q75 for value in evidence),
    )
    if not q25 < q50 < q75 or not all(region_mass):
        raise CalibrationDegeneracyError(
            "pre-calibration evidence cannot support four distinct calibrated regions"
        )
    evidence_knots = [round(q25, 6), round(q50, 6), round(q75, 6)]
    if not evidence_knots[0] < evidence_knots[1] < evidence_knots[2]:
        raise CalibrationDegeneracyError(
            "pre-calibration evidence cannot support distinct knots at "
            "six-decimal storage precision"
        )
    lower_gain = 0.5 / (q50 - q25)
    upper_gain = 0.5 / (q75 - q50)
    clipped = sum(
        not -1.0
        <= (
            -0.5 + (value - q25) * lower_gain
            if value <= q50
            else (value - q50) * upper_gain
        )
        <= 1.0
        for value in evidence
    )
    calibrated = deepcopy(pre_rule)
    calibrated["score_calibration"] = {
        "method": "frozen_pre_quartile_piecewise",
        "evidence_knots": evidence_knots,
        "score_knots": [round(boundary, 6) for boundary in ACTION_BOUNDARIES],
        "clipped_mass_share": round(clipped / len(evidence), 6),
        "segment_gains": [round(lower_gain, 6), round(upper_gain, 6)],
    }
    return calibrated


def measure_rule_pair(
    old: dict[str, Any],
    new: dict[str, Any],
    screening_arrays: dict[str, list[float]],
    *,
    full_report: bool = False,
    pre_population: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure two rules that already share one frozen calibration."""

    if old.get("score_calibration") != new.get("score_calibration"):
        raise ValueError("pre and post rules must share frozen calibration")
    pre = pre_population or precompute_rule_population(old, screening_arrays)
    old_scores = pre["scores"]
    new_scores = _scores_from_evidence(
        new,
        _raw_evidence_array(new, screening_arrays),
    )
    return _population_stats_from_scores(
        old_scores,
        new_scores,
        full_report=full_report,
        old_actions=pre["actions"],
        old_marginal=pre["action_marginal"],
    )


def rule_scores(
    rule: dict[str, Any], screening_arrays: dict[str, list[float]]
) -> list[float]:
    return _scores_from_evidence(
        rule,
        _raw_evidence_array(rule, screening_arrays),
    )


def precompute_rule_population(
    rule: dict[str, Any], screening_arrays: dict[str, list[float]]
) -> dict[str, Any]:
    scores = rule_scores(rule, screening_arrays)
    actions = [_action(score) for score in scores]
    counts = Counter(actions)
    return {
        "scores": scores,
        "actions": actions,
        "action_marginal": [
            counts[index] / len(actions) for index in ACTION_INDEXES
        ],
    }



def population_gate_failures(
    stats: dict[str, Any],
    population_flip_range: tuple[float, float],
    action_share_range: tuple[float, float],
) -> list[str]:
    failures = []
    flip_low, flip_high = population_flip_range
    if not flip_low <= float(stats["flip_rate"]) <= flip_high:
        failures.append("population_flip")
    action_low, action_high = action_share_range
    if any(
        not action_low <= float(share) <= action_high
        for share in [
            *stats["old_action_marginal"],
            *stats["new_action_marginal"],
        ]
    ):
        failures.append("population_action_share")
    return failures



def dynamic_population_rank(
    stats: dict[str, Any],
    population_flip_range: tuple[float, float],
    action_share_range: tuple[float, float],
    coupling: float,
    reversal: float,
    coupling_baseline: float,
    reversal_baseline: float,
    coupling_index: int,
    reversal_index: int,
) -> tuple[float, float, int, int]:
    metrics = [
        (float(stats["flip_rate"]), population_flip_range),
        *((float(value), action_share_range) for value in stats["old_action_marginal"]),
        *((float(value), action_share_range) for value in stats["new_action_marginal"]),
    ]
    margins = [
        min((value - lower) / (upper - lower), (upper - value) / (upper - lower))
        for value, (lower, upper) in metrics
    ]
    distance = abs(coupling - coupling_baseline) + abs(reversal - reversal_baseline)
    return min(margins), -distance, -coupling_index, -reversal_index





def direction_change_diagnostics(
    old: dict[str, Any],
    new: dict[str, Any],
    target: str,
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    family = str(old["family"])
    pre_controls = [explain_control_state(old, state) for state in states]
    post_controls = [explain_control_state(new, state) for state in states]
    key = {
        "conjunctive_gating": "binding_gate",
        "disjunctive_sufficiency": "triggering_cue",
        "lexicographic_priority": "decisive_cue",
    }.get(family)
    if key is None:
        pre_share = mean(
            abs(float(row["contributions"][target]))
            / max(
                sum(abs(float(value)) for value in row["contributions"].values()),
                1e-9,
            )
            for row in pre_controls
        )
        post_share = mean(
            abs(float(row["contributions"][target]))
            / max(
                sum(abs(float(value)) for value in row["contributions"].values()),
                1e-9,
            )
            for row in post_controls
        )
        target_values_pre = [
            float(row["contributions"][target]) for row in pre_controls
        ]
        target_values_post = [
            float(row["contributions"][target]) for row in post_controls
        ]
    else:
        pre_share = mean(row[key] == target for row in pre_controls)
        post_share = mean(row[key] == target for row in post_controls)
        target_values_pre = [float(row["margins"][target]) for row in pre_controls]
        target_values_post = [float(row["margins"][target]) for row in post_controls]

    def distribution(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean": mean(values),
            "p05": ordered[int(0.05 * (len(ordered) - 1))],
            "p50": ordered[int(0.50 * (len(ordered) - 1))],
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
        }

    expected_changes = {"field_directions", "thresholds"}
    checks = {
        "family_equal": old.get("family") == new.get("family"),
        "relevant_fields_equal": old.get("relevant_fields")
        == new.get("relevant_fields"),
        "target_direction_reversed": int(old["field_directions"][target])
        == -int(new["field_directions"][target]),
        "other_directions_equal": all(
            old["field_directions"][field] == new["field_directions"][field]
            for field in old["relevant_fields"]
            if field != target
        ),
        "score_calibration_equal": old.get("score_calibration")
        == new.get("score_calibration"),
    }
    magnitude_parameter: dict[str, float] = {}
    if family == "additive_aggregation":
        expected_changes = {"field_directions", "weights"}
        stable = [field for field in old["relevant_fields"] if field != target]
        reference = stable[0]
        stable_scale = abs(float(new["weights"][reference])) / abs(
            float(old["weights"][reference])
        )
        multiplier = (
            abs(float(new["weights"][target]))
            / abs(float(new["weights"][reference]))
        ) / (
            abs(float(old["weights"][target]))
            / abs(float(old["weights"][reference]))
        )
        magnitude_parameter["reversal_magnitude_multiplier"] = multiplier
        checks.update(
            {
                "stable_weight_scaling_equal": all(
                    abs(
                        abs(float(new["weights"][field]))
                        / abs(float(old["weights"][field]))
                        - stable_scale
                    )
                    <= 1e-5
                    for field in stable
                ),
                "reversal_magnitude_positive": multiplier > 0.0,
                "weights_normalized": abs(
                    sum(abs(float(value)) for value in old["weights"].values())
                    - 1.0
                ) <= 1e-5
                and abs(
                    sum(abs(float(value)) for value in new["weights"].values())
                    - 1.0
                ) <= 1e-5,
                "bias_equal": old.get("bias") == new.get("bias"),
            }
        )
    else:
        checks["stable_thresholds_equal"] = all(
            old["thresholds"][field] == new["thresholds"][field]
            for field in old["relevant_fields"]
            if field != target
        )
        if family == "lexicographic_priority":
            checks["priority_order_equal"] = old.get("priority_order") == new.get(
                "priority_order"
            )
            checks["indifference_band_equal"] = old.get(
                "indifference_band"
            ) == new.get("indifference_band")
    changed_top_level = {
        key
        for key in set(old) | set(new)
        if old.get(key) != new.get(key)
    }
    return {
        "target_field": target,
        "pre_direction": int(old["field_directions"][target]),
        "post_direction": int(new["field_directions"][target]),
        "target_control_share": {
            "pre_dynamic": pre_share,
            "post_dynamic": post_share,
        },
        "target_effect_distribution": {
            "pre_dynamic": distribution(target_values_pre),
            "post_dynamic": distribution(target_values_post),
        },
        "magnitude_parameter": magnitude_parameter,
        "checks": checks,
        "changed_top_level_rule_keys": sorted(changed_top_level),
        "expected_changed_top_level_rule_keys": sorted(expected_changes),
        "single_field_structural_diff_pass": (
            "field_directions" in changed_top_level
            and changed_top_level <= expected_changes
            and all(checks.values())
        ),
    }
