from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


FAMILIES = (
    "additive_aggregation",
    "conjunctive_gating",
    "disjunctive_sufficiency",
    "lexicographic_priority",
)

ACTION_ANCHORS = (-0.75, -0.25, 0.25, 0.75)
UTILITY_BETA = 8.0

ACTION_COUNT = len(ACTION_ANCHORS)
ACTION_INDEXES = tuple(range(ACTION_COUNT))
ACTION_IDS = tuple(f"action_{index}" for index in ACTION_INDEXES)
ACTION_BOUNDARIES = tuple(
    (left + right) / 2.0 for left, right in zip(ACTION_ANCHORS, ACTION_ANCHORS[1:])
)
# Two sides per decision boundary: boundary-targeted strata divide across these.
BOUNDARY_SIDE_COUNT = 2 * len(ACTION_BOUNDARIES)


def hydrate_rule(
    rule: dict[str, Any], fields: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Attach evaluator constants and selected field bounds to a release rule."""

    hydrated = deepcopy(rule)
    hydrated["action_levels"] = list(ACTION_ANCHORS)
    hydrated["field_ranges"] = {
        field: {
            "minimum": fields[field]["minimum"],
            "maximum": fields[field]["maximum"],
        }
        for field in hydrated["relevant_fields"]
    }
    return hydrated


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _normalize(mechanism: dict[str, Any], field: str, raw: float) -> float:
    bounds = mechanism["field_ranges"][field]
    low, high = float(bounds["minimum"]), float(bounds["maximum"])
    return _clip(2.0 * (float(raw) - low) / (high - low) - 1.0)


def _oriented(mechanism: dict[str, Any], field: str, raw: float) -> float:
    return float(mechanism["field_directions"][field]) * _normalize(mechanism, field, raw)


def _margin(mechanism: dict[str, Any], field: str, values: dict[str, float]) -> float:
    return _oriented(mechanism, field, values[field]) - _oriented(
        mechanism, field, mechanism["thresholds"][field]
    )


def explain_control_state(mechanism: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    family = str(mechanism["family"])
    fields = list(mechanism["relevant_fields"])
    if family == "additive_aggregation":
        contributions = {
            field: float(mechanism["weights"][field]) * _normalize(mechanism, field, values[field])
            for field in fields
        }
        return {
            "family": family,
            "contributions": contributions,
            "bias": float(mechanism["bias"]),
            "raw_evidence": float(mechanism["bias"]) + sum(contributions.values()),
        }

    margins = {field: _margin(mechanism, field, values) for field in fields}
    if family == "conjunctive_gating":
        controller = min(fields, key=lambda field: (margins[field], fields.index(field)))
        return {
            "family": family,
            "margins": margins,
            "binding_gate": controller,
            "raw_evidence": margins[controller],
        }
    if family == "disjunctive_sufficiency":
        controller = max(fields, key=lambda field: (margins[field], -fields.index(field)))
        return {
            "family": family,
            "margins": margins,
            "triggering_cue": controller,
            "satisfied_triggers": [field for field in fields if margins[field] > 0],
            "raw_evidence": margins[controller],
        }
    if family == "lexicographic_priority":
        order = list(mechanism["priority_order"])
        band = float(mechanism["indifference_band"])
        decisive = next((field for field in order if abs(margins[field]) > band), order[-1])
        return {
            "family": family,
            "margins": margins,
            "decisive_cue": decisive,
            "stopping_depth": order.index(decisive) + 1,
            "fallback_used": decisive == order[-1] and all(abs(margins[field]) <= band for field in order[:-1]),
            "raw_evidence": margins[decisive],
        }
    raise ValueError(f"unsupported mechanism family: {family}")


def raw_evidence(mechanism: dict[str, Any], values: dict[str, float]) -> float:
    return float(explain_control_state(mechanism, values)["raw_evidence"])


def intervention_score(mechanism: dict[str, Any], values: dict[str, float]) -> float:
    evidence = raw_evidence(mechanism, values)
    calibration = mechanism.get("score_calibration") or {}
    if calibration.get("method") == "frozen_pre_quartile_piecewise":
        knots = [float(knot) for knot in calibration["evidence_knots"]]
        targets = [float(target) for target in calibration["score_knots"]]
        if evidence <= knots[1]:
            slope = (targets[1] - targets[0]) / (knots[1] - knots[0])
            return _clip(targets[0] + (evidence - knots[0]) * slope)
        slope = (targets[2] - targets[1]) / (knots[2] - knots[1])
        return _clip(targets[1] + (evidence - knots[1]) * slope)
    negative_anchor = abs(float(calibration.get("negative_anchor", -1.0)))
    positive_anchor = float(calibration.get("positive_anchor", 1.0))
    if evidence < 0:
        return max(-1.0, evidence / max(negative_anchor, 1e-9))
    if evidence > 0:
        return min(1.0, evidence / max(positive_anchor, 1e-9))
    return 0.0


def utilities_from_score(score: float) -> list[float]:
    logits = [-UTILITY_BETA * (float(score) - anchor) ** 2 for anchor in ACTION_ANCHORS]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [
        (ACTION_COUNT * weight / total - 1.0) / (ACTION_COUNT - 1.0)
        for weight in weights
    ]


def evaluate_utilities(mechanism: dict[str, Any], values: dict[str, float]) -> list[float]:
    return utilities_from_score(intervention_score(mechanism, values))


def best_action(mechanism: dict[str, Any], values: dict[str, float]) -> int:
    utilities = evaluate_utilities(mechanism, values)
    return max(range(len(utilities)), key=utilities.__getitem__)


def validate_mechanism(
    mechanism: dict[str, Any], field_schema: dict[str, dict[str, Any]] | None = None
) -> None:
    family = mechanism.get("family")
    fields = list(mechanism.get("relevant_fields") or [])
    if family not in FAMILIES:
        raise ValueError("mechanism has invalid family")
    if list(mechanism.get("action_levels") or []) != list(ACTION_ANCHORS):
        raise ValueError("AdaExploreBench supports exactly four ordered actions")
    if len(fields) < 2 or len(fields) != len(set(fields)):
        raise ValueError("mechanism requires distinct relevant fields")
    if field_schema is not None and any(field not in field_schema for field in fields):
        raise ValueError("mechanism references an unknown field")
    if family == "additive_aggregation":
        weights = mechanism.get("weights") or {}
        if set(weights) != set(fields) or any(abs(float(weights[field])) <= 0 for field in fields):
            raise ValueError("additive mechanism requires one non-zero weight per field")
        directions = mechanism.get("field_directions") or {}
        if any(
            (1 if float(weights[field]) > 0 else -1)
            != int(directions.get(field, 0))
            for field in fields
        ):
            raise ValueError("additive weight signs must match field directions")
        shares = [abs(float(weights[field])) / sum(abs(float(value)) for value in weights.values()) for field in fields]
        if max(shares) > 0.8 + 1e-9:
            raise ValueError("additive mechanism is dominated by one field")
    else:
        if set(mechanism.get("thresholds") or {}) != set(fields):
            raise ValueError("threshold mechanism requires one threshold per field")
    if family == "lexicographic_priority":
        order = list(mechanism.get("priority_order") or [])
        if len(fields) < 3 or set(order) != set(fields) or len(order) != len(fields):
            raise ValueError("lexicographic mechanism requires a complete distinct priority order")
        if not 0.05 <= float(mechanism.get("indifference_band", 0)) <= 0.75:
            raise ValueError("lexicographic indifference band is out of range")
