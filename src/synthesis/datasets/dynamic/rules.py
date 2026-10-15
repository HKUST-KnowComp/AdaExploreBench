"""Seeded pre/post rule construction for dynamic episodes."""

from __future__ import annotations

import math
import random
from typing import Any

from ..rules import base_rule, effect_strength, rule_parameters
from ..template_pool import TemplateSelection


REPLACEMENT_PARAMETERIZATION = "jittered_role_seeded_v3"
COMPOUND_PARAMETERIZATION = "joint_replacement_reversal_v3"


def _normalized_weights(
    selection: TemplateSelection,
    fields: list[str],
    magnitudes: dict[str, float],
    directions: dict[str, int] | None = None,
) -> dict[str, float]:
    total = sum(float(magnitudes[field]) for field in fields)
    if total <= 0:
        raise ValueError("additive rule requires positive raw strengths")
    return {
        field: round(
            int(
                (directions or {}).get(
                    field, selection.fields[field]["pressure_direction"]
                )
            )
            * float(magnitudes[field])
            / total,
            6,
        )
        for field in fields
    }


def _replacement_strength(
    selection: TemplateSelection, outgoing: str, incoming: str
) -> float:
    return math.sqrt(
        effect_strength(selection, outgoing)
        * effect_strength(selection, incoming)
    )


def _raw_quantile_from_pressure(direction: int, pressure_quantile: float) -> float:
    return pressure_quantile if direction > 0 else 1.0 - pressure_quantile


def _threshold_at_pressure(
    selection: TemplateSelection, field: str, pressure_quantile: float
) -> int | float:
    direction = int(selection.fields[field]["pressure_direction"])
    meta = selection.fields[field]
    quantile = _raw_quantile_from_pressure(direction, pressure_quantile)
    return float(meta["minimum"]) + quantile * (
        float(meta["maximum"]) - float(meta["minimum"])
    )


def _shift_reversed_threshold(
    selection: TemplateSelection,
    field: str,
    pre_threshold: int | float,
    pressure_shift: float,
) -> int | float:
    meta = selection.fields[field]
    low = float(meta["minimum"])
    span = float(meta["maximum"]) - low
    raw_quantile = (float(pre_threshold) - low) / span
    post_direction = -int(meta["pressure_direction"])
    symmetric_post_pressure = (
        raw_quantile if post_direction > 0 else 1.0 - raw_quantile
    )
    post_pressure = max(0.0, min(1.0, symmetric_post_pressure + pressure_shift))
    post_raw_quantile = _raw_quantile_from_pressure(post_direction, post_pressure)
    return low + post_raw_quantile * span


def _reversal_controls(
    family: str, control: float
) -> tuple[float, float]:
    if family == "additive_aggregation":
        return float(control), 0.0
    return 1.0, float(control)


def sample_rule_pair(
    selection: TemplateSelection,
    structure: dict[str, Any],
    family: str,
    seed: int,
    dynamic_type: str,
    *,
    reversal_control: float,
    replacement_role_strength_jitter_range: tuple[float, float],
    replacement_threshold_quantile_jitter_range: tuple[float, float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Instantiate one seeded rule pair; metrics never influence its baseline draw."""

    if dynamic_type not in {
        "field_replacement",
        "direction_reversal",
        "compound",
    }:
        raise ValueError(f"unsupported dynamic type: {dynamic_type}")
    rng = random.Random(seed)
    has_replacement = dynamic_type in {"field_replacement", "compound"}
    has_reversal = dynamic_type in {"direction_reversal", "compound"}
    shared = list(structure["shared_fields"])
    if has_replacement:
        outgoing = str(structure["outgoing_field"])
        incoming = str(structure["incoming_field"])
        old_fields = [*shared, outgoing]
        new_fields = [*shared, incoming]
    else:
        target = str(structure["target_field"])
        old_fields = [*shared, target]
        new_fields = list(old_fields)
    reversal_target = (
        str(structure["reversal_target"])
        if dynamic_type == "compound"
        else str(structure["target_field"])
        if has_reversal
        else None
    )
    old = base_rule(family, old_fields, selection)
    new = base_rule(family, new_fields, selection)
    multiplier, threshold_shift = _reversal_controls(family, reversal_control)

    parameters: dict[str, Any] = {"rule_seed": seed}
    if family == "additive_aggregation":
        old_magnitudes = {
            field: effect_strength(selection, field) for field in old_fields
        }
        new_magnitudes = {
            field: effect_strength(selection, field) for field in new_fields
        }
        if has_replacement:
            role_strength = _replacement_strength(selection, outgoing, incoming)
            jitter = rng.triangular(
                *replacement_role_strength_jitter_range, 0.0
            )
            old_magnitudes[outgoing] = role_strength
            new_magnitudes[incoming] = role_strength * (1.0 + jitter)
            parameters["replacement_role_strength"] = round(role_strength, 6)
            parameters["replacement_role_strength_relative_jitter"] = round(
                jitter, 6
            )
        new_directions = {
            field: int(selection.fields[field]["pressure_direction"])
            for field in new_fields
        }
        if reversal_target is not None:
            new_magnitudes[reversal_target] *= multiplier
            new_directions[reversal_target] *= -1
            parameters["reversal_magnitude_multiplier"] = round(multiplier, 6)
        old["weights"] = _normalized_weights(
            selection, old_fields, old_magnitudes
        )
        new["weights"] = _normalized_weights(
            selection, new_fields, new_magnitudes, new_directions
        )
        new["field_directions"] = new_directions
        low, high = map(float, rule_parameters(selection, family)["bias_range"])
        bias = round(rng.uniform(low, high), 6)
        old["bias"] = new["bias"] = bias
        parameters["bias"] = bias
    else:
        low, high = map(
            float, rule_parameters(selection, family)["threshold_quantiles"]
        )
        pressure_quantiles: dict[str, float] = {}
        for field in shared:
            pressure_quantiles[field] = rng.uniform(low, high)
        if has_replacement:
            role_quantile = rng.uniform(low, high)
            threshold_jitter = rng.triangular(
                *replacement_threshold_quantile_jitter_range, 0.0
            )
            pressure_quantiles[outgoing] = role_quantile
            pressure_quantiles[incoming] = max(
                0.0, min(1.0, role_quantile + threshold_jitter)
            )
            parameters["replacement_role_pressure_quantile"] = round(
                role_quantile, 6
            )
            parameters["replacement_threshold_quantile_jitter"] = round(
                pressure_quantiles[incoming] - role_quantile, 6
            )
        else:
            pressure_quantiles[str(structure["target_field"])] = rng.uniform(
                low, high
            )
        all_fields = list(dict.fromkeys([*old_fields, *new_fields]))
        thresholds = {
            field: _threshold_at_pressure(
                selection, field, pressure_quantiles[field]
            )
            for field in all_fields
        }
        old["thresholds"] = {
            field: thresholds[field] for field in old_fields
        }
        new["thresholds"] = {
            field: thresholds[field] for field in new_fields
        }
        if reversal_target is not None:
            new["field_directions"][reversal_target] *= -1
            new["thresholds"][reversal_target] = _shift_reversed_threshold(
                selection,
                reversal_target,
                old["thresholds"][reversal_target],
                threshold_shift,
            )
            parameters["reversal_threshold_pressure_shift"] = round(
                threshold_shift, 6
            )
        parameters["threshold_pressure_quantiles"] = {
            field: round(value, 6)
            for field, value in pressure_quantiles.items()
        }
        if family == "lexicographic_priority":
            band_low, band_high = map(
                float,
                rule_parameters(selection, family)[
                    "indifference_band_range"
                ],
            )
            band = round(rng.uniform(band_low, band_high), 6)
            if has_replacement:
                old["priority_order"] = list(structure["old_priority_order"])
                new["priority_order"] = list(structure["new_priority_order"])
            else:
                old["priority_order"] = list(structure["priority_order"])
                new["priority_order"] = list(structure["priority_order"])
            old["indifference_band"] = new["indifference_band"] = band
            parameters["indifference_band"] = band

    if dynamic_type == "field_replacement":
        parameters["parameterization"] = REPLACEMENT_PARAMETERIZATION
    elif dynamic_type == "compound":
        parameters["parameterization"] = COMPOUND_PARAMETERIZATION
    else:
        parameters["parameterization"] = "seeded_reversal_magnitude_v2"
    return old, new, parameters
