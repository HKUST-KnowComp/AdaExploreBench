"""Seeded rule instantiation for static episodes."""

from __future__ import annotations

import random
from typing import Any

from ..rules import base_rule, effect_strength, raw_at_quantile, rule_parameters
from ..template_pool import TemplateSelection


def sample_rule(
    selection: TemplateSelection,
    structure: dict[str, Any],
    family: str,
    seed: int,
) -> dict[str, Any]:
    """Draw one reproducible rule from the template prior, without metric feedback."""

    relevant = list(structure["relevant_fields"])
    rng = random.Random(seed)
    rule = base_rule(family, relevant, selection)
    if family == "additive_aggregation":
        magnitudes = {field: effect_strength(selection, field) for field in relevant}
        total = sum(magnitudes.values())
        rule["weights"] = {
            field: round(
                int(selection.fields[field]["pressure_direction"])
                * magnitudes[field]
                / total,
                6,
            )
            for field in relevant
        }
        low, high = map(float, rule_parameters(selection, family)["bias_range"])
        rule["bias"] = round(rng.uniform(low, high), 6)
        return rule

    low, high = map(float, rule_parameters(selection, family)["threshold_quantiles"])
    rule["thresholds"] = {}
    for field in relevant:
        pressure_quantile = rng.uniform(low, high)
        direction = int(selection.fields[field]["pressure_direction"])
        raw_quantile = pressure_quantile if direction > 0 else 1.0 - pressure_quantile
        rule["thresholds"][field] = raw_at_quantile(selection.fields[field], raw_quantile)
    if family == "lexicographic_priority":
        band_low, band_high = map(
            float, rule_parameters(selection, family)["indifference_band_range"]
        )
        rule["priority_order"] = list(structure["priority_order"])
        rule["indifference_band"] = round(rng.uniform(band_low, band_high), 6)
    return rule
