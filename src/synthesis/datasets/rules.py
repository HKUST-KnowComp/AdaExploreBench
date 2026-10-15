"""Rule metadata helpers shared by static and dynamic generation."""

from __future__ import annotations

from typing import Any

from protocol.mechanism import ACTION_ANCHORS
from .template_pool import TemplateSelection, family_parameters


def base_rule(
    family: str,
    relevant: list[str],
    selection: TemplateSelection,
) -> dict[str, Any]:
    fields = selection.fields
    return {
        "family": family,
        "relevant_fields": list(relevant),
        "action_levels": list(ACTION_ANCHORS),
        "field_ranges": {
            field: {
                "minimum": fields[field]["minimum"],
                "maximum": fields[field]["maximum"],
            }
            for field in relevant
        },
        "field_directions": {
            field: int(fields[field]["pressure_direction"]) for field in relevant
        },
        "domain_template_id": (
            f"{selection.domain}.{selection.template_id}.{family}"
        ),
        "domain_rationale": str(
            selection.template["rule_families"][family]["description"]
        ),
    }


def raw_at_quantile(meta: dict[str, Any], quantile: float) -> int | float:
    raw = float(meta["minimum"]) + float(quantile) * (
        float(meta["maximum"]) - float(meta["minimum"])
    )
    return int(round(raw)) if meta["type"] == "integer" else round(raw, 2)


def effect_strength(selection: TemplateSelection, field: str) -> float:
    return float(selection.fields[field]["effect_strength"])


def rule_parameters(selection: TemplateSelection, family: str) -> dict[str, Any]:
    return family_parameters(selection, family)
