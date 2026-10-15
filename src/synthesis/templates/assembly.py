from __future__ import annotations

from copy import deepcopy
from typing import Any


def _by_id(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        identifier = str(item[key])
        if identifier in output:
            raise ValueError(f"duplicate {key}: {identifier}")
        output[identifier] = {name: deepcopy(value) for name, value in item.items() if name != key}
    return output


def merge_field_index(
    catalog: dict[str, Any], semantics: dict[str, Any]
) -> list[dict[str, Any]]:
    semantic_map = _by_id(semantics["fields"], "field_id")
    return [
        {**deepcopy(card), **deepcopy(semantic_map[card["field_id"]])}
        for card in catalog["fields"]
    ]


def assemble_fields(
    catalog: dict[str, Any], semantics: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    semantic_map = _by_id(semantics["fields"], "field_id")
    parameter_map = _by_id(parameters["fields"], "field_id")
    fields: dict[str, Any] = {}
    for card in catalog["fields"]:
        field_id = card["field_id"]
        semantic = semantic_map[field_id]
        params = deepcopy(parameter_map[field_id])
        params.pop("parameter_basis", None)
        fields[field_id] = {
            "type": card["type"],
            "minimum": params["minimum"],
            "maximum": params["maximum"],
            "typical_range": params["typical_range"],
            "display_name": card["display_name"],
            "description": card["description"],
            "unit": card["unit"],
            "pressure_direction": card["pressure_direction"],
            "is_reversal": semantic["is_reversal"],
            "effect_strength": semantic["effect_strength"],
            "eligible_families": semantic["eligible_families"],
            "profile_beta": params["profile_beta"],
        }
    return fields


def assemble_template(
    core: dict[str, Any], catalog: dict[str, Any], semantics: dict[str, Any],
    parameters: dict[str, Any],
    constraints: dict[str, Any], lexicographic_orders: dict[str, Any],
    replacements: dict[str, Any],
) -> dict[str, Any]:
    template = {key: deepcopy(value) for key, value in core.items() if key != "template_id"}
    template["fields"] = assemble_fields(catalog, semantics, parameters)
    lexicographic_field_ids = {
        field_id
        for order in lexicographic_orders["approved_priority_orders"]
        for field_id in order["fields"]
    }
    for field_id in lexicographic_field_ids:
        eligible_families = template["fields"][field_id]["eligible_families"]
        if "lexicographic_priority" not in eligible_families:
            eligible_families.append("lexicographic_priority")
    if constraints["feasibility_constraints"]:
        template["feasibility_constraints"] = [
            {
                "id": item["id"],
                "description": item["description"],
                "expr": deepcopy(item["expr"]),
            }
            for item in constraints["feasibility_constraints"]
        ]
    template["rule_families"]["lexicographic_priority"]["approved_priority_orders"] = [
        {key: deepcopy(value) for key, value in order.items() if key != "order_id"}
        for order in lexicographic_orders["approved_priority_orders"]
    ]
    replacement_map = _by_id(replacements["replacements"], "group_id")
    for group in replacement_map.values():
        group["joint_sampling"] = {"coupling_strength": group.pop("coupling_strength")}
        group["authority_by_field"] = {
            item["field_id"]: item["authority"] for item in group["authority_by_field"]
        }
        group["transitions"] = _by_id(group["transitions"], "transition_id")
    direction_reversals = {
        item["field_id"]: {
            key: item[key]
            for key in ("trigger", "semantics_before", "semantics_after", "reversal_reason")
        }
        for item in semantics["fields"]
        if item["is_reversal"]
    }
    template["dynamics"] = {
        "replacements": replacement_map,
        "direction_reversals": direction_reversals,
    }
    return template


def assemble_pool(skeleton: dict[str, Any], domains: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = deepcopy(skeleton)
    output["domains"] = deepcopy(domains)
    return output
