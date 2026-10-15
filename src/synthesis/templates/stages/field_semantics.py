from __future__ import annotations

from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


SEMANTIC_FAMILIES = (
    "additive_aggregation",
    "conjunctive_gating",
    "disjunctive_sufficiency",
)


def supported_family_meanings(family_meanings: dict[str, Any]) -> dict[str, Any]:
    return {
        family: {"description": family_meanings[family]["description"]}
        for family in SEMANTIC_FAMILIES
    }


def generation(
    renderer: PromptRenderer,
    *,
    template_id: str,
    scenario: dict[str, Any],
    field_catalog: list[dict[str, Any]],
    family_meanings: dict[str, Any],
) -> CallSpec:
    field_ids = tuple(field["field_id"] for field in field_catalog)
    return CallSpec(
        stage="field_semantics",
        component_id=template_id,
        schema="field_semantics",
        system=renderer.system("fields/semantics_system.txt"),
        user=renderer.render(
            "fields/semantics_user.j2",
            scenario=scenario,
            field_catalog=field_catalog,
            family_meanings=supported_family_meanings(family_meanings),
        ),
        enum_bindings={
            "/properties/template_id": (template_id,),
            "/properties/fields/items/properties/field_id": field_ids,
        },
    )
