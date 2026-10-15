from __future__ import annotations

from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


def generation(renderer: PromptRenderer, *, template_id: str, scenario: dict[str, Any], fields: dict[str, Any]) -> CallSpec:
    field_ids = tuple(fields)
    return CallSpec(
        stage="replacements", component_id=template_id, schema="replacements",
        system=renderer.system("dynamics/replacements_system.txt"),
        user=renderer.render("dynamics/replacements_user.j2", scenario=scenario, fields=fields),
        enum_bindings={
            "/properties/template_id": (template_id,),
            "/properties/replacements/items/properties/members/items": field_ids,
            "/properties/replacements/items/properties/authority_by_field/items/properties/field_id": field_ids,
            "/properties/replacements/items/properties/transitions/items/properties/outgoing_field": field_ids,
            "/properties/replacements/items/properties/transitions/items/properties/incoming_field": field_ids,
        },
    )
