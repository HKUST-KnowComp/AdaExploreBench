from __future__ import annotations

from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


def generation(
    renderer: PromptRenderer,
    *,
    template_id: str,
    scenario: dict[str, Any],
    fields: dict[str, Any],
) -> CallSpec:
    candidate_ids = tuple(fields)
    return CallSpec(
        stage="lexicographic_orders",
        component_id=template_id,
        schema="lexicographic_orders",
        system=renderer.system("relations/lexicographic_orders_system.txt"),
        user=renderer.render(
            "relations/lexicographic_orders_user.j2",
            scenario=scenario,
            fields=fields,
            candidate_field_ids=candidate_ids,
        ),
        enum_bindings={
            "/properties/template_id": (template_id,),
            "/properties/approved_priority_orders/items/properties/fields/items": candidate_ids,
        },
    )
