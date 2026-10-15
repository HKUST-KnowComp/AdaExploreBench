from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


_INPUT_PROPERTIES = (
    "display_name",
    "description",
    "type",
    "unit",
    "minimum",
    "maximum",
)


def input_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Project assembled fields onto the constraint stage's input contract."""
    return {
        field_id: {
            property_name: deepcopy(metadata[property_name])
            for property_name in _INPUT_PROPERTIES
        }
        for field_id, metadata in fields.items()
    }


def generation(
    renderer: PromptRenderer,
    *,
    template_id: str,
    scenario: dict[str, Any],
    fields: dict[str, Any],
) -> CallSpec:
    field_ids = tuple(fields)
    return CallSpec(
        stage="field_constraints",
        component_id=template_id,
        schema="field_constraints",
        system=renderer.system("relations/field_constraints_system.txt"),
        user=renderer.render(
            "relations/field_constraints_user.j2",
            scenario=scenario,
            fields=fields,
        ),
        enum_bindings={
            "/properties/template_id": (template_id,),
            "/$defs/number_atom/anyOf/0/properties/field": field_ids,
        },
    )
