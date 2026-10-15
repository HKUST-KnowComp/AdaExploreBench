from __future__ import annotations

from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


def generation(
    renderer: PromptRenderer, *, template_id: str, batch_id: str,
    scenario: dict[str, Any], field_index: list[dict[str, Any]], field_ids: list[str],
    defaults: dict[str, Any],
) -> CallSpec:
    return CallSpec(
        stage="field_parameters", component_id=f"{template_id}/{batch_id}", schema="field_parameters",
        system=renderer.system("fields/parameters_system.txt"),
        user=renderer.render(
            "fields/parameters_user.j2", scenario=scenario, field_index=field_index,
            field_ids=field_ids, defaults=defaults,
        ),
        enum_bindings={
            "/properties/template_id": (template_id,),
            "/properties/fields/items/properties/field_id": tuple(field_ids),
        },
    )
