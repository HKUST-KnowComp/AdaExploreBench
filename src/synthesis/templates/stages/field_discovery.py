from __future__ import annotations

from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


def generation(
    renderer: PromptRenderer, *, template_id: str, scenario: dict[str, Any]
) -> CallSpec:
    return CallSpec(
        stage="field_catalog", component_id=template_id, schema="field_catalog",
        system=renderer.system("fields/discover_system.txt"),
        user=renderer.render("fields/discover_user.j2", scenario=scenario),
        enum_bindings={"/properties/template_id": (template_id,)},
    )
