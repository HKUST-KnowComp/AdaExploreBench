from __future__ import annotations

from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


AUDIT_PROMPTS = {
    "scenario_cores": "scenarios/audit.j2",
    "field_catalog": "fields/discover_audit.j2",
    "field_semantics": "fields/semantics_audit.j2",
    "field_parameters": "fields/parameters_audit.j2",
    "field_constraints": "relations/field_constraints_audit.j2",
    "lexicographic_orders": "relations/lexicographic_orders_audit.j2",
    "replacements": "dynamics/replacements_audit.j2",
}


def semantic(
    renderer: PromptRenderer, *, stage: str, component_id: str, component: Any,
    context: dict[str, Any],
) -> CallSpec:
    if stage == "field_catalog":
        schema = "field_catalog_local_audit"
    elif stage == "field_semantics":
        schema = "field_semantics_audit"
    else:
        schema = "semantic_audit"
    return CallSpec(
        stage=f"{stage}_audit", component_id=component_id, schema=schema, mode="audit",
        system=renderer.system("audit/system.txt"),
        user=renderer.render(AUDIT_PROMPTS[stage], component=component, **context),
    )


def field_catalog_set(
    renderer: PromptRenderer, *, component_id: str, component: dict[str, Any],
    scenario: dict[str, Any],
) -> CallSpec:
    return CallSpec(
        stage="field_catalog_set_audit",
        component_id=component_id,
        schema="field_catalog_set_audit",
        mode="audit",
        system=renderer.system("audit/system.txt"),
        user=renderer.render(
            "fields/discover_set_audit.j2",
            scenario=scenario,
            component=component,
        ),
        enum_bindings={
            "/properties/issues/items/properties/field_ids/items": tuple(
                field["field_id"] for field in component["fields"]
            ),
            "/properties/issues/items/properties/drop_field_ids/items": tuple(
                field["field_id"] for field in component["fields"]
            ),
        },
    )


def domain(
    renderer: PromptRenderer, *, domain_id: str,
    accepted_templates: list[dict[str, Any]], candidate: dict[str, Any],
) -> CallSpec:
    return CallSpec(
        stage="domain_diversity_audit", component_id=candidate["template_id"], schema="overall_audit", mode="audit",
        system=renderer.system("audit/system.txt"),
        user=renderer.render(
            "audit/domain_diversity.j2", domain_id=domain_id,
            accepted_templates=accepted_templates, candidate=candidate,
        ),
        enum_bindings={
            "/properties/issues/items/properties/component_id": (
                candidate["template_id"],
            ),
        },
    )
