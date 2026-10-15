from __future__ import annotations

from typing import Any

from ..prompting import PromptRenderer
from ..structured_call import CallSpec


def family_descriptions(families: dict[str, Any]) -> dict[str, str]:
    return {
        family: str(value["description"])
        for family, value in families.items()
    }


def generation(
    renderer: PromptRenderer,
    *,
    domain_id: str,
    domain_description: str,
    template_id: str,
    benchmark_boundary: str,
    family_meanings: dict[str, Any],
    blueprint: dict[str, Any] | None,
    existing_templates: list[dict[str, Any]],
) -> CallSpec:
    return CallSpec(
        stage="scenario_core",
        component_id=template_id,
        schema=("scenario_expansion" if blueprint is not None else "scenario_generation"),
        system=renderer.system("scenarios/system.txt"),
        user=renderer.render(
            "scenarios/user.j2",
            domain_id=domain_id,
            domain_description=domain_description,
            benchmark_boundary=benchmark_boundary,
            family_meanings=family_descriptions(family_meanings),
            blueprint=blueprint,
            existing_templates=existing_templates,
        ),
        enum_bindings={},
    )


def assemble(
    value: dict[str, Any],
    *,
    domain_id: str,
    domain_description: str,
    template_id: str,
    task_id: str,
    blueprint: dict[str, Any] | None,
) -> dict[str, Any]:
    fixed = blueprint or value
    core = {
        "template_id": template_id,
        "task_id": task_id,
        "description": fixed["description"],
        "agent_role": fixed["agent_role"],
        "actions": fixed["actions"],
        "operational_profiles": value["operational_profiles"],
        "rule_families": value["rule_families"],
    }
    return {
        "domain_id": domain_id,
        "domain_description": domain_description,
        "templates": [core],
    }


def review(renderer: PromptRenderer, *, template_id: str) -> CallSpec:
    """Describe the assembled single-core contract used by audit and repair."""
    return CallSpec(
        stage="scenario_cores",
        component_id=template_id,
        schema="scenario_cores",
        system=renderer.system("scenarios/system.txt"),
        user="",
        enum_bindings={
            "/properties/templates/items/properties/template_id": (template_id,),
        },
    )
