from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from protocol.constraints import compile_core_constraints, constraint_violations
from synthesis.datasets.config import load_config
from synthesis.datasets.dynamic.structures import DynamicCoreSpace
from synthesis.datasets.planning import stable_seed
from synthesis.datasets.population import (
    PopulationBudgetExhausted,
    PopulationPool,
    prepare_base_proposals,
    sample_states,
)
from synthesis.datasets.template_pool import TemplateSelection, load_template_pool, select_template


def _selection() -> TemplateSelection:
    pool = load_template_pool(Path("tests/fixtures/domain_templates_four_action.json"))
    domain = next(iter(pool["domains"]))
    template_id = next(iter(pool["domains"][domain]["templates"]))
    source = select_template(pool, domain, template_id)
    template = deepcopy(source.template)
    fields = list(template["fields"])[:3]
    template["fields"] = {field: template["fields"][field] for field in fields}
    template["feasibility_constraints"] = []
    return TemplateSelection(source.pool, source.domain, source.template_id, template)


def _comparison(left: str, right: str) -> dict:
    return {
        "id": "ordered",
        "description": "left does not exceed right",
        "expr": {"op": "le", "left": {"field": left}, "right": {"field": right}},
    }


def _compiled(selection: TemplateSelection, fields: list[str], constraints: list[dict]) -> dict:
    selection.template["feasibility_constraints"] = constraints
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    return compile_core_constraints(
        selection,
        fields,
        saved_decimal_places=config.search.saved_decimal_places,
        solver_timeout_ms=config.search.solver_timeout_ms,
        conditional_branch_limit=config.search.conditional_branch_limit,
    )


def test_affine_two_field_constraint_uses_direct_conditional_sampling():
    selection = _selection()
    fields = list(selection.fields)[:2]
    compiled = _compiled(selection, fields, [_comparison(*fields)])
    assert compiled["sampling"]["kind"] == "conditional"
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    base = prepare_base_proposals(selection, fields, 7)
    pool = PopulationPool()
    states = sample_states(
        selection, fields, base, compiled,
        replacement_group_id=None, effective_coupling=None,
        config=config.search, count=128, pool=pool,
    )
    assert pool.raw_draw_count == 128
    assert not any(constraint_violations(compiled["constraints"], state) for state in states)


def test_three_field_constraint_is_classified_as_bounded_fallback():
    selection = _selection()
    fields = list(selection.fields)
    constraint = {
        "id": "three",
        "description": "three field sum is bounded",
        "expr": {
            "op": "le",
            "left": {"op": "add", "args": [{"field": field} for field in fields]},
            "right": {"const": sum(float(selection.fields[field]["maximum"]) for field in fields)},
        },
    }
    compiled = _compiled(selection, fields, [constraint])
    assert compiled["sampling"]["kind"] == "fallback"
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    pool = PopulationPool()
    states = sample_states(
        selection, fields, prepare_base_proposals(selection, fields, 9), compiled,
        replacement_group_id=None, effective_coupling=None,
        config=config.search, count=32, pool=pool,
    )
    assert len(states) == 32
    assert pool.raw_draw_count == 32


def test_fallback_stops_at_exact_forty_times_requested_increment():
    selection = _selection()
    field = next(iter(selection.fields))
    impossible = {
        "id": "none",
        "description": "outside the authored range",
        "expr": {"op": "gt", "left": {"field": field}, "right": {"const": selection.fields[field]["maximum"]}},
    }
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    pool = PopulationPool()
    with pytest.raises(PopulationBudgetExhausted):
        sample_states(
            selection, [field], prepare_base_proposals(selection, [field], 13),
            {"constraints": [impossible], "sampling": {"kind": "fallback"}},
            replacement_group_id=None, effective_coupling=None,
            config=config.search, count=5, pool=pool,
        )
    assert pool.raw_draw_count == 5 * 40


def test_evaluation_pool_is_extended_without_changing_its_prefix():
    selection = _selection()
    fields = list(selection.fields)[:2]
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    compiled = _compiled(selection, fields, [])
    base = prepare_base_proposals(selection, fields, 17)
    pool = PopulationPool()
    first = sample_states(
        selection, fields, base, compiled,
        replacement_group_id=None, effective_coupling=None,
        config=config.search, count=16, pool=pool,
    )
    prefix = deepcopy(first)
    extended = sample_states(
        selection, fields, base, compiled,
        replacement_group_id=None, effective_coupling=None,
        config=config.search, count=64, pool=pool,
    )
    assert extended[:16] == prefix
    assert len(pool.records) == 64


def test_common_base_proposals_are_reused_by_separate_coupling_pools():
    pool_data = load_template_pool(Path("datasets/templates/domain_templates.json"))
    selection = select_template(pool_data, "finance", next(iter(pool_data["domains"]["finance"]["templates"])))
    group_id, group = next(iter(selection.template["dynamics"]["replacements"].items()))
    fields = list(group["members"])
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    compiled = _compiled(selection, fields, [])
    base = prepare_base_proposals(selection, fields, 23)
    low_pool, high_pool = PopulationPool(), PopulationPool()
    low = sample_states(selection, fields, base, compiled, replacement_group_id=group_id, effective_coupling=0.1, config=config.search, count=64, pool=low_pool)
    high = sample_states(selection, fields, base, compiled, replacement_group_id=group_id, effective_coupling=0.9, config=config.search, count=64, pool=high_pool)
    assert len(base.uniforms) == 64
    assert low_pool is not high_pool
    assert low != high


def test_conditional_lookup_cannot_escape_selected_interval():
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    pool = load_template_pool(Path("datasets/templates/domain_templates.json"))
    selection = select_template(pool, "finance", "finance_margin_control")
    family = "disjunctive_sufficiency"
    core = DynamicCoreSpace(selection, family, "compound").at(16504)
    fields = list(dict.fromkeys([
        *core["shared_fields"], core["outgoing_field"], core["incoming_field"]
    ]))
    compiled = _compiled(
        selection, fields, selection.template["feasibility_constraints"]
    )
    cell = ("finance", selection.template_id, family, "dynamic", "compound")
    numeric_seed = stable_seed(
        config.dataset.seed, *cell, core["core_space_index"], "numeric"
    )
    states = sample_states(
        selection,
        fields,
        prepare_base_proposals(
            selection, fields, stable_seed(numeric_seed, "evaluation")
        ),
        compiled,
        replacement_group_id=core["group_id"],
        effective_coupling=0.1,
        config=config.search,
        count=config.search.evaluation_sample_count,
    )
    assert not any(
        constraint_violations(compiled["constraints"], state) for state in states
    )
