from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest

from protocol.mechanism import FAMILIES
from synthesis.datasets.config import load_config
from synthesis.datasets.dynamic.rules import sample_rule_pair
from synthesis.datasets.dynamic.structures import (
    DynamicCoreSpace,
    distractor_sources,
    materialize_distractors,
    visible_fields,
)
from synthesis.datasets.evaluation import compound_invariant_diagnostics
from synthesis.datasets.planning import BoundedPermutation, generation_cells, quota_targets
from synthesis.datasets.semantic_review import review_candidate_batch
from synthesis.datasets.static.structures import StaticCoreSpace
from synthesis.datasets.template_pool import (
    TemplateValidationError,
    domain_ids,
    load_template_pool,
    select_template,
    template_hash,
)

TEMPLATE = Path("tests/fixtures/domain_templates_four_action.json")
EIGHT_FIELD_TEMPLATE = Path("datasets/templates/domain_templates.json")


def test_four_action_fixture_pool_validates():
    pool = load_template_pool(TEMPLATE)
    assert len(pool["domains"]) == 4
    assert all(
        len(template["actions"]) == 4
        for domain in pool["domains"].values()
        for template in domain["templates"].values()
    )


def test_static_and_dynamic_configs_are_independent_and_explicit():
    dynamic = load_config(Path("configs/synthesis/dynamic.yaml"))
    static = load_config(Path("configs/synthesis/static.yaml"))
    assert dynamic.dataset.mode == "dynamic" and dynamic.static is None
    assert static.dataset.mode == "static" and static.dynamic is None
    assert dynamic.dynamic.enabled_types == ("compound",)
    assert dynamic.search.q_flip_targets == (0.25, 0.5, 0.75)
    assert len(dynamic.search.coupling_grid) == 9
    assert dynamic.resources.max_core_visits_per_cell == 999999
    assert dynamic.search.fallback_raw_draw_multiplier == 40
    assert dynamic.semantic_review.batch_size == 12
    assert dynamic.semantic_review.request_attempt_limit == 3
    assert dynamic.semantic_review.max_distractor_trials_per_core == 3
    assert dynamic.episode_layout.case_counts_per_batch == {"iid": 8}
    assert static.static.visible_field_count == 8


def test_published_template_manifest_rejects_direct_canonical_edits(tmp_path):
    pool = json.loads(TEMPLATE.read_text())
    audit = {
        "schema_version": "adaexplorebench_template_synthesis_audit_v2",
        "canonical_sha256": template_hash(pool),
        "accepted": True,
        "components": {},
    }
    manifest = {
        "schema_version": "adaexplorebench_template_synthesis_manifest_v2",
        "canonical_sha256": template_hash(pool),
        "audit_sha256": template_hash(audit),
    }
    template_path = tmp_path / "domain_templates.json"
    audit_path = tmp_path / "domain_templates.audit.json"
    manifest_path = tmp_path / "domain_templates.manifest.json"
    template_path.write_text(json.dumps(pool))
    audit_path.write_text(json.dumps(audit))
    manifest_path.write_text(json.dumps(manifest))
    load_template_pool(template_path, manifest_path=manifest_path, audit_path=audit_path)
    first_domain = next(iter(pool["domains"].values()))
    next(iter(first_domain["templates"].values()))["description"] += " changed"
    template_path.write_text(json.dumps(pool))
    with pytest.raises(TemplateValidationError, match="published manifest"):
        load_template_pool(template_path, manifest_path=manifest_path, audit_path=audit_path)


def test_generation_cells_and_group_quotas_are_reproducible_and_balanced():
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    pool = load_template_pool(TEMPLATE)
    first = generation_cells(config, pool)
    assert first == generation_cells(config, pool)
    targets = quota_targets(config, first)
    assert all(target == 24 for target in targets.values())
    for group, target in targets.items():
        members = [cell for cell in first if cell.group_key == group]
        assert sum(cell.initial_quota for cell in members) == target
        assert max(cell.initial_quota for cell in members) - min(cell.initial_quota for cell in members) <= 1


def test_static_core_space_matches_independent_field_combinations():
    pool = load_template_pool(TEMPLATE)
    domain = domain_ids(pool)[0]
    template_id = next(iter(pool["domains"][domain]["templates"]))
    selection = select_template(pool, domain, template_id)
    space = StaticCoreSpace(selection, "additive_aggregation")
    eligible = [
        field_id for field_id, field in selection.fields.items()
        if "additive_aggregation" in field["eligible_families"]
    ]
    expected = {tuple(values) for values in combinations(eligible, 3)}
    observed = {tuple(space.at(index)["relevant_fields"]) for index in range(space.count)}
    assert observed == expected


def test_bounded_permutation_materialized_and_lazy_resume_without_duplicates(tmp_path):
    materialized = BoundedPermutation(12, 7, 999999, materialize_threshold=999999, order_path=tmp_path / "order.bin")
    values = list(materialized)
    assert len(values) == len(set(values)) == 12
    first = BoundedPermutation(40, 11, 17, materialize_threshold=10)
    prefix = [first.next() for _ in range(6)]
    resumed = BoundedPermutation(40, 11, 17, materialize_threshold=10, state=first.snapshot())
    suffix = list(resumed)
    assert len(prefix + suffix) == len(set(prefix + suffix)) == 17
    assert all(0 <= value < 40 for value in prefix + suffix)


def test_dynamic_distractors_use_constraint_free_sources_and_three_quartets():
    pool = load_template_pool(EIGHT_FIELD_TEMPLATE)
    selection = select_template(pool, "finance", next(iter(pool["domains"]["finance"]["templates"])))
    core = DynamicCoreSpace(selection, "additive_aggregation", "compound").at(0)
    sources = distractor_sources(selection, core, "compound")
    assert all(not selection.fields[field].get("constraints") for field in sources)
    quartets = []
    for trial in range(3):
        transition = materialize_distractors(selection, core, "compound", 11, 8, trial)
        quartet = tuple(transition["distractor_fields"])
        assert len(quartet) == len(set(quartet)) == 4
        assert len(visible_fields(transition, "compound")) == 8
        quartets.append(quartet)
    assert len(set(quartets)) == 3


@pytest.mark.asyncio
async def test_semantic_review_accepts_a_tail_batch_when_disabled():
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(config, semantic_review=replace(config.semantic_review, enabled=False))
    pool = load_template_pool(EIGHT_FIELD_TEMPLATE)
    selection = select_template(pool, "finance", next(iter(pool["domains"]["finance"]["templates"])))
    rows = []
    space = DynamicCoreSpace(selection, "additive_aggregation", "compound")
    for index in range(3):
        core = space.at(index)
        transition = materialize_distractors(selection, core, "compound", 41, 8)
        rows.append({
            "core_id": f"core-{index}",
            "mode": "dynamic",
            "dynamic_type": "compound",
            "transition": transition,
            "visible_fields": visible_fields(transition, "compound"),
        })
    accepted, retryable, audited, errors = await review_candidate_batch([(selection, rows)], config)
    assert len(accepted) == len(audited) == 3
    assert retryable == [] and errors == []


@pytest.mark.parametrize("domain", domain_ids(load_template_pool(TEMPLATE)))
def test_lexicographic_compound_preserves_approved_old_order(domain):
    pool = load_template_pool(TEMPLATE)
    template_id = next(iter(pool["domains"][domain]["templates"]))
    selection = select_template(pool, domain, template_id)
    approved = {
        tuple(item["fields"])
        for item in selection.template["rule_families"]["lexicographic_priority"]["approved_priority_orders"]
    }
    space = DynamicCoreSpace(selection, "lexicographic_priority", "compound")
    for index in range(min(space.count, 30)):
        core = space.at(index)
        assert tuple(core["old_priority_order"]) in approved
        assert core["reversal_target"] in core["shared_fields"]


def test_compound_invariant_evaluation_accepts_joint_magnitude_control():
    pool = load_template_pool(EIGHT_FIELD_TEMPLATE)
    selection = select_template(pool, "finance", next(iter(pool["domains"]["finance"]["templates"])))
    core = DynamicCoreSpace(selection, "additive_aggregation", "compound").at(0)
    transition = materialize_distractors(selection, core, "compound", 23, 8)
    old, joint_post, parameters = sample_rule_pair(
        selection, transition, "additive_aggregation", 29, "compound",
        reversal_control=1.15,
        replacement_role_strength_jitter_range=(-0.05, 0.05),
        replacement_threshold_quantile_jitter_range=(-0.02, 0.02),
    )
    report = compound_invariant_diagnostics(
        selection, old, joint_post, transition, parameters,
        (-0.05, 0.05), (-0.02, 0.02),
    )
    assert report["compound_structural_pass"] is True
    assert report["replacement_role_parameter"]["reversal_magnitude_multiplier"] == pytest.approx(1.15, abs=1e-4)
