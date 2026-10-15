from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from synthesis.templates.batching import BatchPlanner, ContextCapacityError
from synthesis.templates.assembly import assemble_template
from synthesis.templates.config import load_config, validate_resources
from synthesis.templates.cli import parser as synthesis_parser
from synthesis.templates.contracts import ContractError, Contracts, provider_schema
from synthesis.templates.graph import DependencyGraph, Status
from synthesis.templates.pipeline import (
    AUDIT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ScenarioRegenerationRequired,
    SemanticAuditError,
    TemplateSynthesisPipeline,
)
from synthesis.templates.prompting import PromptRenderer
from synthesis.templates.repair import (
    PatchError,
    apply_component_patch,
    apply_field_constraint_repairs,
    component_patch_targets,
    validate_field_assignment_patch,
)
from synthesis.templates.scheduler import gather_jobs
from synthesis.templates.state import RunState, canonical_hash
from synthesis.templates.structured_call import CallSpec, StructuredCaller
from synthesis.templates.stages import (
    audits,
    field_constraints,
    field_semantics,
    lexicographic_orders,
    replacements as replacement_stage,
    scenarios,
)
from synthesis.templates.validation import (
    StageValidationError,
    field_parameter_diagnostics,
    semantic_audit_ids,
    validate_field_catalog,
    validate_field_catalog_local_audit,
    validate_field_catalog_set_audit,
    validate_field_constraints,
    validate_field_parameters,
    validate_field_semantics,
    validate_lexicographic_orders,
    validate_replacements,
    validate_scenario_cores,
    validate_scenario_output,
)
from synthesis.datasets.template_pool import (
    TEMPLATE_SCHEMA_VERSION,
    TemplateValidationError,
    load_template_pool,
    validate_template_pool,
)


def resources():
    config = load_config()
    validate_resources(config)
    return config, Contracts(config.paths.canonical_schema, config.paths.stage_schemas)


def published_pool(config):
    if (
        json.loads(config.paths.canonical.read_text()).get("schema_version")
        != TEMPLATE_SCHEMA_VERSION
    ):
        pytest.skip(
            "published canonical is the legacy three-action release; "
            "regenerate via scripts/synthesize.py templates --fresh --publish"
        )
    return load_template_pool(config.paths.canonical)


def test_cli_exposes_deterministic_field_constraint_stop_point():
    args = synthesis_parser().parse_args([
        "--run-id", "resume-constraints", "--resume",
        "--stop-after", "field_constraints",
    ])
    assert args.resume is True
    assert args.stop_after == "field_constraints"


def test_all_contracts_and_current_canonical_are_valid():
    config, contracts = resources()
    assert config.domains == (
        "finance",
        "healthcare",
        "cybersecurity",
        "manufacturing",
    )
    assert {
        domain: tuple(item.task_id for item in config.task_blueprints[domain])
        for domain in config.domains
    } == {
        "finance": (
            "finance_card_authorization",
            "finance_intraday_liquidity",
            "finance_margin_control",
            "finance_securities_settlement",
        ),
        "healthcare": (
            "healthcare_emergency_arrival_triage",
            "healthcare_inpatient_bed_placement",
            "healthcare_lab_specimen_disposition",
            "healthcare_operating_room_schedule_recovery",
        ),
        "cybersecurity": (
            "cybersecurity_privileged_access_authorization",
            "cybersecurity_endpoint_containment",
            "cybersecurity_vulnerability_remediation",
            "cybersecurity_service_dependency_response",
        ),
        "manufacturing": (
            "manufacturing_machine_maintenance",
            "manufacturing_production_resequencing",
            "manufacturing_quality_lot_disposition",
            "manufacturing_material_shortage_response",
        ),
    }
    assert config.minimum_reversal_fields == 10
    assert config.field_parameters.beta_concentration_range == (4.0, 12.0)
    contracts.check_all()
    pool = published_pool(config)
    contracts.validate("canonical", pool)
    assert all(template["description"] for domain in pool["domains"].values() for template in domain["templates"].values())


def test_scenario_generator_receives_one_authoritative_task_blueprint():
    config, contracts = resources()
    observed_count = 0
    for domain_id in config.domains:
        for item in config.task_blueprints[domain_id]:
            blueprint = item.prompt_value()
            spec = scenarios.generation(
                PromptRenderer(config.paths.prompts),
                domain_id=domain_id,
                domain_description=config.domain_descriptions[domain_id],
                template_id=item.task_id,
                benchmark_boundary="boundary",
                family_meanings=json.loads(
                    config.paths.skeleton.read_text()
                )["defaults"]["rule_families"],
                blueprint=blueprint,
                existing_templates=[],
            )

            assert "Authoritative task blueprint:" in spec.user
            blueprint_line = next(
                line for line in spec.user.splitlines()
                if line.startswith("Authoritative task blueprint: ")
            )
            assert json.loads(blueprint_line.split(": ", 1)[1]) == blueprint
            assert len(blueprint["actions"]) == 4
            assert spec.schema == "scenario_expansion"
            family_line = next(
                line for line in spec.user.splitlines()
                if line.startswith("Fixed rule-family meanings: ")
            )
            meanings = json.loads(family_line.split(": ", 1)[1])
            assert set(meanings) == {
                "additive_aggregation", "conjunctive_gating",
                "disjunctive_sufficiency", "lexicographic_priority",
            }
            assert all(isinstance(description, str) for description in meanings.values())
            observed_count += 1
    assert observed_count == 16
    assert set(contracts.schema("scenario_expansion")["properties"]) == {
        "operational_profiles", "rule_families",
    }


def test_blueprint_path_controls_mode_and_ignores_batch_size(tmp_path):
    raw = yaml.safe_load(Path("configs/synthesis/templates.yaml").read_text())
    raw["scenario_batch_size"] = 99
    config_path = tmp_path / "templates.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    configured = load_config(config_path)
    assert configured.blueprint_path is not None
    assert all(len(configured.task_blueprints[domain]) == 4 for domain in configured.domains)

    raw["blueprint_path"] = None
    config_path.write_text(yaml.safe_dump(raw))
    free = load_config(config_path)
    assert free.blueprint_path is None
    assert free.task_blueprints == {}
    assert free.scenario_batch_size == 99


@pytest.mark.asyncio
async def test_blueprint_mode_rejects_append_before_pipeline_work():
    config = load_config()
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = config
    pipeline.resume = False
    with pytest.raises(ValueError, match="does not support append"):
        await pipeline.run(append=True)


def scenario_model_output(*, free: bool = False):
    value = {
        "operational_profiles": {
            "low": {"weight": 0.3, "description": "low"},
            "mixed": {"weight": 0.4, "description": "mixed"},
            "high": {"weight": 0.3, "description": "high"},
        },
        "rule_families": {
            family: {"description": family}
            for family in (
                "additive_aggregation", "conjunctive_gating",
                "disjunctive_sufficiency", "lexicographic_priority",
            )
        },
    }
    if free:
        value.update({
            "description": "A generated operational decision scenario.",
            "agent_role": "An operations specialist selects an intervention.",
            "actions": [
                {"display_name": f"Action {index}", "description": f"Level {index}."}
                for index in range(4)
            ],
        })
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blueprint_mode", "expected_calls"),
    [(True, 4), (False, 3)],
)
async def test_scenario_slots_are_independent_and_select_diversity_by_mode(
    tmp_path, blueprint_mode, expected_calls,
):
    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runs=tmp_path),
        domains=("finance",),
        domain_descriptions={"finance": base.domain_descriptions["finance"]},
        blueprint_path=(base.blueprint_path if blueprint_mode else None),
        task_blueprints=(
            {"finance": base.task_blueprints["finance"]} if blueprint_mode else {}
        ),
        scenario_batch_size=3,
    )
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = config
    pipeline.resume = blueprint_mode
    pipeline.skeleton = json.loads(base.paths.skeleton.read_text())
    pipeline.state = RunState(
        tmp_path / "slot-run", resume=False, resource_identity="test"
    )
    pipeline.renderer = PromptRenderer(base.paths.prompts)
    pipeline.audit_records = {}
    pipeline.caller = SimpleNamespace(
        call=AsyncMock(return_value=scenario_model_output(free=not blueprint_mode))
    )

    async def accept_initial(*, initial_value, **_):
        return initial_value, {"verdicts": [], "overall_summary": "accepted"}

    pipeline._checked = AsyncMock(side_effect=accept_initial)
    pipeline._template_job = AsyncMock(
        side_effect=lambda **kwargs: (kwargs["core"]["template_id"], None)
    )

    await pipeline.run(
        domains=("finance",), append=blueprint_mode, stop_after="field_constraints"
    )

    assert pipeline.caller.call.await_count == expected_calls
    assert pipeline._template_job.await_count == expected_calls
    diversity_values = [
        call.kwargs["diversity_check"]
        for call in pipeline._template_job.await_args_list
    ]
    assert all(
        value is None if blueprint_mode else callable(value)
        for value in diversity_values
    )


def test_free_scenario_generator_omits_blueprint_section():
    config, _ = resources()
    spec = scenarios.generation(
        PromptRenderer(config.paths.prompts),
        domain_id="finance",
        domain_description=config.domain_descriptions["finance"],
        template_id="finance_scenario_001",
        benchmark_boundary="boundary",
        family_meanings=json.loads(config.paths.skeleton.read_text())["defaults"]["rule_families"],
        blueprint=None,
        existing_templates=[],
    )
    assert "Authoritative task blueprint:" not in spec.user
    assert "No task blueprint is supplied" in spec.user
    assert spec.schema == "scenario_generation"


def test_scenario_expansion_is_combined_with_authoritative_blueprint():
    blueprint = {
        "task_id": "healthcare_lab_specimen_disposition",
        "description": "Review an accessioned specimen.",
        "agent_role": "A laboratory quality technologist owns the disposition.",
        "actions": [
            {"display_name": "Accept", "description": "Accept for testing."},
            {"display_name": "Hold", "description": "Hold for verification."},
            {"display_name": "Reject", "description": "Request recollection."},
            {"display_name": "Escalate", "description": "Escalate to the laboratory director."},
        ],
    }
    expansion = {
        "operational_profiles": {
            "low": {"weight": 0.3, "description": "low"},
            "mixed": {"weight": 0.4, "description": "mixed"},
            "high": {"weight": 0.3, "description": "high"},
        },
        "rule_families": {},
    }
    validate_scenario_output(expansion)
    value = scenarios.assemble(
        expansion,
        domain_id="healthcare",
        domain_description="Healthcare operations.",
        template_id="specimen_review_v1",
        task_id=blueprint["task_id"],
        blueprint=blueprint,
    )
    validate_scenario_cores(
        value,
        "healthcare",
        domain_description="Healthcare operations.",
        template_id="specimen_review_v1",
        task_id=blueprint["task_id"],
        blueprint=blueprint,
    )
    core = value["templates"][0]
    assert core["task_id"] == blueprint["task_id"]
    assert core["description"] == blueprint["description"]
    assert core["agent_role"] == blueprint["agent_role"]
    assert core["actions"] == blueprint["actions"]


def test_scenario_validation_rejects_blueprint_rewrites():
    blueprint = {
        "task_id": "healthcare_lab_specimen_disposition",
        "description": "Review an accessioned specimen.",
        "agent_role": "A laboratory quality technologist owns the disposition.",
        "actions": [
            {"display_name": "Accept", "description": "Accept for testing."},
            {"display_name": "Hold", "description": "Hold for verification."},
            {"display_name": "Reject", "description": "Request recollection."},
            {"display_name": "Escalate", "description": "Escalate to the laboratory director."},
        ],
    }
    expansion = {
        "operational_profiles": {
            "low": {"weight": 0.3, "description": "low"},
            "mixed": {"weight": 0.4, "description": "mixed"},
            "high": {"weight": 0.3, "description": "high"},
        },
        "rule_families": {},
    }
    value = scenarios.assemble(
        expansion,
        domain_id="healthcare",
        domain_description="Healthcare operations.",
        template_id="specimen_review_v1",
        task_id=blueprint["task_id"],
        blueprint=blueprint,
    )
    value["templates"][0]["agent_role"] = "The model rewrote this role."
    with pytest.raises(StageValidationError, match="blueprint_role_mismatch"):
        validate_scenario_cores(
            value,
            "healthcare",
            domain_description="Healthcare operations.",
            template_id="specimen_review_v1",
            task_id=blueprint["task_id"],
            blueprint=blueprint,
        )


def test_scenario_expansion_schema_rejects_fixed_blueprint_properties():
    _, contracts = resources()
    expansion = {
        "task_id": "must_not_be_returned",
        "operational_profiles": {
            "low": {"weight": 0.3, "description": "low"},
            "mixed": {"weight": 0.4, "description": "mixed"},
            "high": {"weight": 0.3, "description": "high"},
        },
        "rule_families": {
            family: {"description": family}
            for family in (
                "additive_aggregation", "conjunctive_gating",
                "disjunctive_sufficiency", "lexicographic_priority",
            )
        },
    }
    with pytest.raises(ContractError, match="task_id"):
        contracts.validate("scenario_expansion", expansion)


def test_template_append_requires_one_bound_published_release(tmp_path):
    config, contracts = resources()
    pool = published_pool(config)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "canonical_sha256": canonical_hash(pool),
        "accepted": True,
        "components": {},
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_sha256": canonical_hash(pool),
        "audit_sha256": canonical_hash(audit),
    }
    canonical_path = tmp_path / "domain_templates.json"
    audit_path = tmp_path / "domain_templates.audit.json"
    manifest_path = tmp_path / "domain_templates.manifest.json"
    canonical_path.write_text(json.dumps(pool))
    audit_path.write_text(json.dumps(audit))
    manifest_path.write_text(json.dumps(manifest))
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = replace(
        config,
        paths=replace(
            config.paths,
            canonical=canonical_path,
            audit=audit_path,
            manifest=manifest_path,
        ),
    )
    pipeline.contracts = contracts
    pipeline.skeleton = json.loads(config.paths.skeleton.read_text())
    pipeline.audit_records = {}
    pipeline.parent_manifest_sha256 = None
    assert pipeline._published_base() == pool

    first_domain = next(iter(pool["domains"].values()))
    next(iter(first_domain["templates"].values()))["description"] += " changed"
    canonical_path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="does not match canonical"):
        pipeline._published_base()


@pytest.mark.asyncio
async def test_semantics_reversal_floor_restarts_before_llm_audit_or_repair():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(
        minimum_reversal_fields=10,
        parse_retry_limit=0,
        catalog_set_drop_limit=0,
        repair=SimpleNamespace(limit=5),
    )
    _, pipeline.contracts = resources()
    pipeline.state = SimpleNamespace(
        add_node=lambda *args, **kwargs: None,
        accepted_component=lambda *args, **kwargs: None,
        mark_stale=lambda *args, **kwargs: None,
        accept=AsyncMock(),
    )
    pipeline.audit_records = {}
    pipeline._semantic_audit = AsyncMock()
    pipeline._repair = AsyncMock()
    fields = [
        {
            "field_id": f"field_{index}",
            "effect_strength": 0.5,
            "eligible_families": ["additive_aggregation"],
            "is_reversal": True,
            "reversal_rationale": "",
            "trigger": "A realistic external condition changes.",
            "semantics_before": "Higher values increase intervention pressure.",
            "semantics_after": "Higher values reduce intervention pressure.",
            "reversal_reason": "The condition changes the field's decision meaning.",
        }
        for index in range(9)
    ]
    value = {"template_id": "scene", "fields": fields}

    with pytest.raises(ScenarioRegenerationRequired) as exc_info:
        await pipeline._checked(
            spec=CallSpec(
                stage="field_semantics", component_id="scene",
                schema="field_semantics", system="", user="",
            ),
            validator=lambda item: validate_field_semantics(
                item, "scene", {field["field_id"] for field in fields}
            ),
            audit_stage="field_semantics",
            audit_context={},
            repair_template="fields/semantics_repair.j2",
            repair_context={},
            node_id="scene/field_semantics",
            initial_value=value,
        )

    assert exc_info.value.reversal_count == 9
    assert exc_info.value.required_count == 10
    pipeline._semantic_audit.assert_not_awaited()
    pipeline._repair.assert_not_awaited()
    pipeline.state.accept.assert_not_awaited()


def test_canonical_rejects_replacement_coupling_above_point_six():
    config, contracts = resources()
    pool = published_pool(config)
    template = next(iter(pool["domains"]["finance"]["templates"].values()))
    group = next(iter(template["dynamics"]["replacements"].values()))
    group["joint_sampling"]["coupling_strength"] = 0.61
    with pytest.raises(ContractError) as exc_info:
        contracts.validate("canonical", pool)
    message = str(exc_info.value)
    assert "coupling_strength" in message
    assert "maximum" in message


def test_release_validator_rejects_beta_concentration_outside_configured_range():
    config, _ = resources()
    pool = published_pool(config)
    template = next(iter(pool["domains"]["finance"]["templates"].values()))
    field = next(iter(template["fields"].values()))
    field["profile_beta"]["low"] = [6.5, 6.5]
    with pytest.raises(TemplateValidationError, match="invalid beta"):
        validate_template_pool(pool)


def test_release_validator_ignores_profile_beta_object_key_order():
    config, _ = resources()
    pool = published_pool(config)
    for domain in pool["domains"].values():
        for template in domain["templates"].values():
            for field in template["fields"].values():
                beta = field["profile_beta"]
                field["profile_beta"] = {
                    "high": beta["high"],
                    "low": beta["low"],
                    "mixed": beta["mixed"],
                }
    validate_template_pool(pool)


def test_canonical_validator_does_not_reject_text_by_word_count_or_phrase():
    config, _ = resources()
    pool = published_pool(config)
    template = next(iter(pool["domains"]["finance"]["templates"].values()))
    reversal = next(iter(template["dynamics"]["direction_reversals"].values()))
    reversal["trigger"] = "Event"
    reversal["reversal_reason"] = "Direction reverses"
    group = next(iter(template["dynamics"]["replacements"].values()))
    group["effective_scope"] = "Operations"
    transition = next(iter(group["transitions"].values()))
    transition["trigger"] = "Field is replaced"
    transition["incoming_authority_reason"] = "Source is authoritative"
    transition["expected_duration"] = "Temporarily"

    validate_template_pool(pool)


def test_canonical_rejects_removed_alignment_sidecar():
    config, contracts = resources()
    pool = published_pool(config)
    template = next(iter(pool["domains"]["finance"]["templates"].values()))
    source = [
        "store_intraday_demand_pressure", "checkout_queue_lag_seconds",
        "supplier_fill_commit_hours",
    ]
    derived = [
        "store_close_demand_pressure", "checkout_queue_lag_seconds",
        "supplier_fill_commit_hours",
    ]
    group_id = "store_demand_priority_group"
    transition_id = "store_demand_priority_group_forward"
    template["dynamics"]["lexicographic_alignments"] = {
        "alignment_fixture": {
            "source_order_id": "source_fixture",
            "replacement_group_id": group_id,
            "replacement_transition_id": transition_id,
            "replacement_slot": 0,
            "outgoing_field": source[0], "incoming_field": derived[0],
            "source_fields": source, "derived_fields": derived,
            "alignment_rationale": "The accepted transition changes the first cue.",
            "derived_order_rationale": "Close demand decides, then queue lag, then fill time.",
        },
    }
    with pytest.raises(ContractError) as exc_info:
        contracts.validate("canonical", pool)
    message = str(exc_info.value)
    assert "lexicographic_alignments" in message
    assert "unexpected" in message


@pytest.mark.asyncio
async def test_catalog_wide_audit_blocks_local_field_audits_until_it_passes():
    config, _ = resources()
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(
        batching=SimpleNamespace(max_field_audit_per_batch=1),
    )
    pipeline.renderer = PromptRenderer(config.paths.prompts)
    stages = []

    async def reject_set(spec, *, field_ids, frozen_ids):
        stages.append(spec.stage)
        return ({
            "accepted": False,
            "issues": [{
                "code": "strict_duplicate",
                "field_ids": ["a", "b"],
                "drop_field_ids": ["a"],
                "message": "a is exactly recoverable from b",
            }],
            "summary": "set rejected",
        }, {"a"})

    pipeline._run_field_catalog_set_audit_spec = reject_set
    pipeline._run_semantic_audit_spec = AsyncMock()
    value = {
        "template_id": "scene",
        "fields": [
            {"field_id": "a", "display_name": "A"},
            {"field_id": "b", "display_name": "B"},
        ],
    }
    _, issues = await pipeline._field_catalog_audit(
        component_id="scene",
        value=value,
        context={"scenario": {"agent_role": "operator"}},
        frozen_ids=set(),
        prior_issues=[],
        run_set_audit=True,
    )
    assert stages == ["field_catalog_set_audit"]
    assert {issue["component_id"] for issue in issues} == {"a"}
    pipeline._run_semantic_audit_spec.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_field_audits_run_only_after_catalog_wide_acceptance():
    config, _ = resources()
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(
        batching=SimpleNamespace(max_field_audit_per_batch=1),
    )
    pipeline.renderer = PromptRenderer(config.paths.prompts)
    stages = []

    async def accept_set(spec, *, field_ids, frozen_ids):
        stages.append(spec.stage)
        return ({"accepted": True, "issues": [], "summary": "accepted"}, set())

    async def accept_all(spec, *, field_ids):
        stages.append(spec.stage)
        return {
            "issues": [],
            "summary": "accepted",
        }

    pipeline._run_field_catalog_set_audit_spec = accept_set
    pipeline._run_field_catalog_local_audit_spec = accept_all
    value = {
        "template_id": "scene",
        "fields": [
            {"field_id": "a", "display_name": "A"},
            {"field_id": "b", "display_name": "B"},
        ],
    }
    _, issues = await pipeline._field_catalog_audit(
        component_id="scene",
        value=value,
        context={"scenario": {"agent_role": "operator"}},
        frozen_ids=set(),
        prior_issues=[],
        run_set_audit=True,
    )
    assert stages[0] == "field_catalog_set_audit"
    assert stages[1:] == ["field_catalog_audit", "field_catalog_audit"]
    assert issues == []

    stages.clear()
    await pipeline._field_catalog_audit(
        component_id="scene",
        value=value,
        context={"scenario": {"agent_role": "operator"}},
        frozen_ids={"a"},
        prior_issues=[],
        run_set_audit=False,
    )
    assert stages == ["field_catalog_audit"]


@pytest.mark.asyncio
async def test_repaired_fields_are_reaudited_together_without_batching():
    config, _ = resources()
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(
        batching=SimpleNamespace(max_field_audit_per_batch=10),
    )
    pipeline.renderer = PromptRenderer(config.paths.prompts)
    audited_batches = []

    async def accept_local(spec, *, field_ids):
        audited_batches.append(set(field_ids))
        return {"issues": [], "summary": "accepted"}

    pipeline._run_field_catalog_local_audit_spec = accept_local
    fields = [
        {"field_id": f"field_{index}", "display_name": f"Field {index}"}
        for index in range(12)
    ]
    _, issues = await pipeline._field_catalog_audit(
        component_id="scene",
        value={"template_id": "scene", "fields": fields},
        context={"scenario": {"agent_role": "operator"}},
        frozen_ids=set(),
        prior_issues=[{
            "component_id": "field_0",
            "resolution": "repair_field_assignment",
        }],
        run_set_audit=False,
    )
    assert audited_batches == [{field["field_id"] for field in fields}]
    assert issues == []


def test_parameter_batching_respects_budget_and_quality_limit():
    config, _ = resources()
    planner = BatchPlanner(config.batching)
    ids = [f"field_{index}" for index in range(37)]
    batches = planner.field_parameters(shared_context={"catalog": ids}, field_ids=ids)
    flattened = [item for batch in batches for item in batch.ids]
    assert flattened == ids
    assert len(flattened) == len(set(flattened))
    assert all(batch.estimated_tokens <= planner.usable_tokens for batch in batches)
    assert all(len(batch.ids) <= config.batching.max_field_parameters_per_batch for batch in batches)
    assert [len(batch.ids) for batch in batches] == [16, 16, 5]


def test_batching_fails_instead_of_truncating_global_context():
    config, _ = resources()
    planner = BatchPlanner(config.batching)
    oversized = {"catalog": "x" * (planner.usable_tokens * 4)}
    with pytest.raises(ContextCapacityError):
        planner.field_parameters(shared_context=oversized, field_ids=["field"])


def test_component_patch_changes_only_requested_path():
    original = {"template_id": "fixture", "fields": [{"field_id": "a", "minimum": 0}, {"field_id": "b", "minimum": 1}]}
    patched = apply_component_patch(original, {
        "component_id": "fixture", "operations": [{"op": "replace", "path": "/fields/a/minimum", "value_json": "-1"}],
    })
    assert patched["fields"][0]["minimum"] == -1
    assert patched["fields"][1] == original["fields"][1]
    assert original["fields"][0]["minimum"] == 0


def test_component_patch_rejects_invalid_pointer():
    with pytest.raises(PatchError):
        apply_component_patch({"a": 1}, {"component_id": "x", "operations": [{"op": "replace", "path": "/missing/value", "value_json": "2"}]})


def test_component_patch_rejects_numeric_field_index():
    with pytest.raises(PatchError, match="FIELD_ID"):
        apply_component_patch(
            {"fields": [{"field_id": "a", "minimum": 0}]},
            {"component_id": "x", "operations": [{
                "op": "replace", "path": "/fields/0/minimum", "value_json": "2",
            }]},
        )


def test_component_patch_rejects_whole_component_replacement():
    with pytest.raises(PatchError, match="cannot patch the component root"):
        apply_component_patch({"a": 1}, {
            "component_id": "x",
            "operations": [{"op": "replace", "path": "/", "value_json": '{"a": 2}'}],
        })


def test_component_patch_targets_smallest_dynamic_unit():
    value = {
        "replacements": [{
            "group_id": "handoff",
            "rationale": "old",
            "transitions": [
                {"transition_id": "forward", "trigger": "old"},
                {"transition_id": "reverse", "trigger": "old"},
            ],
        }, {
            "group_id": "unrelated",
            "rationale": "keep",
            "transitions": [{"transition_id": "other", "trigger": "old"}],
        }],
    }
    assert component_patch_targets(value, {
        "component_id": "scene",
        "operations": [{
            "op": "replace",
            "path": "/replacements/0/transitions/0/trigger",
            "value_json": '"new"',
        }],
    }, "replacements") == {"forward"}
    assert component_patch_targets(value, {
        "component_id": "scene",
        "operations": [{
            "op": "replace",
            "path": "/replacements/1/rationale",
            "value_json": '"changed"',
        }],
    }, "replacements") == {"unrelated"}
    with pytest.raises(PatchError, match="identity"):
        component_patch_targets(value, {
            "component_id": "scene",
            "operations": [{
                "op": "replace",
                "path": "/replacements/1/group_id",
                "value_json": '"changed"',
            }],
        }, "replacements")


def test_dependency_invalidation_propagates_only_downstream():
    graph = DependencyGraph()
    graph.add("scenario")
    graph.add("fields", {"scenario"})
    graph.add("parameters", {"fields"})
    graph.add("unrelated")
    for node_id in graph.nodes:
        graph.accept(node_id, node_id)
    invalidated = graph.invalidate("fields")
    assert invalidated == {"parameters"}
    assert graph.nodes["parameters"].status == Status.STALE
    assert graph.nodes["scenario"].status == Status.ACCEPTED
    assert graph.nodes["unrelated"].status == Status.ACCEPTED


@pytest.mark.asyncio
async def test_scheduler_waits_for_parallel_jobs_and_groups_failures():
    completed = []

    async def succeeds():
        completed.append("finished")
        return "ok"

    async def fails():
        raise ValueError("broken")

    with pytest.raises(ExceptionGroup) as caught:
        await gather_jobs([fails, succeeds])
    assert completed == ["finished"]
    assert isinstance(caught.value.exceptions[0], ValueError)


@pytest.mark.asyncio
async def test_run_state_resumes_only_matching_inputs_and_component_hashes(tmp_path):
    run_dir = tmp_path / "run"
    initial = RunState(run_dir, resume=False, resource_identity="resources-v1")
    initial.bind_inputs({"domains": ["finance"], "append": False, "base_sha256": "base"})
    initial.add_node("finance/scenarios")
    audit = {"accepted": True, "issues": []}
    await initial.accept(
        "finance/scenarios", {"domain_id": "finance"}, audit_record=audit
    )

    resumed = RunState(run_dir, resume=True, resource_identity="resources-v1")
    resumed.bind_inputs({"domains": ["finance"], "append": False, "base_sha256": "base"})
    assert resumed.accepted_component("finance/scenarios") == {"domain_id": "finance"}
    assert resumed.audit_record("finance/scenarios") == audit

    with pytest.raises(ValueError, match="resources changed"):
        RunState(run_dir, resume=True, resource_identity="resources-v2")
    with pytest.raises(ValueError, match="inputs changed"):
        resumed.bind_inputs({"domains": ["healthcare"], "append": False, "base_sha256": "base"})

def test_no_stage_schema_limits_field_count():
    config, contracts = resources()
    for name in ("field_catalog", "field_semantics", "field_parameters"):
        fields_schema = contracts.schema(name)["properties"]["fields"]
        assert "maxItems" not in fields_schema
    assert config.scenario_batch_size == 4
    assert not hasattr(config, "cross_stage_drop_limit")
    assert config.catalog_set_drop_limit == 5
    assert config.semantics_drop_limit == 5
    assert config.template_retry_limit == 2
    assert config.scenario_refill_limit == 8
    assert config.repair.limit == 5
    assert config.generation.api_mode == "responses_background"
    assert config.audit.api_mode == "responses_background"
    assert config.repair.api_mode == "responses_background"


def test_field_inventory_and_semantics_have_separate_contracts():
    _, contracts = resources()
    inventory = contracts.schema("field_catalog")["properties"]["fields"]["items"]["properties"]
    semantics = contracts.schema("field_semantics")["properties"]["fields"]["items"]["properties"]
    assert set(inventory) == {
        "field_id", "display_name", "description", "type", "unit",
        "pressure_direction",
    }
    assert "is_reversal" in semantics
    assert "trigger" in semantics
    assert "pressure_direction" not in semantics
    assert "minimum" not in semantics

    valid = {
        "template_id": "scene",
        "fields": [{
            "field_id": "pressure",
            "effect_strength": 0.5,
            "eligible_families": ["additive_aggregation"],
            "is_reversal": False,
            "reversal_rationale": "The measurement remains monotonic in this workflow.",
            "trigger": "",
            "semantics_before": "",
            "semantics_after": "",
            "reversal_reason": "",
        }],
    }
    validate_field_semantics(valid, "scene", {"pressure"})
    valid["fields"][0]["trigger"] = "a named regime change"
    with pytest.raises(StageValidationError, match="unexpected_reversal_event"):
        validate_field_semantics(valid, "scene", {"pressure"})
    valid["fields"][0]["trigger"] = ""
    valid["fields"][0]["reversal_rationale"] = ""
    with pytest.raises(StageValidationError, match="missing_reversal_rationale"):
        validate_field_semantics(valid, "scene", {"pressure"})
    reversible = {
        "template_id": "scene",
        "fields": [{
            **valid["fields"][0],
            "is_reversal": True,
            "trigger": "a named regime change",
            "semantics_before": "Higher values raise intervention pressure.",
            "semantics_after": "Higher values reduce intervention pressure.",
            "reversal_reason": "The regime flips what the same measurement means.",
        }],
    }
    validate_field_semantics(reversible, "scene", {"pressure"})
    reversible["fields"][0]["reversal_reason"] = ""
    with pytest.raises(StageValidationError, match="missing_reversal_event"):
        validate_field_semantics(reversible, "scene", {"pressure"})


def test_single_template_contract_rejects_empty_priority_orders_after_drop():
    config, contracts = resources()
    pool = published_pool(config)
    template = deepcopy(next(iter(pool["domains"]["finance"]["templates"].values())))
    contracts.validate_template(template)
    template["rule_families"]["lexicographic_priority"][
        "approved_priority_orders"
    ] = []
    with pytest.raises(ValueError, match="canonical template contract failed"):
        contracts.validate_template(template)


def test_parameter_diagnostics_map_beta_through_legal_not_typical_range():
    diagnostics = field_parameter_diagnostics({
        "fields": [{
            "field_id": "distance",
            "minimum": -20004,
            "maximum": 20004,
            "typical_range": [-50, 500],
            "profile_beta": {
                "low": [20, 19.9],
                "mixed": [22, 21.5],
                "high": [8, 6.7],
            },
        }],
    })
    means = diagnostics[0]["mapped_profile_means"]
    assert means["low"] == pytest.approx(50.1353, abs=0.001)
    assert means["high"] > 1700


def test_split_contracts_bind_structured_expressions_and_order_field_ids():
    _, contracts = resources()
    constraint_schema = contracts.bound_schema("field_constraints", {
        "/properties/template_id": ("scene",),
        "/$defs/number_atom/anyOf/0/properties/field": ("a", "b", "c"),
    })
    order_schema = contracts.bound_schema("lexicographic_orders", {
        "/properties/template_id": ("scene",),
        "/properties/approved_priority_orders/items/properties/fields/items": (
            "a", "b", "c",
        ),
    })
    assert constraint_schema["properties"]["template_id"]["enum"] == ["scene"]
    assert constraint_schema["$defs"]["number_atom"]["anyOf"][0][
        "properties"
    ]["field"]["enum"] == ["a", "b", "c"]
    assert order_schema["properties"]["approved_priority_orders"]["items"][
        "properties"
    ]["fields"]["items"]["enum"] == ["a", "b", "c"]
    constraints = {
        "template_id": "scene",
        "feasibility_constraints": [{
            "id": "contained",
            "description": "a cannot exceed b",
            "expr": {
                "op": "le", "left": {"field": "a"}, "right": {"field": "b"},
            },
        }],
    }
    orders = {
        "template_id": "scene",
        "approved_priority_orders": [{
            "order_id": "primary", "fields": ["a", "b", "c"],
            "rationale": "a precedes b, then c resolves the remaining cases.",
        }],
    }
    contracts.validate("field_constraints", constraints)
    contracts.validate("lexicographic_orders", orders)
    fields = {
        field_id: {"eligible_families": ["lexicographic_priority"]}
        for field_id in ("a", "b", "c")
    }
    validate_field_constraints(constraints, "scene", fields)
    validate_lexicographic_orders(orders, "scene", fields)


def test_field_constraint_input_contract_excludes_unrelated_field_metadata():
    fields = {
        "risk": {
            "display_name": "Risk",
            "description": "Measured risk.",
            "type": "decimal",
            "unit": "fraction",
            "minimum": 0,
            "maximum": 1,
            "typical_range": [0.1, 0.9],
            "pressure_direction": 1,
            "is_reversal": True,
            "effect_strength": 0.8,
            "eligible_families": ["additive_aggregation"],
            "profile_beta": {"low": [1, 2], "mixed": [2, 2], "high": [2, 1]},
        },
    }

    projected = field_constraints.input_fields(fields)

    assert projected == {"risk": {
        "display_name": "Risk",
        "description": "Measured risk.",
        "type": "decimal",
        "unit": "fraction",
        "minimum": 0,
        "maximum": 1,
    }}
    assert "profile_beta" in fields["risk"]


def test_field_constraint_ids_must_be_unique_before_audit():
    value = {
        "template_id": "scene",
        "feasibility_constraints": [
            {
                "id": "duplicate",
                "description": "a cannot exceed b",
                "expr": {
                    "op": "le", "left": {"field": "a"},
                    "right": {"field": "b"},
                },
            },
            {
                "id": "duplicate",
                "description": "b cannot exceed a",
                "expr": {
                    "op": "le", "left": {"field": "b"},
                    "right": {"field": "a"},
                },
            },
        ],
    }

    with pytest.raises(StageValidationError) as exc_info:
        validate_field_constraints(value, "scene", {"a": {}, "b": {}})
    assert exc_info.value.issues == [{
        "path": "/feasibility_constraints",
        "code": "duplicate_id",
        "message": "constraint IDs must be unique: ['duplicate']",
        "component_id": "__field_constraints__",
    }]
    with pytest.raises(ValueError, match="unique before audit"):
        semantic_audit_ids("field_constraints", value)


def test_field_constraint_program_issues_identify_the_owned_constraint():
    value = {
        "template_id": "scene",
        "feasibility_constraints": [{
            "id": "owned",
            "description": "references an unknown field",
            "expr": {
                "op": "le", "left": {"field": "known"},
                "right": {"field": "unknown"},
            },
        }],
    }

    with pytest.raises(StageValidationError) as exc_info:
        validate_field_constraints(value, "scene", {"known": {}})
    assert exc_info.value.issues[0]["component_id"] == "owned"


def test_field_constraint_repair_contract_and_application_are_id_keyed():
    _, contracts = resources()
    repair_schema = contracts.bound_schema("field_constraints_repair", {
        "/properties/component_id": ("scene",),
        "/properties/repairs/items/properties/constraint_id": (
            "remove", "replace", "split",
        ),
    })
    assert repair_schema["properties"]["component_id"]["enum"] == ["scene"]
    assert repair_schema["properties"]["repairs"]["items"]["properties"][
        "constraint_id"
    ]["enum"] == ["remove", "replace", "split"]

    value = {
        "template_id": "scene",
        "feasibility_constraints": [
            {"id": "remove", "description": "remove me", "expr": {}},
            {"id": "keep", "description": "keep me", "expr": {}},
            {"id": "replace", "description": "old", "expr": {}},
            {"id": "split", "description": "bundled", "expr": {}},
        ],
    }
    repair = {
        "component_id": "scene",
        "repairs": [
            {
                "constraint_id": "remove",
                "action": "remove",
                "replacement_json": "null",
            },
            {
                "constraint_id": "replace",
                "action": "replace",
                "replacement_json": json.dumps({
                    "description": "new",
                    "expr": {
                        "op": "le", "left": {"field": "a"},
                        "right": {"field": "b"},
                    },
                }),
            },
            {
                "constraint_id": "split",
                "action": "split",
                "replacement_json": json.dumps([
                    {
                        "description": "first",
                        "expr": {
                            "op": "le", "left": {"field": "a"},
                            "right": {"field": "b"},
                        },
                    },
                    {
                        "description": "second",
                        "expr": {
                            "op": "le", "left": {"field": "c"},
                            "right": {"field": "d"},
                        },
                    },
                ]),
            },
        ],
    }
    contracts.validate("field_constraints_repair", repair)

    repaired = apply_field_constraint_repairs(
        value, repair, {"remove", "replace", "split"}, {"split"}
    )

    assert repaired["feasibility_constraints"] == [
        {"id": "keep", "description": "keep me", "expr": {}},
        {
            "id": "replace",
            "description": "new",
            "expr": {
                "op": "le", "left": {"field": "a"},
                "right": {"field": "b"},
            },
        },
        {
            "id": "split",
            "description": "first",
            "expr": {
                "op": "le", "left": {"field": "a"},
                "right": {"field": "b"},
            },
        },
        {
            "id": "split__part_2",
            "description": "second",
            "expr": {
                "op": "le", "left": {"field": "c"},
                "right": {"field": "d"},
            },
        },
    ]
    assert value["feasibility_constraints"][2]["description"] == "old"


def test_field_constraint_repair_requires_exact_rejected_id_coverage():
    value = {
        "feasibility_constraints": [
            {"id": "rejected", "description": "old", "expr": {}},
            {"id": "accepted", "description": "keep", "expr": {}},
        ],
    }
    repair = {
        "component_id": "scene",
        "repairs": [{
            "constraint_id": "accepted",
            "action": "remove",
            "replacement_json": "null",
        }],
    }

    with pytest.raises(PatchError, match="coverage mismatch"):
        apply_field_constraint_repairs(value, repair, {"rejected"})


def test_field_constraint_provider_schema_is_finite_and_rejects_recursive_boolean_depth():
    _, contracts = resources()
    schema = contracts.bound_schema("field_constraints", {
        "/properties/template_id": ("scene",),
        "/$defs/number_atom/anyOf/0/properties/field": ("a", "b", "c"),
    })

    references: dict[str, set[str]] = {}
    for name, definition in schema["$defs"].items():
        found: set[str] = set()

        def collect(value):
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str) and reference.startswith("#/$defs/"):
                    found.add(reference.rsplit("/", 1)[-1])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(definition)
        references[name] = found

    def visit(name: str, active: set[str], complete: set[str]) -> None:
        if name in active:
            raise AssertionError(f"recursive provider definition: {name}")
        if name in complete:
            return
        active.add(name)
        for child in references.get(name, set()):
            visit(child, active, complete)
        active.remove(name)
        complete.add(name)

    complete: set[str] = set()
    for name in references:
        visit(name, set(), complete)

    recursively_nested = {
        "template_id": "scene",
        "feasibility_constraints": [{
            "id": "unbounded",
            "description": "Repeated Boolean recursion is forbidden.",
            "expr": {
                "op": "and",
                "args": [
                    {
                        "op": "and",
                        "args": [
                            {
                                "op": "and",
                                "args": [
                                    {"op": "le", "left": {"field": "a"}, "right": {"field": "b"}},
                                    {"op": "le", "left": {"field": "b"}, "right": {"field": "c"}},
                                ],
                            },
                            {"op": "le", "left": {"field": "a"}, "right": {"field": "c"}},
                        ],
                    },
                    {"op": "le", "left": {"field": "a"}, "right": {"field": "c"}},
                ],
            },
        }],
    }
    with pytest.raises(ContractError):
        contracts.validate("field_constraints", recursively_nested)


def test_relation_generation_is_split_into_independent_constraint_and_order_specs():
    config, contracts = resources()
    renderer = PromptRenderer(config.paths.prompts)
    fields = {
        field_id: {"eligible_families": ["lexicographic_priority"]}
        for field_id in ("a", "b", "c")
    }
    constraint_spec = field_constraints.generation(
        renderer, template_id="scene", scenario={}, fields=fields
    )
    order_spec = lexicographic_orders.generation(
        renderer, template_id="scene", scenario={}, fields=fields
    )

    assert constraint_spec.stage == constraint_spec.schema == "field_constraints"
    assert order_spec.stage == order_spec.schema == "lexicographic_orders"
    assert "Approved priority orders" not in constraint_spec.system
    assert "Feasibility constraints" not in order_spec.system
    assert "emit all six permutations by default" in order_spec.system
    assert "inherent, universal" in order_spec.system
    assert order_spec.enum_bindings[
        "/properties/approved_priority_orders/items/properties/fields/items"
    ] == ("a", "b", "c")

    constraints = {
        "template_id": "scene",
        "feasibility_constraints": [{
            "id": "contained",
            "description": "a cannot exceed b",
            "expr": {
                "op": "le", "left": {"field": "a"}, "right": {"field": "b"},
            },
        }],
    }
    orders = {
        "template_id": "scene",
        "approved_priority_orders": [{
            "order_id": "primary",
            "fields": ["a", "b", "c"],
            "rationale": "a precedes b, then c resolves the remaining cases.",
        }],
    }
    contracts.validate("field_constraints", constraints)
    contracts.validate("lexicographic_orders", orders)
    validate_field_constraints(constraints, "scene", fields)
    validate_lexicographic_orders(orders, "scene", fields)
    duplicate_order = deepcopy(orders)
    duplicate_order["approved_priority_orders"].append({
        "order_id": "same_fields_again",
        "fields": ["a", "b", "c"],
        "rationale": "The same exact order must not be emitted twice.",
    })
    with pytest.raises(StageValidationError, match="duplicate_order"):
        validate_lexicographic_orders(duplicate_order, "scene", fields)


def test_replacement_generation_does_not_author_priority_slots_or_families():
    config, contracts = resources()
    renderer = PromptRenderer(config.paths.prompts)
    fields = {
        field_id: {
            "eligible_families": ["additive_aggregation"],
        }
        for field_id in ("a", "b")
    }

    spec = replacement_stage.generation(
        renderer, template_id="scene", scenario={}, fields=fields
    )
    schema = contracts.bound_schema(spec.schema, spec.enum_bindings)
    group_schema = schema["properties"]["replacements"]["items"]

    assert "allowed_priority_slots" not in spec.system
    assert "allowed_priority_slots" not in group_schema["required"]
    assert "allowed_priority_slots" not in group_schema["properties"]
    assert "applicable_families" not in spec.system
    assert "applicable_families" not in group_schema["required"]
    assert "applicable_families" not in group_schema["properties"]
    assert group_schema["properties"]["coupling_strength"]["maximum"] == 0.6
    assert "pair-specific `rationale`" in spec.system
    assert "governed business objects or population" in spec.system
    assert "not a list of rule families" in spec.system


def test_field_constraint_split_rejects_too_few_parts_and_id_collisions():
    value = {
        "template_id": "scene",
        "feasibility_constraints": [
            {"id": "bad", "description": "bundled", "expr": {}},
            {"id": "bad__part_2", "description": "already used", "expr": {}},
        ],
    }
    too_short = {
        "component_id": "scene",
        "repairs": [{
            "constraint_id": "bad",
            "action": "split",
            "replacement_json": json.dumps([{"description": "one", "expr": {}}]),
        }],
    }
    with pytest.raises(PatchError, match="at least two"):
        apply_field_constraint_repairs(value, too_short, {"bad"}, {"bad"})

    with pytest.raises(PatchError, match="only for bundled"):
        apply_field_constraint_repairs(value, too_short, {"bad"})

    collision = deepcopy(too_short)
    collision["repairs"][0]["replacement_json"] = json.dumps([
        {"description": "one", "expr": {}},
        {"description": "two", "expr": {}},
    ])
    with pytest.raises(PatchError, match="existing constraint ID"):
        apply_field_constraint_repairs(value, collision, {"bad"}, {"bad"})


def test_semantics_excludes_lexicographic_and_orders_consider_all_fields():
    config, contracts = resources()
    renderer = PromptRenderer(config.paths.prompts)
    family_meanings = {
        family: {"description": family}
        for family in (
            "additive_aggregation",
            "conjunctive_gating",
            "disjunctive_sufficiency",
            "lexicographic_priority",
        )
    }
    catalog = [{
        "field_id": "a", "description": "A", "type": "decimal",
        "unit": "fraction", "pressure_direction": 1,
    }]
    semantics_spec = field_semantics.generation(
        renderer, template_id="scene", scenario={}, field_catalog=catalog,
        family_meanings=family_meanings,
    )
    semantics_schema = contracts.bound_schema(
        semantics_spec.schema, semantics_spec.enum_bindings
    )
    family_enum = semantics_schema["properties"]["fields"]["items"][
        "properties"
    ]["eligible_families"]["items"]["enum"]
    assert "lexicographic_priority" not in family_enum
    assert "lexicographic_priority" not in semantics_spec.user

    fields = {
        "a": {"eligible_families": ["additive_aggregation"]},
        "b": {"eligible_families": []},
        "c": {"eligible_families": ["disjunctive_sufficiency"]},
    }
    order_spec = lexicographic_orders.generation(
        renderer, template_id="scene", scenario={}, fields=fields
    )
    assert order_spec.enum_bindings[
        "/properties/approved_priority_orders/items/properties/fields/items"
    ] == ("a", "b", "c")


def test_assembly_derives_lexicographic_eligibility_from_accepted_orders():
    catalog = {"fields": [
        {
            "field_id": field_id, "type": "decimal", "unit": "fraction",
            "display_name": field_id, "description": field_id,
            "pressure_direction": 1,
        }
        for field_id in ("a", "b", "c", "d")
    ]}
    semantics = {"fields": [
        {
            "field_id": field_id, "effect_strength": 0.5,
            "eligible_families": ["additive_aggregation"],
            "is_reversal": False, "trigger": "", "semantics_before": "",
            "semantics_after": "", "reversal_reason": "",
        }
        for field_id in ("a", "b", "c", "d")
    ]}
    parameters = {"fields": [
        {
            "field_id": field_id, "minimum": 0, "maximum": 1,
            "typical_range": [0.2, 0.8],
            "profile_beta": {"low": [1, 2], "mixed": [2, 2], "high": [2, 1]},
        }
        for field_id in ("a", "b", "c", "d")
    ]}
    template = assemble_template(
        {"template_id": "scene", "rule_families": {
            "lexicographic_priority": {"description": "priority"},
        }},
        catalog, semantics, parameters,
        {"template_id": "scene", "feasibility_constraints": []},
        {"template_id": "scene", "approved_priority_orders": [{
            "order_id": "abc", "fields": ["a", "b", "c"],
            "rationale": "a, then b, then c",
        }]},
        {"template_id": "scene", "replacements": []},
    )
    assert all(
        "lexicographic_priority" in template["fields"][field_id]["eligible_families"]
        for field_id in ("a", "b", "c")
    )
    assert "lexicographic_priority" not in template["fields"]["d"]["eligible_families"]
    assert "lexicographic_alignments" not in template["dynamics"]
    assert "compound_events" not in template["dynamics"]


def test_split_relation_audits_and_repairs_use_owned_component_ids():
    config, _ = resources()
    renderer = PromptRenderer(config.paths.prompts)
    constraints = {
        "feasibility_constraints": [{"id": "constraint"}],
    }
    orders = {
        "approved_priority_orders": [{
            "order_id": "order",
            "rationale": "original",
        }],
    }
    assert semantic_audit_ids("field_constraints", constraints) == {"constraint"}
    assert semantic_audit_ids("lexicographic_orders", orders) == {"order"}
    audit_spec = audits.semantic(
        renderer,
        stage="lexicographic_orders",
        component_id="scene",
        component=orders,
        context={"scenario": {}, "fields": {}},
    )
    assert "Do not audit diversity, catalog coverage" in audit_spec.user
    assert "Return exactly one verdict for every supplied order_id" in audit_spec.user
    assert "__lexicographic_orders__" not in audit_spec.user
    assert component_patch_targets(orders, {
        "component_id": "scene",
        "operations": [{
            "op": "replace",
            "path": "/approved_priority_orders/0/rationale",
            "value_json": '"repaired"',
        }],
    }, "lexicographic_orders") == {"order"}


def test_proxy_replacement_may_use_different_raw_units():
    fields = {
        "risk_share": {
            "type": "decimal", "unit": "fraction", "pressure_direction": 1,
            "eligible_families": ["additive_aggregation"],
        },
        "flex_hours": {
            "type": "integer", "unit": "hours", "pressure_direction": 1,
            "eligible_families": ["additive_aggregation"],
        },
    }
    value = {
        "template_id": "scene",
        "replacements": [{
            "group_id": "capacity_proxy", "members": ["risk_share", "flex_hours"],
            "relation_type": "proxy_substitution",
            "authority_by_field": [
                {"field_id": "risk_share", "authority": "live risk view"},
                {"field_id": "flex_hours", "authority": "staffing control"},
            ],
            "transitions": [
                {"transition_id": "to_hours", "outgoing_field": "risk_share", "incoming_field": "flex_hours"},
                {"transition_id": "to_risk", "outgoing_field": "flex_hours", "incoming_field": "risk_share"},
            ],
        }],
    }
    validate_replacements(value, "scene", fields)

    fields["flex_hours"]["pressure_direction"] = -1
    with pytest.raises(StageValidationError, match="incompatible_direction"):
        validate_replacements(value, "scene", fields)


def test_replacement_generation_sweeps_all_fields_without_identical_role_gate():
    config, _ = resources()
    renderer = PromptRenderer(config.paths.prompts)
    fields = {
        "direct_measure": {
            "type": "decimal", "unit": "fraction", "pressure_direction": 1,
            "eligible_families": ["additive_aggregation"],
        },
        "proxy_measure": {
            "type": "decimal", "unit": "hours", "pressure_direction": 1,
            "eligible_families": ["additive_aggregation"],
        },
    }
    spec = replacement_stage.generation(
        renderer, template_id="scene", scenario={}, fields=fields
    )
    assert "Consider every field pair in both directions" in spec.system
    assert "do not need to be identical" in spec.system
    assert "same frozen `pressure_direction`" in spec.system
    assert spec.enum_bindings[
        "/properties/replacements/items/properties/members/items"
    ] == ("direct_measure", "proxy_measure")


def test_field_parameter_validation_matches_runtime_beta_direction():
    concentration_range = load_config().field_parameters.beta_concentration_range
    value = {
        "template_id": "scene",
        "fields": [{
            "field_id": "pressure",
            "minimum": 0,
            "maximum": 100,
            "typical_range": [10, 90],
            "effect_strength": 0.5,
            "eligible_families": ["additive_aggregation"],
            "profile_beta": {
                "low": [2, 8],
                "mixed": [5, 5],
                "high": [8, 2],
            },
            "parameter_basis": "Increasing values indicate increasing pressure.",
        }],
    }
    validate_field_parameters(
        value, "scene", {"pressure"}, {"pressure": 1}, concentration_range
    )
    value["fields"][0]["profile_beta"]["high"] = [1.2, 8]
    with pytest.raises(StageValidationError, match="profile_direction_mismatch") as exc_info:
        validate_field_parameters(
            value, "scene", {"pressure"}, {"pressure": 1}, concentration_range
        )
    assert exc_info.value.issues[0]["component_id"] == "pressure"


def test_field_parameter_validation_keeps_low_and_mixed_means_typical():
    concentration_range = load_config().field_parameters.beta_concentration_range
    value = {
        "template_id": "scene",
        "fields": [{
            "field_id": "pressure",
            "minimum": 0,
            "maximum": 100,
            "typical_range": [10, 60],
            "effect_strength": 0.5,
            "eligible_families": ["additive_aggregation"],
            "profile_beta": {
                "low": [0.5, 9.5],
                "mixed": [4, 6],
                "high": [8, 2],
            },
            "parameter_basis": "Increasing values indicate increasing pressure.",
        }],
    }
    with pytest.raises(StageValidationError) as exc_info:
        validate_field_parameters(
            value, "scene", {"pressure"}, {"pressure": 1}, concentration_range
        )
    assert exc_info.value.issues == [{
        "path": "/fields/0/profile_beta/low",
        "code": "profile_mean_outside_typical_range",
            "message": "low mapped mean 5 must lie inside typical_range [10, 60]",
        "component_id": "pressure",
    }]


def test_field_parameter_validation_enforces_positive_shapes_and_concentration():
    concentration_range = load_config().field_parameters.beta_concentration_range
    value = {
        "template_id": "scene",
        "fields": [{
            "field_id": "pressure",
            "minimum": 0,
            "maximum": 100,
            "typical_range": [0, 60],
            "profile_beta": {
                "low": [0.5, 3.5],
                "mixed": [5, 5],
                "high": [11.5, 0.5],
            },
            "parameter_basis": "Increasing values indicate increasing pressure.",
        }],
    }
    validate_field_parameters(
        value, "scene", {"pressure"}, {"pressure": 1}, concentration_range
    )

    value["fields"][0]["profile_beta"]["low"] = [0, 4]
    value["fields"][0]["profile_beta"]["high"] = [12.5, 0.5]
    with pytest.raises(StageValidationError) as exc_info:
        validate_field_parameters(
            value, "scene", {"pressure"}, {"pressure": 1}, concentration_range
        )
    assert [item["code"] for item in exc_info.value.issues] == [
        "invalid_beta_shape",
        "invalid_beta_shape",
    ]

    value["fields"][0]["profile_beta"]["low"] = [0, 0]
    with pytest.raises(StageValidationError) as exc_info:
        validate_field_parameters(
            value, "scene", {"pressure"}, {"pressure": 1}, concentration_range
        )
    assert exc_info.value.issues[0]["code"] == "invalid_beta_shape"


def test_dependency_graph_replans_changed_dependencies_and_invalidates_children():
    graph = DependencyGraph()
    graph.add("batch_1")
    graph.add("batch_2")
    graph.add("parameter_set", {"batch_1", "batch_2"})
    graph.add("field_constraints", {"parameter_set"})
    graph.accept("parameter_set", "old")
    graph.accept("field_constraints", "old-child")

    node = graph.add("parameter_set", {"batch_1"})

    assert node.dependencies == {"batch_1"}
    assert node.status == Status.STALE
    assert graph.nodes["field_constraints"].status == Status.STALE
    assert "parameter_set" not in graph.children["batch_2"]


def test_stage_schemas_use_provider_compatible_array_items():
    _, contracts = resources()

    def boolean_items(value):
        if isinstance(value, dict):
            if isinstance(value.get("items"), bool):
                return True
            return any(boolean_items(child) for child in value.values())
        if isinstance(value, list):
            return any(boolean_items(child) for child in value)
        return False

    for name in contracts._paths:
        if name != "canonical":
            assert not boolean_items(contracts.schema(name)), name


def test_provider_schema_omits_unsupported_keywords_without_mutating_local_contract():
    local = {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "string"},
    }
    projected = provider_schema(local)
    assert "uniqueItems" not in projected
    assert local["uniqueItems"] is True


def test_semantic_audit_identity_contract_covers_items_and_sets():
    assert semantic_audit_ids("field_catalog", {
        "fields": [{"field_id": "a"}, {"field_id": "b"}],
    }) == {"a", "b"}
    assert semantic_audit_ids("field_semantics", {
        "fields": [{"field_id": "a"}, {"field_id": "b"}],
    }) == {"a", "b", "__field_semantics_set__"}
    assert semantic_audit_ids("replacements", {
        "replacements": [{
            "group_id": "g",
            "transitions": [{"transition_id": "g_forward"}, {"transition_id": "g_reverse"}],
        }],
    }) == {"g", "g_forward", "g_reverse", "__replacements__"}


def test_catalog_set_audit_protocol_authorizes_only_safe_drops():
    result = {
        "accepted": False,
        "issues": [{
            "code": "strict_duplicate",
            "field_ids": ["raw", "derived"],
            "drop_field_ids": ["derived"],
            "message": "derived is exactly two times raw",
        }, {
            "code": "unavailable_to_agent",
            "field_ids": ["future"],
            "drop_field_ids": ["future"],
            "message": "future is observed only after the decision",
        }],
        "summary": "two blocking issues",
    }
    assert validate_field_catalog_set_audit(
        result, {"raw", "derived", "future"}
    ) == {"derived", "future"}

    invalid = deepcopy(result)
    invalid["issues"][0]["drop_field_ids"] = ["raw", "derived"]
    with pytest.raises(ValueError, match="retain at least one"):
        validate_field_catalog_set_audit(
            invalid, {"raw", "derived", "future"}
        )


def test_local_field_audit_protocol_returns_only_blocking_field_issues():
    result = {
        "issues": [{
            "field_id": "ratio",
            "code": "ambiguous_definition",
            "message": "The ratio does not identify its numerator and denominator.",
        }],
        "summary": "one field requires repair",
    }
    grouped = validate_field_catalog_local_audit(result, {"ratio", "count"})
    assert grouped["count"] == []
    assert grouped["ratio"] == result["issues"]


def test_local_field_repair_cannot_replace_or_remove_a_field():
    validate_field_assignment_patch({
        "component_id": "scene",
        "operations": [{
            "op": "replace",
            "path": "/fields/ratio/description",
            "value_json": '"finite ratio"',
        }],
    }, {"ratio"})
    with pytest.raises(PatchError, match="local field repair may only replace"):
        validate_field_assignment_patch({
            "component_id": "scene",
            "operations": [{
                "op": "remove",
                "path": "/fields/ratio",
                "value_json": "null",
            }],
        }, {"ratio"})


def test_field_semantics_audit_has_explicit_drop_or_repair_resolution():
    config, contracts = resources()
    spec = audits.semantic(
        PromptRenderer(config.paths.prompts),
        stage="field_semantics",
        component_id="scene",
        component={"fields": []},
        context={"scenario": {}, "field_catalog": [], "family_meanings": {}},
    )
    assert spec.schema == "field_semantics_audit"
    resolution = contracts.schema(spec.schema)["properties"]["verdicts"]["items"][
        "properties"
    ]["issues"]["items"]["properties"]["resolution"]
    assert resolution["enum"] == ["repair_current", "drop_field"]


class _Response:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.model = "fake"
        self.usage = {"input_tokens": 1, "output_tokens": 1}
        self.finish_reason = finish_reason


class _FakeClient:
    def __init__(self, responses: list[str | _Response]) -> None:
        self.responses = list(responses)

    async def generate(self, **_: object) -> _Response:
        response = self.responses.pop(0)
        return response if isinstance(response, _Response) else _Response(response)


@pytest.mark.asyncio
async def test_semantic_audit_retries_accepted_verdict_with_blocking_issues():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=1)
    pipeline.caller = SimpleNamespace(call=AsyncMock(side_effect=[
        {
            "verdicts": [{
                "component_id": "field",
                "accepted": True,
                "issues": [{"path": "/", "code": "bad", "message": "blocking"}],
                "summary": "internally inconsistent",
            }],
            "overall_summary": "invalid protocol",
        },
        {
            "verdicts": [{
                "component_id": "field",
                "accepted": True,
                "issues": [],
                "summary": "accepted",
            }],
            "overall_summary": "valid protocol",
        },
    ]))
    result = await pipeline._run_semantic_audit_spec(
        CallSpec(
            stage="field_catalog_audit", component_id="scene",
            schema="semantic_audit", mode="audit", system="", user="audit",
        ),
        expected={"field"},
    )
    assert result["verdicts"][0]["issues"] == []
    assert pipeline.caller.call.await_count == 2


@pytest.mark.asyncio
async def test_checked_records_terminal_audit_issues_after_repair_limit():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(repair=SimpleNamespace(limit=1))
    pipeline.contracts = SimpleNamespace(validate=lambda *_: None)
    pipeline.audit_records = {}
    pipeline.state = SimpleNamespace(
        add_node=lambda *_: None,
        accepted_component=lambda *_: None,
        accept=AsyncMock(),
        record_error=AsyncMock(),
    )
    issue = {
        "component_id": "field",
        "path": "/fields/field",
        "code": "still_invalid",
        "message": "the repaired field is still invalid",
    }
    pipeline._semantic_audit = AsyncMock(return_value=(
        {"verdicts": [], "overall_summary": "rejected"},
        [issue],
    ))
    pipeline._repair = AsyncMock(return_value={"template_id": "scene", "fields": []})

    with pytest.raises(SemanticAuditError):
        await pipeline._checked(
            spec=CallSpec(
                stage="field_catalog", component_id="scene",
                schema="field_catalog", system="", user="",
            ),
            validator=lambda _: None,
            audit_stage="field_catalog",
            audit_context={},
            repair_template="fields/repair.j2",
            repair_context={},
            node_id="scene/field_catalog",
            initial_value={"template_id": "scene", "fields": []},
        )

    terminal = pipeline.state.record_error.await_args.args[0]
    assert terminal == {
        "stage": "field_catalog",
        "component_id": "scene",
        "audit_attempt": 2,
        "repairs_applied": 1,
        "terminal": True,
        "issues": [issue],
    }


@pytest.mark.asyncio
async def test_checked_drops_catalog_set_issues_without_llm_repair():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(
        repair=SimpleNamespace(limit=1),
        catalog_set_drop_limit=1,
    )
    pipeline.contracts = SimpleNamespace(validate=lambda *_: None)
    pipeline.audit_records = {}
    pipeline.state = SimpleNamespace(
        add_node=lambda *_: None,
        accepted_component=lambda *_: None,
        accept=AsyncMock(),
        record_error=AsyncMock(),
    )
    drop_issue = {
        "component_id": "duplicate",
        "path": "/fields/duplicate",
        "code": "strict_duplicate",
        "field_ids": ["kept", "duplicate"],
        "drop_field_ids": ["duplicate"],
        "message": "duplicate is exactly recoverable from kept",
        "resolution": "drop_catalog_field",
    }
    pipeline._semantic_audit = AsyncMock(side_effect=[
        (
            {"accepted": False, "issues": [drop_issue], "summary": "drop one"},
            [drop_issue],
        ),
        (
            {"verdicts": [{
                "component_id": "kept",
                "accepted": True,
                "issues": [],
                "summary": "accepted",
            }], "overall_summary": "accepted"},
            [],
        ),
    ])
    pipeline._repair = AsyncMock()
    value, _ = await pipeline._checked(
        spec=CallSpec(
            stage="field_catalog", component_id="scene",
            schema="field_catalog", system="", user="",
        ),
        validator=lambda _: None,
        audit_stage="field_catalog",
        audit_context={},
        repair_template="fields/repair.j2",
        repair_context={},
        node_id="scene/field_catalog",
        initial_value={
            "template_id": "scene",
            "fields": [{"field_id": "kept"}, {"field_id": "duplicate"}],
        },
    )
    assert [field["field_id"] for field in value["fields"]] == ["kept"]
    pipeline._repair.assert_not_awaited()
    assert pipeline.state.record_error.await_args.args[0]["dropped_field_ids"] == [
        "duplicate"
    ]


@pytest.mark.asyncio
async def test_checked_reaudits_only_unresolved_fields_after_local_repair():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(
        repair=SimpleNamespace(limit=1),
        catalog_set_drop_limit=1,
    )
    pipeline.contracts = SimpleNamespace(validate=lambda *_: None)
    pipeline.audit_records = {}
    pipeline.state = SimpleNamespace(
        add_node=lambda *_: None,
        accepted_component=lambda *_: None,
        accept=AsyncMock(),
        record_error=AsyncMock(),
    )
    issue = {
        "component_id": "bad",
        "path": "/fields/bad",
        "code": "unit_mismatch",
        "message": "unit does not match the assigned measurement",
        "resolution": "repair_field_assignment",
    }
    pipeline._semantic_audit = AsyncMock(side_effect=[
        ({
            "verdicts": [{
                "component_id": "good", "accepted": True,
                "issues": [], "summary": "accepted",
            }, {
                "component_id": "bad", "accepted": False,
                "issues": [{
                    "path": "/fields/bad", "code": "unit_mismatch",
                    "message": "unit does not match the assigned measurement",
                }], "summary": "requires assignment repair",
            }],
            "overall_summary": "local issue",
        }, [issue]),
        ({
            "verdicts": [{
                "component_id": "good", "accepted": True,
                "issues": [], "summary": "frozen",
            }, {
                "component_id": "bad", "accepted": True,
                "issues": [], "summary": "accepted",
            }],
            "overall_summary": "accepted",
        }, []),
    ])
    repaired = {
        "template_id": "scene",
        "fields": [{"field_id": "good"}, {"field_id": "bad", "unit": "units"}],
    }
    pipeline._repair = AsyncMock(return_value=repaired)
    await pipeline._checked(
        spec=CallSpec(
            stage="field_catalog", component_id="scene",
            schema="field_catalog", system="", user="",
        ),
        validator=lambda _: None,
        audit_stage="field_catalog",
        audit_context={},
        repair_template="fields/repair.j2",
        repair_context={},
        node_id="scene/field_catalog",
        initial_value={
            "template_id": "scene",
            "fields": [{"field_id": "good"}, {"field_id": "bad"}],
        },
    )
    calls = pipeline._semantic_audit.await_args_list
    assert calls[0].kwargs["run_catalog_set_audit"] is True
    assert calls[1].kwargs["run_catalog_set_audit"] is False
    assert calls[1].kwargs["frozen_ids"] == {"good"}
    assert pipeline._repair.await_args.kwargs["repair_template"] == (
        "fields/discover_assignment_repair.j2"
    )


@pytest.mark.asyncio
async def test_structured_caller_retries_invalid_fake_response_without_network():
    _, contracts = resources()
    client = _FakeClient([
        '{"accepted": true}',
        json.dumps({"accepted": True, "issues": [], "summary": "accepted"}),
    ])
    caller = StructuredCaller(
        generation_client=client, audit_client=client, repair_client=client,
        contracts=contracts,
        concurrency={"generation": 1, "audit": 1, "repair": 1},
        parse_retry_limit=1,
    )
    result = await caller.call(CallSpec(
        stage="fixture", component_id="fixture", schema="field_catalog_set_audit",
        system="fixture", user="fixture",
    ))
    assert result["accepted"] is True
    assert client.responses == []


@pytest.mark.asyncio
async def test_structured_caller_does_not_retry_completion_budget_exhaustion():
    _, contracts = resources()
    client = _FakeClient([
        _Response("", finish_reason="length"),
        json.dumps({"accepted": True, "issues": [], "summary": "must not run"}),
    ])
    caller = StructuredCaller(
        generation_client=client, audit_client=client, repair_client=client,
        contracts=contracts,
        concurrency={"generation": 1, "audit": 1, "repair": 1},
        parse_retry_limit=2,
    )

    with pytest.raises(RuntimeError, match="exhausted its completion budget"):
        await caller.call(CallSpec(
            stage="fixture", component_id="fixture", schema="field_catalog_set_audit",
            system="fixture", user="fixture",
        ))

    assert len(client.responses) == 1


@pytest.mark.asyncio
async def test_repair_prompt_freezes_internal_component_id_and_required_properties():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=0)
    _, pipeline.contracts = resources()
    pipeline.renderer = SimpleNamespace(render=lambda *args, **kwargs: "repair context")
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    pipeline.caller = SimpleNamespace(call=AsyncMock(return_value={
        "component_id": "scene/batch_001",
        "operations": [{"op": "replace", "path": "/value", "value_json": "2"}],
    }))
    spec = CallSpec(
        stage="field_parameters", component_id="scene/batch_001",
        schema="field_parameters", system="", user="",
    )

    repaired = await pipeline._repair(
        spec=spec,
        value={"value": 1},
        issues=[{"path": "/value", "code": "bad", "message": "bad value"}],
        repair_template="unused.j2",
        repair_context={},
        attempt=1,
    )

    system = pipeline.caller.call.await_args.args[0].system
    assert '"scene/batch_001"' in system
    assert "Never remove a required property" in system
    assert repaired == {"value": 2}


@pytest.mark.asyncio
async def test_field_constraint_repair_uses_id_keyed_decisions():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=0)
    _, pipeline.contracts = resources()
    pipeline.renderer = SimpleNamespace(render=lambda *args, **kwargs: "repair context")
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    pipeline.caller = SimpleNamespace(call=AsyncMock(return_value={
        "component_id": "scene",
        "repairs": [{
            "constraint_id": "bad",
            "action": "replace",
            "replacement_json": json.dumps({
                "description": "fixed",
                "expr": {
                    "op": "le", "left": {"field": "a"},
                    "right": {"field": "b"},
                },
            }),
        }],
    }))
    kept_expr = {
        "op": "ge", "left": {"field": "a"}, "right": {"const": 0},
    }
    value = {
        "template_id": "scene",
        "feasibility_constraints": [
            {"id": "good", "description": "keep", "expr": kept_expr},
            {"id": "bad", "description": "old", "expr": {}},
        ],
    }

    repaired = await pipeline._repair(
        spec=CallSpec(
            stage="field_constraints", component_id="scene",
            schema="field_constraints", system="", user="",
        ),
        value=value,
        issues=[{
            "component_id": "bad", "path": "/feasibility_constraints/1",
            "code": "incorrect", "message": "repair this constraint",
        }],
        repair_template="unused.j2",
        repair_context={},
        attempt=1,
    )

    repair_spec = pipeline.caller.call.await_args.args[0]
    assert repair_spec.schema == "field_constraints_repair"
    assert "constraint_id" in repair_spec.system
    assert "action replace, remove, or split" in repair_spec.system
    assert "Return the smallest RFC-6902" not in repair_spec.system
    assert "Do not use RFC-6902" in repair_spec.system
    assert repair_spec.enum_bindings[
        "/properties/repairs/items/properties/constraint_id"
    ] == ("bad",)
    assert repaired["feasibility_constraints"] == [
        {"id": "good", "description": "keep", "expr": kept_expr},
        {
            "id": "bad",
            "description": "fixed",
            "expr": {
                "op": "le", "left": {"field": "a"},
                "right": {"field": "b"},
            },
        },
    ]


@pytest.mark.asyncio
async def test_field_constraint_repair_retries_malformed_embedded_component():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=1)
    _, pipeline.contracts = resources()
    pipeline.renderer = SimpleNamespace(render=lambda *args, **kwargs: "repair context")
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    valid_expr = {
        "op": "le", "left": {"field": "a"}, "right": {"field": "b"},
    }
    pipeline.caller = SimpleNamespace(call=AsyncMock(side_effect=[
        {
            "component_id": "scene",
            "repairs": [{
                "constraint_id": "bad",
                "action": "replace",
                "replacement_json": json.dumps({
                    "description": "still malformed", "expr": {},
                }),
            }],
        },
        {
            "component_id": "scene",
            "repairs": [{
                "constraint_id": "bad",
                "action": "replace",
                "replacement_json": json.dumps({
                    "description": "fixed", "expr": valid_expr,
                }),
            }],
        },
    ]))

    repaired = await pipeline._repair(
        spec=CallSpec(
            stage="field_constraints", component_id="scene",
            schema="field_constraints", system="", user="",
        ),
        value={
            "template_id": "scene",
            "feasibility_constraints": [{
                "id": "bad",
                "description": "valid shape before semantic repair",
                "expr": {
                    "op": "ge", "left": {"field": "a"},
                    "right": {"const": 0},
                },
            }],
        },
        issues=[{
            "component_id": "bad", "path": "/feasibility_constraints/0/expr",
            "code": "incorrect", "message": "repair this constraint",
        }],
        repair_template="unused.j2",
        repair_context={},
        attempt=1,
    )

    first_spec, retry_spec = [call.args[0] for call in pipeline.caller.call.await_args_list]
    assert retry_spec.enum_bindings == first_spec.enum_bindings
    assert "field_constraints contract failed" in retry_spec.user
    assert repaired["feasibility_constraints"] == [{
        "id": "bad", "description": "fixed", "expr": valid_expr,
    }]


@pytest.mark.asyncio
async def test_field_constraint_repair_retry_preserves_id_binding():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=1)
    _, pipeline.contracts = resources()
    pipeline.renderer = SimpleNamespace(render=lambda *args, **kwargs: "repair context")
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    pipeline.caller = SimpleNamespace(call=AsyncMock(side_effect=[
        {
            "component_id": "scene",
            "repairs": [{
                "constraint_id": "bad",
                "action": "remove",
                "replacement_json": "{}",
            }],
        },
        {
            "component_id": "scene",
            "repairs": [{
                "constraint_id": "bad",
                "action": "remove",
                "replacement_json": "null",
            }],
        },
    ]))

    repaired = await pipeline._repair(
        spec=CallSpec(
            stage="field_constraints", component_id="scene",
            schema="field_constraints", system="", user="",
        ),
        value={
            "template_id": "scene",
            "feasibility_constraints": [{
                "id": "bad", "description": "invalid", "expr": {},
            }],
        },
        issues=[{
            "component_id": "bad", "path": "/feasibility_constraints/0",
            "code": "incorrect", "message": "remove this constraint",
        }],
        repair_template="unused.j2",
        repair_context={},
        attempt=1,
    )

    first_spec, retry_spec = [call.args[0] for call in pipeline.caller.call.await_args_list]
    assert retry_spec.enum_bindings == first_spec.enum_bindings
    assert retry_spec.schema == "field_constraints_repair"
    assert repaired["feasibility_constraints"] == []


@pytest.mark.asyncio
async def test_checked_regenerates_duplicate_field_constraint_ids_before_audit():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(
        parse_retry_limit=1,
        catalog_set_drop_limit=0,
        repair=SimpleNamespace(limit=0),
    )
    _, pipeline.contracts = resources()
    duplicate = {
        "template_id": "scene",
        "feasibility_constraints": [
            {
                "id": "same", "description": "a <= b",
                "expr": {
                    "op": "le", "left": {"field": "a"},
                    "right": {"field": "b"},
                },
            },
            {
                "id": "same", "description": "b <= a",
                "expr": {
                    "op": "le", "left": {"field": "b"},
                    "right": {"field": "a"},
                },
            },
        ],
    }
    unique = {
        "template_id": "scene",
        "feasibility_constraints": [duplicate["feasibility_constraints"][0]],
    }
    pipeline.caller = SimpleNamespace(call=AsyncMock(side_effect=[duplicate, unique]))
    pipeline.state = SimpleNamespace(
        add_node=lambda *args, **kwargs: None,
        accepted_component=lambda *args, **kwargs: None,
        mark_stale=lambda *args, **kwargs: None,
        record_error=AsyncMock(),
        accept=AsyncMock(),
    )
    pipeline.audit_records = {}
    pipeline._semantic_audit = AsyncMock(return_value=(
        {"verdicts": [], "overall_summary": "accepted"}, []
    ))
    fields = {"a": {}, "b": {}}

    value, _ = await pipeline._checked(
        spec=CallSpec(
            stage="field_constraints", component_id="scene",
            schema="field_constraints", system="system", user="user",
            enum_bindings={
                "/properties/template_id": ("scene",),
                "/$defs/number_atom/anyOf/0/properties/field": ("a", "b"),
            },
        ),
        validator=lambda item: validate_field_constraints(item, "scene", fields),
        audit_stage="field_constraints",
        audit_context={},
        repair_template="unused.j2",
        repair_context={},
        node_id="field_constraints",
        generation_retry_codes={"duplicate_id"},
    )

    assert value == unique
    assert pipeline.caller.call.await_count == 2
    retry_spec = pipeline.caller.call.await_args_list[1].args[0]
    assert "unique stable IDs" in retry_spec.user
    pipeline._semantic_audit.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_retries_instead_of_changing_an_accepted_field():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=1)
    _, pipeline.contracts = resources()
    pipeline.renderer = SimpleNamespace(render=lambda *args, **kwargs: "repair context")
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    pipeline.caller = SimpleNamespace(call=AsyncMock(side_effect=[
        {
                "component_id": "scene",
                "operations": [{
                    "op": "replace", "path": "/fields/good/value", "value_json": "99",
                }],
        },
        {
                "component_id": "scene",
                "operations": [{
                    "op": "replace", "path": "/fields/bad/value", "value_json": "3",
            }],
        },
    ]))
    accepted = {"field_id": "good", "value": 1}
    repaired = await pipeline._repair(
        spec=CallSpec(
            stage="field_catalog", component_id="scene",
            schema="field_catalog", system="", user="",
        ),
        value={"fields": [accepted, {"field_id": "bad", "value": 2}]},
        issues=[{
                "component_id": "bad", "path": "/fields/bad/value",
            "code": "incorrect", "message": "repair this field",
        }],
        repair_template="unused.j2",
        repair_context={},
        attempt=1,
        frozen_fields={"good": accepted},
    )
    assert repaired["fields"] == [accepted, {"field_id": "bad", "value": 3}]
    assert pipeline.caller.call.await_count == 2


@pytest.mark.asyncio
async def test_assignment_repair_prompt_receives_only_rejected_fields():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=0)
    _, pipeline.contracts = resources()
    rendered = {}

    def render(*args, **kwargs):
        rendered.update(kwargs)
        return "repair context"

    pipeline.renderer = SimpleNamespace(render=render)
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    pipeline.caller = SimpleNamespace(call=AsyncMock(return_value={
        "component_id": "scene",
        "operations": [{
            "op": "replace",
            "path": "/fields/bad/description",
            "value_json": '"specific finite measurement"',
        }],
    }))
    accepted = {"field_id": "good", "description": "accepted"}
    rejected = {"field_id": "bad", "description": "ambiguous"}
    repaired = await pipeline._repair(
        spec=CallSpec(
            stage="field_catalog", component_id="scene",
            schema="field_catalog", system="", user="",
        ),
        value={"template_id": "scene", "fields": [accepted, rejected]},
        issues=[{
            "component_id": "bad",
            "path": "/fields/bad",
            "code": "ambiguous_definition",
            "message": "assign one numeric meaning",
            "resolution": "repair_field_assignment",
        }],
        repair_template="fields/discover_assignment_repair.j2",
        repair_context={"scenario": {}},
        attempt=1,
        frozen_fields={"good": accepted},
    )
    assert rendered["component"] == {
        "template_id": "scene", "fields": [rejected],
    }
    system = pipeline.caller.call.await_args.args[0].system
    assert "good" not in system
    assert repaired["fields"][0] == accepted
    assert repaired["fields"][1]["description"] == "specific finite measurement"


@pytest.mark.asyncio
async def test_repair_retries_instead_of_changing_an_unrejected_dynamic_component():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=1)
    _, pipeline.contracts = resources()
    pipeline.renderer = SimpleNamespace(render=lambda *args, **kwargs: "repair context")
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    pipeline.caller = SimpleNamespace(call=AsyncMock(side_effect=[
        {
            "component_id": "scene",
            "operations": [{
                "op": "replace", "path": "/replacements/1/rationale",
                "value_json": '"wrong target"',
            }],
        },
        {
            "component_id": "scene",
            "operations": [{
                "op": "replace", "path": "/replacements/0/rationale",
                "value_json": '"fixed target"',
            }],
        },
    ]))
    value = {
        "replacements": [
            {"group_id": "rejected", "rationale": "old", "transitions": []},
            {"group_id": "accepted", "rationale": "keep", "transitions": []},
        ],
    }
    repaired = await pipeline._repair(
        spec=CallSpec(
            stage="replacements", component_id="scene",
            schema="replacements", system="", user="",
        ),
        value=value,
        issues=[{
            "component_id": "rejected", "path": "/replacements/0/rationale",
            "code": "incorrect", "message": "repair this group",
        }],
        repair_template="unused.j2",
        repair_context={},
        attempt=1,
    )
    assert repaired["replacements"][0]["rationale"] == "fixed target"
    assert repaired["replacements"][1]["rationale"] == "keep"
    assert pipeline.caller.call.await_count == 2


@pytest.mark.asyncio
async def test_repair_accepts_wholesale_patch_that_only_changes_rejected_component():
    pipeline = object.__new__(TemplateSynthesisPipeline)
    pipeline.config = SimpleNamespace(parse_retry_limit=1)
    _, pipeline.contracts = resources()
    pipeline.renderer = SimpleNamespace(render=lambda *args, **kwargs: "repair context")
    pipeline.state = SimpleNamespace(record_error=AsyncMock())
    pipeline.caller = SimpleNamespace(call=AsyncMock(return_value={
        "component_id": "scene",
        "operations": [{
            "op": "replace", "path": "/replacements",
            "value_json": json.dumps([
                {"group_id": "rejected", "rationale": "fixed", "transitions": []},
                {"group_id": "accepted", "rationale": "keep", "transitions": []},
            ]),
        }],
    }))
    value = {
        "replacements": [
            {"group_id": "rejected", "rationale": "old", "transitions": []},
            {"group_id": "accepted", "rationale": "keep", "transitions": []},
        ],
    }
    repaired = await pipeline._repair(
        spec=CallSpec(
            stage="replacements", component_id="scene",
            schema="replacements", system="", user="",
        ),
        value=value,
        issues=[{
            "component_id": "rejected", "path": "/replacements/0/rationale",
            "code": "incorrect", "message": "repair this group",
        }],
        repair_template="unused.j2",
        repair_context={},
        attempt=1,
    )
    assert repaired["replacements"][0]["rationale"] == "fixed"
    assert repaired["replacements"][1]["rationale"] == "keep"
    assert pipeline.caller.call.await_count == 1
