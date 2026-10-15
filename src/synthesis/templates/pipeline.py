from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml
from generator import GeneratorError

from scripts._utils import build_generator, merge_model_config
from .assembly import (
    assemble_fields,
    assemble_pool,
    assemble_template,
    merge_field_index,
)
from .batching import BatchPlanner
from .config import SynthesisConfig, validate_resources
from .contracts import Contracts
from .prompting import PromptRenderer
from .repair import (
    PatchError,
    apply_component_patch,
    apply_field_constraint_repairs,
    component_contents,
    component_patch_targets,
    field_patch_targets,
    validate_field_assignment_patch,
)
from .scheduler import gather_jobs
from .state import RunState, atomic_json, canonical_hash
from .structured_call import CallSpec, ProtocolResponseError, StructuredCaller
from .validation import (
    StageValidationError,
    field_parameter_diagnostics,
    validate_canonical,
    validate_field_catalog,
    validate_field_catalog_local_audit,
    validate_field_catalog_set_audit,
    validate_field_parameters,
    validate_field_semantics,
    validate_field_constraints,
    validate_lexicographic_orders,
    validate_replacements,
    validate_references,
    validate_scenario_cores,
    validate_scenario_output,
    semantic_audit_ids,
    validate_overall_audit,
)
from .stages import (
    audits,
    field_constraints,
    field_discovery,
    field_parameters,
    field_semantics,
    lexicographic_orders,
    replacements,
    scenarios,
)


BENCHMARK_BOUNDARY = (
    "One agent observes numeric operational context and selects one of four "
    "ordered intervention actions. All observations in a template must coexist "
    "within one realistic workflow and decision horizon."
)
AUDIT_SCHEMA_VERSION = "adaexplorebench_template_synthesis_audit_v2"
MANIFEST_SCHEMA_VERSION = "adaexplorebench_template_synthesis_manifest_v2"


class SemanticAuditError(ValueError):
    def __init__(
        self, issues: list[dict[str, Any]], *, value: dict[str, Any] | None = None
    ) -> None:
        self.issues = issues
        self.value = value
        super().__init__(f"semantic audit rejected component: {issues}")


class ScenarioRegenerationRequired(RuntimeError):
    def __init__(
        self, *, template_id: str, reversal_count: int, required_count: int,
    ) -> None:
        super().__init__(
            f"{template_id} generated {reversal_count} reversal fields; "
            f"at least {required_count} are required"
        )
        self.template_id = template_id
        self.reversal_count = reversal_count
        self.required_count = required_count


class ScenarioDiversityRejected(RuntimeError):
    def __init__(self, template_id: str, issues: list[dict[str, Any]]) -> None:
        self.template_id = template_id
        self.issues = issues
        super().__init__(f"{template_id} duplicates an accepted domain scenario")


def _model(path: Path, settings: Any) -> Any:
    raw = yaml.safe_load(path.read_text()) or {}
    override = {
        "max_completion_tokens": settings.max_completion_tokens,
        "effort": settings.effort,
        "output_format": settings.output_format,
        "batch_size": settings.concurrency,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "request_retry_limit": settings.request_retry_limit,
        "api_mode": settings.api_mode,
        "background_poll_interval_seconds": settings.background_poll_interval_seconds,
    }
    return build_generator(merge_model_config(raw, override))


def _git_identity() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resource_identity(config: SynthesisConfig) -> str:
    """Hash every tracked input and implementation file that can affect a run."""
    package_root = Path(__file__).resolve().parent
    paths = [
        config.source,
        config.model_config,
        config.audit_model_config,
        config.paths.skeleton,
        config.paths.canonical_schema,
        package_root.parents[2] / "scripts" / "_utils.py",
    ]
    if config.blueprint_path is not None:
        paths.append(config.blueprint_path)
    paths.extend(sorted(package_root.rglob("*.py")))
    paths.extend(sorted(path for path in config.paths.stage_schemas.rglob("*") if path.is_file()))
    paths.extend(sorted(path for path in config.paths.prompts.rglob("*") if path.is_file()))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(config.source.parents[2])).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class TemplateSynthesisPipeline:
    def __init__(self, config: SynthesisConfig, *, run_id: str, resume: bool = False) -> None:
        validate_resources(config)
        self.config = config
        self.resume = resume
        self.contracts = Contracts(config.paths.canonical_schema, config.paths.stage_schemas)
        self.contracts.check_all()
        self.renderer = PromptRenderer(config.paths.prompts)
        self.planner = BatchPlanner(config.batching)
        self.resource_identity = _resource_identity(config)
        self.state = RunState(
            config.paths.runs / run_id,
            resume=resume,
            resource_identity=self.resource_identity,
        )
        generation_client = _model(config.model_config, config.generation)
        audit_client = _model(config.audit_model_config, config.audit)
        repair_client = _model(config.audit_model_config, config.repair)
        for client in (generation_client, audit_client, repair_client):
            client.set_background_store(self.state)
        self.caller = StructuredCaller(
            generation_client=generation_client,
            audit_client=audit_client,
            repair_client=repair_client,
            contracts=self.contracts,
            concurrency={
                "generation": config.generation.concurrency,
                "audit": config.audit.concurrency,
                "repair": config.repair.concurrency,
            },
            parse_retry_limit=config.parse_retry_limit,
            record_call=self.state.record_call,
            record_error=self.state.record_error,
        )
        self.skeleton = json.loads(config.paths.skeleton.read_text())
        self.audit_records = self.state.audit_records()
        self.parent_manifest_sha256: str | None = None

    def _published_base(self) -> dict[str, Any]:
        paths = (
            self.config.paths.canonical,
            self.config.paths.audit,
            self.config.paths.manifest,
        )
        if not any(path.exists() for path in paths):
            return deepcopy(self.skeleton)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(
                f"published template release is incomplete; missing {missing}"
            )
        candidate = json.loads(self.config.paths.canonical.read_text())
        audit = json.loads(self.config.paths.audit.read_text())
        manifest = json.loads(self.config.paths.manifest.read_text())
        self.contracts.validate("canonical", candidate)
        validate_canonical(candidate)
        candidate_hash = canonical_hash(candidate)
        errors = []
        if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
            errors.append("unsupported template audit schema")
        if audit.get("accepted") is not True:
            errors.append("published template audit is not accepted")
        if audit.get("canonical_sha256") != candidate_hash:
            errors.append("template audit does not match canonical template")
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            errors.append("unsupported template manifest schema")
        if manifest.get("canonical_sha256") != candidate_hash:
            errors.append("template manifest does not match canonical template")
        if manifest.get("audit_sha256") != canonical_hash(audit):
            errors.append("template manifest does not match template audit")
        if errors:
            raise ValueError("; ".join(errors))
        self.audit_records = {
            **dict(audit.get("components") or {}),
            **self.audit_records,
        }
        self.parent_manifest_sha256 = canonical_hash(manifest)
        return candidate

    async def _semantic_audit(
        self, *, stage: str, component_id: str, value: dict[str, Any],
        context: dict[str, Any], frozen_ids: set[str] | None = None,
        prior_issues: list[dict[str, Any]] | None = None,
        run_catalog_set_audit: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        frozen_ids = frozen_ids or set()
        if stage == "field_catalog":
            return await self._field_catalog_audit(
                component_id=component_id,
                value=value,
                context=context,
                frozen_ids=frozen_ids,
                prior_issues=prior_issues or [],
                run_set_audit=run_catalog_set_audit,
            )
        if stage == "field_parameters":
            context = {
                **context,
                "parameter_diagnostics": field_parameter_diagnostics(value),
            }
        spec = audits.semantic(
            self.renderer, stage=stage, component_id=component_id,
            component=value, context=context,
        )
        expected = semantic_audit_ids(stage, value)
        spec = CallSpec(
            stage=spec.stage,
            component_id=spec.component_id,
            schema=spec.schema,
            mode=spec.mode,
            system=spec.system,
            user=spec.user,
            enum_bindings={
                "/properties/verdicts/items/properties/component_id": tuple(sorted(expected))
            },
        )
        if frozen_ids:
            spec = CallSpec(
                stage=spec.stage,
                component_id=spec.component_id,
                schema=spec.schema,
                mode=spec.mode,
                system=spec.system,
                user=(
                    spec.user
                    + "\nThese field IDs were accepted by an earlier audit and are immutable: "
                    + json.dumps(sorted(frozen_ids))
                    + ". Return accepted=true with an empty issues list for them without "
                    "re-evaluating them. Audit only unresolved fields and set-level coherence."
                ),
                enum_bindings=spec.enum_bindings,
            )
        if prior_issues:
            spec = CallSpec(
                stage=spec.stage,
                component_id=spec.component_id,
                schema=spec.schema,
                mode=spec.mode,
                system=spec.system,
                user=(
                    spec.user
                    + "\nThis component was repaired against these exact prior issues: "
                    + json.dumps(prior_issues, ensure_ascii=False)
                    + ". First verify every supplied corrected condition against the current "
                    "value. Keep all unchanged, previously accepted properties accepted. Do "
                    "not introduce a new numerical threshold, semantic requirement, or "
                    "interpretation that contradicts the previous verdict. A genuinely new "
                    "blocking issue is valid only when the repair changed the evidence that "
                    "supports it; identify that changed evidence explicitly."
                ),
                enum_bindings=spec.enum_bindings,
            )
        result = await self._run_semantic_audit_spec(
            spec, expected=expected, frozen_ids=frozen_ids
        )
        rejected = [
            {**issue, "component_id": verdict["component_id"]}
            for verdict in result["verdicts"]
            if not verdict["accepted"] or verdict["issues"]
            for issue in verdict["issues"]
        ]
        return result, rejected

    async def _field_catalog_audit(
        self, *, component_id: str, value: dict[str, Any], context: dict[str, Any],
        frozen_ids: set[str], prior_issues: list[dict[str, Any]],
        run_set_audit: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        fields = value["fields"]
        unresolved = [field for field in fields if field["field_id"] not in frozen_ids]
        capacity = self.config.batching.max_field_audit_per_batch

        def with_history(spec: CallSpec, relevant: list[dict[str, Any]]) -> CallSpec:
            if not relevant:
                return spec
            return CallSpec(
                stage=spec.stage,
                component_id=spec.component_id,
                schema=spec.schema,
                mode=spec.mode,
                system=spec.system,
                user=(
                    spec.user
                    + "\nThis catalog was repaired against these prior issues: "
                    + json.dumps(relevant, ensure_ascii=False)
                    + ". Verify those exact corrected conditions without introducing a "
                    "contradictory requirement for unchanged evidence."
                ),
                enum_bindings=spec.enum_bindings,
            )

        async def audit_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
            ids = {field["field_id"] for field in batch}
            spec = audits.semantic(
                self.renderer,
                stage="field_catalog",
                component_id=component_id,
                component={"template_id": value["template_id"], "fields": batch},
                context=context,
            )
            spec = CallSpec(
                stage=spec.stage,
                component_id=spec.component_id,
                schema=spec.schema,
                mode=spec.mode,
                system=spec.system,
                user=spec.user,
                enum_bindings={
                    "/properties/issues/items/properties/field_id": tuple(sorted(ids)),
                },
            )
            relevant = [
                issue for issue in prior_issues
                if issue.get("component_id") in ids
            ]
            return await self._run_field_catalog_local_audit_spec(
                with_history(spec, relevant), field_ids=ids,
            )

        field_ids = {field["field_id"] for field in fields}
        set_summary = "catalog-wide audit already passed"
        if run_set_audit:
            set_spec = audits.field_catalog_set(
                self.renderer,
                component_id=component_id,
                component=value,
                scenario=context["scenario"],
            )
            set_result, set_drop_ids = await self._run_field_catalog_set_audit_spec(
                set_spec, field_ids=field_ids, frozen_ids=frozen_ids,
            )
            if set_drop_ids:
                set_issues = [
                    {
                        **issue,
                        "component_id": field_id,
                        "path": f"/fields/{field_id}",
                        "resolution": "drop_catalog_field",
                    }
                    for issue in set_result["issues"]
                    for field_id in issue["drop_field_ids"]
                ]
                return set_result, set_issues
            set_summary = set_result["summary"]

        batches = (
            [unresolved]
            if unresolved and not run_set_audit
            else [
                unresolved[index:index + capacity]
                for index in range(0, len(unresolved), capacity)
            ]
        )
        local_results = await gather_jobs([
            lambda batch=batch: audit_batch(batch) for batch in batches
        ]) if batches else []
        issues_by_id = {field["field_id"]: [] for field in unresolved}
        for result in local_results:
            for issue in result["issues"]:
                issues_by_id[issue["field_id"]].append({
                    "path": f"/fields/{issue['field_id']}",
                    "code": issue["code"],
                    "message": issue["message"],
                })
        local_by_id = {
            field_id: {
                "component_id": field_id,
                "accepted": not issues,
                "issues": issues,
                "summary": "accepted" if not issues else "requires assignment repair",
            }
            for field_id, issues in issues_by_id.items()
        }
        local_by_id.update({
            field_id: {
                "component_id": field_id,
                "accepted": True,
                "issues": [],
                "summary": "accepted and frozen by an earlier local audit",
            }
            for field_id in frozen_ids
        })

        verdicts = [local_by_id[field["field_id"]] for field in fields]
        result = {
            "verdicts": verdicts,
            "overall_summary": " ".join([
                set_summary,
                *(item["summary"] for item in local_results),
            ]),
        }
        rejected = [
            {
                **issue,
                "component_id": verdict["component_id"],
                "resolution": "repair_field_assignment",
            }
            for verdict in verdicts
            if not verdict["accepted"] or verdict["issues"]
            for issue in verdict["issues"]
        ]
        return result, rejected

    async def _run_field_catalog_local_audit_spec(
        self, spec: CallSpec, *, field_ids: set[str],
    ) -> dict[str, Any]:
        for attempt in range(self.config.parse_retry_limit + 1):
            result = await self.caller.call(spec)
            try:
                validate_field_catalog_local_audit(result, field_ids)
            except ValueError as exc:
                if attempt >= self.config.parse_retry_limit:
                    raise ProtocolResponseError(
                        f"local field audit protocol failed for "
                        f"{spec.component_id}: {exc}"
                    ) from exc
                spec = CallSpec(
                    stage=spec.stage,
                    component_id=spec.component_id,
                    schema=spec.schema,
                    mode=spec.mode,
                    system=spec.system,
                    user=(
                        spec.user
                        + "\nYour previous response violated the sparse local-audit "
                        "protocol: " + str(exc) + ". Return corrected issues only."
                    ),
                    enum_bindings=spec.enum_bindings,
                )
                continue
            return result
        raise AssertionError("unreachable")

    async def _run_field_catalog_set_audit_spec(
        self, spec: CallSpec, *, field_ids: set[str], frozen_ids: set[str],
    ) -> tuple[dict[str, Any], set[str]]:
        for attempt in range(self.config.parse_retry_limit + 1):
            result = await self.caller.call(spec)
            try:
                drop_ids = validate_field_catalog_set_audit(
                    result, field_ids, frozen_ids
                )
            except ValueError as exc:
                if attempt >= self.config.parse_retry_limit:
                    raise ProtocolResponseError(
                        f"catalog set audit protocol failed for "
                        f"{spec.component_id}: {exc}"
                    ) from exc
                spec = CallSpec(
                    stage=spec.stage,
                    component_id=spec.component_id,
                    schema=spec.schema,
                    mode=spec.mode,
                    system=spec.system,
                    user=(
                        spec.user
                        + "\nYour previous response violated the catalog-set protocol: "
                        + str(exc)
                        + ". Return a corrected issue-only audit."
                    ),
                    enum_bindings=spec.enum_bindings,
                )
                continue
            return result, drop_ids
        raise AssertionError("unreachable")

    async def _run_semantic_audit_spec(
        self, spec: CallSpec, *, expected: set[str],
        frozen_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        frozen_ids = frozen_ids or set()
        for attempt in range(self.config.parse_retry_limit + 1):
            result = await self.caller.call(spec)
            verdicts = result["verdicts"]
            received = [verdict["component_id"] for verdict in verdicts]
            protocol_issues = []
            if len(received) != len(set(received)):
                protocol_issues.append("duplicate component_id values")
            if set(received) != expected:
                protocol_issues.append(
                    f"coverage mismatch: missing={sorted(expected-set(received))}, "
                    f"extra={sorted(set(received)-expected)}"
                )
            for verdict in verdicts:
                if verdict["accepted"] and verdict["issues"]:
                    protocol_issues.append(
                        f"{verdict['component_id']} is accepted while reporting issues"
                    )
                if not verdict["accepted"] and not verdict["issues"]:
                    protocol_issues.append(f"{verdict['component_id']} is rejected without an issue")
                if verdict["component_id"] in frozen_ids and (
                    not verdict["accepted"] or verdict["issues"]
                ):
                    protocol_issues.append(
                        f"{verdict['component_id']} re-evaluated an immutable accepted field"
                    )
            if not protocol_issues:
                return result
            if attempt >= self.config.parse_retry_limit:
                raise ProtocolResponseError(
                    f"semantic audit protocol failed for {spec.stage}/{spec.component_id}: "
                    + "; ".join(protocol_issues)
                )
            spec = CallSpec(
                stage=spec.stage,
                component_id=spec.component_id,
                schema=spec.schema,
                mode=spec.mode,
                system=spec.system,
                user=spec.user + "\nYour previous audit violated the coverage protocol: "
                + "; ".join(protocol_issues)
                + f". Return exactly one internally consistent verdict for each of: {sorted(expected)}.",
                enum_bindings=spec.enum_bindings,
            )
        raise AssertionError("unreachable")

    async def _run_domain_diversity_audit(
        self, spec: CallSpec, *, candidate_id: str,
    ) -> dict[str, Any]:
        for attempt in range(self.config.parse_retry_limit + 1):
            result = await self.caller.call(spec)
            try:
                validate_overall_audit(result, "domain diversity audit")
                foreign = {
                    str(issue["component_id"])
                    for issue in result["issues"]
                    if issue["component_id"] != candidate_id
                }
                if foreign:
                    raise ValueError(
                        f"issues target non-candidate templates: {sorted(foreign)}"
                    )
            except ValueError as exc:
                if attempt >= self.config.parse_retry_limit:
                    raise ProtocolResponseError(
                        f"domain diversity audit protocol failed for {candidate_id}: {exc}"
                    ) from exc
                spec = CallSpec(
                    stage=spec.stage,
                    component_id=spec.component_id,
                    schema=spec.schema,
                    mode=spec.mode,
                    system=spec.system,
                    user=(
                        spec.user
                        + "\nThe previous response violated the candidate-only verdict "
                        f"protocol: {exc}. Return one corrected verdict for "
                        f"{candidate_id} only."
                    ),
                    enum_bindings=spec.enum_bindings,
                )
                continue
            return result
        raise AssertionError("unreachable")

    async def _repair(
        self, *, spec: CallSpec, value: dict[str, Any], issues: list[dict[str, Any]],
        repair_template: str, repair_context: dict[str, Any], attempt: int,
        frozen_fields: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        frozen_fields = frozen_fields or {}
        assignment_issue_ids = {
            str(issue["component_id"])
            for issue in issues
            if issue.get("resolution") == "repair_field_assignment"
        }
        assignment_only = bool(assignment_issue_ids) and all(
            issue.get("resolution") == "repair_field_assignment" for issue in issues
        )
        repair_component = value
        if assignment_only:
            repair_component = {
                "template_id": value["template_id"],
                "fields": [
                    field for field in value["fields"]
                    if field["field_id"] in assignment_issue_ids
                ],
            }
        if spec.stage == "field_parameters" and "fields" in value:
            repair_context = {
                **repair_context,
                "parameter_diagnostics": field_parameter_diagnostics(value),
            }
        field_constraint_issue_ids: set[str] = set()
        repair_schema = "component_patch"
        repair_enum_bindings: dict[str, tuple[Any, ...]] = {}
        if spec.stage == "field_constraints":
            field_constraint_issue_ids = {
                str(issue["component_id"])
                for issue in issues
                if issue.get("component_id")
            }
            if (
                not field_constraint_issue_ids
                or "__field_constraints__" in field_constraint_issue_ids
            ):
                raise ValueError(
                    "field constraint repair requires unique rejected constraint IDs"
                )
            repair_schema = "field_constraints_repair"
            repair_enum_bindings = {
                "/properties/component_id": (spec.component_id,),
                "/properties/repairs/items/properties/constraint_id": tuple(
                    sorted(field_constraint_issue_ids)
                ),
            }
            repair_system = (
                "Return only the field-constraint repair response required by the output "
                f"schema. Set component_id to exactly {json.dumps(spec.component_id)}. "
                "Return exactly one repairs item for every supplied rejected constraint ID "
                "and no other ID. Each item identifies its target with constraint_id and uses "
                "action replace, remove, or split. For remove, replacement_json must be the JSON "
                "text null. For replace, replacement_json must encode exactly one object with "
                "description and expr. For split, it must encode an array of at least two such "
                "objects. Omit id because the program assigns stable identities. Do not use "
                "RFC-6902 operations, paths, or array indices."
            )
        else:
            repair_system = (
                "Return the smallest RFC-6902-style component patch that fixes only the supplied "
                f"issues. Set component_id to exactly {json.dumps(spec.component_id)}. Preserve "
                "accepted IDs, unrelated content, and every property required by the input "
                "contract. Never remove a required property; replace its value with a valid "
                "value instead. Encode every operation value as JSON text in value_json; for "
                "remove use the JSON text null. Never patch the component root. In value_json "
                "use valid JSON with ASCII punctuation and no U+0000-U+001F control characters; "
                "a string value must include its JSON double quotes. For any component whose "
                "fields property is an array, address an existing item as /fields/FIELD_ID or "
                "/fields/FIELD_ID/property, never by numeric array index; use /fields/- only to "
                "append a new field."
                + (
                    " The following field IDs are already accepted and immutable: "
                    + json.dumps(sorted(frozen_fields))
                    + ". Do not modify, remove, or reorder their field objects."
                    if frozen_fields and not assignment_only
                    else ""
                )
            )
        patch_spec = CallSpec(
            stage=f"{spec.stage}_repair", component_id=spec.component_id,
            schema=repair_schema, mode="repair",
            system=repair_system,
            user=self.renderer.render(
                repair_template,
                component=repair_component,
                issues=issues,
                target_schema=self.contracts.bound_schema(spec.schema, spec.enum_bindings),
                **repair_context,
            ),
            enum_bindings=repair_enum_bindings,
        )
        last_error: Exception | None = None
        for patch_attempt in range(self.config.parse_retry_limit + 1):
            patch = await self.caller.call(patch_spec)
            try:
                if patch["component_id"] != spec.component_id:
                    raise ValueError("repair response changed component identity")
                if spec.stage == "field_constraints":
                    field_constraint_split_ids = {
                        str(issue["component_id"])
                        for issue in issues
                        if issue.get("code") == "bundled_independent_invariants"
                        and issue.get("component_id")
                    }
                    repaired = apply_field_constraint_repairs(
                        value, patch, field_constraint_issue_ids,
                        field_constraint_split_ids,
                    )
                    self.contracts.validate(spec.schema, repaired)
                    break
                if assignment_only:
                    validate_field_assignment_patch(patch, assignment_issue_ids)
                if frozen_fields:
                    issue_ids = {
                        str(issue["component_id"])
                        for issue in issues
                        if issue.get("component_id")
                    }
                    unauthorized = field_patch_targets(patch) - issue_ids
                    if unauthorized:
                        raise ValueError(
                            "repair targeted fields without rejected verdicts: "
                            f"{sorted(unauthorized)}"
                        )
                allowed_components: set[str] | None = None
                if spec.stage in {
                    "lexicographic_orders", "replacements",
                }:
                    issue_ids = {
                        str(issue["component_id"])
                        for issue in issues
                        if issue.get("component_id")
                    }
                    sentinel = {
                        "lexicographic_orders": "__lexicographic_orders__",
                        "replacements": "__replacements__",
                    }[spec.stage]
                    if issue_ids and sentinel not in issue_ids:
                        component_patch_targets(value, patch, spec.stage)
                        allowed_components = set(issue_ids)
                        if spec.stage == "replacements":
                            allowed_components.update(
                                transition["transition_id"]
                                for group in value["replacements"]
                                if group["group_id"] in issue_ids
                                for transition in group["transitions"]
                            )
                repaired = apply_component_patch(value, patch)
                if allowed_components is not None:
                    before = component_contents(value, spec.stage)
                    after = component_contents(repaired, spec.stage)
                    changed_components = {
                        component_id
                        for component_id in before.keys() | after.keys()
                        if before.get(component_id) != after.get(component_id)
                    }
                    overreach = sorted(changed_components - allowed_components)
                    if overreach:
                        raise ValueError(
                            "repair changed components without rejected verdicts: "
                            f"{overreach}"
                        )
                if frozen_fields:
                    repaired_by_id = {
                        field["field_id"]: field for field in repaired.get("fields") or []
                    }
                    changed = [
                        field_id
                        for field_id, accepted in frozen_fields.items()
                        if repaired_by_id.get(field_id) != accepted
                    ]
                    if changed:
                        raise ValueError(
                            f"repair changed immutable accepted fields: {sorted(changed)}"
                        )
            except (PatchError, ValueError) as exc:
                last_error = exc
                if patch_attempt >= self.config.parse_retry_limit:
                    raise ProtocolResponseError(
                        f"{spec.stage}/{spec.component_id} repair protocol failed: {exc}"
                    ) from exc
                patch_spec = CallSpec(
                    stage=patch_spec.stage,
                    component_id=patch_spec.component_id,
                    schema=patch_spec.schema,
                    mode=patch_spec.mode,
                    system=patch_spec.system,
                    user=(
                        patch_spec.user
                        + "\nThe previous repair response could not be applied: "
                        + str(exc)
                        + "\nPrevious repair response: "
                        + json.dumps(patch, ensure_ascii=False)
                        + "\nReturn a corrected response under the same output schema."
                    ),
                    enum_bindings=patch_spec.enum_bindings,
                )
                continue
            break
        else:
            raise AssertionError("unreachable") from last_error
        await self.state.record_error({
            "stage": spec.stage, "component_id": spec.component_id,
            "repair_attempt": attempt, "issues": issues,
        })
        return repaired

    async def _checked(
        self, *, spec: CallSpec, validator: Callable[[dict[str, Any]], None],
        audit_stage: str, audit_context: dict[str, Any],
        repair_template: str, repair_context: dict[str, Any], node_id: str,
        dependencies: set[str] | None = None,
        initial_value: dict[str, Any] | None = None,
        generation_retry_codes: set[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.state.add_node(node_id, dependencies)
        freeze_fields = audit_stage in {
            "field_catalog", "field_semantics", "field_parameters",
        }
        frozen_fields: dict[str, dict[str, Any]] = {}
        prior_issues: list[dict[str, Any]] = []
        resumed = self.state.accepted_component(node_id)
        if resumed is not None:
            try:
                self.contracts.validate(spec.schema, resumed)
                self._require_reversal_floor(audit_stage, resumed)
                validator(resumed)
            except ScenarioRegenerationRequired:
                raise
            except Exception:
                self.state.mark_stale(node_id)
            else:
                audit_result = self.state.audit_record(node_id)
                if audit_result is None:
                    raise ValueError(
                        f"accepted component lacks persisted audit record: {node_id}"
                    )
                self.audit_records[node_id] = audit_result
                return resumed, audit_result
        if initial_value is not None:
            value = deepcopy(initial_value)
        else:
            generation_retry_codes = generation_retry_codes or set()
            generation_spec = spec
            for generation_attempt in range(self.config.parse_retry_limit + 1):
                value = await self.caller.call(generation_spec)
                if not generation_retry_codes:
                    break
                try:
                    validator(value)
                except StageValidationError as exc:
                    issue_codes = {str(item.get("code")) for item in exc.issues}
                    if not issue_codes & generation_retry_codes:
                        break
                    if generation_attempt >= self.config.parse_retry_limit:
                        raise
                    await self.state.record_error({
                        "stage": spec.stage,
                        "component_id": spec.component_id,
                        "mode": "generation_protocol",
                        "attempt": generation_attempt + 1,
                        "issues": exc.issues,
                    })
                    generation_spec = CallSpec(
                        stage=spec.stage,
                        component_id=spec.component_id,
                        schema=spec.schema,
                        mode=spec.mode,
                        system=spec.system,
                        user=(
                            spec.user
                            + "\nThe previous generated component violated this "
                            "deterministic identity protocol: "
                            + json.dumps(exc.issues, ensure_ascii=False)
                            + ". Return a complete regenerated component with unique "
                            "stable IDs."
                        ),
                        enum_bindings=spec.enum_bindings,
                    )
                    continue
                break
        repair_attempt = 0
        catalog_drop_attempt = 0
        catalog_set_passed = audit_stage != "field_catalog"
        while True:
            issues: list[dict[str, Any]] = []
            try:
                self.contracts.validate(spec.schema, value)
                self._require_reversal_floor(audit_stage, value)
                validator(value)
            except ScenarioRegenerationRequired:
                raise
            except StageValidationError as exc:
                issues = exc.issues
            except Exception as exc:
                issues = [{"path": "/", "code": "program_validation", "message": str(exc)}]
            audit_result: dict[str, Any] = {"verdicts": [], "overall_summary": "program validation failed"}
            if not issues:
                audit_result, issues = await self._semantic_audit(
                    stage=audit_stage, component_id=spec.component_id,
                    value=value, context=audit_context,
                    frozen_ids=set(frozen_fields),
                    prior_issues=prior_issues,
                    run_catalog_set_audit=not catalog_set_passed,
                )
                if audit_stage == "field_catalog" and not any(
                    issue.get("resolution") == "drop_catalog_field"
                    for issue in issues
                ):
                    catalog_set_passed = True
                if freeze_fields:
                    verdicts = audit_result.get("verdicts", [])
                    fields_by_id = {
                        field["field_id"]: field for field in value["fields"]
                    }
                    for verdict in verdicts:
                        field_id = verdict["component_id"]
                        if (
                            field_id in fields_by_id
                            and verdict["accepted"]
                            and not verdict["issues"]
                        ):
                            frozen_fields.setdefault(
                                field_id, deepcopy(fields_by_id[field_id])
                            )
            if not issues:
                await self.state.accept(
                    node_id, value, audit_record=audit_result
                )
                self.audit_records[node_id] = audit_result
                return value, audit_result
            catalog_drop_ids = {
                str(issue["component_id"])
                for issue in issues
                if issue.get("resolution") == "drop_catalog_field"
            }
            if catalog_drop_ids:
                if audit_stage != "field_catalog":
                    raise ValueError(
                        "drop_catalog_field resolution is valid only for field_catalog"
                    )
                if catalog_drop_attempt >= self.config.catalog_set_drop_limit:
                    await self.state.record_error({
                        "stage": "field_catalog_set_audit",
                        "component_id": spec.component_id,
                        "drop_attempt": catalog_drop_attempt + 1,
                        "terminal": True,
                        "issues": issues,
                    })
                    raise SemanticAuditError(issues, value=value)
                value = {
                    **value,
                    "fields": [
                        field for field in value["fields"]
                        if field["field_id"] not in catalog_drop_ids
                    ],
                }
                catalog_drop_attempt += 1
                prior_issues = deepcopy(issues)
                await self.state.record_error({
                    "stage": "field_catalog_set_audit",
                    "component_id": spec.component_id,
                    "drop_attempt": catalog_drop_attempt,
                    "dropped_field_ids": sorted(catalog_drop_ids),
                    "issues": issues,
                })
                continue
            if any(issue.get("resolution") == "drop_field" for issue in issues):
                raise SemanticAuditError(issues, value=value)
            if repair_attempt >= self.config.repair.limit:
                await self.state.record_error({
                    "stage": spec.stage,
                    "component_id": spec.component_id,
                    "audit_attempt": repair_attempt + catalog_drop_attempt + 1,
                    "repairs_applied": repair_attempt,
                    "terminal": True,
                    "issues": issues,
                })
                raise SemanticAuditError(issues)
            prior_issues = deepcopy(issues)
            repair_attempt += 1
            selected_repair_template = (
                "fields/discover_assignment_repair.j2"
                if issues and all(
                    issue.get("resolution") == "repair_field_assignment"
                    for issue in issues
                )
                else repair_template
            )
            value = await self._repair(
                spec=spec, value=value, issues=issues,
                repair_template=selected_repair_template,
                repair_context=repair_context,
                attempt=repair_attempt,
                frozen_fields=frozen_fields,
            )

    def _require_reversal_floor(
        self, audit_stage: str, value: dict[str, Any]
    ) -> None:
        if audit_stage != "field_semantics":
            return
        reversal_count = sum(
            field.get("is_reversal") is True for field in value["fields"]
        )
        required_count = self.config.minimum_reversal_fields
        if reversal_count < required_count:
            raise ScenarioRegenerationRequired(
                template_id=str(value["template_id"]),
                reversal_count=reversal_count,
                required_count=required_count,
            )

    def _validate_component_bundle(
        self,
        *,
        template_id: str,
        catalog: dict[str, Any],
        semantics: dict[str, Any],
        parameters: dict[str, Any],
        constraints_value: dict[str, Any],
        lexicographic_value: dict[str, Any],
        replacements_value: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate a complete accepted or deterministically filtered component bundle."""
        self.contracts.validate("field_catalog", catalog)
        self.contracts.validate("field_semantics", semantics)
        self.contracts.validate("field_parameters", parameters)
        self.contracts.validate("field_constraints", constraints_value)
        self.contracts.validate("lexicographic_orders", lexicographic_value)
        self.contracts.validate("replacements", replacements_value)
        field_ids = {item["field_id"] for item in catalog["fields"]}
        directions = {
            item["field_id"]: item["pressure_direction"]
            for item in catalog["fields"]
        }
        validate_field_catalog(catalog, template_id)
        validate_field_semantics(semantics, template_id, field_ids)
        validate_field_parameters(
            parameters,
            template_id,
            field_ids,
            directions,
            self.config.field_parameters.beta_concentration_range,
        )
        fields = assemble_fields(catalog, semantics, parameters)
        validate_field_constraints(constraints_value, template_id, fields)
        validate_lexicographic_orders(lexicographic_value, template_id, fields)
        validate_replacements(replacements_value, template_id, fields)
        validate_references(
            fields=fields,
            lexicographic_orders=lexicographic_value,
            replacements=replacements_value,
        )
        return fields

    async def run(
        self, *, domains: tuple[str, ...] | None = None, append: bool = True,
        publish: bool = False, stop_after: str | None = None,
    ) -> dict[str, Any] | None:
        if stop_after not in {None, "field_constraints"}:
            raise ValueError(f"unsupported stop stage: {stop_after}")
        if publish and stop_after:
            raise ValueError("cannot publish a stage-limited synthesis run")
        selected = domains or self.config.domains
        unknown = set(selected) - set(self.config.domains)
        if unknown:
            raise ValueError(f"unknown configured domains: {sorted(unknown)}")
        blueprint_mode = self.config.blueprint_path is not None
        if blueprint_mode and append:
            if getattr(self, "resume", False):
                append = False
            else:
                raise ValueError(
                    "blueprint mode does not support append; use --fresh for a new run"
                )
        base = self._published_base() if append else deepcopy(self.skeleton)
        self.state.bind_inputs({
            "domains": list(selected),
            "append": append,
            "blueprint_mode": blueprint_mode,
            "base_sha256": canonical_hash(base),
        })
        family_meanings = self.skeleton["defaults"]["rule_families"]

        async def domain_job(domain_id: str) -> tuple[str, dict[str, Any] | None]:
            existing = (base.get("domains") or {}).get(domain_id, {})
            domain_description = self.config.domain_descriptions[domain_id]
            if existing and existing.get("description") != domain_description:
                raise ValueError(
                    f"configured description for {domain_id} does not match the "
                    "published domain description"
                )
            existing_templates = [
                {"template_id": template_id, "description": item.get("description"), "task_id": item.get("task_id"), "agent_role": item.get("agent_role")}
                for template_id, item in (existing.get("templates") or {}).items()
            ]

            def complete_summary(template_id: str, item: dict[str, Any]) -> dict[str, Any]:
                return {
                    "template_id": template_id,
                    "task_id": item["task_id"],
                    "description": item["description"],
                    "agent_role": item["agent_role"],
                    "field_ids": sorted(item["fields"]),
                    "reversal_field_ids": sorted(
                        field_id for field_id, field in item["fields"].items()
                        if field["is_reversal"]
                    ),
                    "reversal_event_ids": sorted(
                        item["dynamics"]["direction_reversals"]
                    ),
                }

            accepted_summaries = [
                complete_summary(template_id, item)
                for template_id, item in (existing.get("templates") or {}).items()
            ]
            accepted_ids = {item["template_id"] for item in accepted_summaries}
            diversity_lock = asyncio.Lock()

            async def check_diversity(
                core: dict[str, Any], catalog: dict[str, Any], semantics: dict[str, Any],
                *, node_id: str,
            ) -> None:
                candidate = {
                    "template_id": core["template_id"],
                    "task_id": core["task_id"],
                    "description": core["description"],
                    "agent_role": core["agent_role"],
                    "field_ids": sorted(field["field_id"] for field in catalog["fields"]),
                    "reversal_field_ids": sorted(
                        field["field_id"] for field in semantics["fields"]
                        if field["is_reversal"]
                    ),
                    "reversal_event_ids": sorted(
                        field["field_id"] for field in semantics["fields"]
                        if field["is_reversal"]
                    ),
                }
                async with diversity_lock:
                    if candidate["template_id"] in accepted_ids:
                        return
                    self.state.add_node(
                        node_id,
                        set(),
                    )
                    result = self.state.accepted_component(node_id)
                    if result is None:
                        result = await self._run_domain_diversity_audit(
                            audits.domain(
                                self.renderer,
                                domain_id=domain_id,
                                accepted_templates=accepted_summaries,
                                candidate=candidate,
                            ),
                            candidate_id=candidate["template_id"],
                        )
                        await self.state.accept(
                            node_id, result, audit_record=result
                        )
                    if not result["accepted"]:
                        await self.state.record_error({
                            "stage": "domain_diversity",
                            "component_id": candidate["template_id"],
                            "action": "drop_scenario",
                            "issues": result["issues"],
                        })
                        raise ScenarioDiversityRejected(
                            candidate["template_id"], result["issues"]
                        )
                    self.audit_records[node_id] = result
                    accepted_summaries.append(candidate)
                    accepted_ids.add(candidate["template_id"])

            def exception_leaves(exc: BaseException) -> list[BaseException]:
                if isinstance(exc, BaseExceptionGroup):
                    return [
                        leaf
                        for child in exc.exceptions
                        for leaf in exception_leaves(child)
                    ]
                return [exc]

            if blueprint_mode:
                slots = [
                    (
                        blueprint.task_id,
                        blueprint.task_id,
                        blueprint.task_id,
                        blueprint.prompt_value(),
                    )
                    for blueprint in self.config.task_blueprints[domain_id]
                ]
            else:
                used_ids = set(existing.get("templates") or {})
                slots = []
                next_index = 1
                while len(slots) < self.config.scenario_batch_size:
                    identity = f"{domain_id}_scenario_{next_index:03d}"
                    next_index += 1
                    if identity in used_ids:
                        continue
                    slots.append((identity, identity, identity, None))

            async def run_slot(
                slot_id: str,
                template_id: str,
                task_id: str,
                blueprint: dict[str, Any] | None,
            ) -> tuple[str, dict[str, Any] | None]:
                for scenario_attempt in range(self.config.scenario_refill_limit + 1):
                    attempt_prefix = (
                        f"{domain_id}/scenario_slots/{slot_id}/"
                        f"attempt_{scenario_attempt + 1:03d}"
                    )
                    scenario_node = f"{attempt_prefix}/scenario_core"
                    rejection_node = f"{attempt_prefix}/rejected"
                    if rejection_node in self.state.graph.nodes and (
                        self.state.accepted_component(rejection_node) is not None
                    ):
                        continue
                    try:
                        self.state.add_node(scenario_node)
                        assembled = self.state.accepted_component(scenario_node)
                        if assembled is None:
                            generation_spec = scenarios.generation(
                                self.renderer,
                                domain_id=domain_id,
                                domain_description=domain_description,
                                template_id=template_id,
                                benchmark_boundary=BENCHMARK_BOUNDARY,
                                family_meanings=family_meanings,
                                blueprint=blueprint,
                                existing_templates=existing_templates,
                            )
                            generated_output = await self.caller.call(generation_spec)
                            validate_scenario_output(generated_output)
                            assembled = scenarios.assemble(
                                generated_output,
                                domain_id=domain_id,
                                domain_description=domain_description,
                                template_id=template_id,
                                task_id=task_id,
                                blueprint=blueprint,
                            )
                        review_spec = scenarios.review(
                            self.renderer, template_id=template_id
                        )
                        assembled, _ = await self._checked(
                            spec=review_spec,
                            validator=lambda item: validate_scenario_cores(
                                item,
                                domain_id,
                                domain_description=domain_description,
                                template_id=template_id,
                                task_id=task_id,
                                blueprint=blueprint,
                            ),
                            audit_stage="scenario_cores",
                            audit_context={
                                "domain_description": domain_description,
                                "family_meanings": scenarios.family_descriptions(
                                    family_meanings
                                ),
                                "blueprint": blueprint,
                                "existing_templates": existing_templates,
                            },
                            repair_template="scenarios/repair.j2",
                            repair_context={
                                "domain_description": domain_description,
                                "family_meanings": scenarios.family_descriptions(
                                    family_meanings
                                ),
                                "blueprint": blueprint,
                            },
                            node_id=scenario_node,
                            initial_value=assembled,
                        )
                        core = assembled["templates"][0]

                        diversity = None
                        if not blueprint_mode:
                            async def diversity(
                                candidate_core: dict[str, Any],
                                catalog: dict[str, Any],
                                semantics: dict[str, Any],
                            ) -> None:
                                await check_diversity(
                                    candidate_core,
                                    catalog,
                                    semantics,
                                    node_id=f"{attempt_prefix}/domain_diversity",
                                )

                        result = None
                        for retry in range(self.config.template_retry_limit + 1):
                            try:
                                result = await self._template_job(
                                    domain_id=domain_id,
                                    domain_description=domain_description,
                                    core=core,
                                    defaults=base["defaults"],
                                    scenario_node=scenario_node,
                                    node_prefix=f"{attempt_prefix}/template_pipeline",
                                    stop_after=stop_after,
                                    diversity_check=diversity,
                                )
                                break
                            except BaseException as exc:
                                leaves = exception_leaves(exc)
                                recoverable = all(isinstance(item, (
                                    ScenarioRegenerationRequired,
                                    ScenarioDiversityRejected,
                                    ProtocolResponseError,
                                    SemanticAuditError,
                                    GeneratorError,
                                    StageValidationError,
                                )) for item in leaves)
                                if not recoverable:
                                    raise
                                immediate_drop = any(isinstance(item, (
                                    ScenarioRegenerationRequired,
                                    ScenarioDiversityRejected,
                                )) for item in leaves)
                                if not immediate_drop and retry < self.config.template_retry_limit:
                                    await self.state.record_error({
                                        "stage": "template_retry",
                                        "component_id": template_id,
                                        "slot_id": slot_id,
                                        "scenario_attempt": scenario_attempt + 1,
                                        "retry": retry + 1,
                                        "errors": [str(item) for item in leaves],
                                    })
                                    continue
                                raise
                        if result is None:
                            raise AssertionError("template retry loop returned no result")
                        if result[1] is not None and template_id not in accepted_ids:
                            async with diversity_lock:
                                if template_id not in accepted_ids:
                                    accepted_summaries.append(
                                        complete_summary(template_id, result[1])
                                    )
                                    accepted_ids.add(template_id)
                        return result
                    except BaseException as exc:
                        leaves = exception_leaves(exc)
                        if not all(isinstance(item, (
                            ScenarioRegenerationRequired,
                            ScenarioDiversityRejected,
                            ProtocolResponseError,
                            SemanticAuditError,
                            GeneratorError,
                            StageValidationError,
                        )) for item in leaves):
                            raise
                        await self.state.record_error({
                            "stage": "template_candidate",
                            "component_id": template_id,
                            "slot_id": slot_id,
                            "scenario_attempt": scenario_attempt + 1,
                            "action": "drop_scenario",
                            "errors": [str(item) for item in leaves],
                        })
                        self.state.add_node(rejection_node)
                        await self.state.accept(rejection_node, {
                            "template_id": template_id,
                            "errors": [str(item) for item in leaves],
                        })
                        async with diversity_lock:
                            accepted_ids.discard(template_id)
                            accepted_summaries[:] = [
                                item for item in accepted_summaries
                                if item["template_id"] != template_id
                            ]
                raise RuntimeError(
                    f"{domain_id}/{slot_id} exhausted "
                    f"{self.config.scenario_refill_limit + 1} scenario attempts"
                )

            results = await gather_jobs([
                lambda slot=slot: run_slot(*slot) for slot in slots
            ])
            if stop_after:
                return domain_id, None
            generated = {
                template_id: template
                for template_id, template in results
                if template is not None
            }
            if len(generated) != len(slots):
                raise RuntimeError(
                    f"{domain_id} generated {len(generated)} of {len(slots)} required templates"
                )
            templates = deepcopy(existing.get("templates") or {})
            templates.update(generated)
            return domain_id, {
                "description": domain_description,
                "templates": templates,
            }

        domain_results = await gather_jobs([lambda domain_id=domain_id: domain_job(domain_id) for domain_id in selected])
        if stop_after:
            return None
        domains_value = deepcopy(base.get("domains") or {})
        domains_value.update(dict(domain_results))
        candidate = assemble_pool(self.skeleton, domains_value)
        self.contracts.validate("canonical", candidate)
        complete = set(candidate["domains"]) == set(self.config.domains)
        if complete:
            validate_canonical(candidate)
        elif publish:
            raise ValueError("cannot publish a partial-domain template candidate")
        self.state.write_candidate(candidate)
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "canonical_sha256": canonical_hash(candidate),
            "accepted": True,
            "components": self.audit_records,
        }
        manifest = self._manifest(candidate, audit)
        atomic_json(self.state.run_dir / "candidate" / "domain_templates.audit.json", audit)
        atomic_json(self.state.run_dir / "candidate" / "domain_templates.manifest.json", manifest)
        if publish:
            atomic_json(self.config.paths.canonical, candidate)
            atomic_json(self.config.paths.audit, audit)
            atomic_json(self.config.paths.manifest, manifest)
        return candidate

    async def _template_job(
        self,
        *,
        domain_id: str,
        domain_description: str,
        core: dict[str, Any],
        defaults: dict[str, Any],
        scenario_node: str,
        node_prefix: str | None = None,
        stop_after: str | None = None,
        diversity_check: Callable[[
            dict[str, Any], dict[str, Any], dict[str, Any]
        ], Awaitable[None]] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        template_id = core["template_id"]
        base_node = node_prefix or f"{domain_id}/{template_id}"
        template_node = f"{base_node}/template"
        if template_node in self.state.graph.nodes:
            resumed_template = self.state.accepted_component(template_node)
            if resumed_template is not None:
                self.contracts.validate_template(resumed_template)
                return template_id, None if stop_after else resumed_template
        scenario_context = {
            "domain_id": domain_id,
            "domain_description": domain_description,
            **core,
        }
        catalog_spec = field_discovery.generation(
            self.renderer, template_id=template_id, scenario=scenario_context,
        )
        catalog, _ = await self._checked(
            spec=catalog_spec,
            validator=lambda value: validate_field_catalog(value, template_id),
            audit_stage="field_catalog", audit_context={"scenario": scenario_context},
            repair_template="fields/repair.j2", repair_context={"scenario": scenario_context},
            node_id=f"{base_node}/field_catalog",
            dependencies={scenario_node},
        )
        semantics_seed: dict[str, Any] | None = None
        semantic_family_meanings = field_semantics.supported_family_meanings(
            defaults["rule_families"]
        )
        for drop_pass in range(self.config.semantics_drop_limit + 1):
            field_ids = [field["field_id"] for field in catalog["fields"]]
            semantics_spec = field_semantics.generation(
                self.renderer,
                template_id=template_id,
                scenario=scenario_context,
                field_catalog=catalog["fields"],
                family_meanings=semantic_family_meanings,
            )
            try:
                semantics, _ = await self._checked(
                    spec=semantics_spec,
                    validator=lambda value: validate_field_semantics(
                        value, template_id, set(field_ids)
                    ),
                    audit_stage="field_semantics",
                    audit_context={
                        "scenario": scenario_context,
                        "field_catalog": catalog["fields"],
                        "family_meanings": semantic_family_meanings,
                    },
                    repair_template="fields/semantics_repair.j2",
                    repair_context={
                        "scenario": scenario_context,
                        "field_catalog": catalog["fields"],
                        "family_meanings": semantic_family_meanings,
                    },
                    node_id=f"{base_node}/field_semantics",
                    dependencies={f"{base_node}/field_catalog"},
                    initial_value=semantics_seed,
                )
            except SemanticAuditError as exc:
                drop_ids = {
                    str(item["component_id"])
                    for item in exc.issues
                    if item.get("resolution") == "drop_field"
                    and item.get("component_id") in set(field_ids)
                }
                if not drop_ids or drop_pass >= self.config.semantics_drop_limit:
                    raise
                if exc.value is None:
                    raise ValueError("cross-stage drop requires the rejected semantics value")
                catalog = {
                    **catalog,
                    "fields": [
                        field for field in catalog["fields"]
                        if field["field_id"] not in drop_ids
                    ],
                }
                self.contracts.validate("field_catalog", catalog)
                validate_field_catalog(catalog, template_id)
                semantics_seed = {
                    **exc.value,
                    "fields": [
                        field for field in exc.value["fields"]
                        if field["field_id"] not in drop_ids
                    ],
                }
                event = {
                    "accepted": True,
                    "dropped_field_ids": sorted(drop_ids),
                    "reason": "field cannot support the downstream semantics contract",
                    "issues": [
                        item for item in exc.issues
                        if item.get("component_id") in drop_ids
                    ],
                }
                await self.state.record_error({
                    "stage": "cross_stage_drop",
                    "component_id": template_id,
                    **event,
                })
                cross_stage_node = f"{base_node}/cross_stage_drop"
                self.audit_records[cross_stage_node] = event
                await self.state.record_audit(
                    cross_stage_node, event
                )
                catalog_node = f"{base_node}/field_catalog"
                catalog_audit = deepcopy(self.audit_records[catalog_node])
                retained_ids = {
                    field["field_id"] for field in catalog["fields"]
                }
                catalog_audit["verdicts"] = [
                    verdict
                    for verdict in catalog_audit.get("verdicts", [])
                    if verdict.get("component_id") in retained_ids
                ]
                catalog_audit["overall_summary"] = (
                    str(catalog_audit.get("overall_summary") or "")
                    + "; downstream semantics removed fields "
                    + ", ".join(sorted(drop_ids))
                ).strip("; ")
                self.audit_records[catalog_node] = catalog_audit
                await self.state.accept(
                    catalog_node, catalog, audit_record=catalog_audit
                )
                continue
            break
        else:
            raise AssertionError("unreachable")
        if diversity_check is not None:
            await diversity_check(core, catalog, semantics)
        field_index = merge_field_index(catalog, semantics)
        field_directions = {
            field["field_id"]: field["pressure_direction"] for field in field_index
        }
        batches = self.planner.field_parameters(
            shared_context={"scenario": scenario_context, "field_index": field_index, "defaults": defaults},
            field_ids=field_ids,
        )

        async def parameter_job(index: int, ids: tuple[str, ...]) -> dict[str, Any]:
            batch_id = f"batch_{index:03d}"
            spec = field_parameters.generation(
                self.renderer, template_id=template_id, batch_id=batch_id,
                scenario=scenario_context, field_index=field_index, field_ids=list(ids), defaults=defaults,
            )
            value, _ = await self._checked(
                spec=spec,
                validator=lambda item: validate_field_parameters(
                    item,
                    template_id,
                    set(ids),
                    field_directions,
                    self.config.field_parameters.beta_concentration_range,
                ),
                audit_stage="field_parameters",
                audit_context={"scenario": scenario_context, "field_index": field_index},
                repair_template="fields/parameters_repair.j2",
                repair_context={"scenario": scenario_context, "field_index": field_index},
                node_id=f"{base_node}/field_parameters/{batch_id}",
                dependencies={f"{base_node}/field_semantics"},
            )
            return value

        parameter_parts = await gather_jobs([
            lambda index=index, batch=batch: parameter_job(index, batch.ids)
            for index, batch in enumerate(batches, 1)
        ])
        merged_parameters = {"template_id": template_id, "fields": [field for part in parameter_parts for field in part["fields"]]}
        validate_field_parameters(
            merged_parameters,
            template_id,
            set(field_ids),
            field_directions,
            self.config.field_parameters.beta_concentration_range,
        )
        merged_node = f"{base_node}/field_parameters/merged"
        self.state.add_node(
            merged_node,
            {f"{base_node}/field_parameters/batch_{index:03d}" for index in range(1, len(batches) + 1)},
        )
        await self.state.accept(merged_node, merged_parameters)
        fields = assemble_fields(catalog, semantics, merged_parameters)

        constraint_input_fields = field_constraints.input_fields(fields)
        constraint_spec = field_constraints.generation(
            self.renderer,
            template_id=template_id,
            scenario=scenario_context,
            fields=constraint_input_fields,
        )
        constraint_job = lambda: self._checked(
            spec=constraint_spec,
            validator=lambda value: validate_field_constraints(
                value, template_id, fields
            ),
            audit_stage="field_constraints",
            audit_context={
                "scenario": scenario_context,
                "fields": constraint_input_fields,
            },
            repair_template="relations/field_constraints_repair.j2",
            repair_context={
                "scenario": scenario_context,
                "fields": constraint_input_fields,
            },
            node_id=f"{base_node}/field_constraints",
            dependencies={merged_node},
            generation_retry_codes={"duplicate_id"},
        )
        if stop_after == "field_constraints":
            await constraint_job()
            return template_id, None

        lexicographic_spec = lexicographic_orders.generation(
            self.renderer,
            template_id=template_id,
            scenario=scenario_context,
            fields=fields,
        )
        replacement_spec = replacements.generation(self.renderer, template_id=template_id, scenario=scenario_context, fields=fields)
        lexicographic_job = lambda: self._checked(
            spec=lexicographic_spec,
            validator=lambda value: validate_lexicographic_orders(
                value, template_id, fields
            ),
            audit_stage="lexicographic_orders",
            audit_context={"scenario": scenario_context, "fields": fields},
            repair_template="relations/lexicographic_orders_repair.j2",
            repair_context={"scenario": scenario_context, "fields": fields},
            node_id=f"{base_node}/lexicographic_orders",
            dependencies={merged_node},
        )
        replacement_job = lambda: self._checked(
            spec=replacement_spec, validator=lambda value: validate_replacements(value, template_id, fields),
            audit_stage="replacements", audit_context={"scenario": scenario_context, "fields": fields},
            repair_template="dynamics/repair.j2", repair_context={"scenario": scenario_context, "fields": fields},
            node_id=f"{base_node}/replacements",
            dependencies={merged_node},
        )
        constraint_pair, lexicographic_pair, replacement_pair = await gather_jobs([
            constraint_job,
            lexicographic_job,
            replacement_job,
        ])
        constraint_value = constraint_pair[0]
        lexicographic_value = lexicographic_pair[0]
        replacement_value = replacement_pair[0]
        validate_references(
            fields=fields,
            lexicographic_orders=lexicographic_value,
            replacements=replacement_value,
        )
        self._validate_component_bundle(
            template_id=template_id,
            catalog=catalog,
            semantics=semantics,
            parameters=merged_parameters,
            constraints_value=constraint_value,
            lexicographic_value=lexicographic_value,
            replacements_value=replacement_value,
        )
        template = assemble_template(
            core, catalog, semantics, merged_parameters, constraint_value,
            lexicographic_value, replacement_value,
        )
        self.contracts.validate_template(template)
        self.state.add_node(template_node, {
            f"{base_node}/field_constraints",
            f"{base_node}/lexicographic_orders",
            f"{base_node}/replacements",
        })
        await self.state.accept(template_node, template)
        return template_id, template

    def _manifest(
        self, candidate: dict[str, Any], audit: dict[str, Any],
    ) -> dict[str, Any]:
        config_hash = hashlib.sha256(self.config.source.read_bytes()).hexdigest()
        call_stats: dict[str, Any] = {
            "total": 0,
            "by_mode": {},
            "models": [],
            "usage": {},
        }
        models: set[str] = set()
        if self.state.calls_path.exists():
            for line in self.state.calls_path.read_text().splitlines():
                if not line.strip():
                    continue
                call = json.loads(line)
                call_stats["total"] += 1
                mode = str(call.get("mode") or "unknown")
                call_stats["by_mode"][mode] = call_stats["by_mode"].get(mode, 0) + 1
                if call.get("model"):
                    models.add(str(call["model"]))
                for key, amount in (call.get("usage") or {}).items():
                    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                        call_stats["usage"][key] = call_stats["usage"].get(key, 0) + amount
        call_stats["models"] = sorted(models)
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_identity(),
            "config": str(self.config.source.relative_to(self.config.source.parents[2])),
            "config_sha256": config_hash,
            "resource_sha256": self.resource_identity,
            "canonical_sha256": canonical_hash(candidate),
            "audit_sha256": canonical_hash(audit),
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "model_config": str(self.config.model_config),
            "audit_model_config": str(self.config.audit_model_config),
            "call_stats": call_stats,
        }
