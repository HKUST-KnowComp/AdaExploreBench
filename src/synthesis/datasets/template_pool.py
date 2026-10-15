"""Strict canonical-template loading and direct generation projections."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:
    Draft202012Validator = None  # type: ignore[assignment]


TEMPLATE_SCHEMA_VERSION = "adaexplorebench_domain_templates_v4_0"
TEMPLATE_VERSION = "4.0.0"
TEMPLATE_AUDIT_SCHEMA_VERSION = "adaexplorebench_template_synthesis_audit_v2"
TEMPLATE_MANIFEST_SCHEMA_VERSION = "adaexplorebench_template_synthesis_manifest_v2"
EXPECTED_DOMAIN_COUNT = 4
AGENT_FIELD_KEYS = (
    "display_name",
    "description",
    "type",
    "unit",
    "minimum",
    "maximum",
    "typical_range",
)


class TemplateValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class TemplateSelection:
    pool: dict[str, Any]
    domain: str
    template_id: str
    template: dict[str, Any]

    @property
    def fields(self) -> dict[str, dict[str, Any]]:
        return self.template["fields"]

    @property
    def defaults(self) -> dict[str, Any]:
        return self.pool["defaults"]


def template_hash(pool: dict[str, Any]) -> str:
    payload = json.dumps(
        pool, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_template_manifest(
    pool: dict[str, Any], manifest_path: str | Path, audit_path: str | Path,
) -> None:
    manifest_source = Path(manifest_path)
    audit_source = Path(audit_path)
    try:
        manifest = json.loads(manifest_source.read_text())
        audit = json.loads(audit_source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TemplateValidationError([
            "published template audit or manifest is unreadable"
        ]) from exc
    errors = []
    if audit.get("schema_version") != TEMPLATE_AUDIT_SCHEMA_VERSION:
        errors.append("unsupported published template audit schema")
    if audit.get("accepted") is not True:
        errors.append("published template audit is not accepted")
    if manifest.get("schema_version") != TEMPLATE_MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported published template manifest schema")
    observed = template_hash(pool)
    if audit.get("canonical_sha256") != observed:
        errors.append("published template audit differs from canonical template")
    if manifest.get("canonical_sha256") != observed:
        errors.append(
            "canonical template differs from its published manifest: "
            f"expected {manifest.get('canonical_sha256')}, observed {observed}"
        )
    if manifest.get("audit_sha256") != template_hash(audit):
        errors.append("published template audit differs from its manifest")
    if errors:
        raise TemplateValidationError(errors)


def validate_template_pool(pool: dict[str, Any]) -> None:
    schema_path = (
        Path(__file__).parents[1]
        / "templates"
        / "domain_templates.schema.json"
    )
    rendered: list[str] = []
    if Draft202012Validator is not None:
        schema = json.loads(schema_path.read_text())
        errors = sorted(
            Draft202012Validator(schema).iter_errors(pool),
            key=lambda error: list(error.absolute_path),
        )
        rendered.extend(
            f"/{'/'.join(map(str, error.absolute_path))}: {error.message}"
            for error in errors[:20]
        )
    elif not isinstance(pool, dict) or set(pool) != {
        "schema_version",
        "template_version",
        "defaults",
        "domains",
    }:
        rendered.append("canonical template pool has invalid top-level keys")
    if pool.get("schema_version") != TEMPLATE_SCHEMA_VERSION:
        rendered.append("unsupported canonical template schema")
    if pool.get("template_version") != TEMPLATE_VERSION:
        rendered.append(f"template_version must be {TEMPLATE_VERSION}")
    domains = pool.get("domains") or {}
    if not isinstance(domains, dict) or len(domains) != EXPECTED_DOMAIN_COUNT:
        rendered.append(
            f"canonical template pool must define exactly {EXPECTED_DOMAIN_COUNT} domains"
        )
    concentration = (pool.get("defaults") or {}).get("sampling", {}).get(
        "beta_concentration_range", [0, 0]
    )
    if isinstance(domains, dict) and len(concentration) == 2:
        low, high = map(float, concentration)
        for domain_id, domain in domains.items():
            for template_id, template in (domain.get("templates") or {}).items():
                for field_id, field in (template.get("fields") or {}).items():
                    for profile_id, beta in (field.get("profile_beta") or {}).items():
                        if (
                            not isinstance(beta, list)
                            or len(beta) != 2
                            or not low <= sum(map(float, beta)) <= high
                        ):
                            rendered.append(
                                f"{domain_id}/{template_id}/{field_id}/{profile_id} "
                                "has invalid beta concentration"
                            )
    if rendered:
        raise TemplateValidationError(rendered)


def load_template_pool(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    audit_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.suffix in {".yaml", ".yml"}:
        import yaml

        pool = yaml.safe_load(source.read_text())
    else:
        pool = json.loads(source.read_text())
    validate_template_pool(pool)
    if (manifest_path is None) != (audit_path is None):
        raise TemplateValidationError([
            "published template validation requires both audit and manifest"
        ])
    if manifest_path is not None and audit_path is not None:
        validate_template_manifest(pool, manifest_path, audit_path)
    return pool


def template_ids(pool: dict[str, Any], domain: str) -> tuple[str, ...]:
    if domain not in pool["domains"]:
        raise ValueError(f"unsupported domain: {domain}")
    return tuple(sorted(pool["domains"][domain]["templates"]))


def domain_ids(pool: dict[str, Any]) -> tuple[str, ...]:
    """Return the domains bound to this canonical template release."""
    return tuple(pool["domains"])


def select_template(
    pool: dict[str, Any], domain: str, template_id: str
) -> TemplateSelection:
    templates = pool["domains"][domain]["templates"]
    if template_id not in templates:
        raise ValueError(f"unknown template {domain}/{template_id}")
    return TemplateSelection(
        pool=pool,
        domain=domain,
        template_id=template_id,
        template=templates[template_id],
    )


def ordered_actions(selection: TemplateSelection) -> list[dict[str, str]]:
    return [
        {
            "action_id": f"action_{index}",
            "display_name": str(action["display_name"]),
            "description": str(action["description"]),
        }
        for index, action in enumerate(selection.template["actions"])
    ]


def agent_fields(
    selection: TemplateSelection, field_ids: list[str]
) -> dict[str, dict[str, Any]]:
    return {
        field_id: {
            key: deepcopy(selection.fields[field_id][key])
            for key in AGENT_FIELD_KEYS
        }
        for field_id in field_ids
    }


def family_parameters(selection: TemplateSelection, family: str) -> dict[str, Any]:
    authored = selection.template["rule_families"][family]
    parameters = deepcopy(selection.defaults["rule_families"][family])
    parameters.pop("description", None)
    parameters.update(deepcopy(authored.get("parameter_overrides") or {}))
    return parameters


def family_field_eligible(
    selection: TemplateSelection, field_id: str, family: str
) -> bool:
    if family == "lexicographic_priority":
        return any(
            field_id in order["fields"]
            for order in selection.template["rule_families"][family][
                "approved_priority_orders"
            ]
        )
    return family in selection.fields[field_id]["eligible_families"]


def approved_priority_orders(selection: TemplateSelection) -> list[dict[str, Any]]:
    return deepcopy(
        selection.template["rule_families"]["lexicographic_priority"][
            "approved_priority_orders"
        ]
    )


def direction_reversal_options(
    selection: TemplateSelection, family: str
) -> list[dict[str, Any]]:
    options = []
    for field_id, event in selection.template["dynamics"][
        "direction_reversals"
    ].items():
        if not family_field_eligible(selection, field_id, family):
            continue
        direction = int(selection.fields[field_id]["pressure_direction"])
        options.append(
            {
                "target_field": field_id,
                "event_id": field_id,
                "pre_direction": direction,
                "post_direction": -direction,
                "operational_trigger": event["trigger"],
                "decision_semantics_before": event["semantics_before"],
                "decision_semantics_after": event["semantics_after"],
                "why_direction_reverses": event["reversal_reason"],
            }
        )
    return options


def replacement_edges(
    selection: TemplateSelection, family: str
) -> list[dict[str, Any]]:
    if family == "lexicographic_priority":
        orders = approved_priority_orders(selection)
        eligible_pairs = {
            (outgoing, incoming)
            for order in orders
            for outgoing in order["fields"]
            for incoming in selection.fields
            if incoming not in order["fields"]
        }
    else:
        eligible_pairs = set()
    output = []
    for group_id, group in selection.template["dynamics"]["replacements"].items():
        for transition_id, transition in group["transitions"].items():
            outgoing = str(transition["outgoing_field"])
            incoming = str(transition["incoming_field"])
            eligible = (
                (outgoing, incoming) in eligible_pairs
                if family == "lexicographic_priority"
                else family_field_eligible(selection, outgoing, family)
                and family_field_eligible(selection, incoming, family)
            )
            if not eligible:
                continue
            output.append(
                {
                    "group_id": group_id,
                    "transition_id": transition_id,
                    "outgoing_field": outgoing,
                    "incoming_field": incoming,
                    "decision_role": group["decision_role"],
                    "relation_type": group["relation_type"],
                    "coupling_strength": float(
                        group["joint_sampling"]["coupling_strength"]
                    ),
                    "trigger": transition["trigger"],
                    "precondition": transition.get("precondition"),
                    "outgoing_retirement_reason": transition[
                        "outgoing_retirement_reason"
                    ],
                    "incoming_authority_reason": transition[
                        "incoming_authority_reason"
                    ],
                    "authority_before": group["authority_by_field"][outgoing],
                    "authority_after": group["authority_by_field"][incoming],
                    "effective_scope": group["effective_scope"],
                    "expected_duration": transition["expected_duration"],
                    "ending_condition": transition["ending_condition"],
                    "rationale": group["rationale"],
                }
            )
    return output


def template_supports_cell(
    selection: TemplateSelection,
    family: str,
    dynamic_type: str | None,
    visible_field_count: int = 5,
) -> bool:
    fields = selection.fields
    if family == "lexicographic_priority":
        has_static = (
            bool(approved_priority_orders(selection))
            and len(fields) >= visible_field_count
        )
    else:
        eligible = [
            field_id
            for field_id in fields
            if family_field_eligible(selection, field_id, family)
        ]
        has_static = len(eligible) >= 3 and len(fields) >= visible_field_count
    if not has_static or dynamic_type is None:
        return has_static
    replacements = replacement_edges(selection, family)
    reversals = direction_reversal_options(selection, family)
    if dynamic_type == "field_replacement":
        return bool(replacements)
    if dynamic_type == "direction_reversal":
        return bool(reversals)
    if dynamic_type == "compound":
        return any(
            option["target_field"]
            not in {edge["outgoing_field"], edge["incoming_field"]}
            for edge in replacements
            for option in reversals
        )
    raise ValueError(f"unsupported dynamic type: {dynamic_type}")
