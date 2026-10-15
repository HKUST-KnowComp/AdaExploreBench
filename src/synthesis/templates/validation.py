from __future__ import annotations

from typing import Any

from protocol.mechanism import FAMILIES
from synthesis.datasets.template_pool import validate_template_pool


class StageValidationError(ValueError):
    def __init__(self, stage: str, issues: list[dict[str, str]]) -> None:
        self.stage = stage
        self.issues = issues
        super().__init__(f"{stage} validation failed: {issues}")


def issue(
    path: str, code: str, message: str, *, component_id: str | None = None
) -> dict[str, str]:
    value = {"path": path, "code": code, "message": message}
    if component_id is not None:
        value["component_id"] = component_id
    return value


def validate_scenario_cores(
    value: dict[str, Any],
    domain_id: str,
    *,
    domain_description: str,
    template_id: str,
    task_id: str,
    blueprint: dict[str, Any] | None,
) -> None:
    issues = []
    if value["domain_id"] != domain_id:
        issues.append(issue("/domain_id", "identity_mismatch", "domain_id changed"))
    if value["domain_description"] != domain_description:
        issues.append(issue(
            "/domain_description", "frozen_value", "domain description changed"
        ))
    if len(value["templates"]) != 1:
        issues.append(issue(
            "/templates", "count_mismatch", "expected one scenario"
        ))
    else:
        template = value["templates"][0]
        if template["template_id"] != template_id:
            issues.append(issue(
                "/templates/0/template_id", "identity_mismatch", "template_id changed"
            ))
        if str(template["task_id"]) != task_id:
            issues.append(issue(
                "/templates/0/task_id", "identity_mismatch", "task_id changed"
            ))
        if blueprint is not None and template["agent_role"] != blueprint["agent_role"]:
            issues.append(issue(
                "/templates/0/agent_role",
                "blueprint_role_mismatch",
                "agent_role must match the authoritative task blueprint verbatim",
                component_id=str(template["template_id"]),
            ))
        if blueprint is not None and template["description"] != blueprint["description"]:
            issues.append(issue(
                "/templates/0/description",
                "blueprint_description_mismatch",
                "description must match the authoritative task blueprint verbatim",
                component_id=str(template["template_id"]),
            ))
        if blueprint is not None and template["actions"] != blueprint["actions"]:
            issues.append(issue(
                "/templates/0/actions",
                "blueprint_actions_mismatch",
                "actions must match the authoritative task blueprint verbatim",
                component_id=str(template["template_id"]),
            ))
        try:
            validate_scenario_output(template)
        except StageValidationError as exc:
            issues.extend(exc.issues)
    if issues:
        raise StageValidationError("scenario_cores", issues)


def validate_scenario_output(value: dict[str, Any]) -> None:
    """Validate deterministic invariants shared by both scenario output modes."""
    issues = []
    weights = [
        value["operational_profiles"][key]["weight"]
        for key in ("low", "mixed", "high")
    ]
    if abs(sum(weights) - 1.0) > 1e-9:
        issues.append(issue(
            "/operational_profiles", "invalid_weight_sum",
            "profile weights must sum to one",
        ))
    if "actions" in value:
        names = [str(action["display_name"]) for action in value["actions"]]
        if len(names) != len(set(names)):
            issues.append(issue(
                "/actions", "duplicate_action", "action display names must be unique"
            ))
    if issues:
        raise StageValidationError("scenario_output", issues)


def validate_field_catalog(value: dict[str, Any], template_id: str) -> None:
    issues = []
    if value["template_id"] != template_id:
        issues.append(issue("/template_id", "identity_mismatch", "template_id changed"))
    ids = [item["field_id"] for item in value["fields"]]
    if len(ids) != len(set(ids)):
        issues.append(issue("/fields", "duplicate_id", "field IDs must be unique"))
    if issues:
        raise StageValidationError("field_catalog", issues)


def validate_field_semantics(
    value: dict[str, Any], template_id: str, expected_ids: set[str]
) -> None:
    issues = []
    if value["template_id"] != template_id:
        issues.append(issue("/template_id", "identity_mismatch", "template_id changed"))
    ids = [item["field_id"] for item in value["fields"]]
    if len(ids) != len(set(ids)):
        issues.append(issue("/fields", "duplicate_id", "field IDs must be unique"))
    if set(ids) != expected_ids:
        issues.append(issue(
            "/fields",
            "coverage_mismatch",
            f"missing={sorted(expected_ids-set(ids))}, extra={sorted(set(ids)-expected_ids)}",
        ))
    event_keys = ("trigger", "semantics_before", "semantics_after", "reversal_reason")
    for index, field in enumerate(value["fields"]):
        if field["is_reversal"]:
            empty = [key for key in event_keys if not field[key].strip()]
            if empty:
                issues.append(issue(
                    f"/fields/{index}",
                    "missing_reversal_event",
                    f"is_reversal=true requires non-empty {', '.join(empty)}",
                    component_id=field["field_id"],
                ))
        else:
            filled = [key for key in event_keys if field[key].strip()]
            if filled:
                issues.append(issue(
                    f"/fields/{index}",
                    "unexpected_reversal_event",
                    f"is_reversal=false requires empty {', '.join(filled)}",
                    component_id=field["field_id"],
                ))
            if not field["reversal_rationale"].strip():
                issues.append(issue(
                    f"/fields/{index}",
                    "missing_reversal_rationale",
                    "is_reversal=false requires reversal_rationale explaining the stable tendency",
                    component_id=field["field_id"],
                ))
    if issues:
        raise StageValidationError("field_semantics", issues)


def validate_field_parameters(
    value: dict[str, Any],
    template_id: str,
    expected_ids: set[str],
    field_directions: dict[str, int],
    beta_concentration_range: tuple[float, float],
) -> None:
    issues = []
    if value["template_id"] != template_id:
        issues.append(issue("/template_id", "identity_mismatch", "template_id changed"))
    ids = [item["field_id"] for item in value["fields"]]
    if len(ids) != len(set(ids)):
        issues.append(issue("/fields", "duplicate_id", "field IDs must be unique"))
    if set(ids) != expected_ids:
        issues.append(issue("/fields", "coverage_mismatch", f"missing={sorted(expected_ids-set(ids))}, extra={sorted(set(ids)-expected_ids)}"))
    for index, field in enumerate(value["fields"]):
        field_id = field["field_id"]
        low, high = field["typical_range"]
        if not field["minimum"] < field["maximum"] or not (
            field["minimum"] <= low < high <= field["maximum"]
        ):
            issues.append(issue(
                f"/fields/{index}/typical_range",
                "invalid_range",
                "typical range must be inside legal range",
                component_id=field_id,
            ))
        means: list[float | None] = []
        concentration_minimum, concentration_maximum = beta_concentration_range
        for profile in ("low", "mixed", "high"):
            alpha, beta = field["profile_beta"][profile]
            concentration = alpha + beta
            if (
                alpha <= 0
                or beta <= 0
                or not concentration_minimum <= concentration <= concentration_maximum
            ):
                issues.append(issue(
                    f"/fields/{index}/profile_beta/{profile}",
                    "invalid_beta_shape",
                    "Beta alpha and beta must each be strictly positive and "
                    f"alpha+beta must be within [{concentration_minimum:g}, "
                    f"{concentration_maximum:g}]",
                    component_id=field_id,
                ))
            means.append(alpha / concentration if alpha > 0 and beta > 0 else None)
        raw_means = [
            None if mean is None else field["minimum"] + mean * (
                field["maximum"] - field["minimum"]
            )
            for mean in means
        ]
        for profile, raw_mean in zip(("low", "mixed"), raw_means[:2], strict=True):
            if raw_mean is not None and not low <= raw_mean <= high:
                issues.append(issue(
                    f"/fields/{index}/profile_beta/{profile}",
                    "profile_mean_outside_typical_range",
                    f"{profile} mapped mean {raw_mean:.6g} must lie inside "
                    f"typical_range [{low}, {high}]",
                    component_id=field_id,
                ))
        direction = field_directions.get(field_id)
        complete_means = all(mean is not None for mean in means)
        numeric_means = [float(mean) for mean in means if mean is not None]
        ordered = complete_means and (
            numeric_means[0] <= numeric_means[1] <= numeric_means[2]
            if direction == 1
            else numeric_means[0] >= numeric_means[1] >= numeric_means[2]
        )
        if complete_means and (direction not in {-1, 1} or not ordered):
            issues.append(issue(
                f"/fields/{index}/profile_beta",
                "profile_direction_mismatch",
                "normalized Beta means "
                f"{[round(mean, 6) for mean in means]} must follow low/mixed/high "
                f"pressure for pressure_direction={direction}",
                component_id=field_id,
            ))
    if issues:
        raise StageValidationError("field_parameters", issues)


def field_parameter_diagnostics(value: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = []
    for field in value["fields"]:
        minimum = field["minimum"]
        span = field["maximum"] - minimum
        means = {
            profile: minimum + alpha / (alpha + beta) * span
            for profile, (alpha, beta) in field["profile_beta"].items()
        }
        concentrations = {
            profile: alpha + beta
            for profile, (alpha, beta) in field["profile_beta"].items()
        }
        diagnostics.append({
            "field_id": field["field_id"],
            "legal_range": [minimum, field["maximum"]],
            "typical_range": field["typical_range"],
            "mapped_profile_means": means,
            "profile_concentrations": concentrations,
        })
    return diagnostics


def _field_constraint_issues(
    value: dict[str, Any], template_id: str, fields: dict[str, Any]
) -> list[dict[str, str]]:
    issues = []
    field_ids = set(fields)
    if value["template_id"] != template_id:
        issues.append(issue(
            "/template_id", "identity_mismatch", "template_id changed",
            component_id="__field_constraints__",
        ))
    constraint_ids = [
        str(constraint["id"])
        for constraint in value["feasibility_constraints"]
    ]
    seen_constraint_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for constraint_id in constraint_ids:
        if constraint_id in seen_constraint_ids:
            duplicate_ids.add(constraint_id)
        seen_constraint_ids.add(constraint_id)
    if duplicate_ids:
        issues.append(issue(
            "/feasibility_constraints", "duplicate_id",
            f"constraint IDs must be unique: {sorted(duplicate_ids)}",
            component_id="__field_constraints__",
        ))
    for index, constraint in enumerate(value["feasibility_constraints"]):
        expression = constraint["expr"]
        referenced = _expression_fields(expression)
        if referenced - field_ids:
            issues.append(issue(
                f"/feasibility_constraints/{index}/expr", "unknown_field",
                str(sorted(referenced-field_ids)), component_id=str(constraint["id"]),
            ))
    return issues


def validate_field_constraints(
    value: dict[str, Any], template_id: str, fields: dict[str, Any]
) -> None:
    issues = _field_constraint_issues(value, template_id, fields)
    if issues:
        raise StageValidationError("field_constraints", issues)


def _lexicographic_order_issues(
    value: dict[str, Any], template_id: str, fields: dict[str, Any]
) -> list[dict[str, str]]:
    issues = []
    field_ids = set(fields)
    if value["template_id"] != template_id:
        issues.append(issue("/template_id", "identity_mismatch", "template_id changed"))
    order_ids = [order["order_id"] for order in value["approved_priority_orders"]]
    if len(order_ids) != len(set(order_ids)):
        issues.append(issue("/approved_priority_orders", "duplicate_id", "priority order IDs must be unique"))
    ordered_triples = [tuple(order["fields"]) for order in value["approved_priority_orders"]]
    if len(ordered_triples) != len(set(ordered_triples)):
        issues.append(issue(
            "/approved_priority_orders", "duplicate_order",
            "identical ordered field triples must not be repeated",
        ))
    for index, order in enumerate(value["approved_priority_orders"]):
        unknown = set(order["fields"]) - field_ids
        if unknown:
            issues.append(issue(f"/approved_priority_orders/{index}/fields", "unknown_field", str(sorted(unknown))))
    return issues


def validate_lexicographic_orders(
    value: dict[str, Any], template_id: str, fields: dict[str, Any]
) -> None:
    issues = _lexicographic_order_issues(value, template_id, fields)
    if issues:
        raise StageValidationError("lexicographic_orders", issues)


def _expression_fields(value: Any) -> set[str]:
    if isinstance(value, dict):
        output = {str(value["field"])} if "field" in value else set()
        for child in value.values():
            output.update(_expression_fields(child))
        return output
    if isinstance(value, list):
        output: set[str] = set()
        for child in value:
            output.update(_expression_fields(child))
        return output
    return set()


def validate_replacements(value: dict[str, Any], template_id: str, fields: dict[str, Any]) -> None:
    issues = []
    if value["template_id"] != template_id:
        issues.append(issue("/template_id", "identity_mismatch", "template_id changed"))
    group_ids: set[str] = set()
    transition_ids: set[str] = set()
    for index, group in enumerate(value["replacements"]):
        group_id = group["group_id"]
        if group_id in group_ids:
            issues.append(issue(f"/replacements/{index}/group_id", "duplicate_id", group_id))
        group_ids.add(group_id)
        members = group["members"]
        if any(member not in fields for member in members):
            issues.append(issue(f"/replacements/{index}/members", "unknown_field", "unknown endpoint"))
            continue
        authorities = [item["field_id"] for item in group["authority_by_field"]]
        if len(authorities) != len(set(authorities)) or set(authorities) != set(members):
            issues.append(issue(f"/replacements/{index}/authority_by_field", "coverage_mismatch", "authority entries must cover both members exactly once"))
        left, right = (fields[member] for member in members)
        if left["pressure_direction"] != right["pressure_direction"]:
            issues.append(issue(
                f"/replacements/{index}/members",
                "incompatible_direction",
                "replacement endpoints must have the same frozen pressure_direction",
            ))
        directions = set()
        for transition in group["transitions"]:
            transition_id = transition["transition_id"]
            if transition_id in transition_ids:
                issues.append(issue(f"/replacements/{index}/transitions", "duplicate_id", transition_id))
            transition_ids.add(transition_id)
            directions.add((transition["outgoing_field"], transition["incoming_field"]))
        expected = {(members[0], members[1]), (members[1], members[0])}
        if directions != expected:
            issues.append(issue(f"/replacements/{index}/transitions", "missing_direction", "current contract requires two independently authored directions"))
    if issues:
        raise StageValidationError("replacements", issues)


def validate_references(
    *,
    fields: dict[str, Any],
    lexicographic_orders: dict[str, Any],
    replacements: dict[str, Any],
) -> None:
    issues = []
    field_ids = set(fields)
    for index, order in enumerate(lexicographic_orders["approved_priority_orders"]):
        unknown = set(order["fields"]) - field_ids
        if unknown:
            issues.append(issue(f"/approved_priority_orders/{index}", "unknown_field", str(sorted(unknown))))
    for group in replacements["replacements"]:
        members = set(group["members"])
        if not members <= field_ids:
            issues.append(issue(f"/replacements/{group['group_id']}/members", "unknown_field", "replacement references unknown fields"))
        for transition in group["transitions"]:
            if {transition["outgoing_field"], transition["incoming_field"]} != members:
                issues.append(issue(f"/replacements/{group['group_id']}/transitions", "invalid_endpoints", "transition endpoints must equal group members"))
    if issues:
        raise StageValidationError("references", issues)


def validate_canonical(value: dict[str, Any]) -> None:
    validate_template_pool(value)


def semantic_audit_ids(stage: str, value: dict[str, Any]) -> set[str]:
    """Return the exact component IDs an auditor must cover for a stage."""
    if stage == "scenario_cores":
        return {item["template_id"] for item in value["templates"]}
    if stage == "field_catalog":
        return {item["field_id"] for item in value["fields"]}
    if stage == "field_semantics":
        return {
            item["field_id"] for item in value["fields"]
        } | {"__field_semantics_set__"}
    if stage == "field_parameters":
        return {item["field_id"] for item in value["fields"]}
    if stage == "field_constraints":
        identifiers = [
            str(item["id"]) for item in value["feasibility_constraints"]
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("field constraint IDs must be unique before audit")
        return set(identifiers) or {"__field_constraints__"}
    if stage == "lexicographic_orders":
        return {item["order_id"] for item in value["approved_priority_orders"]}
    if stage == "replacements":
        ids = {item["group_id"] for item in value["replacements"]}
        ids.update(
            transition["transition_id"]
            for group in value["replacements"]
            for transition in group["transitions"]
        )
        return ids | {"__replacements__"}
    raise KeyError(f"no semantic audit identity contract for stage {stage}")


def validate_field_catalog_set_audit(
    value: dict[str, Any], field_ids: set[str], frozen_ids: set[str] | None = None,
) -> set[str]:
    """Validate issue-only catalog audit output and return authorized drop IDs."""
    frozen_ids = frozen_ids or set()
    issues = value["issues"]
    if value["accepted"] != (not issues):
        raise ValueError("catalog set audit accepted flag must equal absence of issues")
    drop_ids: set[str] = set()
    for index, item in enumerate(issues):
        implicated = set(item["field_ids"])
        dropped = set(item["drop_field_ids"])
        unknown = (implicated | dropped) - field_ids
        if unknown:
            raise ValueError(
                f"catalog set audit issue {index} references unknown fields {sorted(unknown)}"
            )
        if not dropped <= implicated:
            raise ValueError(
                f"catalog set audit issue {index} drops fields outside its implicated group"
            )
        immutable = dropped & frozen_ids
        if immutable:
            raise ValueError(
                f"catalog set audit issue {index} drops immutable fields {sorted(immutable)}"
            )
        if item["code"] == "strict_duplicate":
            if len(implicated) < 2:
                raise ValueError("strict duplicate issue must identify at least two fields")
            if dropped == implicated:
                raise ValueError("strict duplicate issue must retain at least one field")
        elif dropped != implicated:
            raise ValueError("unavailable fields must all be selected for removal")
        overlap = drop_ids & dropped
        if overlap:
            raise ValueError(
                f"catalog set audit selects fields more than once {sorted(overlap)}"
            )
        drop_ids.update(dropped)
    return drop_ids


def validate_field_catalog_local_audit(
    value: dict[str, Any], field_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Validate sparse local field issues and group them by field ID."""
    grouped = {field_id: [] for field_id in field_ids}
    seen: set[tuple[str, str]] = set()
    for item in value["issues"]:
        field_id = item["field_id"]
        if field_id not in field_ids:
            raise ValueError(f"local field audit references unknown field {field_id}")
        identity = (field_id, item["code"])
        if identity in seen:
            raise ValueError(
                f"local field audit repeats {item['code']} for {field_id}"
            )
        seen.add(identity)
        grouped[field_id].append(item)
    return grouped


def validate_overall_audit(value: dict[str, Any], stage: str) -> None:
    if value["accepted"] and value["issues"]:
        raise ValueError(f"{stage} accepted the component while reporting issues")
    if not value["accepted"] and not value["issues"]:
        raise ValueError(f"{stage} rejected the component without a repairable issue")
