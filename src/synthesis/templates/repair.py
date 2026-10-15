from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


class PatchError(ValueError):
    pass


def _tokens(path: str) -> list[str]:
    if not path.startswith("/"):
        raise PatchError("JSON pointer must start with /")
    if path == "/":
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def field_patch_targets(patch: dict[str, Any]) -> set[str]:
    targets = set()
    for operation in patch["operations"]:
        parts = _tokens(operation["path"])
        if len(parts) >= 2 and parts[0] == "fields" and parts[1] != "-":
            targets.add(parts[1])
    return targets


def validate_field_assignment_patch(
    patch: dict[str, Any], field_ids: set[str],
) -> None:
    """Restrict local discovery repair to mutable field-assignment properties."""
    allowed_properties = {"description", "type", "unit", "pressure_direction"}
    for operation in patch["operations"]:
        parts = _tokens(operation["path"])
        if (
            operation["op"] != "replace"
            or len(parts) != 3
            or parts[0] != "fields"
            or parts[1] not in field_ids
            or parts[2] not in allowed_properties
        ):
            raise PatchError(
                "local field repair may only replace description, type, unit, or "
                "pressure_direction on an audited rejected field"
            )


def component_patch_targets(
    value: dict[str, Any], patch: dict[str, Any], stage: str
) -> set[str]:
    """Resolve non-field patch paths to their smallest authored component IDs."""
    roots = {
        "lexicographic_orders": {
            "approved_priority_orders": ("order_id", "__lexicographic_orders__"),
        },
        "replacements": {"replacements": ("group_id", "__replacements__")},
    }
    if stage not in roots:
        return set()
    current = deepcopy(value)
    targets: set[str] = set()
    for operation in patch["operations"]:
        parts = _tokens(operation["path"])
        root = parts[0] if parts else ""
        definition = roots[stage].get(root)
        if definition is None or len(parts) < 2 or parts[1] == "-":
            targets.add(next(iter(roots[stage].values()))[1])
        else:
            id_key, sentinel = definition
            try:
                item = current[root][int(parts[1])]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise PatchError(f"invalid component selector {operation['path']}") from exc
            if (
                stage == "replacements"
                and len(parts) >= 4
                and parts[2] == "transitions"
                and parts[3] != "-"
            ):
                try:
                    transition = item["transitions"][int(parts[3])]
                    targets.add(str(transition["transition_id"]))
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise PatchError(
                        f"invalid transition selector {operation['path']}"
                    ) from exc
                if len(parts) >= 5 and parts[4] == "transition_id":
                    raise PatchError("repair cannot change a transition identity")
            elif stage == "replacements" and len(parts) >= 3 and parts[2] == "transitions":
                targets.add(sentinel)
            else:
                try:
                    targets.add(str(item[id_key]))
                except (KeyError, TypeError) as exc:
                    raise PatchError(
                        f"component lacks identity at {operation['path']}"
                    ) from exc
                if len(parts) >= 3 and parts[2] == id_key:
                    raise PatchError("repair cannot change a component identity")
        current = apply_component_patch(current, {
            "component_id": patch["component_id"],
            "operations": [operation],
        })
    return targets


def component_contents(value: dict[str, Any], stage: str) -> dict[str, Any]:
    """Map every authored component ID to its content for change comparison."""
    if stage == "lexicographic_orders":
        return {
            str(item["order_id"]): item
            for item in value["approved_priority_orders"]
        }
    if stage == "replacements":
        contents = {}
        for group in value["replacements"]:
            contents[str(group["group_id"])] = {
                key: item for key, item in group.items() if key != "transitions"
            }
            for transition in group["transitions"]:
                contents[str(transition["transition_id"])] = transition
        return contents
    return {}


def apply_field_constraint_repairs(
    value: dict[str, Any], repair: dict[str, Any], allowed_ids: set[str],
    split_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Apply complete constraint replacements, removals, or splits by stable ID."""
    split_ids = split_ids or set()
    constraints = value.get("feasibility_constraints")
    if not isinstance(constraints, list):
        raise PatchError("field constraint component lacks feasibility_constraints")
    existing_ids = [str(item.get("id")) for item in constraints]
    if len(existing_ids) != len(set(existing_ids)):
        raise PatchError("field constraint IDs must be unique before repair")

    decisions = repair.get("repairs")
    if not isinstance(decisions, list) or not decisions:
        raise PatchError("field constraint repair must contain at least one decision")
    decision_ids = [str(decision.get("constraint_id")) for decision in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise PatchError("field constraint repair repeated a constraint ID")
    if set(decision_ids) != allowed_ids:
        raise PatchError(
            "field constraint repair coverage mismatch: "
            f"missing={sorted(allowed_ids-set(decision_ids))}, "
            f"extra={sorted(set(decision_ids)-allowed_ids)}"
        )
    unknown = set(decision_ids) - set(existing_ids)
    if unknown:
        raise PatchError(f"field constraint repair references unknown IDs: {sorted(unknown)}")

    by_id = {str(decision["constraint_id"]): decision for decision in decisions}
    repaired_constraints = []
    for constraint in constraints:
        constraint_id = str(constraint["id"])
        decision = by_id.get(constraint_id)
        if decision is None:
            repaired_constraints.append(deepcopy(constraint))
            continue
        try:
            replacement = json.loads(decision["replacement_json"])
        except json.JSONDecodeError as exc:
            raise PatchError(
                f"replacement_json is invalid for {constraint_id}"
            ) from exc
        action = decision["action"]
        if action == "remove":
            if replacement is not None:
                raise PatchError("remove requires replacement_json to be null")
            continue
        if action == "split":
            if constraint_id not in split_ids:
                raise PatchError(
                    "split is allowed only for bundled_independent_invariants"
                )
            if not isinstance(replacement, list) or len(replacement) < 2:
                raise PatchError("split requires at least two replacement constraints")
            parts = replacement
        elif action == "replace":
            parts = [replacement]
        else:
            raise PatchError(f"unknown field constraint repair action: {action}")
        for part_index, part in enumerate(parts, 1):
            if not isinstance(part, dict) or set(part) != {"description", "expr"}:
                raise PatchError(
                    f"{action} requires description and expr, with no identity property"
                )
            part_id = (
                constraint_id
                if part_index == 1
                else f"{constraint_id}__part_{part_index}"
            )
            if part_id != constraint_id and part_id in existing_ids:
                raise PatchError(f"split generated an existing constraint ID: {part_id}")
            repaired_constraints.append({
                "id": part_id,
                "description": deepcopy(part["description"]),
                "expr": deepcopy(part["expr"]),
            })

    return {
        **deepcopy(value),
        "feasibility_constraints": repaired_constraints,
    }


def apply_component_patch(value: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    output: Any = deepcopy(value)
    for operation in patch["operations"]:
        try:
            operation_value = json.loads(operation["value_json"])
        except json.JSONDecodeError as exc:
            raise PatchError("patch value_json is not valid JSON") from exc
        parts = _tokens(operation["path"])
        if not parts:
            raise PatchError("cannot patch the component root; repair only the rejected paths")
        if (
            len(parts) >= 2
            and parts[0] == "fields"
            and isinstance(output.get("fields"), list)
            and parts[1] != "-"
        ):
            if parts[1].isdigit():
                raise PatchError(
                    "field-array patches must address /fields/FIELD_ID, not a numeric index"
                )
            matches = [
                index for index, field in enumerate(output["fields"])
                if field.get("field_id") == parts[1]
            ]
            if len(matches) != 1:
                raise PatchError(f"field selector must resolve exactly once: {parts[1]}")
            parts[1] = str(matches[0])
        parent = output
        for token in parts[:-1]:
            try:
                parent = parent[int(token)] if isinstance(parent, list) else parent[token]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise PatchError(f"invalid JSON pointer {operation['path']}") from exc
        leaf = parts[-1]
        op = operation["op"]
        if isinstance(parent, list):
            try:
                index = len(parent) if leaf == "-" else int(leaf)
            except ValueError as exc:
                raise PatchError(f"invalid list index {leaf}") from exc
            if op == "add":
                parent.insert(index, deepcopy(operation_value))
            elif op == "replace":
                parent[index] = deepcopy(operation_value)
            else:
                parent.pop(index)
        elif isinstance(parent, dict):
            if op == "remove":
                if leaf not in parent:
                    raise PatchError(f"cannot remove absent path {operation['path']}")
                del parent[leaf]
            else:
                if op == "replace" and leaf not in parent:
                    raise PatchError(f"cannot replace absent path {operation['path']}")
                parent[leaf] = deepcopy(operation_value)
        else:
            raise PatchError(f"path parent is not a container: {operation['path']}")
    if not isinstance(output, dict):
        raise PatchError("patched component must remain an object")
    return output
