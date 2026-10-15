"""Evaluate authored feasibility constraints against sampled field values."""

from __future__ import annotations

import math
import hashlib
import json
from copy import deepcopy
from typing import Any, Protocol

import z3


class TemplateSelectionLike(Protocol):
    template: dict[str, Any]
    fields: dict[str, dict[str, Any]]


def _number(expr: dict[str, Any], values: dict[str, Any]) -> float:
    if "const" in expr:
        return float(expr["const"])
    if "field" in expr:
        return float(values[expr["field"]])
    args = [_number(item, values) for item in expr.get("args") or []]
    op = expr.get("op")
    if op == "add":
        return sum(args)
    if op == "sub" and len(args) == 2:
        return args[0] - args[1]
    if op == "mul":
        return math.prod(args)
    if op == "div" and len(args) == 2 and abs(args[1]) > 1e-12:
        return args[0] / args[1]
    raise ValueError(f"invalid numeric feasibility expression: {expr}")


def evaluate_constraint(expr: dict[str, Any], values: dict[str, Any]) -> bool:
    op = expr.get("op")
    if op in {"and", "or"}:
        results = [evaluate_constraint(item, values) for item in expr.get("args") or []]
        return all(results) if op == "and" else any(results)
    if op == "not":
        return not evaluate_constraint(expr["arg"], values)
    if op == "implies":
        return not evaluate_constraint(expr["if"], values) or evaluate_constraint(
            expr["then"], values
        )
    left, right = _number(expr["left"], values), _number(expr["right"], values)
    return {
        "le": left <= right + 1e-9,
        "lt": left < right,
        "ge": left + 1e-9 >= right,
        "gt": left > right,
        "eq": abs(left - right) <= 1e-9,
    }.get(str(op), False)


def constraint_fields(expr: Any) -> set[str]:
    if isinstance(expr, dict):
        result = {str(expr["field"])} if "field" in expr else set()
        for value in expr.values():
            result.update(constraint_fields(value))
        return result
    if isinstance(expr, list):
        result: set[str] = set()
        for item in expr:
            result.update(constraint_fields(item))
        return result
    return set()


def selected_constraints(
    selection: TemplateSelectionLike, field_ids: list[str]
) -> list[dict[str, Any]]:
    selected = set(field_ids)
    return [
        deepcopy(item)
        for item in selection.template.get("feasibility_constraints") or []
        if constraint_fields(item["expr"]).issubset(selected)
    ]


def constraint_violations(
    constraints: list[dict[str, Any]], values: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {"constraint_id": str(item["id"]), "message": str(item["description"])}
        for item in constraints
        if constraint_fields(item["expr"]).issubset(values)
        and not evaluate_constraint(item["expr"], values)
    ]


_COMPILED_CACHE: dict[str, dict[str, Any]] = {}


def _affine(expr: dict[str, Any]) -> tuple[dict[str, float], float] | None:
    if "const" in expr:
        return {}, float(expr["const"])
    if "field" in expr:
        return {str(expr["field"]): 1.0}, 0.0
    op = expr.get("op")
    args = [_affine(item) for item in expr.get("args") or []]
    if any(item is None for item in args):
        return None
    values = [item for item in args if item is not None]
    if op in {"add", "sub"}:
        if op == "sub" and len(values) != 2:
            return None
        coefficients: dict[str, float] = {}
        constant = 0.0
        for index, (part, offset) in enumerate(values):
            sign = -1.0 if op == "sub" and index == 1 else 1.0
            constant += sign * offset
            for field, coefficient in part.items():
                coefficients[field] = coefficients.get(field, 0.0) + sign * coefficient
        return coefficients, constant
    if op in {"mul", "div"}:
        if op == "div" and len(values) == 2 and not values[1][0] and values[1][1] != 0:
            factor = 1.0 / values[1][1]
            return ({key: factor * value for key, value in values[0][0].items()}, factor * values[0][1])
        variable = [item for item in values if item[0]]
        constants = [item[1] for item in values if not item[0]]
        if len(variable) != 1:
            return None
        factor = math.prod(constants)
        return ({key: factor * value for key, value in variable[0][0].items()}, factor * variable[0][1])
    return None


def _comparison_atom(expr: dict[str, Any], *, negate: bool = False) -> dict[str, Any] | None:
    op = str(expr.get("op"))
    if op not in {"le", "lt", "ge", "gt", "eq"}:
        return None
    if negate:
        op = {"le": "gt", "lt": "ge", "ge": "lt", "gt": "le", "eq": "ne"}[op]
    left, right = _affine(expr["left"]), _affine(expr["right"])
    if left is None or right is None:
        return None
    coefficients = dict(left[0])
    for field, coefficient in right[0].items():
        coefficients[field] = coefficients.get(field, 0.0) - coefficient
    coefficients = {key: value for key, value in coefficients.items() if abs(value) > 1e-15}
    if len(coefficients) > 2:
        return None
    return {"coefficients": coefficients, "constant": left[1] - right[1], "op": op}


def _constraint_dnf(expr: dict[str, Any], limit: int, *, negate: bool = False) -> list[list[dict[str, Any]]] | None:
    op = str(expr.get("op"))
    if op == "implies":
        return _constraint_dnf(
            {"op": "or", "args": [{"op": "not", "arg": expr["if"]}, expr["then"]]},
            limit,
            negate=negate,
        )
    if op == "not":
        return _constraint_dnf(expr["arg"], limit, negate=not negate)
    if op in {"and", "or"}:
        effective = ({"and": "or", "or": "and"}[op] if negate else op)
        parts = [_constraint_dnf(item, limit, negate=negate) for item in expr.get("args") or []]
        if any(item is None for item in parts):
            return None
        groups = [item for item in parts if item is not None]
        if effective == "or":
            result = [branch for group in groups for branch in group]
        else:
            result: list[list[dict[str, Any]]] = [[]]
            for group in groups:
                result = [left + right for left in result for right in group]
                if len(result) > limit:
                    return None
        return result if len(result) <= limit else None
    atom = _comparison_atom(expr, negate=negate)
    return None if atom is None else [[atom]]


def _sampling_classification(
    constraints: list[dict[str, Any]], fields: tuple[str, ...], branch_limit: int
) -> dict[str, Any]:
    branches: list[list[dict[str, Any]]] = [[]]
    for item in constraints:
        group = _constraint_dnf(item["expr"], branch_limit)
        if group is None:
            return {"kind": "fallback", "reason": "constraint_not_in_conditional_whitelist"}
        branches = [left + right for left in branches for right in group]
        if len(branches) > branch_limit:
            return {"kind": "fallback", "reason": "conditional_branch_limit"}
    involved = {
        field
        for branch in branches
        for atom in branch
        for field in atom["coefficients"]
    }
    if len(involved) > 2 or any(atom["op"] == "ne" for branch in branches for atom in branch):
        return {"kind": "fallback", "reason": "constraint_not_in_conditional_whitelist"}
    if any(field not in fields for field in involved):
        return {"kind": "fallback", "reason": "constraint_field_mismatch"}
    return {
        "kind": "conditional",
        "fields": [field for field in fields if field in involved],
        "branches": branches,
    }


def _z3_number(
    expr: dict[str, Any], variables: dict[str, z3.ArithRef], scales: dict[str, int]
) -> z3.ArithRef:
    if "const" in expr:
        return z3.RealVal(str(expr["const"]))
    if "field" in expr:
        field = str(expr["field"])
        value = variables[field]
        return value if scales[field] == 1 else z3.ToReal(value) / scales[field]
    args = [_z3_number(item, variables, scales) for item in expr.get("args") or []]
    op = expr.get("op")
    if op == "add":
        return sum(args, z3.RealVal(0))
    if op == "sub" and len(args) == 2:
        return args[0] - args[1]
    if op == "mul":
        value = z3.RealVal(1)
        for item in args:
            value *= item
        return value
    if op == "div" and len(args) == 2:
        return args[0] / args[1]
    raise ValueError(f"invalid numeric feasibility expression: {expr}")


def _z3_constraint(
    expr: dict[str, Any], variables: dict[str, z3.ArithRef], scales: dict[str, int]
) -> z3.BoolRef:
    op = str(expr.get("op"))
    if op in {"and", "or"}:
        values = [_z3_constraint(item, variables, scales) for item in expr.get("args") or []]
        return z3.And(values) if op == "and" else z3.Or(values)
    if op == "not":
        return z3.Not(_z3_constraint(expr["arg"], variables, scales))
    if op == "implies":
        return z3.Implies(
            _z3_constraint(expr["if"], variables, scales),
            _z3_constraint(expr["then"], variables, scales),
        )
    left = _z3_number(expr["left"], variables, scales)
    right = _z3_number(expr["right"], variables, scales)
    return {"le": left <= right, "lt": left < right, "ge": left >= right,
            "gt": left > right, "eq": left == right}[op]


def _solver_result(solver: z3.Solver) -> str:
    result = solver.check()
    if result == z3.sat:
        return "sat"
    if result == z3.unsat:
        return "unsat"
    return "unknown"


def compile_core_constraints(
    selection: TemplateSelectionLike,
    field_ids: list[str],
    *,
    saved_decimal_places: int,
    solver_timeout_ms: int,
    conditional_branch_limit: int,
) -> dict[str, Any]:
    fields = tuple(dict.fromkeys(map(str, field_ids)))
    constraints = selected_constraints(selection, list(fields))
    metadata = selection.fields
    payload = {
        "template": selection.template,
        "fields": fields,
        "constraints": constraints,
        "saved_decimal_places": saved_decimal_places,
        "conditional_branch_limit": conditional_branch_limit,
        "dependency_rule": "duplicate_model_v1",
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if key in _COMPILED_CACHE:
        return deepcopy(_COMPILED_CACHE[key])
    scales = {
        field: 1 if metadata[field]["type"] == "integer" else 10**saved_decimal_places
        for field in fields
    }

    def variables(suffix: str) -> dict[str, z3.ArithRef]:
        return {field: z3.Int(f"{key}_{suffix}_{field}") for field in fields}

    def assertions(values: dict[str, z3.ArithRef]) -> list[z3.BoolRef]:
        result: list[z3.BoolRef] = []
        for field in fields:
            scale = scales[field]
            low = math.ceil(float(metadata[field]["minimum"]) * scale - 1e-9)
            high = math.floor(float(metadata[field]["maximum"]) * scale + 1e-9)
            result.extend((values[field] >= low, values[field] <= high))
        result.extend(_z3_constraint(item["expr"], values, scales) for item in constraints)
        return result

    first = variables("a")
    solver = z3.Solver()
    solver.set(timeout=solver_timeout_ms)
    solver.add(*assertions(first))
    status = _solver_result(solver)
    if status == "unknown":
        result = {"key": key, "status": "unknown", "reason": solver.reason_unknown()}
        return result
    if status == "unsat":
        result = {"key": key, "status": "infeasible", "constraints": constraints}
        _COMPILED_CACHE[key] = result
        return deepcopy(result)
    second = variables("b")
    dependent: list[str] = []
    for target in fields:
        dependency = z3.Solver()
        dependency.set(timeout=solver_timeout_ms)
        dependency.add(*assertions(first), *assertions(second))
        dependency.add(*(first[field] == second[field] for field in fields if field != target))
        dependency.add(first[target] != second[target])
        relation = _solver_result(dependency)
        if relation == "unknown":
            return {
                "key": key,
                "status": "unknown",
                "reason": dependency.reason_unknown(),
            }
        if relation == "unsat" and len(fields) > 1:
            dependent.append(target)
    result = {
        "key": key,
        "status": "deterministic_core_dependency" if dependent else "feasible",
        "dependent_fields": dependent,
        "fields": list(fields),
        "constraints": constraints,
        "scales": scales,
        "sampling": _sampling_classification(constraints, fields, conditional_branch_limit),
    }
    _COMPILED_CACHE[key] = result
    return deepcopy(result)
