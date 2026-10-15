from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable


MAX_EXPRESSION_CHARACTERS = 2_048
MAX_AST_NODES = 512
MAX_POWER_EXPONENT = 100


class CalculatorError(ValueError):
    """A safe, Agent-correctable calculator input error."""


def _mean(*values: float) -> float:
    if not values:
        raise CalculatorError("mean requires at least one argument")
    return sum(values) / len(values)


def _round(value: float, digits: float = 0) -> int | float:
    if not float(digits).is_integer() or not -15 <= int(digits) <= 15:
        raise CalculatorError("round digits must be an integer from -15 to 15")
    return round(value, int(digits))


def _pow(base: float, exponent: float) -> int | float:
    if abs(exponent) > MAX_POWER_EXPONENT:
        raise CalculatorError(
            f"absolute exponent must not exceed {MAX_POWER_EXPONENT}"
        )
    return pow(base, exponent)


FUNCTIONS: dict[str, Callable[..., int | float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "min": min,
    "max": max,
    "sum": lambda *values: sum(values),
    "mean": _mean,
    "avg": _mean,
    "average": _mean,
    "round": _round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": _pow,
}

BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _normalize(expression: str) -> str:
    return expression.translate(
        str.maketrans({"×": "*", "÷": "/", "−": "-", "^": "**"})
    )


def _check_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("expression must produce a real number")
    if not math.isfinite(float(value)):
        raise CalculatorError("expression produced a non-finite result")
    if abs(value) > 1e308:
        raise CalculatorError("result magnitude is too large")
    return value


def _evaluate(node: ast.AST, variables: dict[str, int | float]) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, variables)
    if isinstance(node, ast.Constant):
        return _check_number(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return _check_number(variables[node.id])
        raise CalculatorError(f"unknown variable: {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARYOPS:
        return _check_number(UNARYOPS[type(node.op)](_evaluate(node.operand, variables)))
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        left = _evaluate(node.left, variables)
        right = _evaluate(node.right, variables)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise CalculatorError(
                f"absolute exponent must not exceed {MAX_POWER_EXPONENT}"
            )
        return _check_number(BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in FUNCTIONS
            or node.keywords
        ):
            raise CalculatorError("unsupported function call")
        arguments = [_evaluate(argument, variables) for argument in node.args]
        try:
            return _check_number(FUNCTIONS[node.func.id](*arguments))
        except CalculatorError:
            raise
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise CalculatorError(str(exc)) from exc
    raise CalculatorError(
        "use numbers, arithmetic operators, parentheses, assigned variables, and supported functions only"
    )


def calculate(
    expression: str, variables: dict[str, int | float] | None = None
) -> int | float:
    if not isinstance(expression, str):
        raise CalculatorError("expression must be a string")
    expression = expression.strip()
    if not expression:
        raise CalculatorError("expression must not be empty")
    if len(expression) > MAX_EXPRESSION_CHARACTERS:
        raise CalculatorError(
            f"expression exceeds {MAX_EXPRESSION_CHARACTERS} characters"
        )
    try:
        tree = ast.parse(_normalize(expression), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise CalculatorError("invalid numeric expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalculatorError("expression is too structurally complex")
    try:
        return _evaluate(tree, variables or {})
    except ZeroDivisionError as exc:
        raise CalculatorError("division by zero") from exc
    except OverflowError as exc:
        raise CalculatorError("numeric overflow") from exc


_ASSIGNMENT = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)")


def calculate_many(expressions: Any) -> dict[str, Any]:
    """Evaluate one calculator batch with a fresh, call-local variable scope.

    Assignment rows still return their numeric value.  All assignments in the
    batch are resolved before non-assignment expressions, so a variable may be
    referenced before its assignment appears in the input list.  The resulting
    variable bindings are used only within this list and are discarded when
    this function returns; callers must pass a numeric result or assign a
    variable again in a later batch.
    """
    if not isinstance(expressions, list) or not expressions:
        raise ValueError("calculator requires a non-empty expressions list")
    results: list[dict[str, Any] | None] = [None] * len(expressions)
    variables: dict[str, int | float] = {}

    assignments: list[tuple[int, str, Any]] = []
    non_assignments: list[tuple[int, Any]] = []
    for index, expression in enumerate(expressions):
        variable = None
        if isinstance(expression, str):
            match = _ASSIGNMENT.match(expression.strip())
            if match:
                variable = match.group(1)
                expression = expression.strip()[match.end():]
        if variable is None:
            non_assignments.append((index, expression))
        else:
            assignments.append((index, variable, expression))

    # Resolve assignment expressions first.  A small fixed-point pass lets
    # assignments depend on variables assigned later in the list while still
    # reporting genuinely unknown/cyclic references as expression-local errors.
    unresolved = assignments
    while unresolved:
        progress = False
        deferred: list[tuple[int, str, Any]] = []
        for index, variable, expression in unresolved:
            try:
                if variable in FUNCTIONS:
                    raise CalculatorError(
                        f"variable name conflicts with a supported function: {variable}"
                    )
                value = calculate(expression, variables)
            except CalculatorError as exc:
                if str(exc).startswith("unknown variable:"):
                    deferred.append((index, variable, expression))
                    continue
                results[index] = {
                    "index": index,
                    "ok": False,
                    "error": {
                        "type": "invalid_expression",
                        "message": str(exc),
                    },
                }
                progress = True
                continue
            variables[variable] = value
            results[index] = {
                "index": index,
                "ok": True,
                "value": value,
                "variable": variable,
            }
            progress = True
        if not deferred:
            break
        if not progress:
            # Evaluate once with the final available scope to preserve the
            # concrete error message for cycles or unresolved names.
            for index, variable, expression in deferred:
                try:
                    if variable in FUNCTIONS:
                        raise CalculatorError(
                            f"variable name conflicts with a supported function: {variable}"
                        )
                    value = calculate(expression, variables)
                except CalculatorError as exc:
                    results[index] = {
                        "index": index,
                        "ok": False,
                        "error": {
                            "type": "invalid_expression",
                            "message": str(exc),
                        },
                    }
                else:
                    variables[variable] = value
                    results[index] = {
                        "index": index,
                        "ok": True,
                        "value": value,
                        "variable": variable,
                    }
            break
        unresolved = deferred

    # Preserve the caller's expression order in the result even though
    # assignment rows were evaluated in a separate phase.
    for index, expression in non_assignments:
        try:
            value = calculate(expression, variables)
            results[index] = {"index": index, "ok": True, "value": value}
        except CalculatorError as exc:
            results[index] = {
                "index": index,
                "ok": False,
                "error": {
                    "type": "invalid_expression",
                    "message": str(exc),
                },
            }
    ordered_results = [row for row in results if row is not None]
    return {
        "type": "calculator_result",
        "results": ordered_results,
        "variables": variables,
        "successful_expression_count": sum(item["ok"] for item in ordered_results),
        "invalid_expression_count": sum(
            not item["ok"] for item in ordered_results
        ),
    }
