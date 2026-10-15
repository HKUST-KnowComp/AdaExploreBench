from __future__ import annotations

import math

import pytest

from runtime.calculator import CalculatorError, calculate, calculate_many
from runtime.environment import BenchmarkEnvironment


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12.5 + 3) * 4", 62),
        ("sqrt(81) + 3^4", 90),
        ("1.2e-3 * 50000", 60),
        ("10 // 3 + 10 % 3", 4),
        ("abs(-4.5) + min(2, 3) + max(7, 1)", 13.5),
        ("sum(1, 2, 3) + mean(2, 4, 6)", 10),
        ("average(2, 4, 6) + avg(1, 2, 3) + ln(exp(1))", 7),
        ("round((7.41-5.11)/5.11*100, 2)", 45.01),
        ("2×3 + 8÷4 − 1", 7),
        ("log(exp(2)) + log10(100) + floor(2.9) + ceil(2.1)", 9),
    ],
)
def test_calculator_supports_observed_numeric_forms(expression, expected):
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "let a = 2 + 3",
        "a + 3",
        "2 < 3",
        "[1, 2, 3]",
        "__import__(\"os\")",
        "(1).__class__",
        "1 / 0",
        "2 ** 101",
    ],
)
def test_calculator_rejects_non_numeric_or_unsafe_syntax(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_calculator_localizes_invalid_expressions():
    result = calculate_many(["2 + 3", "let a = 4", "sqrt(16)"])
    assert result == {
        "type": "calculator_result",
        "results": [
            {"index": 0, "ok": True, "value": 5},
            {
                "index": 1,
                "ok": False,
                "error": {
                    "type": "invalid_expression",
                    "message": "invalid numeric expression syntax",
                },
            },
            {"index": 2, "ok": True, "value": 4.0},
        ],
        "variables": {},
        "successful_expression_count": 2,
        "invalid_expression_count": 1,
    }


def test_calculator_supports_variables_within_one_call():
    result = calculate_many(
        ["X = (12.5 + 3) * 4", "Y = X * 2", "sqrt(X) + Y", "mean(X, Y)"]
    )
    assert result["results"][0] == {
        "index": 0,
        "ok": True,
        "value": 62,
        "variable": "X",
    }
    assert result["results"][1] == {
        "index": 1,
        "ok": True,
        "value": 124,
        "variable": "Y",
    }
    assert result["results"][2]["value"] == pytest.approx(124 + math.sqrt(62))
    assert result["results"][3]["value"] == pytest.approx(93)
    assert result["variables"] == {"X": 62, "Y": 124}


def test_calculator_resolves_assignments_before_other_expressions():
    result = calculate_many(
        ["Y = X * 2", "X = 4", "Y + X", "X + 1"]
    )
    assert [row["value"] for row in result["results"]] == [8, 4, 12, 5]
    assert result["results"][0]["variable"] == "Y"
    assert result["results"][1]["variable"] == "X"
    assert result["variables"] == {"X": 4, "Y": 8}


def test_calculator_variables_are_stateless_between_batches():
    first = calculate_many(["X = 2 * 3", "X + 1"])
    assert [row["value"] for row in first["results"]] == [6, 7]
    assert first["variables"] == {"X": 6}

    direct_reference = calculate_many(["X + 1"])
    assert direct_reference["results"] == [
        {
            "index": 0,
            "ok": False,
            "error": {
                "type": "invalid_expression",
                "message": "unknown variable: X",
            },
        }
    ]
    assert direct_reference["variables"] == {}

    reassigned = calculate_many(["X = 5", "X + 1"])
    assert [row["value"] for row in reassigned["results"]] == [5, 6]
    assert reassigned["results"][0]["variable"] == "X"
    assert reassigned["variables"] == {"X": 5}


def test_calculator_variable_errors_stay_local_to_each_expression():
    result = calculate_many(["Y = 1 / 0", "Y + 1", "abs = 3", "Z = 2 + 2", "Z"])
    rows = result["results"]
    assert rows[0]["ok"] is False
    assert rows[1]["ok"] is False
    assert "unknown variable" in rows[1]["error"]["message"]
    assert rows[2]["ok"] is False
    assert "conflicts with a supported function" in rows[2]["error"]["message"]
    assert rows[3]["ok"] is True
    assert rows[4]["ok"] is True
    assert rows[4]["value"] == 4
    assert result["variables"] == {"Z": 4}


def test_calculator_rejects_consecutive_calls_until_another_tool_runs(
    dynamic_episode,
    runtime_interaction_policy,
):
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    result = env.execute("calculator", {"expressions": ["1 + 1"]})
    assert result["results"][0]["value"] == 2
    usability = {
        entry["tool"]: entry for entry in env.state_card()["tool_usability"]
    }
    assert usability["calculator"]["available"] is False
    assert "one calculator call" in usability["calculator"]["reason"]
    with pytest.raises(ValueError, match="calculator was already used"):
        env.execute("calculator", {"expressions": ["2 + 2"]})
    assert env.calculator_calls == 1
    env.execute("request_next_batch", {})
    usability = {
        entry["tool"]: entry for entry in env.state_card()["tool_usability"]
    }
    assert usability["calculator"]["available"] is True
    assert usability["calculator"]["reason"] is None
    result = env.execute("calculator", {"expressions": ["X = 3^2", "X * 10"]})
    assert result["results"][1]["value"] == 90
    assert env.calculator_calls == 2


def test_calculator_remains_available_while_action_batch_is_pending(
    dynamic_episode,
    runtime_interaction_policy,
):
    env = BenchmarkEnvironment(
        dynamic_episode, interaction_policy=runtime_interaction_policy,
        sampling_branch="flip_50",
    )
    env.execute("request_next_batch", {})
    assert env.pending_batch is not None
    assert {
        entry["tool"]: entry["available"]
        for entry in env.state_card()["tool_usability"]
    } == {
        "probe_batch": False,
        "request_next_batch": False,
        "submit_action_batch": True,
        "calculator": True,
    }
    result = env.execute("calculator", {"expressions": ["sqrt(144)"]})
    assert result["results"][0]["value"] == 12.0
    assert {
        entry["tool"]: entry["available"]
        for entry in env.state_card()["tool_usability"]
    }["calculator"] is False
