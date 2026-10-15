from __future__ import annotations

from pathlib import Path

import pytest

from protocol.mechanism import hydrate_rule
from synthesis.datasets import cases as cases_module
from synthesis.datasets.cases import (
    build_bucket_inventory,
    dynamic_batches,
    solve_target_matrices,
    static_batches,
    targeted_dynamic_cases,
)
from synthesis.datasets.config import load_config
from synthesis.datasets.dynamic.certification import heldout_transition_mask


def _rule(direction: int = 1) -> dict:
    return hydrate_rule(
        {
            "family": "additive_aggregation",
            "relevant_fields": ["x"],
            "field_directions": {"x": direction},
            "weights": {"x": float(direction)},
            "bias": 0.0,
            "score_calibration": {
                "method": "frozen_pre_quartile_piecewise",
                "evidence_knots": [-0.5, 0.0, 0.5],
                "score_knots": [-0.5, 0.0, 0.5],
                "segment_gains": [1.0, 1.0],
                "clipped_mass_share": 0.0,
            },
        },
        {"x": {"minimum": 0.0, "maximum": 1.0}},
    )


def _records(repetitions: int = 4) -> list[tuple[dict[str, float], str]]:
    values = (0.1, 0.3, 0.6, 0.9)
    return [
        ({"x": value}, f"row-{action}-{copy}")
        for action, value in enumerate(values)
        for copy in range(repetitions)
    ]


def test_iid_layout_and_batches_have_no_removed_strata():
    config = load_config(Path("configs/synthesis/static.yaml"))
    assert config.episode_layout.case_counts == {"iid": 96}
    assert config.episode_layout.case_counts_per_batch == {"iid": 8}
    cases = [
        {"context_id": f"seq_case_{index:03d}", "stratum": "iid"}
        for index in range(1, 97)
    ]
    batches, ordered = static_batches(cases, 17, config.episode_layout)
    assert len(ordered) == 96
    assert all(batch["strata_counts"] == {"iid": 8} for batch in batches)


def test_all_sixteen_transition_buckets_support_q25_q50_q75():
    matrices = solve_target_matrices(
        (True,) * 16, (0.25, 0.5, 0.75), (0.15, 0.35), 1e-7
    )
    assert isinstance(matrices, dict)
    for target in (0.25, 0.5, 0.75):
        matrix = matrices[f"{target:.6f}"]
        assert sum(sum(row) for row in matrix) == pytest.approx(1.0)
        assert sum(
            matrix[row][column]
            for row in range(4)
            for column in range(4)
            if row != column
        ) == pytest.approx(target)
        assert all(value > 0.0 for row in matrix for value in row)


def test_full_mask_feasibility_short_circuits_z3(monkeypatch):
    cases_module.target_mask_is_feasible.cache_clear()
    monkeypatch.setattr(
        cases_module,
        "_target_matrix",
        lambda *args: pytest.fail("full support must not invoke Z3"),
    )
    assert cases_module.target_mask_is_feasible(
        (True,) * 16, (0.25, 0.5, 0.75), (0.15, 0.35), 1e-7
    )


def test_non_full_mask_feasibility_is_cached(monkeypatch):
    cases_module.target_mask_is_feasible.cache_clear()
    calls = []
    monkeypatch.setattr(
        cases_module,
        "solve_target_matrices",
        lambda *args, **kwargs: calls.append(args[0]) or True,
    )
    mask = (False, *(True for _ in range(15)))
    arguments = (mask, (0.25, 0.5, 0.75), (0.15, 0.35), 1e-7)
    assert cases_module.target_mask_is_feasible(*arguments)
    assert cases_module.target_mask_is_feasible(*arguments)
    assert calls == [mask]


def test_diagonal_only_mask_is_infeasible_for_positive_flip_targets():
    mask = tuple(row == column for row in range(4) for column in range(4))
    assert solve_target_matrices(mask, (0.25,), (0.15, 0.35), 1e-7) is None


def test_exact_z3_matrix_is_not_rejected_by_post_solver_reconstruction():
    # A released v6 held-out mask for which Z3 was satisfiable at all three Q
    # targets, but the removed iterative reconstruction returned None.
    mask = tuple(value == "1" for value in "1110111011100011")
    matrices = solve_target_matrices(
        mask, (0.25, 0.5, 0.75), (0.15, 0.35), 1e-7
    )
    assert isinstance(matrices, dict)
    for target in (0.25, 0.5, 0.75):
        matrix = matrices[f"{target:.6f}"]
        assert sum(sum(row) for row in matrix) == pytest.approx(1.0)
        assert all(
            0.15 <= sum(row) <= 0.35
            for row in matrix
        )
        assert all(
            0.15 <= sum(matrix[row][column] for row in range(4)) <= 0.35
            for column in range(4)
        )
        assert sum(
            matrix[row][column]
            for row in range(4)
            for column in range(4)
            if row != column
        ) == pytest.approx(target)
        assert all(
            matrix[index // 4][index % 4] == 0.0
            for index, allowed in enumerate(mask)
            if not allowed
        )


def test_heldout_transition_mask_keeps_82_and_drops_81_occurrences():
    counts = [[0] * 4 for _ in range(4)]
    counts[0][0] = 81
    counts[0][1] = 82
    mask = heldout_transition_mask(counts, 16_384, 0.005)
    assert mask[0] is False
    assert mask[1] is True


def test_inventory_retains_duplicate_numeric_records_as_distinct_rows():
    records = [({"x": 0.3}, "same"), ({"x": 0.3}, "same")]
    inventory = build_bucket_inventory(records, _rule(), _rule())
    assert len(inventory["records"]) == 2
    assert inventory["records"][0]["raw_values"] == inventory["records"][1]["raw_values"]
    bucket = next(indexes for indexes in inventory["buckets"].values() if indexes)
    assert bucket == [0, 1]


def test_targeted_branch_draws_pool_rows_without_replacement():
    inventory = build_bucket_inventory(_records(), _rule(), _rule(-1))
    matrix = [[1.0 if (row, column) == (0, 3) else 0.0 for column in range(4)] for row in range(4)]
    cases, diagnostics = targeted_dynamic_cases(
        inventory, matrix, _rule(), _rule(-1), 4, 31
    )
    assert len({case["operational_profile"] for case in cases}) == 4
    assert diagnostics["realized_flip_rate"] == 1.0


def test_exhausted_bucket_is_removed_and_remaining_weight_is_renormalized():
    records = [({"x": 0.1}, "only-first"), *[({"x": 0.3}, f"rest-{index}") for index in range(4)]]
    inventory = build_bucket_inventory(records, _rule(), _rule())
    matrix = [[0.0] * 4 for _ in range(4)]
    matrix[0][0] = 0.99
    matrix[1][1] = 0.01
    cases, _ = targeted_dynamic_cases(inventory, matrix, _rule(), _rule(), 4, 1)
    assert sum(case["raw_values"]["x"] == 0.1 for case in cases) == 1
    assert sum(case["raw_values"]["x"] == 0.3 for case in cases) == 3


def test_saved_numeric_values_determine_the_same_inventory_bucket():
    old, new = _rule(), _rule(-1)
    records = [({"x": round(value, 6)}, f"row-{index}") for index, value in enumerate((0.2499999, 0.2500001, 0.7499999, 0.7500001))]
    first = build_bucket_inventory(records, old, new)
    second = build_bucket_inventory(
        [(dict(row["raw_values"]), row["operational_profile"]) for row in first["records"]],
        old,
        new,
    )
    assert first["buckets"] == second["buckets"]


def test_realized_batch_ratios_are_diagnostics_not_rejection_conditions():
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    cases = [
        {"context_id": f"case-{index}", "stratum": "iid"}
        for index in range(config.episode_layout.action_case_count)
    ]
    batches, ordered = dynamic_batches(
        cases[:40], cases[40:], 19, 5, config.episode_layout
    )
    assert len(batches) == 12
    assert len(ordered) == 96
