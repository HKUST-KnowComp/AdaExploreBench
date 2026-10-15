"""Finite held-out bucketing and case construction."""

from __future__ import annotations

import math
import random
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from statistics import mean
from typing import Any

import z3

from protocol.mechanism import best_action, explain_control_state, intervention_score, utilities_from_score
from .config import DatasetGenerationConfig, EpisodeLayout
from .population import prepare_base_proposals, sample_population_records, sample_profile_values


def _real(value: z3.ArithRef) -> float:
    rendered = value.as_decimal(24)
    if rendered.endswith("?"):
        rendered = rendered[:-1]
    if rendered.endswith("/"):
        return float(value.numerator_as_long()) / value.denominator_as_long()
    return float(rendered)


def _target_matrix(mask: tuple[bool, ...], flip: float, share_range: tuple[float, float], tolerance: float) -> list[list[float]] | None:
    del tolerance
    values = [z3.Real(f"q_{index}") for index in range(16)]
    optimizer = z3.Optimize()
    for index, value in enumerate(values):
        optimizer.add(value >= 0, value == 0 if not mask[index] else value <= 1)
    optimizer.add(z3.Sum(values) == 1)
    optimizer.add(z3.Sum(value for index, value in enumerate(values) if index // 4 != index % 4) == z3.RealVal(str(flip)))
    deviations = []
    for groups in (
        ([values[row * 4 + column] for column in range(4)] for row in range(4)),
        ([values[row * 4 + column] for row in range(4)] for column in range(4)),
    ):
        for index, group in enumerate(groups):
            total = z3.Sum(group)
            optimizer.add(total >= z3.RealVal(str(share_range[0])), total <= z3.RealVal(str(share_range[1])))
            deviation = z3.Real(f"d_{len(deviations)}")
            optimizer.add(deviation >= total - z3.RealVal("0.25"), deviation >= z3.RealVal("0.25") - total)
            deviations.append(deviation)
    optimizer.minimize(z3.Sum(deviations))
    if optimizer.check() != z3.sat:
        return None
    model = optimizer.model()
    return [
        [
            _real(model.eval(values[row * 4 + column], model_completion=True))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _full_support_matrix(flip: float) -> list[list[float]]:
    diagonal = (1.0 - flip) / 4.0
    off_diagonal = flip / 12.0
    return [
        [diagonal if row == column else off_diagonal for column in range(4)]
        for row in range(4)
    ]


@lru_cache(maxsize=65_536)
def target_mask_is_feasible(
    mask: tuple[bool, ...],
    flip_targets: tuple[float, ...],
    share_range: tuple[float, float],
    tolerance: float,
) -> bool:
    if all(mask) and share_range[0] <= 0.25 <= share_range[1]:
        return all(0.0 <= target <= 1.0 for target in flip_targets)
    return solve_target_matrices(
        mask, flip_targets, share_range, tolerance, feasibility_only=True
    ) is True


def solve_target_matrices(mask: tuple[bool, ...], flip_targets: tuple[float, ...], share_range: tuple[float, float], tolerance: float, *, feasibility_only: bool = False) -> dict[str, list[list[float]]] | bool | None:
    if all(mask) and share_range[0] <= 0.25 <= share_range[1]:
        if not all(0.0 <= target <= 1.0 for target in flip_targets):
            return None
        if feasibility_only:
            return True
        return {
            f"{float(target):.6f}": _full_support_matrix(float(target))
            for target in flip_targets
        }
    matrices: dict[str, list[list[float]]] = {}
    for target in flip_targets:
        matrix = _target_matrix(mask, float(target), share_range, tolerance)
        if matrix is None:
            return None
        matrices[f"{float(target):.6f}"] = matrix
    return True if feasibility_only else matrices


def _oracle(rule: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    score = intervention_score(rule, values)
    utilities = utilities_from_score(score)
    return {
        "score": score,
        "optimal_action_id": f"action_{best_action(rule, values)}",
        "utilities": {f"action_{index}": value for index, value in enumerate(utilities)},
        "control_state": explain_control_state(rule, values),
    }


def build_bucket_inventory(records: list[tuple[dict[str, Any], str]], old_rule: dict[str, Any], new_rule: dict[str, Any]) -> dict[str, Any]:
    buckets = {f"{row},{column}": [] for row in range(4) for column in range(4)}
    rows = []
    for index, (values, profile) in enumerate(records):
        old_action = best_action(old_rule, values)
        new_action = best_action(new_rule, values)
        row = {"pool_index": index, "raw_values": values, "operational_profile": profile, "old_action": old_action, "new_action": new_action}
        rows.append(row)
        buckets[f"{old_action},{new_action}"].append(index)
    return {"records": rows, "buckets": buckets}


def _case(row: dict[str, Any], index: int, old_rule: dict[str, Any], new_rule: dict[str, Any] | None, phase: str) -> dict[str, Any]:
    item = {
        "context_id": f"seq_case_{index:03d}",
        "phase": phase,
        "stratum": "iid",
        "raw_values": deepcopy(row["raw_values"]),
        "operational_profile": row["operational_profile"],
        "private_oracle": {"pre_dynamic" if new_rule is not None else "static": _oracle(old_rule, row["raw_values"])},
    }
    if new_rule is not None:
        item["private_oracle"]["post_dynamic"] = _oracle(new_rule, row["raw_values"])
    return item


def add_distractor_values(
    cases: list[dict[str, Any]],
    selection: Any,
    distractor_fields: list[str],
    seed: int,
    config: DatasetGenerationConfig,
) -> None:
    for index, case in enumerate(cases):
        case["raw_values"].update(
            sample_profile_values(
                selection,
                distractor_fields,
                str(case["operational_profile"]),
                seed + index,
                config.search,
            )
        )


def natural_dynamic_cases(inventory: dict[str, Any], old_rule: dict[str, Any], new_rule: dict[str, Any], count: int, seed: int, *, phase: str, excluded_pool_indexes: set[int] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = excluded_pool_indexes or set()
    rows = [row for row in inventory["records"] if row["pool_index"] not in excluded]
    if len(rows) < count:
        raise ValueError("held-out pool is smaller than the requested natural case count")
    chosen = random.Random(seed).sample(rows, count)
    cases = [_case(row, index, old_rule, new_rule, phase) for index, row in enumerate(chosen, 1)]
    return cases, {"source": "held_out", "row_count": count, "pool_indices": [row["pool_index"] for row in chosen]}


def targeted_dynamic_cases(inventory: dict[str, Any], matrix: list[list[float]], old_rule: dict[str, Any], new_rule: dict[str, Any], count: int, seed: int, *, excluded_pool_indexes: set[int] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    excluded = excluded_pool_indexes or set()
    available = {key: [index for index in indexes if index not in excluded] for key, indexes in inventory["buckets"].items()}
    for indexes in available.values():
        rng.shuffle(indexes)
    selected = []
    for _ in range(count):
        choices = []
        for row in range(4):
            for column in range(4):
                key = f"{row},{column}"
                if available[key] and matrix[row][column] > 0:
                    choices.append((key, matrix[row][column]))
        if not choices:
            raise ValueError("held-out buckets exhausted before branch sampling completed")
        key = _weighted_bucket(choices, rng.random())
        selected.append(inventory["records"][available[key].pop()])
    cases = [_case(row, index, old_rule, new_rule, "post_dynamic") for index, row in enumerate(selected, 1)]
    flips = sum(row["old_action"] != row["new_action"] for row in selected)
    return cases, {"source": "held_out_q", "row_count": count, "realized_flip_rate": flips / count, "pool_indices": [row["pool_index"] for row in selected]}


def _weighted_bucket(items: list[tuple[str, float]], probability: float) -> str:
    total = sum(weight for _, weight in items)
    target, running = probability * total, 0.0
    for key, weight in items:
        running += weight
        if target <= running:
            return key
    return items[-1][0]


def action_pool(rule: dict[str, Any], records: list[tuple[dict[str, Any], str]], count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = build_bucket_inventory(records, rule, rule)
    rows = random.Random(seed).sample(inventory["records"], count)
    return [_case(row, index, rule, None, "static") for index, row in enumerate(rows, 1)], {"source": "natural_population", "row_count": count}


def static_cases(rule: dict[str, Any], selection: Any, visible: list[str], seed: int, config: DatasetGenerationConfig, compiled_constraints: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = config.episode_layout.action_case_count
    base = prepare_base_proposals(selection, visible, seed)
    records = sample_population_records(selection, visible, base, compiled_constraints, replacement_group_id=None, effective_coupling=None, config=config.search, count=count)
    return action_pool(rule, records, count, seed)


def _build_batches(cases: list[dict[str, Any]], seed: int, layout: EpisodeLayout, phases: list[tuple[str, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    source = list(cases)
    rng.shuffle(source)
    batches, ordered, cursor = [], [], 0
    for phase, batch_count in phases:
        for _ in range(batch_count):
            chosen = source[cursor:cursor + layout.action_batch_size]
            cursor += layout.action_batch_size
            number = len(batches) + 1
            batches.append({"batch_id": f"batch_{number:02d}", "batch_number": number, "phase": phase, "context_ids": [item["context_id"] for item in chosen], "strata_counts": {"iid": len(chosen)}})
            ordered.extend(chosen)
    return batches, ordered


def dynamic_batches(pre_cases: list[dict[str, Any]], post_cases: list[dict[str, Any]], seed: int, after_action_batch: int, layout: EpisodeLayout) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pre, ordered_pre = _build_batches(pre_cases, seed, layout, [("pre_dynamic", after_action_batch)])
    post, ordered_post = _build_batches(post_cases, seed + 1, layout, [("post_dynamic", layout.action_batch_count - after_action_batch)])
    for offset, batch in enumerate(post, len(pre) + 1):
        batch["batch_number"] = offset
        batch["batch_id"] = f"batch_{offset:02d}"
    return pre + post, ordered_pre + ordered_post


def static_batches(cases: list[dict[str, Any]], seed: int, layout: EpisodeLayout) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _build_batches(cases, seed, layout, [("static", layout.action_batch_count)])


def dynamic_batch_quality(batches: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["context_id"]: item for item in cases}
    return [{"batch_number": batch["batch_number"], "phase": batch["phase"], "oracle_action_distribution": dict(Counter(by_id[key]["private_oracle"][batch["phase"]]["optimal_action_id"] for key in batch["context_ids"]))} for batch in batches]


def static_batch_quality(batches: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["context_id"]: item for item in cases}
    output = []
    for batch in batches:
        oracles = [by_id[key]["private_oracle"]["static"] for key in batch["context_ids"]]
        margins = [sorted(map(float, oracle["utilities"].values()), reverse=True) for oracle in oracles]
        output.append({"batch_number": batch["batch_number"], "oracle_action_distribution": dict(Counter(oracle["optimal_action_id"] for oracle in oracles)), "score_margin": {"minimum": min(row[0] - row[1] for row in margins), "mean": mean(row[0] - row[1] for row in margins), "maximum": max(row[0] - row[1] for row in margins)}})
    return output
