"""Search and certify one dynamic core against its natural population."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from statistics import NormalDist
from typing import Any

from protocol.constraints import compile_core_constraints, constraint_fields
from protocol.mechanism import validate_mechanism
from ..cases import build_bucket_inventory, solve_target_matrices, target_mask_is_feasible
from ..config import DatasetGenerationConfig
from ..evaluation import CalibrationDegeneracyError, dynamic_population_rank, fit_frozen_pre_calibration, measure_rule_pair, normalized_state_arrays, population_gate_failures, precompute_rule_population, replacement_pair_diagnostics
from ..planning import stable_seed
from ..population import PopulationBudgetExhausted, PopulationPool, prepare_base_proposals, sample_states
from ..template_pool import TemplateSelection
from .rules import sample_rule_pair
from .structures import distractor_sources


def _record_rejection(
    rejection_counts: dict[str, int] | None, stage: str
) -> None:
    if rejection_counts is not None:
        rejection_counts[stage] = rejection_counts.get(stage, 0) + 1


def _core_identity(core: dict[str, Any], dynamic_type: str) -> tuple[Any, ...]:
    identity: list[Any] = [dynamic_type, tuple(map(str, core["shared_fields"]))]
    if dynamic_type in {"field_replacement", "compound"}:
        identity.extend((str(core["outgoing_field"]), str(core["incoming_field"])))
    if dynamic_type in {"direction_reversal", "compound"}:
        identity.append(str(core.get("reversal_target") or core["target_field"]))
    identity.append(tuple(map(str, core.get("old_priority_order") or core.get("priority_order") or ())))
    return tuple(identity)


def candidate_core_id(template_id: str, core: dict[str, Any], dynamic_type: str) -> str:
    payload = json.dumps([template_id, *_core_identity(core, dynamic_type)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _core_fields(core: dict[str, Any], dynamic_type: str) -> list[str]:
    fields = list(map(str, core["shared_fields"]))
    if dynamic_type == "direction_reversal":
        fields.append(str(core["target_field"]))
    else:
        fields.extend((str(core["outgoing_field"]), str(core["incoming_field"])))
    return list(dict.fromkeys(fields))


def _control_grids(family: str, dynamic_type: str, config: DatasetGenerationConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    coupling = tuple(map(float, config.search.coupling_grid)) if dynamic_type in {"field_replacement", "compound"} else (0.0,)
    if dynamic_type not in {"direction_reversal", "compound"}:
        reversal = (1.0 if family == "additive_aggregation" else 0.0,)
    elif family == "additive_aggregation":
        reversal = tuple(map(float, config.search.reversal_additive_grid))
    else:
        reversal = tuple(map(float, config.search.reversal_threshold_grid))
    return coupling, reversal


def _binomial_interval(successes: int, count: int, confidence: float, comparisons: int) -> tuple[float, float]:
    alpha = (1.0 - confidence) / max(1, comparisons)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    share = successes / count
    denominator = 1.0 + z * z / count
    center = (share + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(share * (1.0 - share) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _optimistically_possible(stats: dict[str, Any], count: int, config: DatasetGenerationConfig, setting_count: int) -> bool:
    search = config.search
    metrics = [(float(stats["flip_rate"]), search.population_flip_range), *((float(value), search.action_share_range) for value in stats["old_action_marginal"]), *((float(value), search.action_share_range) for value in stats["new_action_marginal"])]
    transition_count = sum(len(row) for row in stats["action_transition_matrix"])
    comparisons = setting_count * (len(metrics) + transition_count)
    for share, (lower, upper) in metrics:
        interval = _binomial_interval(round(share * count), count, search.early_rejection_confidence, comparisons)
        if interval[1] < lower or interval[0] > upper:
            return False
    return True


def _optimistic_transition_mask(
    stats: dict[str, Any],
    count: int,
    config: DatasetGenerationConfig,
    setting_count: int,
) -> tuple[bool, ...]:
    search = config.search
    metric_count = 1 + len(stats["old_action_marginal"]) + len(stats["new_action_marginal"])
    transition_count = sum(len(row) for row in stats["action_transition_matrix"])
    comparisons = setting_count * (metric_count + transition_count)
    return tuple(
        _binomial_interval(
            int(value), count, search.early_rejection_confidence, comparisons
        )[1]
        >= search.min_transition_probability
        for row in stats["action_transition_matrix"]
        for value in row
    )


def heldout_transition_mask(
    transition_counts: list[list[int]], sample_count: int, minimum_probability: float
) -> tuple[bool, ...]:
    threshold = math.ceil(minimum_probability * sample_count)
    return tuple(
        int(value) >= threshold for row in transition_counts for value in row
    )


def _rule_pair(selection: TemplateSelection, core: dict[str, Any], family: str, dynamic_type: str, seed: int, reversal: float, config: DatasetGenerationConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if config.dynamic is None:
        raise ValueError("dynamic certification requires dynamic config")
    old, new, parameters = sample_rule_pair(
        selection, core, family, seed, dynamic_type,
        reversal_control=reversal,
        replacement_role_strength_jitter_range=config.dynamic.replacement_role_strength_jitter_range,
        replacement_threshold_quantile_jitter_range=config.dynamic.replacement_threshold_quantile_jitter_range,
    )
    validate_mechanism(old, selection.fields)
    validate_mechanism(new, selection.fields)
    return old, new, parameters


def _calibration_depends_on_coupling(
    compiled: dict[str, Any], core: dict[str, Any], dynamic_type: str
) -> bool:
    if dynamic_type not in {"field_replacement", "compound"}:
        return False
    replacement_fields = {str(core["outgoing_field"]), str(core["incoming_field"])}
    return any(
        constraint_fields(constraint["expr"]) & replacement_fields
        for constraint in compiled.get("constraints") or []
    )


def _select_refinement_settings(
    ranked: list[tuple[tuple[float, float, int, int], dict[str, Any]]],
    coupling_count: int,
    reversal_count: int,
    config: DatasetGenerationConfig,
) -> list[dict[str, Any]] | None:
    required_couplings = min(config.search.refinement_coupling_count, coupling_count)
    settings_per_coupling = min(
        config.search.refinement_settings_per_coupling, reversal_count
    )
    coupling_order = tuple(
        dict.fromkeys(float(row["effective_coupling"]) for _, row in ranked)
    )
    selected: list[dict[str, Any]] = []
    for coupling in coupling_order:
        coupling_settings: list[dict[str, Any]] = []
        for _, row in ranked:
            if float(row["effective_coupling"]) != coupling:
                continue
            if target_mask_is_feasible(
                row["optimistic_transition_mask"],
                config.search.q_flip_targets,
                config.search.q_action_share_range,
                config.search.q_tolerance,
            ):
                coupling_settings.append(row)
                if len(coupling_settings) == settings_per_coupling:
                    break
        if len(coupling_settings) == settings_per_coupling:
            selected.extend(coupling_settings)
            if len(selected) == required_couplings * settings_per_coupling:
                return selected
    return None


def _evaluate_structure(
    selection: TemplateSelection,
    core: dict[str, Any],
    family: str,
    dynamic_type: str,
    seed: int,
    config: DatasetGenerationConfig,
    compiled: dict[str, Any],
    rejection_counts: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    coupling_grid, reversal_grid = _control_grids(family, dynamic_type, config)
    fields = _core_fields(core, dynamic_type)
    group_id = str(core["group_id"]) if dynamic_type in {"field_replacement", "compound"} else None
    evaluation_base = prepare_base_proposals(selection, fields, stable_seed(seed, "evaluation"))
    coupling_pools: dict[float, PopulationPool] = {}
    coupling_dependent_calibration = _calibration_depends_on_coupling(
        compiled, core, dynamic_type
    )
    shared_calibration: dict[str, Any] | None = None
    screening: list[tuple[tuple[float, float, int, int], dict[str, Any]]] = []
    for coupling_index, coupling in enumerate(coupling_grid):
        evaluation_pool = PopulationPool()
        coupling_pools[coupling] = evaluation_pool
        try:
            evaluation_states = sample_states(selection, fields, evaluation_base, compiled, replacement_group_id=group_id, effective_coupling=coupling, config=config.search, count=config.search.parameter_screening_sample_count, pool=evaluation_pool)
            arrays = normalized_state_arrays(selection, fields, evaluation_states)
            first_old, _, _ = _rule_pair(selection, core, family, dynamic_type, seed, reversal_grid[0], config)
            if coupling_dependent_calibration or shared_calibration is None:
                calibrated_old = fit_frozen_pre_calibration(first_old, arrays)
                if not coupling_dependent_calibration:
                    shared_calibration = deepcopy(calibrated_old["score_calibration"])
            else:
                calibrated_old = deepcopy(first_old)
                calibrated_old["score_calibration"] = deepcopy(shared_calibration)
            pre_population = precompute_rule_population(calibrated_old, arrays)
        except PopulationBudgetExhausted:
            _record_rejection(rejection_counts, "screening_population_sampling")
            return None
        except CalibrationDegeneracyError:
            _record_rejection(rejection_counts, "screening_calibration")
            return None
        for reversal_index, reversal in enumerate(reversal_grid):
            old, new, _ = _rule_pair(selection, core, family, dynamic_type, seed, reversal, config)
            old["score_calibration"] = deepcopy(calibrated_old["score_calibration"])
            new["score_calibration"] = deepcopy(calibrated_old["score_calibration"])
            stats = measure_rule_pair(old, new, arrays, pre_population=pre_population)
            if not _optimistically_possible(stats, len(evaluation_states), config, len(coupling_grid) * len(reversal_grid)):
                continue
            rank = dynamic_population_rank(stats, config.search.population_flip_range, config.search.action_share_range, coupling, reversal, config.search.coupling_baseline, config.search.reversal_additive_baseline if family == "additive_aggregation" else config.search.reversal_threshold_baseline, coupling_index, reversal_index)
            screening.append((rank, {"effective_coupling": coupling, "reversal_control": reversal, "optimistic_transition_mask": _optimistic_transition_mask(stats, len(evaluation_states), config, len(coupling_grid) * len(reversal_grid))}))
    screening.sort(key=lambda item: item[0], reverse=True)
    if not screening:
        _record_rejection(rejection_counts, "screening_population_gate")
        return None
    selected_settings = _select_refinement_settings(
        screening, len(coupling_grid), len(reversal_grid), config
    )
    if selected_settings is None:
        _record_rejection(rejection_counts, "screening_q_or_diversity_gate")
        return None

    shared_calibration = None
    refinement_resources: dict[float, tuple[dict[str, list[float]], dict[str, Any], dict[str, Any], dict[str, Any] | None, PopulationPool]] = {}
    refined: list[tuple[tuple[float, float, int, int], dict[str, Any]]] = []
    for coupling in dict.fromkeys(float(row["effective_coupling"]) for row in selected_settings):
        evaluation_pool = coupling_pools[coupling]
        try:
            evaluation_states = sample_states(selection, fields, evaluation_base, compiled, replacement_group_id=group_id, effective_coupling=coupling, config=config.search, count=config.search.evaluation_sample_count, pool=evaluation_pool)
            arrays = normalized_state_arrays(selection, fields, evaluation_states)
            pair_diagnostics = replacement_pair_diagnostics(selection, str(core["outgoing_field"]), str(core["incoming_field"]), evaluation_states, minimum_absolute_correlation=config.search.achieved_correlation_range[0], maximum_absolute_correlation=config.search.achieved_correlation_range[1], minimum_mean_divergence=config.search.minimum_mean_normalized_divergence) if group_id is not None else None
            first_reversal = next(
                float(row["reversal_control"])
                for row in selected_settings
                if float(row["effective_coupling"]) == coupling
            )
            first_old, _, _ = _rule_pair(selection, core, family, dynamic_type, seed, first_reversal, config)
            if coupling_dependent_calibration or shared_calibration is None:
                calibrated_old = fit_frozen_pre_calibration(first_old, arrays)
                if not coupling_dependent_calibration:
                    shared_calibration = deepcopy(calibrated_old["score_calibration"])
            else:
                calibrated_old = deepcopy(first_old)
                calibrated_old["score_calibration"] = deepcopy(shared_calibration)
            pre_population = precompute_rule_population(calibrated_old, arrays)
        except PopulationBudgetExhausted:
            _record_rejection(rejection_counts, "refinement_population_sampling")
            return None
        except CalibrationDegeneracyError:
            _record_rejection(rejection_counts, "refinement_calibration")
            return None
        refinement_resources[coupling] = (arrays, calibrated_old, pre_population, pair_diagnostics, evaluation_pool)
    for setting in selected_settings:
        coupling = float(setting["effective_coupling"])
        reversal = float(setting["reversal_control"])
        arrays, calibrated_old, pre_population, pair_diagnostics, evaluation_pool = refinement_resources[coupling]
        old, new, parameters = _rule_pair(selection, core, family, dynamic_type, seed, reversal, config)
        old["score_calibration"] = deepcopy(calibrated_old["score_calibration"])
        new["score_calibration"] = deepcopy(calibrated_old["score_calibration"])
        stats = measure_rule_pair(old, new, arrays, pre_population=pre_population)
        if not _optimistically_possible(stats, config.search.evaluation_sample_count, config, len(selected_settings)):
            continue
        rank = dynamic_population_rank(stats, config.search.population_flip_range, config.search.action_share_range, coupling, reversal, config.search.coupling_baseline, config.search.reversal_additive_baseline if family == "additive_aggregation" else config.search.reversal_threshold_baseline, coupling_grid.index(coupling), reversal_grid.index(reversal))
        refined.append((rank, {"old": old, "new": new, "parameters": parameters, "effective_coupling": coupling, "reversal_control": reversal, "evaluation": stats, "replacement_pair_diagnostics": pair_diagnostics, "optimistic_transition_mask": _optimistic_transition_mask(stats, config.search.evaluation_sample_count, config, len(selected_settings)), "_evaluation_arrays": arrays, "_pre_population": pre_population, "_evaluation_base": evaluation_base, "_evaluation_pool": evaluation_pool}))
    if not refined:
        _record_rejection(rejection_counts, "refinement_population_gate")
        return None
    refined.sort(key=lambda item: item[0], reverse=True)
    for _, row in refined:
        if target_mask_is_feasible(row.pop("optimistic_transition_mask"), config.search.q_flip_targets, config.search.q_action_share_range, config.search.q_tolerance):
            arrays = row.pop("_evaluation_arrays")
            pre_population = row.pop("_pre_population")
            row["evaluation"] = measure_rule_pair(
                row["old"],
                row["new"],
                arrays,
                full_report=True,
                pre_population=pre_population,
            )
            return row
    _record_rejection(rejection_counts, "refinement_q_gate")
    return None


def certify_candidate(selection: TemplateSelection, core: dict[str, Any], family: str, dynamic_type: str, config: DatasetGenerationConfig, compiled: dict[str, Any], selected: dict[str, Any], rejection_counts: dict[str, int] | None = None) -> dict[str, Any] | None:
    fields = _core_fields(core, dynamic_type)
    group_id = str(core["group_id"]) if dynamic_type in {"field_replacement", "compound"} else None
    pool: PopulationPool = selected.pop("_evaluation_pool")
    base = selected.pop("_evaluation_base")
    try:
        heldout = sample_states(selection, fields, base, compiled, replacement_group_id=group_id, effective_coupling=selected["effective_coupling"], config=config.search, count=config.search.held_out_pool_size, pool=pool)
    except PopulationBudgetExhausted:
        _record_rejection(rejection_counts, "heldout_population_sampling")
        return None
    stats = measure_rule_pair(selected["old"], selected["new"], normalized_state_arrays(selection, fields, heldout), full_report=True)
    if population_gate_failures(stats, config.search.population_flip_range, config.search.action_share_range):
        _record_rejection(rejection_counts, "heldout_population_gate")
        return None
    if group_id is not None:
        pair_diagnostics = replacement_pair_diagnostics(
            selection,
            str(core["outgoing_field"]),
            str(core["incoming_field"]),
            heldout,
            minimum_absolute_correlation=config.search.achieved_correlation_range[0],
            maximum_absolute_correlation=config.search.achieved_correlation_range[1],
            minimum_mean_divergence=config.search.minimum_mean_normalized_divergence,
        )
        if not pair_diagnostics["joint_realism_pass"]:
            _record_rejection(rejection_counts, "heldout_replacement_realism_gate")
            return None
        selected["replacement_pair_diagnostics"] = pair_diagnostics
    mask = heldout_transition_mask(
        stats["action_transition_matrix"],
        config.search.held_out_pool_size,
        config.search.min_transition_probability,
    )
    matrices = solve_target_matrices(mask, config.search.q_flip_targets, config.search.q_action_share_range, config.search.q_tolerance)
    if not isinstance(matrices, dict):
        _record_rejection(rejection_counts, "heldout_q_gate")
        return None
    return {**selected, "heldout": stats, "transition_mask": list(mask), "target_matrices": matrices, "inventory": build_bucket_inventory(pool.records, selected["old"], selected["new"]), "population_raw_draw_count": pool.raw_draw_count}


def evaluate_core(selection: TemplateSelection, core: dict[str, Any], family: str, dynamic_type: str, config: DatasetGenerationConfig, *, seed: int, rejection_counts: dict[str, int] | None = None) -> dict[str, Any] | None:
    if config.dynamic is None:
        raise ValueError("dynamic core requires dynamic config")
    sources = distractor_sources(selection, core, dynamic_type)
    core_field_count = len(_core_fields(core, dynamic_type))
    if len(sources) < config.dynamic.dataset_design.visible_field_count - core_field_count:
        _record_rejection(rejection_counts, "distractor_capacity")
        return None
    compiled = compile_core_constraints(selection, _core_fields(core, dynamic_type), saved_decimal_places=config.search.saved_decimal_places, solver_timeout_ms=config.search.solver_timeout_ms, conditional_branch_limit=config.search.conditional_branch_limit)
    if compiled["status"] in {"infeasible", "deterministic_core_dependency"}:
        _record_rejection(rejection_counts, f"constraint_{compiled['status']}")
        return None
    if compiled["status"] != "feasible":
        raise RuntimeError(f"constraint solver did not establish feasibility: {compiled}")
    try:
        selected = _evaluate_structure(selection, core, family, dynamic_type, seed, config, compiled, rejection_counts)
        certified = None if selected is None else certify_candidate(selection, core, family, dynamic_type, config, compiled, selected, rejection_counts)
    except (PopulationBudgetExhausted, CalibrationDegeneracyError):
        _record_rejection(rejection_counts, "unexpected_population_or_calibration")
        return None
    if certified is None:
        return None
    return {"core_id": candidate_core_id(selection.template_id, core, dynamic_type), "template_id": selection.template_id, "catalog_seed": seed, "distractor_source_ids": list(sources), "compiled_constraints": compiled, "certification": certified}


def catalog_candidate(candidate: dict[str, Any], semantic_review: dict[str, Any]) -> dict[str, Any]:
    return {key: candidate[key] for key in ("core_id", "template_id", "catalog_seed", "transition", "visible_fields", "distractor_source_ids", "certification")} | {"semantic_review": semantic_review}
