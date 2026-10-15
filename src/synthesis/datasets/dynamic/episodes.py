"""Build a dynamic episode from one frozen, reviewed candidate."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from protocol.constraints import selected_constraints
from ..cases import add_distractor_values, dynamic_batch_quality, dynamic_batches, natural_dynamic_cases, targeted_dynamic_cases
from ..config import DatasetGenerationConfig
from ..contracts import DYNAMIC_EPISODE_SCHEMA_VERSION, EpisodeBuild, EpisodeSpec
from ..template_pool import agent_fields, ordered_actions, select_template, template_hash

_RULE_KEYS = {"family", "relevant_fields", "field_directions", "weights", "bias", "thresholds", "priority_order", "indifference_band", "score_calibration"}


def _release_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in rule.items() if key in _RULE_KEYS}


def _release_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"context_id": str(item["context_id"]), "raw_values": deepcopy(item["raw_values"]), "stratum": "iid"} for item in cases]


def _release_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"batch_id": str(item["batch_id"]), "batch_number": int(item["batch_number"]), "phase": str(item["phase"]), "context_ids": list(item["context_ids"])} for item in batches]


def _operations(dynamic_type: str, transition: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    if dynamic_type in {"field_replacement", "compound"}:
        output.append({"type": "field_replacement", "group_id": str(transition["group_id"]), "transition_id": str(transition.get("replacement_transition_id") or transition["transition_id"]), "outgoing_field": str(transition["outgoing_field"]), "incoming_field": str(transition["incoming_field"])})
    if dynamic_type in {"direction_reversal", "compound"}:
        operation = transition.get("reversal_operation") or transition
        output.append({"type": "direction_reversal", "event_id": str(operation["event_id"]), "target_field": str(transition.get("reversal_target") or transition["target_field"])})
    return output


def _rename(cases: list[dict[str, Any]], prefix: str) -> None:
    for index, item in enumerate(cases, 1):
        item["context_id"] = f"{prefix}_case_{index:03d}"


def _semantic_change(
    dynamic_type: str, transition: dict[str, Any], review: dict[str, Any]
) -> dict[str, str]:
    if dynamic_type == "compound":
        return {
            "trigger_event": str(review["trigger_event"]),
            "change_reason": str(review["change_reason"]),
        }
    if dynamic_type == "field_replacement":
        return {
            "trigger_event": str(transition["trigger"]),
            "change_reason": " ".join(
                (
                    str(transition["outgoing_retirement_reason"]),
                    str(transition["incoming_authority_reason"]),
                )
            ),
        }
    reversal = transition.get("reversal_operation") or transition
    return {
        "trigger_event": str(reversal["trigger"]),
        "change_reason": str(reversal["reversal_reason"]),
    }


def make_dynamic_episode(pool: dict[str, Any], spec: EpisodeSpec, config: DatasetGenerationConfig, candidate: dict[str, Any]) -> EpisodeBuild:
    if config.dynamic is None or spec.after_action_batch is None or spec.dynamic_type is None:
        raise ValueError("dynamic episode requires dynamic config, type, and change point")
    selection = select_template(pool, spec.domain, spec.template_id)
    certification = candidate["certification"]
    old, new = deepcopy(certification["old"]), deepcopy(certification["new"])
    inventory = certification["inventory"]
    layout = config.episode_layout
    pre_count = spec.after_action_batch * layout.action_batch_size
    post_count = (layout.action_batch_count - spec.after_action_batch) * layout.action_batch_size
    pre_cases, pre_diagnostics = natural_dynamic_cases(inventory, old, new, pre_count, spec.seed + 700_001, phase="pre_dynamic")
    pre_pool_indexes = set(pre_diagnostics["pool_indices"])
    distractors = list(candidate["transition"]["distractor_fields"])
    add_distractor_values(
        pre_cases, selection, distractors, spec.seed + 800_001, config
    )
    _rename(pre_cases, "pre")
    branches: dict[str, dict[str, Any]] = {}
    private: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    branch_diagnostics = {}
    branch_specs: list[tuple[str, float | None]] = [("static_iid", None), *((f"flip_{round(target * 100):02d}", float(target)) for target in config.search.q_flip_targets)]
    shared_batches = None
    shared_ordered = None
    for branch_index, (name, target) in enumerate(branch_specs):
        if target is None:
            post_cases, diagnostics = natural_dynamic_cases(inventory, old, old, post_count, spec.seed + 900_001, phase="post_dynamic", excluded_pool_indexes=pre_pool_indexes)
        else:
            post_cases, diagnostics = targeted_dynamic_cases(inventory, certification["target_matrices"][f"{target:.6f}"], old, new, post_count, spec.seed + 900_001 + branch_index * 100_003, excluded_pool_indexes=pre_pool_indexes)
        add_distractor_values(
            post_cases,
            selection,
            distractors,
            spec.seed + 1_000_001 + branch_index * 100_003,
            config,
        )
        _rename(post_cases, name)
        batches, ordered = dynamic_batches(pre_cases, post_cases, spec.seed + 1_100_001 + branch_index, spec.after_action_batch, layout)
        if shared_batches is None:
            shared_batches = batches[:spec.after_action_batch]
            shared_ordered = ordered[:pre_count]
        post_batches = batches[spec.after_action_batch:]
        ordered_post = ordered[pre_count:]
        branches[name] = {"condition": "static" if target is None else "dynamic", "sampling_design": {"mode": "natural_iid" if target is None else "target_flip", "target_flip_rate": target}, "cases": _release_cases(ordered_post), "batches": _release_batches(post_batches)}
        private[name] = post_batches, ordered_post
        branch_diagnostics[name] = diagnostics
    assert shared_batches is not None and shared_ordered is not None
    visible = list(candidate["visible_fields"])
    random.Random(spec.seed).shuffle(visible)
    transition = candidate["transition"]
    review = candidate["semantic_review"]
    heldout = certification["heldout"]
    interaction = {"action_batch_count": layout.action_batch_count, "action_batch_size": layout.action_batch_size, "probe_budget_per_batch": layout.probe_budget_per_batch, "max_probe_cases_per_batch": layout.max_probe_cases_per_batch, "max_probe_budget": layout.max_probe_budget}
    episode = {
        "schema_version": DYNAMIC_EPISODE_SCHEMA_VERSION, "dataset_mode": "dynamic", "episode_id": spec.episode_id, "seed": spec.seed,
        "provenance": {"domain": spec.domain, "template_id": spec.template_id, "sequence": spec.sequence},
        "agent_task": {"agent_role": selection.template["agent_role"], "scenario_description": selection.template["description"], "actions": ordered_actions(selection), "fields": agent_fields(selection, visible), "feasibility_constraints": selected_constraints(selection, visible), "interaction_protocol": interaction},
        "evaluation": {"rule_family": spec.rule_family, "transition": {"type": spec.dynamic_type, "after_action_batch": spec.after_action_batch, "pre_rule": _release_rule(old), "post_rule": _release_rule(new), "operations": _operations(spec.dynamic_type, transition), "population_reference": {key: deepcopy(heldout[key]) for key in ("flip_rate", "old_action_marginal", "new_action_marginal", "action_transition_matrix", "mean_score_shift", "mean_absolute_score_shift")}, "sampling_design": {"candidate_slot_id": candidate["core_id"], "available_post_branches": list(branches)}, "semantic_change": _semantic_change(spec.dynamic_type, transition, review)}, "batch_contract": {"strata_per_batch": {"iid": layout.action_batch_size}}},
        "shared_prefix": {"cases": _release_cases(shared_ordered), "batches": _release_batches(shared_batches)}, "post_branches": branches,
    }
    diagnostics = {"episode_id": spec.episode_id, "template_pool_hash": template_hash(pool), "template_id": spec.template_id, "population_change": {key: deepcopy(value) for key, value in certification.items() if key != "inventory"}, "pre_dynamic_action_pool": pre_diagnostics, "post_branch_action_pools": branch_diagnostics, "private_shared_cases": shared_ordered, "private_shared_batches": shared_batches, "private_post_branches": private, "branch_quality": {name: dynamic_batch_quality([*shared_batches, *rows[0]], [*shared_ordered, *rows[1]]) for name, rows in private.items()}, "model_calls": 0}
    return EpisodeBuild(episode=episode, diagnostics=diagnostics)
