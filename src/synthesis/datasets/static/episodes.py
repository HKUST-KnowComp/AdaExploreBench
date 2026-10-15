"""Build a static episode from one reviewed candidate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from protocol.constraints import selected_constraints
from ..cases import static_batch_quality, static_batches, static_cases
from ..config import DatasetGenerationConfig
from ..contracts import EpisodeBuild, EpisodeSpec, STATIC_EPISODE_SCHEMA_VERSION
from ..template_pool import agent_fields, ordered_actions, select_template

_RULE_KEYS = {"family", "relevant_fields", "field_directions", "weights", "bias", "thresholds", "priority_order", "indifference_band", "score_calibration"}


def _release_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in rule.items() if key in _RULE_KEYS}


def make_static_episode(pool: dict[str, Any], spec: EpisodeSpec, config: DatasetGenerationConfig, candidate: dict[str, Any]) -> EpisodeBuild:
    selection = select_template(pool, spec.domain, spec.template_id)
    rule = deepcopy(candidate["rule"])
    visible = list(candidate["visible_fields"])
    cases, sampling_diagnostics = static_cases(rule, selection, visible, spec.seed + 700_001, config, candidate["compiled_constraints"])
    batches, ordered = static_batches(cases, spec.seed + 1_100_001, config.episode_layout)
    layout = config.episode_layout
    episode = {
        "schema_version": STATIC_EPISODE_SCHEMA_VERSION, "dataset_mode": "static", "episode_id": spec.episode_id, "seed": spec.seed,
        "provenance": {"domain": spec.domain, "template_id": spec.template_id, "sequence": spec.sequence},
        "agent_task": {"agent_role": selection.template["agent_role"], "scenario_description": selection.template["description"], "actions": ordered_actions(selection), "fields": agent_fields(selection, visible), "feasibility_constraints": selected_constraints(selection, visible), "interaction_protocol": {"action_batch_count": layout.action_batch_count, "action_batch_size": layout.action_batch_size, "probe_budget_per_batch": layout.probe_budget_per_batch, "max_probe_cases_per_batch": layout.max_probe_cases_per_batch, "max_probe_budget": layout.max_probe_budget}},
        "evaluation": {"rule_family": spec.rule_family, "rule": _release_rule(rule), "batch_contract": {"strata_per_batch": {"iid": layout.action_batch_size}}},
        "cases": [{"context_id": item["context_id"], "raw_values": deepcopy(item["raw_values"]), "stratum": "iid"} for item in ordered],
        "batches": [{"batch_id": item["batch_id"], "batch_number": item["batch_number"], "phase": "static", "context_ids": list(item["context_ids"])} for item in batches],
    }
    return EpisodeBuild(episode=episode, diagnostics={"episode_id": spec.episode_id, "template_id": spec.template_id, "action_pool": sampling_diagnostics, "private_cases": ordered, "private_batches": batches, "batch_quality": static_batch_quality(batches, ordered), "semantic_review": candidate["semantic_review"], "model_calls": 0})
