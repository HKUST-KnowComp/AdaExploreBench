from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from synthesis.datasets.config import load_config
from protocol.constraints import compile_core_constraints, selected_constraints
from synthesis.datasets.contracts import DYNAMIC_EPISODE_SCHEMA_VERSION
from synthesis.datasets.dynamic.structures import (
    DynamicCoreSpace,
    materialize_distractors,
    visible_fields,
)
from synthesis.datasets.population import sample_raw_values
from synthesis.datasets.template_pool import (
    agent_fields,
    load_template_pool,
    ordered_actions,
    select_template,
)


DATASET_CONFIG = load_config(Path("configs/synthesis/dynamic.yaml"))
ACTION_BATCH_COUNT = DATASET_CONFIG.episode_layout.action_batch_count
STRATA_PER_BATCH = DATASET_CONFIG.episode_layout.case_counts_per_batch
ACTION_BATCH_SIZE = sum(STRATA_PER_BATCH.values())
DEFAULT_DYNAMIC_AFTER_BATCH = DATASET_CONFIG.dynamic.change_after_batches[1]
RUNTIME_INTERACTION_POLICY = (
    yaml.safe_load(
        (
            Path("configs/runtime/default.yaml")
            if Path("configs/runtime/default.yaml").is_file()
            else Path("configs/runtime/default-only.yaml")
        ).read_text()
    )["interaction"]
)


def _rule(selection, family, relevant):
    fields = selection.fields
    weights = {
        field: 0.25 * fields[field]["pressure_direction"] for field in relevant
    }
    weights[relevant[-1]] = 0.5 * fields[relevant[-1]]["pressure_direction"]
    return {
        "family": family,
        "relevant_fields": relevant,
        "field_directions": {
            field: fields[field]["pressure_direction"] for field in relevant
        },
        "weights": weights,
        "bias": 0.0,
        "score_calibration": {
            "method": "test_fixture",
            "negative_anchor": -1.0,
            "positive_anchor": 1.0,
        },
    }


def _fixture_selection():
    pool = load_template_pool("tests/fixtures/domain_templates_four_action.json")
    fixture_domain = next(iter(pool["domains"]))
    template_id = next(iter(pool["domains"][fixture_domain]["templates"]))
    template = pool["domains"][fixture_domain]["templates"][template_id]
    source_fields = list(template["fields"].values())[:2]
    for index, source in enumerate(source_fields, start=1):
        field = deepcopy(source)
        field["display_name"] = f"Fixture distractor {index}"
        field["description"] = (
            f"Independent fixture-only measurement {index} available at decision time."
        )
        field["eligible_families"] = []
        template["fields"][f"fixture_distractor_{index}"] = field
    return select_template(pool, fixture_domain, template_id)


def _episode():
    selection = _fixture_selection()
    fixture_domain = selection.domain
    template_id = selection.template_id
    family = "additive_aggregation"
    core = DynamicCoreSpace(selection, family, "field_replacement").at(0)
    structure = materialize_distractors(
        selection, core, "field_replacement", 71, 8
    )
    shared = structure["shared_fields"]
    outgoing = structure["outgoing_field"]
    incoming = structure["incoming_field"]
    visible = visible_fields(structure, "field_replacement")
    pre_rule = _rule(selection, family, [*shared, outgoing])
    post_rule = _rule(selection, family, [*shared, incoming])
    rng = random.Random(71)
    compiled = compile_core_constraints(
        selection,
        visible,
        saved_decimal_places=DATASET_CONFIG.search.saved_decimal_places,
        solver_timeout_ms=DATASET_CONFIG.search.solver_timeout_ms,
        conditional_branch_limit=DATASET_CONFIG.search.conditional_branch_limit,
    )
    cases = []
    batches = []
    for batch_index in range(ACTION_BATCH_COUNT):
        phase = (
            "pre_dynamic"
            if batch_index < DEFAULT_DYNAMIC_AFTER_BATCH
            else "post_dynamic"
        )
        context_ids = []
        for _ in range(ACTION_BATCH_SIZE):
            values = sample_raw_values(
                selection,
                visible,
                rng.randrange(2**63),
                compiled,
                replacement_group_id=structure["group_id"],
                effective_coupling=DATASET_CONFIG.search.coupling_baseline,
                config=DATASET_CONFIG.search,
            )[0]
            context_id = f"seq_case_{len(cases) + 1:03d}"
            cases.append(
                {
                    "context_id": context_id,
                    "raw_values": values,
                    "stratum": "iid",
                }
            )
            context_ids.append(context_id)
        batches.append(
            {
                "batch_id": f"batch_{batch_index + 1:02d}",
                "batch_number": batch_index + 1,
                "phase": phase,
                "context_ids": context_ids,
            }
        )
    pre_case_count = DEFAULT_DYNAMIC_AFTER_BATCH * ACTION_BATCH_SIZE
    shared_cases = cases[:pre_case_count]
    shared_batches = batches[:DEFAULT_DYNAMIC_AFTER_BATCH]
    source_post_cases = cases[pre_case_count:]
    source_post_batches = batches[DEFAULT_DYNAMIC_AFTER_BATCH:]
    post_branches = {}
    for branch_name, condition, target in (
        ("static_iid", "static", None),
        ("flip_25", "dynamic", 0.25),
        ("flip_50", "dynamic", 0.5),
        ("flip_75", "dynamic", 0.75),
    ):
        id_map = {
            case["context_id"]: f"{branch_name}_{case['context_id']}"
            for case in source_post_cases
        }
        post_branches[branch_name] = {
            "condition": condition,
            "sampling_design": {
                "mode": "natural_iid" if target is None else "target_flip",
                "target_flip_rate": target,
            },
            "cases": [
                {**deepcopy(case), "context_id": id_map[case["context_id"]]}
                for case in source_post_cases
            ],
            "batches": [
                {
                    **deepcopy(batch),
                    "context_ids": [id_map[value] for value in batch["context_ids"]],
                }
                for batch in source_post_batches
            ],
        }
    layout = DATASET_CONFIG.episode_layout
    return {
        "schema_version": DYNAMIC_EPISODE_SCHEMA_VERSION,
        "dataset_mode": "dynamic",
        "episode_id": "additive_domain_fixture",
        "seed": 71,
        "provenance": {
            "domain": fixture_domain,
            "template_id": template_id,
            "sequence": 1,
        },
        "agent_task": {
            "agent_role": selection.template["agent_role"],
            "scenario_description": selection.template["description"],
            "actions": ordered_actions(selection),
            "fields": agent_fields(selection, visible),
            "feasibility_constraints": selected_constraints(selection, visible),
            "interaction_protocol": {
                "action_batch_count": ACTION_BATCH_COUNT,
                "action_batch_size": ACTION_BATCH_SIZE,
                "probe_budget_per_batch": layout.probe_budget_per_batch,
                "max_probe_cases_per_batch": layout.max_probe_cases_per_batch,
                "max_probe_budget": layout.max_probe_budget,
            },
        },
        "evaluation": {
            "rule_family": family,
            "transition": {
                "type": "field_replacement",
                "after_action_batch": DEFAULT_DYNAMIC_AFTER_BATCH,
                "pre_rule": pre_rule,
                "post_rule": post_rule,
                "operations": [
                    {
                        "type": "field_replacement",
                        "group_id": structure["group_id"],
                        "transition_id": structure["transition_id"],
                        "outgoing_field": outgoing,
                        "incoming_field": incoming,
                    }
                ],
                "population_reference": {
                    "flip_rate": 0.5,
                    "old_action_marginal": [0.25, 0.25, 0.25, 0.25],
                    "new_action_marginal": [0.25, 0.25, 0.25, 0.25],
                    "action_transition_matrix": [
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 1, 1, 1],
                        [0, 0, 1, 1],
                    ],
                    "mean_score_shift": 0.0,
                    "mean_absolute_score_shift": 0.5,
                },
                "sampling_design": {
                    "candidate_slot_id": "candidate_001",
                    "available_post_branches": list(post_branches),
                },
                "semantic_change": {
                    "trigger_event": "A test operational policy update occurs.",
                    "change_reason": "The test replacement changes the active evidence source.",
                },
            },
            "batch_contract": {"strata_per_batch": dict(STRATA_PER_BATCH)},
        },
        "shared_prefix": {"cases": shared_cases, "batches": shared_batches},
        "post_branches": post_branches,
    }


@pytest.fixture(scope="session")
def dynamic_base_episode():
    return _episode()


@pytest.fixture(scope="session")
def dynamic_selection():
    return _fixture_selection()


@pytest.fixture()
def dynamic_episode(dynamic_base_episode):
    return deepcopy(dynamic_base_episode)


@pytest.fixture()
def runtime_interaction_policy():
    return deepcopy(RUNTIME_INTERACTION_POLICY)
