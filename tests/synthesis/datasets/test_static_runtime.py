from __future__ import annotations

from copy import deepcopy

from synthesis.datasets.contracts import STATIC_EPISODE_SCHEMA_VERSION
from protocol.constraints import constraint_fields

from runtime.metrics import aggregate_static_metrics
from runtime.runner import run_baseline_episode


def test_static_release_protocol_runs(dynamic_episode, runtime_interaction_policy):
    episode = deepcopy(dynamic_episode)
    episode["schema_version"] = STATIC_EPISODE_SCHEMA_VERSION
    transition = episode["evaluation"]["transition"]
    episode["dataset_mode"] = "static"
    episode["evaluation"] = {
        "rule_family": episode["evaluation"]["rule_family"],
        "rule": transition["pre_rule"],
        "batch_contract": episode["evaluation"]["batch_contract"],
    }
    relevant = list(transition["pre_rule"]["relevant_fields"])
    keep = relevant + [
        field for field in episode["agent_task"]["fields"] if field not in relevant
    ][:5]
    episode["agent_task"]["fields"] = {
        field: episode["agent_task"]["fields"][field] for field in keep
    }
    episode["agent_task"]["feasibility_constraints"] = [
        item
        for item in episode["agent_task"]["feasibility_constraints"]
        if constraint_fields(item["expr"]).issubset(keep)
    ]
    shared = episode.pop("shared_prefix")
    static_post = episode.pop("post_branches")["static_iid"]
    episode["cases"] = [*shared["cases"], *static_post["cases"]]
    episode["batches"] = [*shared["batches"], *static_post["batches"]]
    for case in episode["cases"]:
        case["raw_values"] = {
            field: case["raw_values"][field] for field in keep
        }
    for batch in episode["batches"]:
        batch["phase"] = "static"
    artifact = run_baseline_episode(
        episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        seed=7,
        condition="static",
    )
    assert artifact["run_success"] is True
    metrics = aggregate_static_metrics([artifact])
    assert metrics["summary"]["mean_accuracy"] == 1.0
