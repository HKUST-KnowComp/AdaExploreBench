from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # viewer data can still be built without run configs.
    yaml = None
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from protocol.mechanism import evaluate_utilities as v3_evaluate_utilities
from protocol.mechanism import hydrate_rule
from protocol.mechanism import intervention_score as v3_intervention_score
from protocol.precision import round_field_values
from runtime.metrics import episode_metrics as v3_episode_metrics
from runtime.metrics import first_learn_rule_comparison
from protocol.constraints import constraint_violations as v3_constraint_violations
from runtime.conversation import ConversationSession
from runtime.runner import render_system_prompt
from runtime.schema import SEQUENTIAL_TRACE_SCHEMA_VERSIONS
from generator.conversation_adapters import environment_message_text
from viewer.run_summary import (
    RELEASE_VERSION,
    RUN_SUMMARY_CACHE_NAME,
    SUPPORTED_BRANCHED_TRACE_SCHEMA_VERSIONS,
    _aggregate_relevant_fields_assessments,
    _build_run_summary,
    _canonical_sampling_branch,
    _is_dynamic_episode,
    _load_json,
    _load_run_artifacts,
    _load_run_summary_cache,
    _load_validated_dataset,
    _log,
    _run_label,
    _write_json,
    _write_run_summary_cache,
    precompute_summaries,
)


RELEASE_NAME = "AdaExploreBench Dynamic"
RELEASE_SLUG = "dynamic"
DEFAULT_DATASET = (
    REPO_ROOT / "datasets" / "full-384" / "dynamic_episodes.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "viewer" / "public" / "data"
LEGACY_HISTORY_POLICY_NAMES = {
    "full_history": "eval_case_given",
    "outcome_only": "eval_case_hidden",
}


def _runtime_interaction_policy() -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load the runtime configuration")
    config_paths = [
        REPO_ROOT / "configs" / "runtime" / "default.yaml",
        REPO_ROOT / "configs" / "runtime" / "default-only.yaml",
    ]
    for config_path in config_paths:
        if not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text()) or {}
        policy = config.get("interaction")
        if isinstance(policy, dict):
            return dict(policy)
    raise ValueError("runtime config requires an interaction mapping")


def write_gzip_mirror(output: Path) -> int:
    """Write deterministic gzip mirrors for every published JSON payload."""

    for stale in output.rglob("*.json.gz"):
        stale.unlink()
    count = 0
    paths = sorted(output.rglob("*.json"))
    _log(f"writing gzip mirrors for {len(paths)} payloads")
    for path in paths:
        compressed = gzip.compress(path.read_bytes(), compresslevel=9, mtime=0)
        path.with_suffix(path.suffix + ".gz").write_bytes(compressed)
        count += 1
    return count


def _normalize(value: float, meta: dict[str, Any]) -> float:
    low, high = float(meta["minimum"]), float(meta["maximum"])
    return (float(value) - low) / (high - low) if high > low else 0.0


def _sample_value(meta: dict[str, Any], fraction: float) -> int | float:
    value = float(meta["minimum"]) + fraction * (float(meta["maximum"]) - float(meta["minimum"]))
    return int(round(value)) if meta["type"] == "integer" else round(value, 2)


def _reference_values(episode: dict[str, Any]) -> dict[str, int | float]:
    fields = episode["agent_task"]["fields"]
    records = [
        item["raw_values"] for item in episode["shared_prefix"]["cases"]
    ] + [
        item["raw_values"]
        for branch in episode["post_branches"].values()
        for item in branch["cases"]
    ]
    output = {}
    for field, meta in fields.items():
        value = median(float(item[field]) for item in records)
        output[field] = int(round(value)) if meta["type"] == "integer" else round(value, 2)
    return output


def _point(mechanism: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    utilities = v3_evaluate_utilities(mechanism, values)
    score = v3_intervention_score(mechanism, values)
    return {
        "score": round(score, 6),
        "utilities": [round(value, 6) for value in utilities],
        "oracle_action_index": max(range(len(utilities)), key=utilities.__getitem__),
    }


def _mechanism_summary(mechanism: dict[str, Any]) -> dict[str, Any]:
    family = mechanism["family"]
    if family == "additive_aggregation":
        detail = {
            "weights": mechanism["weights"],
            "bias": mechanism["bias"],
            "score_calibration": mechanism.get("score_calibration"),
        }
    elif family in {"conjunctive_gating", "disjunctive_sufficiency"}:
        detail = {
            "thresholds": mechanism["thresholds"],
            "score_calibration": mechanism.get("score_calibration"),
        }
    elif family == "lexicographic_priority":
        detail = {
            "thresholds": mechanism["thresholds"],
            "priority_order": mechanism["priority_order"],
            "indifference_band": mechanism["indifference_band"],
            "score_calibration": mechanism.get("score_calibration"),
        }
    else:
        detail = {}
    return {"family": family, "detail": detail, "domain_rationale": mechanism.get("domain_rationale", "")}


def _curves(
    episode: dict[str, Any],
    reference: dict[str, Any],
    mechanism: dict[str, Any],
    relevant_fields: list[str],
) -> list[dict[str, Any]]:
    fields = episode["agent_task"]["fields"]
    action_count = len(episode["agent_task"].get("actions") or [])
    constraints = episode["agent_task"].get("feasibility_constraints") or []
    output = []
    for field in relevant_fields:
        meta = fields[field]
        samples = []
        for index in range(101):
            values = dict(reference)
            values[field] = _sample_value(meta, index / 100)
            feasible = not v3_constraint_violations(constraints, values)
            samples.append({
                "x": values[field],
                "normalized_x": round(_normalize(values[field], meta), 6),
                "feasible": feasible,
                **(_point(mechanism, values) if feasible else {"score": None, "utilities": [None] * action_count}),
            })
        output.append({"field_id": field, "reference_value": reference[field], "samples": samples})
    return output


def _heatmap(
    episode: dict[str, Any],
    reference: dict[str, Any],
    mechanism: dict[str, Any],
    relevant_fields: list[str],
) -> dict[str, Any] | None:
    x_field, y_field = relevant_fields[:2]
    region = None
    fields = episode["agent_task"]["fields"]
    action_count = len(episode["agent_task"].get("actions") or [])
    constraints = episode["agent_task"].get("feasibility_constraints") or []
    cells = []
    for yi in range(25):
        for xi in range(25):
            values = dict(reference)
            values[x_field] = _sample_value(fields[x_field], xi / 24)
            values[y_field] = _sample_value(fields[y_field], yi / 24)
            feasible = not v3_constraint_violations(constraints, values)
            cells.append({
                "x": values[x_field], "y": values[y_field], "xi": xi, "yi": yi, "feasible": feasible,
                **(_point(mechanism, values) if feasible else {"score": None, "utilities": [None] * action_count}),
            })
    return {"x_field": x_field, "y_field": y_field, "size": 25, "region": region, "cells": cells}


def build_episode_payload(
    episode: dict[str, Any],
    interaction_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _is_dynamic_episode(episode):
        raise ValueError("viewer data generation requires a dynamic episode")
    transition = episode["evaluation"]["transition"]
    fields = episode["agent_task"]["fields"]
    pre_rule = hydrate_rule(transition["pre_rule"], fields)
    post_rule = hydrate_rule(transition["post_rule"], fields)
    pre_relevant = list(pre_rule["relevant_fields"])
    post_relevant = list(post_rule["relevant_fields"])
    reference = _reference_values(episode)
    actions = episode["agent_task"]["actions"]
    phase_by_id = {
        context_id: batch["phase"]
        for batch in episode["shared_prefix"]["batches"]
        for context_id in batch["context_ids"]
    }
    branch_by_id = {context_id: "shared_prefix" for context_id in phase_by_id}
    for branch_name, branch in episode["post_branches"].items():
        for batch in branch["batches"]:
            for context_id in batch["context_ids"]:
                phase_by_id[context_id] = batch["phase"]
                branch_by_id[context_id] = branch_name
    release_cases = [*episode["shared_prefix"]["cases"]] + [
        item
        for branch in episode["post_branches"].values()
        for item in branch["cases"]
    ]
    cases = []
    for item in release_cases:
        phase = phase_by_id[item["context_id"]]
        active_rule = pre_rule if phase == "pre_dynamic" else post_rule
        runtime_values = round_field_values(item["raw_values"], fields)
        computed = _point(active_rule, runtime_values)
        cases.append({
            "context_id": item["context_id"],
            "stratum": item["stratum"],
            "phase": phase,
            "sampling_branch": branch_by_id[item["context_id"]],
            "raw_values": runtime_values,
            "normalized_values": {field: round(_normalize(runtime_values[field], fields[field]), 6) for field in fields},
            "oracle_action_id": actions[computed["oracle_action_index"]]["action_id"],
            **computed,
        })
    prompt = (REPO_ROOT / "prompts" / "runtime" / "agent_system.txt").read_text()
    episode_interaction = episode["agent_task"]["interaction_protocol"]
    policy = interaction_policy or _runtime_interaction_policy()
    task_spec = {
        "agent_role": episode["agent_task"]["agent_role"],
        "actions": actions,
        "interaction_protocol": {
            "action_batch_size": episode_interaction["action_batch_size"],
            "max_probe_cases_per_batch": policy["max_probe_cases_per_batch"],
        },
    }
    system_prompt = render_system_prompt(prompt, task_spec)
    interaction_spec = {
        "action_batch_count": episode_interaction["action_batch_count"],
        "action_batch_size": episode_interaction["action_batch_size"],
        "action_case_count": (
            episode_interaction["action_batch_count"]
            * episode_interaction["action_batch_size"]
        ),
        **policy,
        "probe_budget": episode_interaction["max_probe_budget"],
    }
    mechanism_views = {
        "pre_change": {
            "relevant_fields": pre_relevant,
            "mechanism": _mechanism_summary(pre_rule),
            "mechanism_raw": pre_rule,
            "field_curves": _curves(episode, reference, pre_rule, pre_relevant),
            "heatmap": _heatmap(episode, reference, pre_rule, pre_relevant),
        },
        "post_change": {
            "relevant_fields": post_relevant,
            "mechanism": _mechanism_summary(post_rule),
            "mechanism_raw": post_rule,
            "field_curves": _curves(episode, reference, post_rule, post_relevant),
            "heatmap": _heatmap(episode, reference, post_rule, post_relevant),
        },
    }
    replacement = next(
        (
            operation
            for operation in transition["operations"]
            if operation["type"] == "field_replacement"
        ),
        {},
    )
    reversal = next(
        (
            operation
            for operation in transition["operations"]
            if operation["type"] == "direction_reversal"
        ),
        {},
    )
    target = reversal.get("target_field")
    viewer_event = {
        **deepcopy(transition),
        "dynamic_type": transition["type"],
        "outgoing_field": replacement.get("outgoing_field"),
        "incoming_field": replacement.get("incoming_field"),
        "target_field": target,
        "pre_direction": pre_rule["field_directions"].get(target),
        "post_direction": post_rule["field_directions"].get(target),
        "operations": [
            {**operation, "operation_type": operation["type"]}
            for operation in transition["operations"]
        ],
    }
    return {
        "episode_id": episode["episode_id"],
        "schema_version": episode["schema_version"],
        "setting": "sequential_dynamic",
        "taxonomy": {
            "rule_family": episode["evaluation"]["rule_family"],
            "dynamic_type": transition["type"],
        },
        "scenario": {
            "domain": episode["provenance"]["domain"],
            "template_id": episode["provenance"]["template_id"],
            "agent_role": episode["agent_task"]["agent_role"],
            "actions": actions,
        },
        "interaction_spec": interaction_spec,
        "fields": fields,
        "relevant_fields": sorted(set(pre_relevant + post_relevant)),
        "distractor_fields": sorted(
            set(fields) - set(pre_relevant) - set(post_relevant)
        ),
        "constraints": (
            episode["agent_task"].get("feasibility_constraints")
        )
        or [],
        "mechanism": _mechanism_summary(pre_rule),
        "mechanism_raw": pre_rule,
        "mechanism_views": mechanism_views,
        "dynamic": {
            "event": viewer_event,
            "pre_dynamic": {
                "relevant_fields": pre_relevant,
                "mechanism": _mechanism_summary(pre_rule),
                "mechanism_raw": pre_rule,
            },
            "post_dynamic": {
                "relevant_fields": post_relevant,
                "mechanism": _mechanism_summary(post_rule),
                "mechanism_raw": post_rule,
            },
            "population_change": transition["population_reference"],
        },
        "reference_values": reference,
        "system_prompt": system_prompt,
        "cases": cases,
        "post_branches": {
            name: deepcopy(branch["sampling_design"])
            for name, branch in episode["post_branches"].items()
        },
        "field_curves": mechanism_views["pre_change"]["field_curves"],
        "heatmap": mechanism_views["pre_change"]["heatmap"],
        "generation_diagnostics": {},
    }


def _resolve_probe_values(
    probe: Any, contexts: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if not isinstance(probe, dict):
        return None
    if isinstance(probe.get("values"), dict):
        return dict(probe["values"])
    context_id = str(probe.get("base_context_id") or "")
    if context_id not in contexts or not isinstance(probe.get("changes"), dict):
        return None
    values = dict(contexts[context_id]["raw_values"])
    values.update(probe["changes"])
    return values


def _is_unanswered_recovery(artifact: dict[str, Any]) -> bool:
    """Identify legacy artifacts where exhausted repairs became unanswered cases.

    Older runtime versions called this terminal outcome
    ``completed_with_failures`` even though the cases were deliberately scored
    as unanswered with utility 0.  It is safe to normalize only this explicit
    reason; provider/runtime failures retain their failed status.
    """

    termination = artifact.get("termination") or {}
    return (
        termination.get("reason") == "invalid_action_batch_after_repair"
        and termination.get("status") in {"completed", "completed_with_failures"}
    )


def _normalized_completion(
    artifact: dict[str, Any],
) -> tuple[str, bool | None, dict[str, Any] | None]:
    """Return viewer-facing completion fields with legacy unanswered fixed."""

    termination = artifact.get("termination")
    if _is_unanswered_recovery(artifact):
        normalized_termination = (
            dict(termination) if isinstance(termination, dict) else None
        )
        if normalized_termination is not None:
            normalized_termination["status"] = "completed"
        return "completed", True, normalized_termination

    # Branched run envelopes from the old runtime inherited a failure status
    # from branches that were only unanswered.  Recompute the envelope from
    # its branches while leaving any genuine provider failure untouched.
    branches = artifact.get("branches")
    if isinstance(branches, dict) and branches:
        branch_completions = [
            _normalized_completion(branch) for branch in branches.values()
            if isinstance(branch, dict)
        ]
        if branch_completions and all(
            completion[1] is True for completion in branch_completions
        ):
            return "completed", True, termination if isinstance(termination, dict) else None

    status = artifact.get("run_status")
    success = artifact.get("run_success")
    if status is None:
        status = "completed" if success else "failed"
    return str(status), success if isinstance(success, bool) else None, (
        termination if isinstance(termination, dict) else None
    )


def build_run_payload(
    artifact: dict[str, Any], episode_payload: dict[str, Any], provider_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dynamic = episode_payload["dynamic"]
    fields = episode_payload["fields"]
    cases = {item["context_id"]: item for item in episode_payload["cases"]}
    contexts = dict(cases)
    probes = []
    probe_attempt = 0
    trace = artifact.get("tool_trace") or []
    private_probes = {
        int(item.get("interaction") or 0): item
        for item in artifact.get("private_history") or []
        if item.get("type") == "probe_batch"
    }
    condition = str(artifact.get("condition") or "dynamic")
    for event in trace:
        response = event.get("response") or {}
        if response.get("tool") != "probe_batch":
            continue
        event_result = event.get("result") or {}
        if event.get("valid") is False or not isinstance(
            event_result.get("results"), list
        ):
            continue
        probe_attempt += 1
        private_probe = private_probes.get(int(event.get("interaction") or 0), {})
        batch_index = int(
            private_probe.get("batch_number", probe_attempt - 1)
        )
        before_action_batch = batch_index + 1
        probe_round = before_action_batch
        change_after_batch = int(dynamic["event"]["after_action_batch"])
        rule_phase = (
            "post_change"
            if condition == "dynamic" and before_action_batch > change_after_batch
            else "pre_change"
        )
        arguments = response.get("arguments") or {}
        submitted = arguments.get("probes") or []
        results = event_result["results"]
        for index, probe in enumerate(submitted):
            values = _resolve_probe_values(probe, contexts)
            result = results[index] if index < len(results) else {}
            if values is None:
                continue
            values = round_field_values(values, fields)
            mechanism = dynamic[
                "post_dynamic" if rule_phase == "post_change" else "pre_dynamic"
            ]["mechanism_raw"]
            computed = _point(mechanism, values)
            returned = dict(result.get("utilities") or {})
            probe_payload = {
                "context_id": result.get("context_id") or f"attempt_{event.get('interaction')}_{index + 1}",
                "interaction": event.get("interaction"),
                "probe_round": probe_round,
                "before_action_batch": before_action_batch,
                "rule_phase": rule_phase,
                "valid": bool(result),
                "mode": result.get("probe_mode") or ("existing_context" if probe.get("base_context_id") else "new_context"),
                "returned_utilities": [returned.get(action["action_id"]) for action in episode_payload["scenario"]["actions"]],
                "raw_values": values,
                "normalized_values": {field: round(_normalize(values[field], fields[field]), 6) for field in fields},
                **computed,
            }
            if probe.get("base_context_id"):
                probe_payload["base_context_id"] = probe["base_context_id"]
                probe_payload["changes"] = probe.get("changes") or {}
            probes.append(probe_payload)
            requested_base_id = str(probe.get("base_context_id") or "")
            probe_context_id = str(result.get("context_id") or "")
            if probe_context_id and probe_context_id != requested_base_id:
                contexts[probe_context_id] = {"raw_values": dict(values)}
    action_results = []
    for result in artifact.get("action_results") or []:
        case = cases.get(result.get("context_id"))
        batch_number = int(result.get("batch_number") or 0)
        change_after_batch = int(dynamic["event"]["after_action_batch"])
        result_phase = (
            "post_change"
            if condition == "dynamic" and batch_number > change_after_batch
            else "pre_change"
        )
        mechanism = dynamic[
            "post_dynamic" if result_phase == "post_change" else "pre_dynamic"
        ]["mechanism_raw"]
        computed = (
            _point(mechanism, case["raw_values"])
            if case and case.get("raw_values")
            else {}
        )
        oracle = result.get("oracle") or {}
        oracle_action_id = oracle.get("optimal_action_id")
        action_results.append({
            **result,
            "selected_action_index": (
                next(
                    (
                        index
                        for index, action in enumerate(
                            episode_payload["scenario"]["actions"]
                        )
                        if action["action_id"] == result.get("action_id")
                    ),
                    None,
                )
            ),
            "oracle_action_index": (
                next(
                    (
                        index
                        for index, action in enumerate(
                            episode_payload["scenario"]["actions"]
                        )
                        if action["action_id"] == oracle_action_id
                    ),
                    result.get("oracle_action_index"),
                )
            ),
            "score": computed.get("score"),
            "raw_values": case.get("raw_values") if case else None,
        })

    enriched_trace = []
    for event in trace:
        provider = provider_records.get(str(event.get("request_id") or ""))
        enriched_trace.append({**event, "provider_trace": provider})
    metrics = (
        v3_episode_metrics(artifact)
        if artifact.get("trace_schema_version")
        in SEQUENTIAL_TRACE_SCHEMA_VERSIONS
        else {}
    )
    metrics["probes"] = (
        (metrics.get("probe_metrics") or {}).get("cases")
        if "probe_metrics" in metrics
        else metrics.get("probes")
    )
    def public_conversation(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        output = dict(value)
        output["events"] = [
            {key: item for key, item in event.items() if key != "provider_continuation"}
            for event in value.get("events") or []
            if isinstance(event, dict)
        ]
        return output

    conversation = public_conversation(artifact.get("conversation"))
    agent_visible_history = public_conversation(artifact.get("agent_visible_history"))
    run_status, run_success, termination = _normalized_completion(artifact)
    initial_user_message = None
    if isinstance(conversation, dict):
        session_payload = dict(conversation)
        session_payload["history_policy"] = LEGACY_HISTORY_POLICY_NAMES.get(
            str(conversation.get("history_policy") or ""),
            conversation.get("history_policy"),
        )
        session = ConversationSession.from_dict(session_payload)
        if session.events:
            initial_user_message = environment_message_text(
                session, session.events[0], initial=True
            )
    return {
        "episode_id": artifact.get("episode_id"),
        "setting_id": artifact.get("setting_id") or artifact.get("sampling_branch"),
        "sampling_branch": artifact.get("sampling_branch"),
        "enable_probe_skip": artifact.get("enable_probe_skip"),
        "least_probe_count": artifact.get("least_probe_count"),
        "history_policy": artifact.get("history_policy") or "legacy_eval_case_given",
        "history_policy_version": artifact.get("history_policy_version"),
        "history_view_status": (
            "available"
            if agent_visible_history is not None
            else (
                "eval_case_given_only"
                if artifact.get("history_policy") in {
                    "eval_case_given",
                    "full_history",
                }
                and artifact.get("history_policy_version")
                else "legacy_eval_case_given_only"
            )
        ),
        "condition": condition,
        "run_status": run_status,
        "run_success": run_success,
        "termination": termination,
        "error": artifact.get("error"),
        "metrics": metrics,
        "probes": probes,
        "action_results": action_results,
        "tool_trace": enriched_trace,
        "conversation": conversation,
        "agent_visible_history": agent_visible_history,
        "initial_user_message": initial_user_message,
        "relevant_fields_assessments": artifact.get(
            "relevant_fields_assessments"
        ) or {},
        "private_history": artifact.get("private_history") or [],
        "private_state": artifact.get("private_state") or {},
        "trace_schema_version": artifact.get("trace_schema_version") or "unknown_trace",
        "episode_schema_version": artifact.get("episode_schema_version"),
    }


def _light_trajectory_episode_ids(episodes: list[dict[str, Any]]) -> set[str]:
    """Pick one episode per (domain, rule family) cell for trajectory views."""

    selected: dict[tuple[str, str], str] = {}
    for episode in episodes:
        cell = (
            str((episode.get("provenance") or {}).get("domain") or ""),
            str((episode.get("evaluation") or {}).get("rule_family") or ""),
        )
        if cell not in selected:
            selected[cell] = str(episode["episode_id"])
    return set(selected.values())


def _load_trajectory_artifacts(
    run_dir: Path, episode_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Load only the requested episode artifacts (filenames are episode IDs)."""

    artifacts: dict[str, dict[str, Any]] = {}
    for episode_id in sorted(episode_ids):
        artifact = _load_json(run_dir / f"{episode_id}.json")
        if str(artifact.get("episode_id") or "") != episode_id:
            continue
        if (
            artifact.get("trace_schema_version")
            not in SUPPORTED_BRANCHED_TRACE_SCHEMA_VERSIONS
        ):
            continue
        artifacts[episode_id] = artifact
    return artifacts


def _provider_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "provider_trace.jsonl"
    if not path.exists():
        return {}
    output = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        request_id = str((record.get("trace_context") or {}).get("request_id") or "")
        if request_id:
            output[request_id] = record
    return output


def generate(
    dataset: Path,
    output: Path,
    run_dirs: list[Path],
    *,
    replace: bool = False,
    append: bool = False,
    light: bool = False,
    example_episode_ids: set[str] | None = None,
) -> dict[str, Any]:
    if replace and append:
        raise ValueError("replace and append modes are mutually exclusive")
    if light and example_episode_ids is not None:
        raise ValueError("--light and --example-episode are mutually exclusive")
    episodes, dataset_sha256 = _load_validated_dataset(dataset)
    if example_episode_ids is not None:
        if not example_episode_ids:
            raise ValueError("at least one example episode is required")
        missing = example_episode_ids - {episode["episode_id"] for episode in episodes}
        if missing:
            raise ValueError(f"example episodes are absent from dataset: {sorted(missing)}")
    published_episodes = (
        [episode for episode in episodes if episode["episode_id"] in example_episode_ids]
        if example_episode_ids is not None else episodes
    )
    interaction_policy = _runtime_interaction_policy()
    _log(f"building episode payloads for {len(published_episodes)} episodes")
    payloads = {
        episode["episode_id"]: build_episode_payload(
            episode, interaction_policy
        )
        for episode in published_episodes
    }
    trajectory_episode_ids = (
        _light_trajectory_episode_ids(episodes) if light else set(payloads)
    )
    existing_manifest: dict[str, Any] = {}
    if output.exists() and any(output.iterdir()):
        if replace:
            shutil.rmtree(output)
        elif append:
            existing_manifest = _load_json(output / "manifest.json")
            if not existing_manifest:
                raise ValueError(
                    f"viewer output has no valid manifest: {output}; "
                    "use --replace"
                )
            if existing_manifest.get("dataset_sha256") != dataset_sha256:
                raise ValueError(
                    "viewer append dataset differs from the existing viewer release; use --replace"
                )
            episodes_dir = output / "episodes"
            if episodes_dir.exists():
                shutil.rmtree(episodes_dir)
            for path in (output / "runs").glob("*/*.json"):
                if path.stem not in payloads:
                    path.unlink()
        else:
            raise ValueError(
                f"viewer output is not empty: {output}; "
                "pass --replace or --append"
            )
    elif output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for episode_id, payload in payloads.items():
        _write_json(output / "episodes" / f"{episode_id}.json", payload)

    current_episode_ids = set(payloads)
    run_summaries = [
        {
            **summary,
            "episode_ids": [
                episode_id
                for episode_id in summary.get("episode_ids") or []
                if episode_id in current_episode_ids
            ],
        }
        for summary in existing_manifest.get("runs") or []
        if set(summary.get("episode_ids") or []) & current_episode_ids
    ]
    retained_run_ids = {
        str(summary["run_id"]) for summary in run_summaries
    }
    availability: dict[str, list[str]] = {episode_id: [] for episode_id in payloads}
    for episode in existing_manifest.get("episodes") or []:
        episode_id = str(episode.get("episode_id") or "")
        if episode_id in availability:
            availability[episode_id] = [
                run_id
                for run_id in episode.get("runs") or []
                if run_id in retained_run_ids
            ]
    episodes_by_id = {str(ep["episode_id"]): ep for ep in episodes}
    protocol = episodes[0]["agent_task"]["interaction_protocol"]
    batch_count = int(protocol["action_batch_count"])
    batch_size = int(protocol["action_batch_size"])
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        run_manifest = _load_json(run_dir / "manifest.json")
        simulation_scenario = run_manifest.get("simulation_scenario")
        simulation_description = run_manifest.get("simulation_description")
        run_dataset_sha = (
            (run_manifest.get("run_identity") or {}).get("dataset_sha256")
            or (run_manifest.get("dataset") or {}).get("sha256")
        )
        if run_dataset_sha != dataset_sha256:
            raise ValueError(
                f"benchmark run does not belong to the viewer dataset: {run_dir}"
            )
        run_id = run_dir.name
        run_summaries = [
            summary
            for summary in run_summaries
            if summary["run_id"] != run_id
        ]
        for episode_runs in availability.values():
            if run_id in episode_runs:
                episode_runs.remove(run_id)
        run_output = output / "runs" / run_id
        if run_output.exists():
            shutil.rmtree(run_output)
        label, model_key = _run_label(run_dir.name)
        provider = _provider_records(run_dir)
        episode_ids = []
        run_summary = _load_run_summary_cache(run_dir, dataset_sha256)
        if run_summary is not None:
            _log(
                f"{run_id}: reusing {RUN_SUMMARY_CACHE_NAME}, "
                f"loading {len(trajectory_episode_ids)} trajectory artifacts"
            )
            artifacts_by_episode = _load_trajectory_artifacts(
                run_dir, trajectory_episode_ids
            )
        else:
            artifacts_by_branch, all_artifacts_by_episode = _load_run_artifacts(
                run_dir, set(episodes_by_id)
            )
            artifacts_by_episode = {
                episode_id: artifact
                for episode_id, artifact in all_artifacts_by_episode.items()
                if episode_id in trajectory_episode_ids
            }
            run_summary = _build_run_summary(
                run_dir,
                run_manifest,
                artifacts_by_branch,
                sorted(all_artifacts_by_episode),
                episodes_by_id,
                batch_count,
                batch_size,
            )
            _write_run_summary_cache(run_dir, dataset_sha256, run_summary)
            # Light publications retain only sampled trajectories once the
            # full summary is cached, including while loading the next run.
            del artifacts_by_branch, all_artifacts_by_episode
        for episode_id in sorted(artifacts_by_episode):
            branched_artifact = artifacts_by_episode[episode_id]
            branches = branched_artifact.get("branches") or {}
            branch_payloads: dict[str, dict[str, Any]] = {}
            for name, branch in branches.items():
                canonical_name = _canonical_sampling_branch(str(name))
                branch_payload = build_run_payload(
                    branch, payloads[episode_id], provider
                )
                branch_payload["source_sampling_branch"] = branch_payload.get(
                    "sampling_branch"
                )
                # A setting may reuse a source sampling timeline while keeping
                # its own policy identity. Publish the setting key and retain
                # the source timeline for case filtering and validation.
                branch_payload["setting_id"] = canonical_name
                branch_payload["sampling_branch"] = canonical_name
                # If a future artifact contains both spellings, prefer the
                # canonical source over its legacy alias.
                if canonical_name not in branch_payloads or name == canonical_name:
                    branch_payloads[canonical_name] = branch_payload
            static_branch = branches.get("static_iid")
            if static_branch is not None:
                for name, branch_payload in branch_payloads.items():
                    raw_branch = branches.get(name)
                    if raw_branch is None and name == "flip_50_eval_case_given":
                        raw_branch = branches.get("flip_50_full_history")
                    if not raw_branch or raw_branch.get("condition") != "dynamic":
                        continue
                    comparison = first_learn_rule_comparison(
                        [raw_branch], [static_branch]
                    )
                    if not comparison["comparisons"]:
                        continue
                    branch_payload["metrics"].update({
                        "first_learn_accuracy": comparison["dynamic_post_rule_accuracy"],
                        "first_learn_rule_accuracy": comparison["static_first_learn_accuracy"],
                        "first_learn_static_accuracy": comparison["static_first_learn_accuracy"],
                        "first_learn_accuracy_delta": comparison["accuracy_delta"],
                        "first_learn_comparison_batch_count": comparison["comparisons"][0]["comparison_batch_count"],
                    })
            for branch_payload in branch_payloads.values():
                branch_payload.update(
                    {
                        "run_id": run_dir.name,
                        "run_label": label,
                        "model_key": model_key,
                    }
                )
            run_status, run_success, _ = _normalized_completion(branched_artifact)
            run_payload = {
                "episode_id": episode_id,
                "run_id": run_dir.name,
                "run_label": label,
                "model_key": model_key,
                "simulation_scenario": simulation_scenario,
                "simulation_description": simulation_description,
                "fork_after_batch": int(
                    branched_artifact.get("fork_after_batch")
                    or payloads[episode_id]["dynamic"]["event"]["after_action_batch"]
                ),
                "shared_prefix_interactions": len(
                    ((branched_artifact.get("shared_prefix") or {}).get("tool_trace") or [])
                ),
                "run_status": run_status,
                "run_success": run_success,
                "trace_schema_version": branched_artifact.get(
                    "trace_schema_version"
                ),
                "episode_schema_version": branched_artifact.get(
                    "episode_schema_version"
                ),
                "branches": branch_payloads,
            }
            _write_json(output / "runs" / run_dir.name / f"{episode_id}.json", run_payload)
            episode_ids.append(episode_id)
            availability[episode_id].append(run_id)
        run_summaries.append({**run_summary, "episode_ids": episode_ids})
        _log(f"{run_id}: wrote {len(episode_ids)} trajectory payloads")

    family_counts = Counter(
        item["evaluation"]["rule_family"] for item in episodes
    )
    dynamic_type_counts = Counter(
        item["evaluation"]["transition"]["type"] for item in episodes
    )
    change_point_counts = Counter(
        int(item["evaluation"]["transition"]["after_action_batch"])
        for item in episodes
    )
    protocol = episodes[0]["agent_task"]["interaction_protocol"]
    manifest = {
        "title": f"{RELEASE_NAME} Research Viewer",
        "release": {
            "name": RELEASE_NAME,
            "slug": RELEASE_SLUG,
            "version": RELEASE_VERSION,
        },
        "benchmark_config": {
            "schema_version": episodes[0]["schema_version"],
            "action_batch_count": protocol["action_batch_count"],
            "action_batch_size": protocol["action_batch_size"],
            "probe_budget": protocol["max_probe_budget"],
            **interaction_policy,
        },
        "dataset": str(
            dataset.relative_to(REPO_ROOT)
            if dataset.is_relative_to(REPO_ROOT)
            else dataset
        ),
        "dataset_sha256": dataset_sha256,
        "episode_count": len(episodes),
        "family_counts": dict(family_counts),
        "dynamic_type_counts": dict(dynamic_type_counts),
        "change_point_counts": dict(change_point_counts),
        "runs": run_summaries,
        "episodes": [{
            "episode_id": item["episode_id"], "domain": item["provenance"]["domain"],
            "rule_family": item["evaluation"]["rule_family"],
            "dynamic_type": item["evaluation"]["transition"]["type"],
            "sampling_branches": deepcopy(
                item["evaluation"]["transition"]["sampling_design"][
                    "available_post_branches"
                ]
            ),
            "after_action_batch": item["evaluation"]["transition"]["after_action_batch"],
            "raw_field_count": len(item["agent_task"]["fields"]),
            "relevant_field_count": 3,
            "distractor_field_count": 2,
            "runs": availability[item["episode_id"]],
        } for item in published_episodes],
    }
    if light or example_episode_ids is not None:
        manifest["trajectory_sample"] = {
            "mode": "selected_examples" if example_episode_ids is not None else "domain_rule_family_cell",
            "episode_ids": [
                item["episode_id"]
                for item in published_episodes
                if item["episode_id"] in trajectory_episode_ids
            ],
        }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AdaExploreBench research-viewer data")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="benchmark run directory to publish; repeat for multiple runs",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--replace",
        action="store_true",
        help="delete the existing viewer output before rebuilding it",
    )
    mode.add_argument(
        "--append",
        action="store_true",
        help="retain existing runs and add or replace the listed run directories",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help=(
            "publish full overview aggregates but per-episode trajectories only "
            "for one episode per (domain, rule family) cell"
        ),
    )
    parser.add_argument(
        "--example-episode",
        action="append",
        help="publish only this episode as an example; repeat to select more while keeping full run summaries",
    )
    parser.add_argument(
        "--precompute-summary",
        action="store_true",
        help=(
            "compute the per-run viewer summary cache (viewer_summary.json) "
            "for each --run-dir and exit; no viewer output is written"
        ),
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="write deterministic .json.gz mirrors for static publication",
    )
    args = parser.parse_args()
    if args.precompute_summary:
        precompute_summaries(args.dataset, args.run_dir)
        return
    manifest = generate(
        args.dataset.resolve(),
        args.output.resolve(),
        [path.resolve() for path in args.run_dir],
        replace=args.replace,
        append=args.append,
        light=args.light,
        example_episode_ids=set(args.example_episode) if args.example_episode else None,
    )
    compressed_count = write_gzip_mirror(args.output.resolve()) if args.gzip else 0
    message = (
        f"Generated viewer data for {manifest['episode_count']} episodes "
        f"and {len(manifest['runs'])} runs"
    )
    if compressed_count:
        message += f" ({compressed_count} gzip payloads)"
    print(message)


if __name__ == "__main__":
    main()
