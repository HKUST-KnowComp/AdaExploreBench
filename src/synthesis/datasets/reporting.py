"""Incremental reports for dynamic dataset publication."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from statistics import mean
from typing import Any, Iterable

from protocol.provenance import DATASET_MANIFEST_SCHEMA_VERSION, code_identity
from .config import DatasetGenerationConfig
from .contracts import DYNAMIC_EPISODE_SCHEMA_VERSION


def _report_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(item) for key, item in value.items() if not key.startswith("private_")}


def build_dynamic_artifacts(episodes: Iterable[dict[str, Any]], diagnostics: Iterable[dict[str, Any]], config: DatasetGenerationConfig, dataset_sha256: str, *, previous_report: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    if config.dynamic is None:
        raise ValueError("dynamic reporting requires dynamic config")
    diagnostics_by_id = {str(item["episode_id"]): item for item in diagnostics}
    previous = {str(item["episode_id"]): item for item in (previous_report or {}).get("cells", [])}
    cells, plan_rows = [], []
    domain_rule: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    change_points: Counter[int] = Counter()
    types: set[str] = set()
    for episode in episodes:
        transition = episode["evaluation"]["transition"]
        episode_id = str(episode["episode_id"])
        diagnostic = diagnostics_by_id.get(episode_id)
        cell = previous.get(episode_id)
        if cell is None:
            cell = {"episode_id": episode_id, "domain": episode["provenance"]["domain"], "template_id": episode["provenance"]["template_id"], "rule_family": episode["evaluation"]["rule_family"], "dynamic_type": transition["type"], "after_action_batch": transition["after_action_batch"], "population_reference": deepcopy(transition["population_reference"]), "generation_diagnostics": _report_diagnostics(diagnostic or {})}
        cells.append(cell)
        domain_rule[f"{cell['domain']}/{cell['rule_family']}"] += 1
        templates[str(cell["template_id"])] += 1
        change_points[int(cell["after_action_batch"])] += 1
        types.add(str(cell["dynamic_type"]))
        core = set(transition["pre_rule"]["relevant_fields"]) | set(transition["post_rule"]["relevant_fields"])
        visible = list(episode["agent_task"]["fields"])
        plan_rows.append({"episode_id": episode_id, "seed": episode["seed"], **deepcopy(episode["provenance"]), "rule_family": cell["rule_family"], "dynamic_type": cell["dynamic_type"], "after_action_batch": cell["after_action_batch"], "sampling_branches": list(episode["post_branches"]), "field_layout": {"semantic_core": [field for field in visible if field in core], "distractors": [field for field in visible if field not in core]}})
    flip_rates = [float(item["population_reference"]["flip_rate"]) for item in cells]
    capacity = {"schema_version": "adaexplorebench_dynamic_capacity_report_v7", "summary": {"cell_count": len(cells), "minimum_flip_rate": min(flip_rates) if flip_rates else None, "maximum_flip_rate": max(flip_rates) if flip_rates else None, "mean_flip_rate": mean(flip_rates) if flip_rates else None}, "cells": cells}
    manifest = {"manifest_schema_version": DATASET_MANIFEST_SCHEMA_VERSION, "schema_version": DYNAMIC_EPISODE_SCHEMA_VERSION, "release": {"name": config.dataset.name, "slug": config.dataset.slug, "version": config.dataset.version}, "dataset": {"id": config.dataset.dataset_id, "version": config.dataset.version, "mode": "dynamic", "file": "dynamic_episodes.jsonl", "sha256": dataset_sha256}, "code": code_identity(), "episode_count": len(cells), "domain_rule_counts": dict(sorted(domain_rule.items())), "template_counts": dict(sorted(templates.items())), "action_batch_count": config.episode_layout.action_batch_count * len(cells), "action_case_count": config.episode_layout.action_case_count * len(cells), "dynamic_enabled": True, "dynamic_event_count": len(cells), "dynamic_types": sorted(types), "dynamic_after_action_batches": dict(change_points)}
    return {"resolved_episode_plan": {"episodes": plan_rows}, "capacity_report": capacity, "manifest": manifest}
