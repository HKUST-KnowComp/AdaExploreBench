"""Incremental reports for static dataset publication."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable

from protocol.mechanism import best_action, hydrate_rule
from protocol.provenance import DATASET_MANIFEST_SCHEMA_VERSION, code_identity
from ..config import DatasetGenerationConfig
from ..contracts import STATIC_EPISODE_SCHEMA_VERSION


def _report_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(item) for key, item in value.items() if not key.startswith("private_")}


def build_static_artifacts(episodes: Iterable[dict[str, Any]], diagnostics: Iterable[dict[str, Any]], config: DatasetGenerationConfig, dataset_sha256: str, *, previous_report: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    diagnostics_by_id = {str(item["episode_id"]): item for item in diagnostics}
    previous = {str(item["episode_id"]): item for item in (previous_report or {}).get("cells", [])}
    cells, plan_rows = [], []
    action_counts: Counter[str] = Counter()
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        rule = hydrate_rule(episode["evaluation"]["rule"], episode["agent_task"]["fields"])
        counts = Counter(f"action_{best_action(rule, case['raw_values'])}" for case in episode["cases"])
        action_counts.update(counts)
        cell = previous.get(episode_id) or {"episode_id": episode_id, "domain": episode["provenance"]["domain"], "template_id": episode["provenance"]["template_id"], "rule_family": episode["evaluation"]["rule_family"], "published_action_counts": dict(sorted(counts.items())), "generation_diagnostics": _report_diagnostics(diagnostics_by_id.get(episode_id) or {})}
        cells.append(cell)
        plan_rows.append({"episode_id": episode_id, "seed": episode["seed"], **deepcopy(episode["provenance"]), "rule_family": episode["evaluation"]["rule_family"]})
    report = {"schema_version": "adaexplorebench_static_balance_report_v4", "summary": {"episode_count": len(cells), "action_case_count": sum(action_counts.values()), "published_action_counts": dict(sorted(action_counts.items()))}, "cells": cells}
    manifest = {"manifest_schema_version": DATASET_MANIFEST_SCHEMA_VERSION, "schema_version": STATIC_EPISODE_SCHEMA_VERSION, "release": {"name": config.dataset.name, "slug": config.dataset.slug, "version": config.dataset.version}, "dataset": {"id": config.dataset.dataset_id, "version": config.dataset.version, "mode": "static", "file": "static_episodes.jsonl", "sha256": dataset_sha256}, "code": code_identity(), "episode_count": len(cells), "action_batch_count": config.episode_layout.action_batch_count * len(cells), "action_case_count": config.episode_layout.action_case_count * len(cells), "dynamic_enabled": False}
    return {"resolved_episode_plan": {"episodes": plan_rows}, "balance_report": report, "manifest": manifest}
