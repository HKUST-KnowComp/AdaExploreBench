"""Deterministic balanced selection from final LLM review scores."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .contracts import validate_episode
from .final_review import DIMENSIONS, SCHEMA_VERSION as REVIEW_SCHEMA_VERSION
from .final_review import dataset_sha256


SCHEMA_VERSION = "adaexplorebench_dataset_selection_v1"


@dataclass(frozen=True)
class DatasetSelectionConfig:
    path: Path
    target_episode_count: int
    dimension_score_cutoff: float
    output_filename: str
    report_filename: str


@dataclass(frozen=True)
class SelectionResult:
    episodes: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _plain_filename(value: Any, label: str) -> str:
    filename = str(value)
    if not filename or Path(filename).name != filename:
        raise ValueError(f"{label} must be a plain filename")
    return filename


def load_selection_config(path: Path) -> DatasetSelectionConfig:
    resolved = path.resolve()
    value = yaml.safe_load(resolved.read_text()) or {}
    expected = {
        "target_episode_count",
        "dimension_score_cutoff",
        "output_filename",
        "report_filename",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"dataset selection config must define exactly {sorted(expected)}")
    target = value["target_episode_count"]
    cutoff = value["dimension_score_cutoff"]
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("target_episode_count must be a positive integer")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise ValueError("dimension_score_cutoff must be numeric")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or not 0 <= cutoff < 5:
        raise ValueError("dimension_score_cutoff must lie in [0, 5)")
    return DatasetSelectionConfig(
        path=resolved,
        target_episode_count=target,
        dimension_score_cutoff=cutoff,
        output_filename=_plain_filename(value["output_filename"], "output_filename"),
        report_filename=_plain_filename(value["report_filename"], "report_filename"),
    )


def _review_index(review: dict[str, Any]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("selection requires a compatible final review")
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for group in review.get("groups") or []:
        key = (str(group["domain"]), str(group["rule_family"]))
        if key in indexed:
            raise ValueError("final review contains a duplicate domain/rule group")
        rows: dict[str, dict[str, Any]] = {}
        for row in group.get("episodes") or []:
            episode_id = str(row["episode_id"])
            if episode_id in rows:
                raise ValueError("final review contains a duplicate episode")
            scores = row.get("scores") or {}
            if set(scores) != set(DIMENSIONS):
                raise ValueError("final review row has invalid score dimensions")
            rows[episode_id] = row
        indexed[key] = rows
    if not indexed:
        raise ValueError("final review contains no groups")
    return indexed


def select_episodes(
    episodes: list[dict[str, Any]],
    review: dict[str, Any],
    config: DatasetSelectionConfig,
) -> SelectionResult:
    if not episodes:
        raise ValueError("dataset selection requires at least one episode")
    for episode in episodes:
        validate_episode(episode, "dynamic")
    source_sha256 = dataset_sha256(episodes)
    reviewed_dataset = review.get("dataset") or {}
    if reviewed_dataset.get("sha256") != source_sha256:
        raise ValueError("final review does not belong to the input dataset")
    rows_by_group = _review_index(review)
    episodes_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if episode_id in seen_ids:
            raise ValueError("input dataset contains duplicate episode IDs")
        seen_ids.add(episode_id)
        key = (
            str(episode["provenance"]["domain"]),
            str(episode["evaluation"]["rule_family"]),
        )
        episodes_by_group.setdefault(key, []).append(episode)
    if set(episodes_by_group) != set(rows_by_group):
        raise ValueError("dataset and final review domain/rule groups differ")
    group_count = len(episodes_by_group)
    if config.target_episode_count % group_count:
        raise ValueError(
            "target_episode_count must be divisible by the observed domain/rule group count"
        )
    requested_per_group = config.target_episode_count // group_count
    eligible_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rejected_by_group: dict[tuple[str, str], list[str]] = {}
    for key, group_episodes in episodes_by_group.items():
        review_rows = rows_by_group[key]
        group_ids = {str(episode["episode_id"]) for episode in group_episodes}
        if set(review_rows) != group_ids:
            raise ValueError("dataset and final review episode IDs differ within a group")
        eligible = []
        rejected = []
        for episode_id in group_ids:
            row = review_rows[episode_id]
            if all(
                float(row["scores"][dimension]["score"])
                > config.dimension_score_cutoff
                for dimension in DIMENSIONS
            ):
                eligible.append(row)
            else:
                rejected.append(episode_id)
        eligible.sort(key=lambda row: (-float(row["overall_score"]), str(row["episode_id"])))
        eligible_by_group[key] = eligible
        rejected_by_group[key] = sorted(rejected)
    effective_per_group = min(
        requested_per_group,
        *(len(rows) for rows in eligible_by_group.values()),
    )
    if effective_per_group <= 0:
        raise ValueError("at least one domain/rule group has no eligible episodes")
    selected_ids: set[str] = set()
    report_groups = []
    for key in sorted(eligible_by_group):
        eligible = eligible_by_group[key]
        chosen = eligible[:effective_per_group]
        chosen_ids = [str(row["episode_id"]) for row in chosen]
        selected_ids.update(chosen_ids)
        report_groups.append(
            {
                "domain": key[0],
                "rule_family": key[1],
                "input_count": len(episodes_by_group[key]),
                "eligible_count": len(eligible),
                "selected_count": len(chosen_ids),
                "threshold_rejected_episode_ids": rejected_by_group[key],
                "rank_omitted_episode_ids": [
                    str(row["episode_id"]) for row in eligible[effective_per_group:]
                ],
                "selected_episode_ids": chosen_ids,
            }
        )
    selected = tuple(episode for episode in episodes if str(episode["episode_id"]) in selected_ids)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": {
            "sha256": source_sha256,
            "episode_count": len(episodes),
        },
        "selected_dataset": {
            "sha256": dataset_sha256(selected),
            "episode_count": len(selected),
        },
        "policy": {
            "target_episode_count": config.target_episode_count,
            "dimension_score_cutoff": config.dimension_score_cutoff,
            "comparison": "score > cutoff for every dimension",
            "group_count": group_count,
            "requested_per_group": requested_per_group,
            "effective_per_group": effective_per_group,
        },
        "groups": report_groups,
    }
    return SelectionResult(selected, report)


def serialized_selection_report(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + "\n"


def selection_manifest_entry(result: dict[str, Any], filename: str) -> dict[str, Any]:
    content = serialized_selection_report(result).encode()
    return {
        "file": filename,
        "schema_version": SCHEMA_VERSION,
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_dataset_sha256": result["source_dataset"]["sha256"],
        "selected_dataset_sha256": result["selected_dataset"]["sha256"],
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def published_dataset_manifest_entry(
    result: SelectionResult, filename: str
) -> dict[str, Any]:
    return {
        "file": filename,
        "mode": "dynamic",
        "sha256": result.report["selected_dataset"]["sha256"],
        "episode_count": len(result.episodes),
        "domain_rule_counts": {
            f"{group['domain']}/{group['rule_family']}": group["selected_count"]
            for group in result.report["groups"]
        },
    }


def serialized_selected_episodes(result: SelectionResult) -> str:
    return "".join(
        json.dumps(episode, ensure_ascii=False) + "\n"
        for episode in result.episodes
    )


def write_selection_outputs(
    result: SelectionResult,
    config: DatasetSelectionConfig,
    output_path: Path,
    report_path: Path,
    manifest_path: Path,
) -> None:
    if len({output_path.parent.resolve(), report_path.parent.resolve(), manifest_path.parent.resolve()}) != 1:
        raise ValueError("published dataset, selection report, and manifest must share one directory")
    if output_path.name == str((json.loads(manifest_path.read_text()).get("dataset") or {}).get("file")):
        raise ValueError("published selection must not overwrite the generated source dataset")
    dataset_content = serialized_selected_episodes(result).encode()
    report_content = serialized_selection_report(result.report).encode()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("dataset") or {}).get("sha256") != result.report["source_dataset"]["sha256"]:
        raise ValueError("manifest does not describe the selected source dataset")
    manifest["published_dataset"] = published_dataset_manifest_entry(result, output_path.name)
    manifest["dataset_selection"] = selection_manifest_entry(result.report, config.report_filename)
    _atomic_write(output_path, dataset_content)
    _atomic_write(report_path, report_content)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
