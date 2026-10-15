#!/usr/bin/env python3
"""Extract the diverse 16-episode runtime test set from a full release."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from synthesis.datasets.contracts import validate_episode


RULE_FAMILIES = (
    "additive_aggregation",
    "conjunctive_gating",
    "disjunctive_sufficiency",
    "lexicographic_priority",
)
CHANGE_POINTS = (5, 6, 7)


def _feature_sets(episode: dict[str, Any]) -> tuple[set[str], set[str]]:
    visible = set(episode["agent_task"]["fields"])
    transition = episode["evaluation"]["transition"]
    semantic_core = set(transition["pre_rule"]["relevant_fields"])
    semantic_core.update(transition["post_rule"]["relevant_fields"])
    return visible, semantic_core


def select_episodes(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one episode per domain/rule and one distinct task per domain.

    Among selections with complete task coverage, prefer balanced transition
    points, maximum visible/core field coverage, and a wide source-position
    spread. Source position is the final deterministic tie-breaker.
    """
    indexed = list(enumerate(episodes))
    domains = sorted({item["provenance"]["domain"] for item in episodes})
    if len(domains) != 4:
        raise ValueError(f"expected exactly four domains, observed {domains}")

    observed_families = {
        item["evaluation"]["rule_family"] for item in episodes
    }
    if observed_families != set(RULE_FAMILIES):
        raise ValueError(
            "expected the four runtime rule families; observed "
            f"{sorted(observed_families)}"
        )

    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, episode in indexed:
        key = (
            episode["provenance"]["domain"],
            episode["evaluation"]["rule_family"],
            episode["provenance"]["template_id"],
        )
        grouped.setdefault(key, []).append((index, episode))

    # Each change-point count pattern needs only its strongest within-domain
    # option. Domains use separate field namespaces, so these scores compose.
    domain_options: dict[
        str,
        dict[
            tuple[int, int, int],
            tuple[tuple[Any, ...], tuple[tuple[int, dict[str, Any]], ...]],
        ],
    ] = {}
    for domain in domains:
        templates = sorted(
            {
                item["provenance"]["template_id"]
                for item in episodes
                if item["provenance"]["domain"] == domain
            }
        )
        if len(templates) != len(RULE_FAMILIES):
            raise ValueError(
                f"{domain} must expose exactly four distinct tasks; observed {templates}"
            )

        best_by_change_pattern: dict[
            tuple[int, int, int],
            tuple[tuple[Any, ...], tuple[tuple[int, dict[str, Any]], ...]],
        ] = {}
        for task_assignment in itertools.permutations(templates):
            candidate_pools = [
                grouped[(domain, family, template)]
                for family, template in zip(RULE_FAMILIES, task_assignment)
            ]
            for selection in itertools.product(*candidate_pools):
                change_counts = Counter(
                    episode["evaluation"]["transition"]["after_action_batch"]
                    for _, episode in selection
                )
                change_pattern = tuple(
                    change_counts[point] for point in CHANGE_POINTS
                )
                visible_fields: set[str] = set()
                semantic_fields: set[str] = set()
                positions: list[int] = []
                for position, episode in selection:
                    visible, semantic = _feature_sets(episode)
                    visible_fields.update(visible)
                    semantic_fields.update(semantic)
                    positions.append(position)
                score = (
                    len(visible_fields),
                    len(semantic_fields),
                    max(positions) - min(positions),
                    tuple(-position for position in positions),
                )
                current = best_by_change_pattern.get(change_pattern)
                if current is None or score > current[0]:
                    best_by_change_pattern[change_pattern] = (score, selection)
        domain_options[domain] = best_by_change_pattern

    best_global: tuple[tuple[Any, ...], tuple[dict[str, Any], ...]] | None = None
    option_lists = [list(domain_options[domain].items()) for domain in domains]
    for combined in itertools.product(*option_lists):
        change_patterns = [pattern for pattern, _ in combined]
        selected = tuple(
            episode
            for _, (_, selection) in combined
            for _, episode in selection
        )
        change_totals = tuple(
            sum(pattern[index] for pattern in change_patterns)
            for index in range(len(CHANGE_POINTS))
        )
        visible_fields: set[str] = set()
        semantic_fields: set[str] = set()
        positions: list[int] = []
        position_by_id = {
            episode["episode_id"]: index for index, episode in indexed
        }
        for episode in selected:
            visible, semantic = _feature_sets(episode)
            visible_fields.update(visible)
            semantic_fields.update(semantic)
            positions.append(position_by_id[episode["episode_id"]])
        score = (
            -(max(change_totals) - min(change_totals)),
            sum(count > 0 for count in change_totals),
            len(visible_fields),
            len(semantic_fields),
            max(positions) - min(positions),
            tuple(-position for position in positions),
        )
        if best_global is None or score > best_global[0]:
            best_global = (score, selected)

    if best_global is None:
        raise ValueError("could not construct a complete runtime test set")
    family_order = {family: index for index, family in enumerate(RULE_FAMILIES)}
    return sorted(
        best_global[1],
        key=lambda item: (
            item["provenance"]["domain"],
            family_order[item["evaluation"]["rule_family"]],
        ),
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_testset(source: Path, output_dir: Path, *, replace: bool) -> None:
    source = source.resolve()
    output_dir = output_dir.resolve()
    output_dataset = output_dir / "dynamic_episodes.jsonl"
    output_manifest = output_dir / "manifest.json"
    if not replace and (output_dataset.exists() or output_manifest.exists()):
        raise FileExistsError(
            f"output already exists beneath {output_dir}; pass --replace to regenerate"
        )

    source_payload = source.read_bytes()
    source_lines = [line for line in source_payload.splitlines() if line.strip()]
    episodes = [json.loads(line) for line in source_lines]
    for episode in episodes:
        validate_episode(episode, "dynamic")
    selected = select_episodes(episodes)
    source_line_by_id = {
        episode["episode_id"]: line
        for episode, line in zip(episodes, source_lines)
    }
    output_payload = b"\n".join(
        source_line_by_id[episode["episode_id"]] for episode in selected
    ) + b"\n"
    output_sha256 = hashlib.sha256(output_payload).hexdigest()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()

    source_manifest_path = source.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    claimed_source_sha = source_manifest["dataset"]["sha256"]
    if claimed_source_sha != source_sha256:
        raise ValueError(
            "source manifest hash does not match the source dataset: "
            f"{claimed_source_sha} != {source_sha256}"
        )

    domain_rule_counts = Counter(
        f"{episode['provenance']['domain']}/{episode['evaluation']['rule_family']}"
        for episode in selected
    )
    template_counts = Counter(
        episode["provenance"]["template_id"] for episode in selected
    )
    change_counts = Counter(
        episode["evaluation"]["transition"]["after_action_batch"]
        for episode in selected
    )
    visible_fields: set[str] = set()
    semantic_fields: set[str] = set()
    for episode in selected:
        visible, semantic = _feature_sets(episode)
        visible_fields.update(visible)
        semantic_fields.update(semantic)

    source_version = str(source_manifest["dataset"]["version"])
    manifest = {
        "manifest_schema_version": source_manifest["manifest_schema_version"],
        "schema_version": source_manifest["schema_version"],
        "release": {
            "name": "AdaExploreBench Dynamic runtime test set",
            "slug": "test-16",
            "version": f"{source_version}-test-16",
        },
        "dataset": {
            "id": "test-16",
            "version": f"{source_version}-test-16",
            "mode": "dynamic",
            "file": output_dataset.name,
            "sha256": output_sha256,
            "source_dataset": _relative(source),
            "source_sha256": source_sha256,
        },
        "code": source_manifest.get("code", {}),
        "episode_count": len(selected),
        "domain_rule_counts": dict(sorted(domain_rule_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "action_batch_count": sum(
            episode["agent_task"]["interaction_protocol"]["action_batch_count"]
            for episode in selected
        ),
        "action_case_count": sum(
            episode["agent_task"]["interaction_protocol"]["action_batch_count"]
            * episode["agent_task"]["interaction_protocol"]["action_batch_size"]
            for episode in selected
        ),
        "dynamic_enabled": True,
        "dynamic_event_count": len(selected),
        "dynamic_types": sorted(
            {
                episode["evaluation"]["transition"]["type"]
                for episode in selected
            }
        ),
        "dynamic_after_action_batches": {
            str(point): change_counts[point] for point in CHANGE_POINTS
        },
        "selection": {
            "operation": "subset",
            "source_dataset": _relative(source),
            "source_sha256": source_sha256,
            "criterion": (
                "one episode per domain x rule family; distinct task per cell; "
                "balance change points; maximize visible and semantic-core field diversity"
            ),
            "task_coverage": len(template_counts),
            "task_coverage_maximum": len(selected),
            "visible_field_coverage": len(visible_fields),
            "semantic_core_field_coverage": len(semantic_fields),
            "episode_ids": [episode["episode_id"] for episode in selected],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_tmp = output_dataset.with_suffix(".jsonl.tmp")
    manifest_tmp = output_manifest.with_suffix(".json.tmp")
    dataset_tmp.write_bytes(output_payload)
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    dataset_tmp.replace(output_dataset)
    manifest_tmp.replace(output_manifest)
    print(
        f"Wrote {len(selected)} episodes to {output_dataset} "
        f"(tasks={len(template_counts)}, visible_fields={len(visible_fields)}, "
        f"semantic_core_fields={len(semantic_fields)}, "
        f"change_points={dict(sorted(change_counts.items()))})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/full-384/dynamic_episodes.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/test-16",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    write_testset(args.source, args.output_dir, replace=args.replace)


if __name__ == "__main__":
    main()
