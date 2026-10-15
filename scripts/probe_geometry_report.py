#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

from runtime.exploration_design import exploration_design
from runtime.probe_geometry import compute_geometry, materialize_probe_points

SKIPPED_FILES = {"manifest.json", "metrics.json", "probe_geometry_report.json"}


def load_episodes(path: Path) -> dict[str, dict[str, Any]]:
    episodes: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        episodes[str(episode["episode_id"])] = episode
    return episodes


def load_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in SKIPPED_FILES or ".interrupted-" in path.name:
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("branches"), dict) and payload["branches"]:
            artifacts.append(payload)
    return artifacts


def _episode_geometry(
    branch_artifact: dict[str, Any], episode: dict[str, Any] | None
) -> dict[str, Any] | None:
    if episode is None:
        return None
    transition = (branch_artifact.get("evaluation") or {}).get("transition") or {}
    change_after_batch = transition.get("after_action_batch")
    if (
        isinstance(change_after_batch, bool)
        or not isinstance(change_after_batch, int)
        or change_after_batch <= 0
    ):
        return None
    condition = str(branch_artifact.get("condition") or "dynamic")
    points, unresolved = materialize_probe_points(branch_artifact, episode)
    geometry = compute_geometry(
        points,
        episode,
        condition=condition,
        change_after_batch=change_after_batch,
    )
    geometry["unresolved_probes"] = unresolved
    geometry["design"] = exploration_design(
        points, episode, condition=condition, change_after_batch=change_after_batch
    )
    return geometry


def _summarize_geometry(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    def leaf(path: tuple[str, ...]) -> float | None:
        values = []
        for block in blocks:
            value: Any = block
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        return round(mean(values), 6) if values else None

    def contributing(path: tuple[str, ...]) -> int:
        count = 0
        for block in blocks:
            value: Any = block
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                count += 1
        return count

    return {
        "probe_points": leaf(("probe_points",)),
        "unresolved_probes": leaf(("unresolved_probes",)),
        "exploration_failure_rate": (
            round(
                mean(
                    1.0 if int(block["probe_points"]) < 2 else 0.0
                    for block in blocks
                ),
                6,
            )
            if blocks
            else None
        ),
        "spread": {"total_variance": leaf(("spread", "total_variance"))},
        "pre": {
            "focus_variance_share": leaf(("pre", "focus_variance_share")),
            "incoming_field_share": leaf(("pre", "incoming_field_share")),
            "outgoing_field_share": leaf(("pre", "outgoing_field_share")),
        },
        "post": {
            "focus_variance_share": leaf(("post", "focus_variance_share")),
            "incoming_field_share": leaf(("post", "incoming_field_share")),
            "outgoing_field_share": leaf(("post", "outgoing_field_share")),
        },
        "valid_episodes": {
            "spread": contributing(("spread", "total_variance")),
            "pre_focus": contributing(("pre", "focus_variance_share")),
            "post_focus": contributing(("post", "focus_variance_share")),
            "pre_incoming": contributing(("pre", "incoming_field_share")),
            "post_incoming": contributing(("post", "incoming_field_share")),
            "pre_outgoing": contributing(("pre", "outgoing_field_share")),
            "post_outgoing": contributing(("post", "outgoing_field_share")),
        },
    }


def _cell(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def print_summary(
    branches: dict[str, dict[str, Any]], field_count: int | None
) -> None:
    headers = (
        "branch",
        "episodes",
        "probe_pts",
        "fail_rate",
        "spread",
        "spread_unif",
        "pre_focus",
        "post_focus",
        "pre_in",
        "post_in",
        "pre_out",
        "post_out",
    )
    rows = []
    for name, branch in branches.items():
        geometry = branch["probe_geometry"]
        spread = geometry["spread"]["total_variance"]
        if spread is None or not field_count:
            spread_unif: Any = None
        else:
            spread_unif = float(spread) / (field_count / 12)
        rows.append((
            name,
            str(branch["episode_count"]),
            _cell(geometry["probe_points"]),
            _cell(geometry.get("exploration_failure_rate")),
            _cell(spread),
            _cell(spread_unif),
            _cell(geometry["pre"]["focus_variance_share"]),
            _cell(geometry["post"]["focus_variance_share"]),
            _cell(geometry["pre"]["incoming_field_share"]),
            _cell(geometry["post"]["incoming_field_share"]),
            _cell(geometry["pre"]["outgoing_field_share"]),
            _cell(geometry["post"]["outgoing_field_share"]),
        ))
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    for row in rows:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(row))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute probe-geometry aggregates for an existing run directory."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    episodes_by_id = load_episodes(args.dataset)
    artifacts = load_artifacts(args.output_dir)
    if not artifacts:
        raise SystemExit(f"no branched episode artifacts found in {args.output_dir}")
    field_count: int | None = None
    for episode in episodes_by_id.values():
        fields = (episode.get("agent_task") or {}).get("fields") or {}
        if fields:
            field_count = len(fields)
            break

    branch_names = sorted({
        name
        for artifact in artifacts
        for name in (artifact.get("branches") or {})
    })
    branches: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    for name in branch_names:
        blocks: list[dict[str, Any]] = []
        episode_count = 0
        for artifact in artifacts:
            branch_artifact = (artifact.get("branches") or {}).get(name)
            if branch_artifact is None:
                continue
            episode_count += 1
            episode_id = str(
                branch_artifact.get("episode_id") or artifact.get("episode_id") or ""
            )
            geometry = _episode_geometry(branch_artifact, episodes_by_id.get(episode_id))
            if geometry is not None:
                blocks.append(geometry)
            episodes.append({
                "episode_id": episode_id,
                "branch": name,
                "rule_family": branch_artifact.get("rule_family"),
                "condition": branch_artifact.get("condition"),
                "geometry": geometry,
            })
        branches[name] = {
            "episode_count": episode_count,
            "probe_geometry": _summarize_geometry(blocks),
        }

    report = {
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "artifact_count": len(artifacts),
        "branches": branches,
        "episodes": episodes,
    }
    report_path = args.output_dir / "probe_geometry_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(f"artifacts: {len(artifacts)}  episodes: {len(episodes)}")
    print_summary(branches, field_count)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
