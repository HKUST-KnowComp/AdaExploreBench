#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def _load(path: Path, compressed: bool) -> dict[str, Any]:
    target = path.with_suffix(path.suffix + ".gz") if compressed else path
    if not target.is_file():
        raise ValueError(f"missing viewer payload: {target}")
    raw = gzip.decompress(target.read_bytes()) if compressed else target.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"viewer payload must be an object: {target}")
    return value


def source_sampling_name(name: str) -> str:
    for suffix in (
        "_no_change_control",
        "_eval_case_given",
        "_full_history",
    ):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def validate(data_dir: Path, *, compressed: bool) -> tuple[int, int]:
    manifest = _load(data_dir / "manifest.json", compressed)
    if not str(manifest.get("dataset_sha256") or ""):
        raise ValueError("viewer manifest lacks dataset SHA provenance")
    publication = manifest.get("publication") or {}
    if publication.get("mode") == "overview":
        runs = manifest.get("runs") or []
        expected = publication.get("model_count")
        if not isinstance(expected, int) or expected <= 0 or len(runs) != expected:
            raise ValueError("overview model count differs from publication contract")
        if len({(run.get("hyperparameters") or {}).get("model") for run in runs}) != expected:
            raise ValueError("overview requires distinct model identities")
        if manifest.get("episodes") or any(run.get("episode_ids") for run in runs):
            raise ValueError("overview publication must contain aggregate data only")
        if manifest.get("episode_count", 0) <= 0:
            raise ValueError("overview lacks evaluated episode count")
        for run in runs:
            view = (run.get("views") or {}).get("all") or {}
            default = (view.get("branches") or {}).get("flip_50") or {}
            if default.get("episode_count") != manifest["episode_count"]:
                raise ValueError(f"overview run lacks complete default results: {run.get('run_id')}")
            if (default.get("metrics") or {}).get("calibrated_accuracy") is None:
                raise ValueError(f"overview run lacks calibrated accuracy: {run.get('run_id')}")
        if any(path.is_dir() for path in data_dir.iterdir()):
            raise ValueError("overview publication contains episode or trajectory directories")
        print(f"Validated overview for {expected} models across {manifest['episode_count']} episodes")
        return manifest["episode_count"], 0
    run_ids = {
        str(run.get("run_id"))
        for run in manifest.get("runs") or []
        if isinstance(run, dict) and run.get("run_id")
    }
    checked_branches = 0
    for summary in manifest.get("episodes") or []:
        episode_id = str(summary.get("episode_id") or "")
        episode = _load(data_dir / "episodes" / f"{episode_id}.json", compressed)
        if episode.get("episode_id") != episode_id:
            raise ValueError(f"episode ID mismatch: {episode_id}")
        available = set(
            ((episode.get("dynamic") or {}).get("event") or {})
            .get("sampling_design", {})
            .get("available_post_branches")
            or []
        )
        if not available:
            raise ValueError(f"episode lacks integrated sampling branches: {episode_id}")
        for run_id in summary.get("runs") or []:
            if run_id not in run_ids:
                raise ValueError(f"episode references unknown run: {run_id}")
            payload = _load(
                data_dir / "runs" / run_id / f"{episode_id}.json", compressed
            )
            branches = payload.get("branches") or {}
            if not branches:
                raise ValueError(f"run has no selected branches: {run_id}/{episode_id}")
            for name, branch in branches.items():
                sampling_name = source_sampling_name(
                    str(
                        branch.get("source_sampling_branch")
                        or branch.get("sampling_branch")
                        or name
                    )
                )
                if sampling_name not in available:
                    raise ValueError(
                        f"run branch is absent from dataset: {run_id}/{episode_id}/{name}"
                    )
                batch_metrics = (branch.get("metrics") or {}).get("batch_metrics") or []
                if branch.get("run_status") == "completed" and len(batch_metrics) != 12:
                    raise ValueError(
                        f"completed branch must contain 12 batches: {run_id}/{episode_id}/{name}"
                    )
                checked_branches += 1
    return len(manifest.get("episodes") or []), checked_branches


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate published viewer payloads")
    parser.add_argument("--data-dir", type=Path, default=Path("viewer/public/data"))
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()
    episodes, branches = validate(args.data_dir, compressed=args.compressed)
    print(f"Validated {episodes} episodes and {branches} run branches")


if __name__ == "__main__":
    main()
