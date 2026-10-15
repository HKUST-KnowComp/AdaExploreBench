"""CLI for scoring an already published dynamic dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .final_review import (
    attach_review_to_manifest,
    load_dynamic_episodes,
    load_final_review_config,
    run_final_review,
    write_review_result,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="synthesize.py dataset-review",
        description="Score every episode in a published dynamic dataset.",
    )
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--manifest", type=Path)
    value.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthesis/final_review.yaml"),
    )
    value.add_argument("--output", type=Path)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parser().parse_args()
    dataset_path = args.dataset.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else dataset_path.parent / "manifest.json"
    )
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise ValueError("dataset-review requires an existing dataset and manifest")
    manifest = json.loads(manifest_path.read_text())
    manifest_dataset = manifest.get("dataset") or {}
    if manifest_dataset.get("file") != dataset_path.name:
        raise ValueError("manifest does not reference the requested dataset file")
    observed_sha256 = _file_sha256(dataset_path)
    if manifest_dataset.get("sha256") != observed_sha256:
        raise ValueError("dataset file differs from its manifest hash")
    config = load_final_review_config(args.config)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else dataset_path.parent / config.output_filename
    )
    episodes = load_dynamic_episodes(dataset_path)
    result = run_final_review(
        episodes,
        config,
        dataset={
            "id": manifest_dataset.get("id"),
            "version": manifest_dataset.get("version"),
            "file": dataset_path.name,
        },
    )
    if result["dataset"]["sha256"] != observed_sha256:
        raise ValueError("reviewed episode serialization differs from the dataset file")
    write_review_result(output_path, result)
    attach_review_to_manifest(manifest_path, result, output_path)
    print(
        f"Reviewed {len(episodes)} episodes into {output_path} "
        f"(overall mean {result['summary']['overall_mean']:.2f})"
    )
