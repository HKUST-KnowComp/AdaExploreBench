"""CLI for balanced score-based selection from a reviewed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .final_review import load_dynamic_episodes
from .selection import load_selection_config, select_episodes, write_selection_outputs


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="synthesize.py dataset-select",
        description="Select an equal top-k from every reviewed domain/rule group.",
    )
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--review", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthesis/dataset_selection.yaml"),
    )
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    dataset_path = args.dataset.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else dataset_path.parent / "manifest.json"
    review_path = args.review.resolve() if args.review else dataset_path.parent / "llm_review.json"
    for path in (dataset_path, manifest_path, review_path):
        if not path.is_file():
            raise ValueError(f"required selection input does not exist: {path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("dataset") or {}).get("file") != dataset_path.name:
        raise ValueError("manifest does not reference the requested dataset")
    config = load_selection_config(args.config)
    output_path = args.output.resolve() if args.output else dataset_path.parent / config.output_filename
    report_path = output_path.parent / config.report_filename
    result = select_episodes(
        load_dynamic_episodes(dataset_path),
        json.loads(review_path.read_text()),
        config,
    )
    write_selection_outputs(result, config, output_path, report_path, manifest_path)
    print(
        f"Selected {len(result.episodes)} episodes into {output_path} "
        f"({result.report['policy']['effective_per_group']} per domain/rule group)"
    )
