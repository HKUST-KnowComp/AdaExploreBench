"""Streaming fresh/append dataset persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from .config import DatasetGenerationConfig
from .contracts import DYNAMIC_EPISODE_SCHEMA_VERSION, STATIC_EPISODE_SCHEMA_VERSION, GeneratedDataset, validate_episode
from .reporting import build_dynamic_artifacts
from .static.reporting import build_static_artifacts


@dataclass(frozen=True)
class ExistingDataset:
    path: Path | None
    episode_count: int
    starting_sequence: int
    manifest: dict[str, Any] | None
    report: dict[str, Any] | None


def _dataset_filename(mode: str) -> str:
    return f"{mode}_episodes.jsonl"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an episode object")
            yield value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_existing_dataset(config: DatasetGenerationConfig) -> ExistingDataset:
    output = config.dataset.output_dir
    if config.dataset.operation == "fresh":
        if output.exists() and any(output.iterdir()) and not config.dataset.overwrite:
            raise ValueError(f"fresh dataset output is not empty: {output}; set dataset.overwrite explicitly")
        return ExistingDataset(None, 0, 1, None, None)
    manifest_path = output / "manifest.json"
    path = output / _dataset_filename(config.dataset.mode)
    if not manifest_path.is_file() or not path.is_file():
        raise ValueError("append requires an existing manifest and matching episode file")
    manifest = json.loads(manifest_path.read_text())
    dataset = manifest.get("dataset") or {}
    expected_schema = STATIC_EPISODE_SCHEMA_VERSION if config.dataset.mode == "static" else DYNAMIC_EPISODE_SCHEMA_VERSION
    for key, expected in {"id": config.dataset.dataset_id, "version": config.dataset.version, "mode": config.dataset.mode, "file": path.name}.items():
        if dataset.get(key) != expected:
            raise ValueError(f"append dataset {key} mismatch")
    if manifest.get("schema_version") != expected_schema:
        raise ValueError("append episode schema version does not match config mode")
    if dataset.get("sha256") != _file_sha256(path):
        raise ValueError("append source episode file does not match its manifest hash")
    count, maximum = 0, 0
    for episode in _iter_jsonl(path):
        validate_episode(episode, config.dataset.mode)
        count += 1
        maximum = max(maximum, int(episode["provenance"]["sequence"]))
    artifact = manifest.get("generation_artifact") or {}
    report_name = "capacity_report.json" if config.dataset.mode == "dynamic" else "balance_report.json"
    report_path = config.dataset.artifacts_dir / str(artifact.get("run_id") or "") / report_name
    return ExistingDataset(path, count, maximum + 1, manifest, json.loads(report_path.read_text()) if report_path.is_file() else None)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_dataset(
    generated: GeneratedDataset,
    config: DatasetGenerationConfig,
    existing: ExistingDataset,
    *,
    run_id: str | None = None,
    publication_files: dict[str, str] | None = None,
    manifest_extensions: dict[str, Any] | None = None,
    expected_dataset_sha256: str | None = None,
) -> None:
    if generated.mode != config.dataset.mode:
        raise ValueError("generated dataset mode does not match config")
    output = config.dataset.output_dir
    output.mkdir(parents=True, exist_ok=True)
    destination = output / _dataset_filename(generated.mode)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=output)
    digest = hashlib.sha256()
    ids: set[str] = set()
    sequences: set[int] = set()
    added_ids: list[str] = []
    publication_files = publication_files or {}
    manifest_extensions = manifest_extensions or {}
    if any(Path(name).name != name or not name for name in publication_files):
        raise ValueError("publication sidecars must use plain filenames")
    try:
        with os.fdopen(descriptor, "wb") as target:
            if existing.path is not None:
                for episode in _iter_jsonl(existing.path):
                    ids.add(str(episode["episode_id"]))
                    sequences.add(int(episode["provenance"]["sequence"]))
                    line = (json.dumps(episode, ensure_ascii=False) + "\n").encode()
                    target.write(line)
                    digest.update(line)
            for episode in generated.episodes:
                validate_episode(episode, generated.mode)
                episode_id = str(episode["episode_id"])
                sequence = int(episode["provenance"]["sequence"])
                if episode_id in ids or sequence in sequences:
                    raise ValueError("append would create duplicate episode ID or sequence")
                ids.add(episode_id)
                sequences.add(sequence)
                added_ids.append(episode_id)
                line = (json.dumps(episode, ensure_ascii=False) + "\n").encode()
                target.write(line)
                digest.update(line)
            target.flush()
            os.fsync(target.fileno())
        dataset_sha256 = digest.hexdigest()
        if expected_dataset_sha256 is not None and dataset_sha256 != expected_dataset_sha256:
            raise ValueError("staged dataset differs from its pre-publication review identity")
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        artifact_dir = config.dataset.artifacts_dir / run_id
        episodes = _iter_jsonl(Path(temporary_name))
        artifacts = (build_dynamic_artifacts if generated.mode == "dynamic" else build_static_artifacts)(episodes, iter(generated.diagnostics), config, dataset_sha256, previous_report=existing.report)
        manifest = artifacts["manifest"]
        overlap = set(manifest) & set(manifest_extensions)
        if overlap:
            raise ValueError(f"manifest extensions cannot replace built fields: {sorted(overlap)}")
        manifest.update(manifest_extensions)
        manifest["generation_artifact"] = {"run_id": run_id}
        generation = {
            "operation": config.dataset.operation,
            "starting_sequence": existing.starting_sequence,
            "episode_count_added": len(added_ids),
            "episode_ids": added_ids,
            "generation_contract_sha256": config.generation_contract_hash(),
            "template_sha256": _file_sha256(config.dataset.template_pool),
            "template_audit_sha256": _file_sha256(config.dataset.template_audit),
            "template_manifest_sha256": _file_sha256(config.dataset.template_manifest),
        }
        manifest["generations"] = [
            *((existing.manifest or {}).get("generations") or []), generation
        ]
        _atomic_write(artifact_dir / "resolved_config.yaml", yaml.safe_dump(config.to_manifest_dict(), sort_keys=False))
        _atomic_write(artifact_dir / "resolved_episode_plan.yaml", yaml.safe_dump(artifacts["resolved_episode_plan"], sort_keys=False))
        report_key = "capacity_report" if generated.mode == "dynamic" else "balance_report"
        report_name = "capacity_report.json" if generated.mode == "dynamic" else "balance_report.json"
        _atomic_write(artifact_dir / report_name, json.dumps(artifacts[report_key], ensure_ascii=False, indent=2) + "\n")
        _atomic_write(artifact_dir / "generation_summary.json", json.dumps({"dataset_id": config.dataset.dataset_id, "mode": generated.mode, "episode_count_added": len(added_ids), "combined_episode_count": len(ids), "review_error_count": len(generated.review_errors)}, indent=2) + "\n")
        for filename, content in publication_files.items():
            _atomic_write(artifact_dir / filename, content)
        os.replace(temporary_name, destination)
        for filename, content in publication_files.items():
            _atomic_write(output / filename, content)
        _atomic_write(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
