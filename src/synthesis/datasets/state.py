"""Atomic, stage-aware checkpoints for dataset synthesis runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DatasetGenerationConfig


def _cell_digest(cell: tuple[str, ...]) -> str:
    return hashlib.sha256("\u0000".join(cell).encode()).hexdigest()[:20]


def _cell_directory(cell: tuple[str, ...]) -> str:
    readable = "--".join(re.sub(r"[^A-Za-z0-9._-]+", "_", part) for part in cell)
    return f"{_cell_digest(cell)}--{readable}"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def candidate_identity(cell: tuple[str, ...], candidate: dict[str, Any]) -> str:
    return "\u0000".join(
        (*cell, str(candidate["core_id"]), str(int(candidate.get("distractor_trial", 0))))
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    value = dict(candidate)
    certification = dict(value.get("certification") or {})
    inventory = certification.get("inventory")
    if not isinstance(inventory, dict) or not inventory.get("records"):
        return candidate
    records = inventory["records"]
    fields = list(records[0]["raw_values"])
    certification["inventory"] = {
        "record_columns": {
            "fields": fields,
            "raw_values": [
                [record["raw_values"][field] for field in fields]
                for record in records
            ],
            "operational_profiles": [
                record["operational_profile"] for record in records
            ],
            "old_actions": [record["old_action"] for record in records],
            "new_actions": [record["new_action"] for record in records],
        },
        "buckets": inventory["buckets"],
    }
    value["certification"] = certification
    return value


def _expand_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    value = dict(candidate)
    certification = dict(value.get("certification") or {})
    inventory = certification.get("inventory")
    columns = inventory.get("record_columns") if isinstance(inventory, dict) else None
    if not isinstance(columns, dict):
        return candidate
    fields = columns["fields"]
    records = []
    for index, raw_values in enumerate(columns["raw_values"]):
        records.append({
            "pool_index": index,
            "raw_values": dict(zip(fields, raw_values)),
            "operational_profile": columns["operational_profiles"][index],
            "old_action": columns["old_actions"][index],
            "new_action": columns["new_actions"][index],
        })
    certification["inventory"] = {
        "records": records,
        "buckets": inventory["buckets"],
    }
    value["certification"] = certification
    return value


class DatasetRunState:
    def __init__(
        self,
        config: DatasetGenerationConfig,
        run_id: str,
        *,
        resume: bool,
    ) -> None:
        self.run_dir = config.dataset.artifacts_dir / run_id
        self.state_path = self.run_dir / "run_state.json"
        identity = {
            "generation_contract_sha256": config.generation_contract_hash(),
            "template_sha256": _file_sha256(config.dataset.template_pool),
            "template_audit_sha256": _file_sha256(config.dataset.template_audit),
            "template_manifest_sha256": _file_sha256(config.dataset.template_manifest),
            "operation": config.dataset.operation,
            "dataset_seed": config.dataset.seed,
            "append_source_sha256": self._append_source_sha256(config),
        }
        if resume:
            if not self.state_path.is_file():
                raise ValueError("dataset resume requires an existing run state")
            self.value = json.loads(self.state_path.read_text())
            if self.value.get("identity") != identity:
                raise ValueError("dataset resume inputs differ from the pinned run identity")
            if self.value.get("published"):
                return
            self.value["status"] = "running"
            self.value.pop("failure", None)
            self._recover_episode_fragments()
            self._write()
            return
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise ValueError(f"dataset run ID already exists: {run_id}")
        self.value = {
            "status": "running",
            "published": False,
            "identity": identity,
            "cells": {},
            "groups": {},
            "episodes": {},
            "candidate_commits": {},
            "review_batches": [],
            "review_errors": [],
        }
        self._write()

    @staticmethod
    def _key(parts: tuple[str, ...]) -> str:
        return "\u0000".join(parts)

    @staticmethod
    def _append_source_sha256(config: DatasetGenerationConfig) -> str | None:
        if config.dataset.operation != "append":
            return None
        path = config.dataset.output_dir / f"{config.dataset.mode}_episodes.jsonl"
        return _file_sha256(path) if path.is_file() else None

    @property
    def published(self) -> bool:
        return bool(self.value.get("published"))

    def cell_checkpoint(self, cell: tuple[str, ...]) -> dict[str, Any] | None:
        entry = (self.value.get("cells") or {}).get(self._key(cell))
        if not entry:
            return None
        path = self.run_dir / entry["checkpoint"]
        if not path.is_file():
            raise ValueError(f"cell checkpoint is missing: {self._key(cell)}")
        return json.loads(path.read_text())

    def cell_dir(self, cell: tuple[str, ...]) -> Path:
        return self.run_dir / "cells" / _cell_directory(cell)

    def cell_order_path(self, cell: tuple[str, ...]) -> Path:
        return self.cell_dir(cell) / "order.bin"

    def save_cell_checkpoint(
        self, cell: tuple[str, ...], stage: str, payload: dict[str, Any]
    ) -> None:
        relative = f"cells/{_cell_directory(cell)}/state.json"
        self._write_payload(relative, {"cell": list(cell), "stage": stage, **payload})
        self.value["cells"][self._key(cell)] = {
            "stage": stage,
            "checkpoint": relative,
        }
        self._write()

    def episode(self, episode_id: str) -> dict[str, Any] | None:
        relative = (self.value.get("episodes") or {}).get(episode_id)
        if not relative:
            return None
        path = self.run_dir / relative
        if not path.is_file():
            raise ValueError(f"episode checkpoint is missing: {episode_id}")
        return json.loads(path.read_text())

    def write_candidate(self, cell: tuple[str, ...], candidate: dict[str, Any]) -> str:
        review_state = "reviewed" if candidate.get("semantic_review") else "pending"
        candidate_id = (
            f"{candidate['core_id']}\0{int(candidate.get('distractor_trial', 0))}"
            f"\0{review_state}"
        )
        digest = hashlib.sha256(candidate_id.encode()).hexdigest()[:20]
        relative = f"cells/{_cell_directory(cell)}/candidates/{digest}.json"
        self._write_payload(relative, _compact_candidate(candidate))
        return relative

    def read_candidate(self, relative: str) -> dict[str, Any]:
        path = self.run_dir / relative
        if not path.is_file():
            raise ValueError(f"candidate checkpoint is missing: {relative}")
        return _expand_candidate(json.loads(path.read_text()))

    def delete_candidate(self, relative: str) -> None:
        path = (self.run_dir / relative).resolve()
        if self.run_dir.resolve() not in path.parents:
            raise ValueError("candidate checkpoint is outside the run directory")
        if path.is_file():
            path.unlink()

    def record_episode(
        self, cell: tuple[str, ...], candidate: dict[str, Any], build: Any
    ) -> str:
        episode_id = str(build.episode["episode_id"])
        digest = hashlib.sha256(episode_id.encode()).hexdigest()[:20]
        relative = f"cells/{_cell_directory(cell)}/episodes/{digest}.json"
        self._write_payload(
            relative,
            {
                "cell": list(cell),
                "candidate_identity": candidate_identity(cell, candidate),
                "episode": build.episode,
                "diagnostics": build.diagnostics,
            },
        )
        self.value["episodes"][episode_id] = relative
        self.value.setdefault("candidate_commits", {})[
            candidate_identity(cell, candidate)
        ] = episode_id
        self._write()
        return relative

    def candidate_committed(
        self, cell: tuple[str, ...], candidate: dict[str, Any]
    ) -> bool:
        return candidate_identity(cell, candidate) in self.value.setdefault(
            "candidate_commits", {}
        )

    def _recover_episode_fragments(self) -> None:
        episodes = self.value.setdefault("episodes", {})
        for path in self.run_dir.glob("cells/*/episodes/*.json"):
            payload = json.loads(path.read_text())
            episode_id = str(payload["episode"]["episode_id"])
            episodes.setdefault(episode_id, str(path.relative_to(self.run_dir)))
            identity = payload.get("candidate_identity")
            if identity:
                self.value.setdefault("candidate_commits", {}).setdefault(identity, episode_id)

    def record_review_error(self, error: dict[str, Any]) -> None:
        batch_id = error.get("batch_id")
        if batch_id and any(
            item.get("batch_id") == batch_id for item in self.value["review_errors"]
        ):
            return
        self.value["review_errors"].append(error)
        self._write()

    def record_review_batch(self, batch: dict[str, Any]) -> None:
        batches = self.value.setdefault("review_batches", [])
        for index, existing in enumerate(batches):
            if existing.get("batch_id") == batch.get("batch_id"):
                batches[index] = batch
                self._write()
                return
        batches.append(batch)
        self._write()

    def mark_group_complete(
        self,
        group: tuple[str, ...],
        episode_ids: list[str],
        resource_paths: list[str] | None = None,
    ) -> None:
        missing = [item for item in episode_ids if item not in self.value["episodes"]]
        if missing:
            raise ValueError(f"group completion precedes episode commit: {missing}")
        self.value["groups"][self._key(group)] = {
            "status": "completed",
            "episode_ids": list(episode_ids),
            "resource_paths": list(resource_paths or []),
            "resources_cleaned": False,
        }
        self._write()

    def clean_group_resources(
        self,
        group: tuple[str, ...],
        relative_paths: list[str] | None = None,
    ) -> None:
        key = self._key(group)
        entry = self.value["groups"].get(key)
        if not entry or entry.get("status") != "completed":
            raise ValueError("group resources can be cleaned only after group completion")
        root = self.run_dir.resolve()
        paths = entry.get("resource_paths", []) if relative_paths is None else relative_paths
        for relative in paths:
            path = (self.run_dir / relative).resolve()
            if root not in path.parents:
                raise ValueError("group resource is outside the run directory")
            if path.is_file():
                path.unlink()
        entry["resources_cleaned"] = True
        entry["resource_paths"] = []
        self._write()

    def complete(self, *, published: bool = True) -> None:
        self.value["status"] = "completed"
        self.value["published"] = published
        self.value.pop("failure", None)
        self._write()

    def fail(self, error: BaseException) -> None:
        self.value["status"] = "failed"
        self.value["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._write()

    def _write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, self.state_path)

    def _write_payload(self, relative: str, value: dict[str, Any]) -> None:
        _atomic_write_json(self.run_dir / relative, value)


def write_worker_cell_checkpoint(
    run_dir: Path,
    cell: tuple[str, ...],
    stage: str,
    payload: dict[str, Any],
) -> None:
    _atomic_write_json(
        run_dir / "cells" / _cell_directory(cell) / "state.json",
        {"cell": list(cell), "stage": stage, **payload},
    )


def write_worker_candidate(
    run_dir: Path, cell: tuple[str, ...], candidate: dict[str, Any]
) -> str:
    review_state = "reviewed" if candidate.get("semantic_review") else "pending"
    identity = (
        f"{candidate['core_id']}\0{int(candidate.get('distractor_trial', 0))}"
        f"\0{review_state}"
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    relative = f"cells/{_cell_directory(cell)}/candidates/{digest}.json"
    _atomic_write_json(run_dir / relative, _compact_candidate(candidate))
    return relative


def capture_worker_error(
    cell: tuple[str, ...],
    core: dict[str, Any],
    error: Exception,
    records: list[dict[str, Any]],
) -> None:
    error_type, message = type(error).__name__, str(error)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    match = next(
        (
            item
            for item in records
            if item["type"] == error_type and item["message"] == message
        ),
        None,
    )
    core_index = int(core["core_space_index"])
    if match is None:
        records.append(
            {
                "type": error_type,
                "message": message,
                "count": 1,
                "first_core_space_index": core_index,
                "last_core_space_index": core_index,
                "first_timestamp": timestamp,
                "last_timestamp": timestamp,
                "traceback": traceback.format_exc(),
            }
        )
    else:
        match["count"] = int(match["count"]) + 1
        match["last_core_space_index"] = core_index
        match["last_timestamp"] = timestamp
        match["traceback"] = traceback.format_exc()
    print(
        f"{timestamp} "
        f"core_failed cell={'/'.join(cell)} core={core_index} "
        f"error={error_type}: {message}",
        flush=True,
    )
