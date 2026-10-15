from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graph import DependencyGraph, Status


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class RunState:
    def __init__(self, run_dir: Path, *, resume: bool, resource_identity: str) -> None:
        self.run_dir = run_dir
        self.components = run_dir / "components"
        self.responses = run_dir / "responses"
        self.candidate = run_dir / "candidate" / "domain_templates.json"
        self.state_path = run_dir / "state.json"
        self.calls_path = run_dir / "calls.jsonl"
        self.errors_path = run_dir / "errors.jsonl"
        self._lock = asyncio.Lock()
        self.graph = DependencyGraph()
        self.metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "resource_identity": resource_identity,
            "components": {},
            "audits": {},
        }
        if resume and self.state_path.exists():
            self._load()
            if self.metadata.get("resource_identity") != resource_identity and not os.environ.get(
                "TEMPLATE_SYNTHESIS_ALLOW_RESOURCE_DRIFT"
            ):
                raise ValueError(
                    "resume resources changed; use a new run ID instead of mixing prompts, "
                    "schemas, or model configuration"
                )
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
        self._call_sequence = (
            sum(1 for line in self.calls_path.read_text().splitlines() if line.strip())
            if self.calls_path.exists()
            else 0
        )

    def _load(self) -> None:
        self.metadata = json.loads(self.state_path.read_text())
        for node_id, item in self.metadata.get("components", {}).items():
            node = self.graph.add(node_id, set(item.get("dependencies") or ()))
            node.status = Status(item["status"])
            node.component_hash = item.get("hash")

    def add_node(self, node_id: str, dependencies: set[str] | None = None) -> None:
        self.graph.add(node_id, dependencies)

    def bind_inputs(self, inputs: dict[str, Any]) -> None:
        """Freeze output-affecting run inputs before any component is reused."""
        prior = self.metadata.get("inputs")
        if prior is not None and prior != inputs:
            raise ValueError(
                "resume inputs changed; domains, append mode, or base template do not "
                "match the original run"
            )
        self.metadata["inputs"] = inputs
        self._save()

    def component_path(self, node_id: str) -> Path:
        safe = [part for part in node_id.split("/") if part not in {"", ".", ".."}]
        return self.components.joinpath(*safe).with_suffix(".json")

    def read_component(self, node_id: str) -> dict[str, Any]:
        value = json.loads(self.component_path(node_id).read_text())
        if not isinstance(value, dict):
            raise ValueError(f"component {node_id} is not an object")
        return value

    def accepted_component(self, node_id: str) -> dict[str, Any] | None:
        node = self.graph.nodes[node_id]
        path = self.component_path(node_id)
        if node.status != Status.ACCEPTED or not path.exists():
            return None
        value = self.read_component(node_id)
        if canonical_hash(value) != node.component_hash:
            node.status = Status.STALE
            node.component_hash = None
            self._save()
            return None
        return value

    def audit_records(self) -> dict[str, Any]:
        return {
            str(node_id): value
            for node_id, value in (self.metadata.get("audits") or {}).items()
            if isinstance(value, dict)
        }

    def audit_record(self, node_id: str) -> dict[str, Any] | None:
        value = (self.metadata.get("audits") or {}).get(node_id)
        return value if isinstance(value, dict) else None

    async def record_audit(self, node_id: str, value: dict[str, Any]) -> None:
        async with self._lock:
            self.metadata.setdefault("audits", {})[node_id] = value
            self._save()

    async def clear_audits(self, prefix: str) -> None:
        async with self._lock:
            audits = self.metadata.get("audits") or {}
            self.metadata["audits"] = {
                node_id: value
                for node_id, value in audits.items()
                if not str(node_id).startswith(prefix)
            }
            self._save()

    def mark_stale(self, node_id: str) -> None:
        node = self.graph.nodes[node_id]
        node.status = Status.STALE
        node.component_hash = None
        self.graph.invalidate(node_id)
        self._save()

    async def accept(
        self,
        node_id: str,
        value: dict[str, Any],
        *,
        audit_record: dict[str, Any] | None = None,
    ) -> None:
        digest = canonical_hash(value)
        async with self._lock:
            prior = self.graph.nodes[node_id].component_hash
            atomic_json(self.component_path(node_id), value)
            self.graph.accept(node_id, digest)
            if audit_record is not None:
                self.metadata.setdefault("audits", {})[node_id] = audit_record
            if prior is not None and prior != digest:
                self.graph.invalidate(node_id)
            self._save()

    async def record_call(self, item: dict[str, Any]) -> None:
        async with self._lock:
            self._call_sequence += 1
            content = item.pop("response_content")
            response_path = self.responses / f"{self._call_sequence:06d}.json"
            atomic_json(response_path, {
                "stage": item["stage"],
                "component_id": item["component_id"],
                "mode": item["mode"],
                "attempt": item["attempt"],
                "content": content,
            })
            item["response_path"] = str(response_path.relative_to(self.run_dir))
            self.calls_path.parent.mkdir(parents=True, exist_ok=True)
            with self.calls_path.open("a") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    async def record_error(self, item: dict[str, Any]) -> None:
        async with self._lock:
            self.errors_path.parent.mkdir(parents=True, exist_ok=True)
            with self.errors_path.open("a") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def pending_response_id(self, request_key: str) -> str | None:
        item = (self.metadata.get("pending_responses") or {}).get(request_key)
        if not isinstance(item, dict):
            return None
        response_id = item.get("response_id")
        return str(response_id) if response_id else None

    async def record_pending_response(self, request_key: str, response_id: str) -> None:
        async with self._lock:
            pending = self.metadata.setdefault("pending_responses", {})
            pending[request_key] = {
                "response_id": response_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()

    async def clear_pending_response(self, request_key: str, response_id: str) -> None:
        async with self._lock:
            pending = self.metadata.get("pending_responses") or {}
            item = pending.get(request_key)
            if isinstance(item, dict) and item.get("response_id") == response_id:
                del pending[request_key]
                self._save()

    def _save(self) -> None:
        self.metadata["components"] = {
            node_id: {
                "dependencies": sorted(node.dependencies),
                "status": node.status.value,
                "hash": node.component_hash,
            }
            for node_id, node in sorted(self.graph.nodes.items())
        }
        atomic_json(self.state_path, self.metadata)

    def write_candidate(self, candidate: dict[str, Any]) -> None:
        atomic_json(self.candidate, candidate)
