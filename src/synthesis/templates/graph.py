from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Status(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    ACCEPTED = "accepted"
    FAILED = "failed"
    STALE = "stale"


@dataclass
class Node:
    node_id: str
    dependencies: set[str] = field(default_factory=set)
    status: Status = Status.PENDING
    component_hash: str | None = None


class DependencyGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.children: dict[str, set[str]] = {}

    def add(self, node_id: str, dependencies: set[str] | None = None) -> Node:
        deps = set(dependencies or ())
        node = self.nodes.get(node_id)
        if node is not None and node.dependencies != deps:
            for dependency in node.dependencies:
                self.children.get(dependency, set()).discard(node_id)
            node.dependencies = deps
            node.status = Status.STALE
            node.component_hash = None
            self.invalidate(node_id)
        elif node is None:
            node = Node(node_id, deps)
            self.nodes[node_id] = node
        for dependency in deps:
            self.children.setdefault(dependency, set()).add(node_id)
        return node

    def ready(self) -> list[str]:
        return sorted(
            node_id for node_id, node in self.nodes.items()
            if node.status in {Status.PENDING, Status.STALE}
            and all(self.nodes[dep].status == Status.ACCEPTED for dep in node.dependencies)
        )

    def accept(self, node_id: str, component_hash: str) -> None:
        node = self.nodes[node_id]
        node.status = Status.ACCEPTED
        node.component_hash = component_hash

    def invalidate(self, node_id: str) -> set[str]:
        invalidated: set[str] = set()
        stack = list(self.children.get(node_id, ()))
        while stack:
            child = stack.pop()
            if child in invalidated:
                continue
            invalidated.add(child)
            self.nodes[child].status = Status.STALE
            self.nodes[child].component_hash = None
            stack.extend(self.children.get(child, ()))
        return invalidated
