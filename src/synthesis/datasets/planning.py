"""Stable identities and quota plans for incremental dataset generation."""

from __future__ import annotations

import hashlib
import os
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from protocol.mechanism import FAMILIES

from .config import DatasetGenerationConfig
from .contracts import EpisodeSpec
from .template_pool import domain_ids, select_template, template_ids, template_supports_cell


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


class BoundedPermutation:
    """A resumable seeded permutation prefix bounded by the configured visit limit."""

    def __init__(
        self,
        total: int,
        seed: int,
        limit: int,
        *,
        materialize_threshold: int,
        order_path: Path | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        if min(total, limit, materialize_threshold) < 0:
            raise ValueError("permutation sizes must be non-negative")
        self.total = total
        self.limit = min(total, limit)
        self.seed = seed
        self.cursor = 0
        self.materialized = total <= materialize_threshold
        self._rng = random.Random(seed)
        self._swaps: dict[int, int] = {}
        self._order: array[int] | None = None
        if self.materialized:
            typecode = "I" if total <= 0xFFFFFFFF else "Q"
            self._order = array(typecode)
            if order_path is not None and order_path.is_file():
                with order_path.open("rb") as stream:
                    self._order.fromfile(stream, total)
                if len(self._order) != total:
                    raise ValueError("persisted permutation has the wrong length")
            else:
                self._order.extend(range(total))
                self._rng.shuffle(self._order)
                if order_path is not None:
                    order_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = order_path.with_suffix(order_path.suffix + ".tmp")
                    with temporary.open("wb") as stream:
                        self._order.tofile(stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, order_path)
        if state is not None:
            self.restore(state)

    def __iter__(self) -> Iterator[int]:
        while self.cursor < self.limit:
            yield self.next()

    def next(self) -> int:
        if self.cursor >= self.limit:
            raise StopIteration
        if self._order is not None:
            value = int(self._order[self.cursor])
        else:
            selected = self._rng.randrange(self.cursor, self.total)
            value = self._swaps.get(selected, selected)
            self._swaps[selected] = self._swaps.get(self.cursor, self.cursor)
        self.cursor += 1
        return value

    def snapshot(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "limit": self.limit,
            "seed": self.seed,
            "cursor": self.cursor,
            "materialized": self.materialized,
            "rng_state": self._rng.getstate() if not self.materialized else None,
            "swaps": sorted(self._swaps.items()) if not self.materialized else [],
        }

    def restore(self, state: dict[str, Any]) -> None:
        expected = (self.total, self.limit, self.seed, self.materialized)
        observed = (
            int(state["total"]),
            int(state["limit"]),
            int(state["seed"]),
            bool(state["materialized"]),
        )
        if observed != expected:
            raise ValueError("permutation state does not match its core space")
        self.cursor = int(state["cursor"])
        if not 0 <= self.cursor <= self.limit:
            raise ValueError("permutation cursor is out of range")
        if not self.materialized:
            self._rng.setstate(_tuple_tree(state["rng_state"]))
            self._swaps = {int(key): int(value) for key, value in state["swaps"]}


@dataclass(frozen=True, order=True)
class GenerationCell:
    priority: int
    domain: str
    template_id: str
    rule_family: str
    mode: str
    dynamic_type: str | None
    initial_quota: int

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.domain,
            self.template_id,
            self.rule_family,
            self.mode,
            self.dynamic_type or "static",
        )

    @property
    def group_key(self) -> tuple[str, ...]:
        return (
            self.domain,
            self.rule_family,
            self.mode,
            self.dynamic_type or "static",
        )


def _short_family_name(family: str) -> str:
    for suffix in ("_aggregation", "_gating", "_sufficiency", "_priority"):
        family = family.removesuffix(suffix)
    return family


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _eligible_templates(
    pool: dict,
    domain: str,
    family: str,
    dynamic_type: str | None,
    visible_field_count: int,
) -> list[str]:
    return [
        template_id
        for template_id in template_ids(pool, domain)
        if template_supports_cell(
            select_template(pool, domain, template_id),
            family,
            dynamic_type,
            visible_field_count,
        )
    ]


def generation_cells(
    config: DatasetGenerationConfig, pool: dict
) -> tuple[GenerationCell, ...]:
    dynamic_types: tuple[str | None, ...]
    if config.dynamic is None:
        dynamic_types = (None,)
        group_target = config.static.episodes_per_cell if config.static else 0
        visible_field_count = config.static.visible_field_count if config.static else 0
    else:
        dynamic_types = tuple(config.dynamic.enabled_types)
        group_target = config.dynamic.dataset_design.episode_groups_per_domain_rule
        visible_field_count = config.dynamic.dataset_design.visible_field_count
    cells: list[GenerationCell] = []
    priority = 0
    for domain in domain_ids(pool):
        for family in FAMILIES:
            for dynamic_type in dynamic_types:
                eligible = _eligible_templates(
                    pool, domain, family, dynamic_type, visible_field_count
                )
                if not eligible:
                    continue
                base, remainder = divmod(group_target, len(eligible))
                for index, template_id in enumerate(eligible):
                    cells.append(
                        GenerationCell(
                            priority=priority,
                            domain=domain,
                            template_id=template_id,
                            rule_family=family,
                            mode=config.dataset.mode,
                            dynamic_type=dynamic_type,
                            initial_quota=base + (index < remainder),
                        )
                    )
                    priority += 1
    return tuple(cells)


def quota_targets(
    config: DatasetGenerationConfig, cells: tuple[GenerationCell, ...]
) -> dict[tuple[str, ...], int]:
    targets: dict[tuple[str, ...], int] = {}
    for cell in cells:
        if cell.group_key in targets:
            continue
        targets[cell.group_key] = (
            config.dynamic.dataset_design.episode_groups_per_domain_rule
            if config.dynamic is not None
            else sum(
                item.initial_quota for item in cells if item.group_key == cell.group_key
            )
        )
    return targets


def episode_spec_for_accept(
    config: DatasetGenerationConfig,
    cell: GenerationCell,
    candidate: dict,
    *,
    sequence: int,
    group_slot: int,
) -> EpisodeSpec:
    token = hashlib.sha256(cell.template_id.encode()).hexdigest()[:8]
    parts = [_short_family_name(cell.rule_family), cell.domain, token]
    if cell.dynamic_type is not None:
        parts.append(cell.dynamic_type)
    parts.extend((f"slot{group_slot + 1:02d}", f"{sequence:03d}"))
    after_action_batch = None
    if config.dynamic is not None:
        after_action_batch = random.Random(
            stable_seed(
                config.dataset.seed,
                *cell.group_key,
                sequence,
                group_slot,
                config.dynamic.change_point_seed,
            )
        ).choice(config.dynamic.change_after_batches)
    return EpisodeSpec(
        episode_id="_".join(parts),
        seed=stable_seed(config.dataset.seed, *cell.key, sequence, group_slot),
        domain=cell.domain,
        template_id=cell.template_id,
        rule_family=cell.rule_family,
        sequence=sequence,
        dynamic_type=cell.dynamic_type,
        after_action_batch=after_action_batch,
        catalog_seed=(
            int(candidate["catalog_seed"]) if "catalog_seed" in candidate else None
        ),
        candidate_slot=group_slot,
    )
