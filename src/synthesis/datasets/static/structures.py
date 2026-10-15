"""Indexed legal core space for stable-rule episodes."""

from __future__ import annotations

import hashlib
import math
import random
from copy import deepcopy
from typing import Any, Iterator

from protocol.constraints import constraint_fields

from ..planning import BoundedPermutation
from ..template_pool import TemplateSelection, approved_priority_orders, family_field_eligible


def _combination_at(items: list[str], size: int, index: int) -> tuple[str, ...]:
    if size < 0 or size > len(items) or index < 0 or index >= math.comb(len(items), size):
        raise IndexError("combination index is out of range")
    selected: list[str] = []
    start = 0
    while size:
        for item_index in range(start, len(items)):
            block = math.comb(len(items) - item_index - 1, size - 1)
            if index < block:
                selected.append(items[item_index])
                start = item_index + 1
                size -= 1
                break
            index -= block
    return tuple(selected)


class StaticCoreSpace:
    def __init__(self, selection: TemplateSelection, family: str) -> None:
        self.selection = selection
        self.family = family
        self._orders = (
            [tuple(map(str, item["fields"])) for item in approved_priority_orders(selection)]
            if family == "lexicographic_priority"
            else None
        )
        self._eligible = (
            []
            if self._orders is not None
            else [
                field
                for field in selection.fields
                if family_field_eligible(selection, field, family)
            ]
        )
        self.count = (
            len(self._orders)
            if self._orders is not None
            else math.comb(len(self._eligible), 3)
        )

    def at(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.count:
            raise IndexError("static core index is out of range")
        relevant = (
            self._orders[index]
            if self._orders is not None
            else _combination_at(self._eligible, 3, index)
        )
        value = {"core_space_index": index, "relevant_fields": list(relevant)}
        if self._orders is not None:
            value["priority_order"] = list(relevant)
        return value


def candidate_structures(
    selection: TemplateSelection,
    family: str,
    traversal: BoundedPermutation,
) -> Iterator[dict[str, Any]]:
    space = StaticCoreSpace(selection, family)
    if traversal.total != space.count:
        raise ValueError("static traversal does not match its core space")
    return (space.at(index) for index in traversal)


def distractor_sources(
    selection: TemplateSelection, core: dict[str, Any]
) -> tuple[str, ...]:
    constrained = {
        str(field)
        for item in selection.template.get("feasibility_constraints", [])
        for field in constraint_fields(item["expr"])
    }
    relevant = set(map(str, core["relevant_fields"]))
    return tuple(
        sorted(field for field in selection.fields if field not in relevant | constrained)
    )


def materialize_distractors(
    selection: TemplateSelection,
    core: dict[str, Any],
    seed: int,
    visible_field_count: int,
    trial: int,
) -> dict[str, Any]:
    count = visible_field_count - len(core["relevant_fields"])
    options = list(distractor_sources(selection, core))
    total = math.comb(len(options), count) if 0 < count <= len(options) else 0
    if not total:
        raise ValueError("static core lacks distinct constraint-free distractor fields")
    if trial < 0 or trial >= total:
        raise IndexError("static distractor trial is out of range")
    identity = "|".join((*map(str, core["relevant_fields"]), str(seed)))
    order = list(range(total))
    random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")).shuffle(order)
    value = deepcopy(core)
    value["distractor_fields"] = list(_combination_at(options, count, order[trial]))
    return value
