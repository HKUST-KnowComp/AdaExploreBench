"""Indexed dynamic-core spaces and deterministic bounded traversal."""

from __future__ import annotations

import hashlib
import math
import random
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

from protocol.constraints import constraint_fields

from ..planning import BoundedPermutation
from ..template_pool import (
    TemplateSelection,
    approved_priority_orders,
    direction_reversal_options,
    family_field_eligible,
    replacement_edges,
)


def _eligible(selection: TemplateSelection, family: str) -> list[str]:
    return [
        field
        for field in selection.fields
        if family_field_eligible(selection, field, family)
    ]


def _combination_at(items: list[str], size: int, index: int) -> tuple[str, ...]:
    if size < 0 or size > len(items) or index < 0 or index >= math.comb(len(items), size):
        raise IndexError("combination index is out of range")
    output: list[str] = []
    start = 0
    while size:
        for item_index in range(start, len(items)):
            block = math.comb(len(items) - item_index - 1, size - 1)
            if index < block:
                output.append(items[item_index])
                start = item_index + 1
                size -= 1
                break
            index -= block
    return tuple(output)


@dataclass(frozen=True)
class _Block:
    count: int
    edge: dict[str, Any] | None = None
    reversal: dict[str, Any] | None = None
    order: tuple[str, ...] | None = None
    choices: tuple[str, ...] = ()


class DynamicCoreSpace:
    """Map a dense integer range to each legal semantic core exactly once."""

    def __init__(
        self,
        selection: TemplateSelection,
        family: str,
        dynamic_type: str,
    ) -> None:
        if dynamic_type not in {
            "compound",
            "field_replacement",
            "direction_reversal",
        }:
            raise ValueError(f"unsupported dynamic type: {dynamic_type}")
        self.selection = selection
        self.family = family
        self.dynamic_type = dynamic_type
        self._blocks = self._build_blocks()
        self._ends: list[int] = []
        total = 0
        for block in self._blocks:
            total += block.count
            self._ends.append(total)
        self.count = total

    def _build_blocks(self) -> list[_Block]:
        eligible = _eligible(self.selection, self.family)
        reversals = direction_reversal_options(self.selection, self.family)
        edges = replacement_edges(self.selection, self.family)
        orders = [
            tuple(map(str, item["fields"]))
            for item in approved_priority_orders(self.selection)
        ]
        blocks: list[_Block] = []
        if self.dynamic_type == "direction_reversal":
            for reversal in reversals:
                target = str(reversal["target_field"])
                if self.family == "lexicographic_priority":
                    blocks.extend(
                        _Block(1, reversal=reversal, order=order)
                        for order in orders
                        if target in order
                    )
                else:
                    choices = tuple(field for field in eligible if field != target)
                    count = math.comb(len(choices), 2)
                    if count:
                        blocks.append(_Block(count, reversal=reversal, choices=choices))
            return blocks
        if self.dynamic_type == "field_replacement":
            for edge in edges:
                outgoing = str(edge["outgoing_field"])
                incoming = str(edge["incoming_field"])
                if self.family == "lexicographic_priority":
                    blocks.extend(
                        _Block(1, edge=edge, order=order)
                        for order in orders
                        if outgoing in order and incoming not in order
                    )
                else:
                    choices = tuple(
                        field for field in eligible if field not in {outgoing, incoming}
                    )
                    count = math.comb(len(choices), 2)
                    if count:
                        blocks.append(_Block(count, edge=edge, choices=choices))
            return blocks
        for edge in edges:
            outgoing = str(edge["outgoing_field"])
            incoming = str(edge["incoming_field"])
            for reversal in reversals:
                target = str(reversal["target_field"])
                if target in {outgoing, incoming}:
                    continue
                if self.family == "lexicographic_priority":
                    blocks.extend(
                        _Block(1, edge=edge, reversal=reversal, order=order)
                        for order in orders
                        if outgoing in order and incoming not in order and target in order
                    )
                else:
                    choices = tuple(
                        field
                        for field in eligible
                        if field not in {outgoing, incoming, target}
                    )
                    if choices:
                        blocks.append(
                            _Block(len(choices), edge=edge, reversal=reversal, choices=choices)
                        )
        return blocks

    def at(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.count:
            raise IndexError("dynamic core index is out of range")
        block_index = bisect_right(self._ends, index)
        start = 0 if block_index == 0 else self._ends[block_index - 1]
        block = self._blocks[block_index]
        offset = index - start
        if self.dynamic_type == "direction_reversal":
            assert block.reversal is not None
            target = str(block.reversal["target_field"])
            shared = (
                [field for field in block.order or () if field != target]
                if block.order is not None
                else list(_combination_at(list(block.choices), 2, offset))
            )
            value = {
                **deepcopy(block.reversal),
                "core_space_index": index,
                "transition_id": str(block.reversal["event_id"]),
                "shared_fields": shared,
            }
            if block.order is not None:
                value["priority_order"] = list(block.order)
            return value
        assert block.edge is not None
        outgoing = str(block.edge["outgoing_field"])
        incoming = str(block.edge["incoming_field"])
        if self.dynamic_type == "field_replacement":
            shared = (
                [field for field in block.order or () if field != outgoing]
                if block.order is not None
                else list(_combination_at(list(block.choices), 2, offset))
            )
            value = {
                **deepcopy(block.edge),
                "core_space_index": index,
                "shared_fields": shared,
            }
            if block.order is not None:
                value.update(
                    {
                        "old_priority_order": list(block.order),
                        "new_priority_order": [
                            incoming if field == outgoing else field
                            for field in block.order
                        ],
                        "replacement_slot": block.order.index(outgoing),
                    }
                )
            return value
        assert block.reversal is not None
        target = str(block.reversal["target_field"])
        stable = (
            next(field for field in block.order or () if field not in {outgoing, target})
            if block.order is not None
            else block.choices[offset]
        )
        replacement_id = str(block.edge["transition_id"])
        value = {
            **deepcopy(block.edge),
            "core_space_index": index,
            "transition_id": f"{replacement_id}__{block.reversal['event_id']}",
            "replacement_transition_id": replacement_id,
            "shared_fields": [stable, target],
            "reversal_target": target,
            "stable_relevant": stable,
            "reversal_operation": deepcopy(block.reversal),
        }
        if block.order is not None:
            value.update(
                {
                    "old_priority_order": list(block.order),
                    "new_priority_order": [
                        incoming if field == outgoing else field for field in block.order
                    ],
                    "replacement_slot": block.order.index(outgoing),
                }
            )
        return value


def _iter_space(
    selection: TemplateSelection,
    family: str,
    dynamic_type: str,
    traversal: BoundedPermutation,
) -> Iterator[dict[str, Any]]:
    space = DynamicCoreSpace(selection, family, dynamic_type)
    if traversal.total != space.count:
        raise ValueError("dynamic traversal does not match its core space")
    return (space.at(index) for index in traversal)


def iter_reversal_cores(
    selection: TemplateSelection, family: str, traversal: BoundedPermutation
) -> Iterator[dict[str, Any]]:
    return _iter_space(selection, family, "direction_reversal", traversal)


def iter_replacement_cores(
    selection: TemplateSelection, family: str, traversal: BoundedPermutation
) -> Iterator[dict[str, Any]]:
    return _iter_space(selection, family, "field_replacement", traversal)


def iter_compound_cores(
    selection: TemplateSelection, family: str, traversal: BoundedPermutation
) -> Iterator[dict[str, Any]]:
    return _iter_space(selection, family, "compound", traversal)


def _core_fields(core: dict[str, Any], dynamic_type: str) -> set[str]:
    if dynamic_type == "direction_reversal":
        return {*map(str, core["shared_fields"]), str(core["target_field"])}
    return {
        *map(str, core["shared_fields"]),
        str(core["outgoing_field"]),
        str(core["incoming_field"]),
    }


def distractor_sources(
    selection: TemplateSelection,
    core: dict[str, Any],
    dynamic_type: str,
) -> tuple[str, ...]:
    constrained = {
        str(field)
        for constraint in selection.template.get("feasibility_constraints", [])
        for field in constraint_fields(constraint["expr"])
    }
    used = _core_fields(core, dynamic_type)
    return tuple(sorted(field for field in selection.fields if field not in used | constrained))


def _seed_for(core: dict[str, Any], seed: int) -> int:
    identity = "|".join(
        [str(core["transition_id"]), *map(str, core["shared_fields"]), str(seed)]
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def materialize_distractors(
    selection: TemplateSelection,
    core: dict[str, Any],
    dynamic_type: str,
    seed: int,
    visible_field_count: int,
    trial: int = 0,
) -> dict[str, Any]:
    value = deepcopy(core)
    count = visible_field_count - len(_core_fields(core, dynamic_type))
    options = list(distractor_sources(selection, core, dynamic_type))
    total = math.comb(len(options), count) if 0 <= count <= len(options) else 0
    if count <= 0 or not total:
        raise ValueError("semantic core lacks distinct constraint-free distractor fields")
    if trial < 0 or trial >= total:
        raise IndexError("distractor quartet trial is out of range")
    order = list(range(total))
    random.Random(_seed_for(core, seed)).shuffle(order)
    value["distractor_fields"] = list(_combination_at(options, count, order[trial]))
    return value


def visible_fields(structure: dict[str, Any], dynamic_type: str) -> list[str]:
    core = (
        [*structure["shared_fields"], structure["target_field"]]
        if dynamic_type == "direction_reversal"
        else [
            *structure["shared_fields"],
            structure["outgoing_field"],
            structure["incoming_field"],
        ]
    )
    return [*core, *structure["distractor_fields"]]
