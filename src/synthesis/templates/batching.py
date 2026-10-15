from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .config import BatchingSettings


class ContextCapacityError(ValueError):
    pass


def estimate_tokens(value: Any) -> int:
    """Conservative tokenizer-free estimate used only for preflight batching."""
    return max(1, (len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) + 2) // 3)


@dataclass(frozen=True)
class Batch:
    ids: tuple[str, ...]
    estimated_tokens: int


class BatchPlanner:
    def __init__(self, settings: BatchingSettings) -> None:
        self.settings = settings

    @property
    def usable_tokens(self) -> int:
        return self.settings.context_window_tokens - self.settings.safety_margin_tokens

    def field_parameters(
        self,
        *,
        shared_context: Any,
        field_ids: Sequence[str],
    ) -> list[Batch]:
        base = estimate_tokens(shared_context)
        per_item = self.settings.estimated_field_parameter_tokens
        if base + per_item > self.usable_tokens:
            raise ContextCapacityError(
                "the complete compact field index exceeds the safe model context"
            )
        context_capacity = max(1, (self.usable_tokens - base) // per_item)
        capacity = min(
            context_capacity,
            self.settings.max_field_parameters_per_batch,
        )
        return [
            Batch(tuple(field_ids[index:index + capacity]), base + per_item * len(field_ids[index:index + capacity]))
            for index in range(0, len(field_ids), capacity)
        ]
