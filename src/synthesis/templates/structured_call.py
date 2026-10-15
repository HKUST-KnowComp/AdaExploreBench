from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Awaitable, Callable

from .contracts import ContractError, Contracts, provider_schema


RecordCall = Callable[[dict[str, Any]], Awaitable[None]]
RecordError = Callable[[dict[str, Any]], Awaitable[None]]


class ProtocolResponseError(RuntimeError):
    """The provider responded, but exhausted bounded structured-protocol retries."""


@dataclass(frozen=True)
class CallSpec:
    stage: str
    component_id: str
    schema: str
    system: str
    user: str
    mode: str = "generation"
    enum_bindings: dict[str, tuple[Any, ...]] = field(default_factory=dict)


class StructuredCaller:
    def __init__(
        self,
        *,
        generation_client: Any,
        audit_client: Any,
        repair_client: Any,
        contracts: Contracts,
        concurrency: dict[str, int],
        parse_retry_limit: int,
        record_call: RecordCall | None = None,
        record_error: RecordError | None = None,
    ) -> None:
        self.clients = {
            "generation": generation_client,
            "audit": audit_client,
            "repair": repair_client,
        }
        self.contracts = contracts
        self.semaphores = {
            mode: asyncio.Semaphore(limit) for mode, limit in concurrency.items()
        }
        self.parse_retry_limit = parse_retry_limit
        self.record_call = record_call
        self.record_error = record_error

    async def call(self, spec: CallSpec) -> dict[str, Any]:
        if spec.mode not in self.clients:
            raise ValueError(f"unknown call mode {spec.mode}")
        messages = [
            {"role": "system", "content": spec.system},
            {"role": "user", "content": spec.user},
        ]
        bound_schema = self.contracts.bound_schema(spec.schema, spec.enum_bindings)
        schema = provider_schema(bound_schema)
        last_error: Exception | None = None
        async with self.semaphores[spec.mode]:
            for attempt in range(self.parse_retry_limit + 1):
                started = monotonic()
                try:
                    response = await self.clients[spec.mode].generate(
                        messages=messages,
                        schema=schema,
                        trace_context={
                            "stage": spec.stage,
                            "component_id": spec.component_id,
                            "mode": spec.mode,
                            "attempt": attempt + 1,
                        },
                    )
                except Exception as exc:
                    if self.record_error is not None:
                        await self.record_error({
                            "stage": spec.stage,
                            "component_id": spec.component_id,
                            "mode": spec.mode,
                            "attempt": attempt + 1,
                            "provider_failure": True,
                            "duration_ms": int((monotonic() - started) * 1000),
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                        })
                    raise
                try:
                    value = json.loads(response.content)
                    if not isinstance(value, dict):
                        raise ValueError("structured response must be an object")
                    self.contracts.validate(spec.schema, value)
                except (json.JSONDecodeError, ValueError, ContractError) as exc:
                    last_error = exc
                    await self._record(spec, response, attempt, started, error=str(exc))
                    if str(response.finish_reason).lower() in {
                        "length", "max_tokens", "max_output_tokens",
                    }:
                        raise ProtocolResponseError(
                            f"{spec.stage}/{spec.component_id} exhausted its completion "
                            "budget before producing a valid structured object"
                        ) from exc
                    messages.extend([
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": f"The response failed its strict contract: {exc}. Return a complete corrected object only."},
                    ])
                    continue
                await self._record(spec, response, attempt, started, error=None)
                return value
        raise ProtocolResponseError(
            f"{spec.stage}/{spec.component_id} failed structured parsing"
        ) from last_error

    async def _record(
        self, spec: CallSpec, response: Any, attempt: int, started: float, error: str | None
    ) -> None:
        if self.record_call is None:
            return
        await self.record_call({
            "stage": spec.stage,
            "component_id": spec.component_id,
            "mode": spec.mode,
            "attempt": attempt + 1,
            "model": response.model,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "duration_ms": int((monotonic() - started) * 1000),
            "error": error,
            "prompt_sha256": hashlib.sha256(
                json.dumps(
                    {"system": spec.system, "user": spec.user},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "schema_sha256": hashlib.sha256(
                json.dumps(
                    provider_schema(
                        self.contracts.bound_schema(spec.schema, spec.enum_bindings)
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "response_content": response.content,
        })
