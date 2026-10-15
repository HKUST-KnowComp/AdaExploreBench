from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar


T = TypeVar("T")


async def gather_jobs(jobs: Iterable[Callable[[], Awaitable[T]]]) -> list[T]:
    """Run an already dependency-ready layer; provider semaphores bound concurrency."""
    results = await asyncio.gather(*(job() for job in jobs), return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise ExceptionGroup("one or more parallel synthesis jobs failed", errors)
    return [result for result in results if not isinstance(result, BaseException)]
