"""Versioned contracts emitted by the benchmark runtime."""

from __future__ import annotations

STATIC_TRACE_SCHEMA_VERSION = "adaexplorebench_static_trace_v5"
DYNAMIC_TRACE_SCHEMA_VERSION = "adaexplorebench_dynamic_trace_v5"
BRANCHED_TRACE_SCHEMA_VERSION = "adaexplorebench_branched_trace_v7"
RUN_MANIFEST_SCHEMA_VERSION = "adaexplorebench_run_manifest_v7"

SEQUENTIAL_TRACE_SCHEMA_VERSIONS = frozenset(
    {
        "adaexplorebench_static_trace_v2",
        "adaexplorebench_dynamic_trace_v2",
        "adaexplorebench_static_trace_v3",
        "adaexplorebench_dynamic_trace_v3",
        "adaexplorebench_static_trace_v4",
        "adaexplorebench_dynamic_trace_v4",
        STATIC_TRACE_SCHEMA_VERSION,
        DYNAMIC_TRACE_SCHEMA_VERSION,
    }
)


def sequential_trace_schema_version(dataset_mode: str) -> str:
    if dataset_mode == "static":
        return STATIC_TRACE_SCHEMA_VERSION
    if dataset_mode == "dynamic":
        return DYNAMIC_TRACE_SCHEMA_VERSION
    raise ValueError(f"unsupported dataset mode: {dataset_mode}")
