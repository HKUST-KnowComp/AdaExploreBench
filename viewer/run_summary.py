"""Run summary computation and caches shared by precompute and publication."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from runtime.metrics import aggregate_metrics as v3_aggregate_metrics
from runtime.metrics import aggregate_episode_metrics
from runtime.metrics import episode_metrics as v3_episode_metrics
from runtime.metrics import first_learn_rule_comparison
from runtime.metrics import partial_credit as v3_partial_credit
from runtime.probe_geometry import probe_phase_point_sets
from runtime.schema import BRANCHED_TRACE_SCHEMA_VERSION, SEQUENTIAL_TRACE_SCHEMA_VERSIONS
from synthesis.datasets.contracts import validate_episode

REPO_ROOT = Path(__file__).resolve().parents[1]


RELEASE_VERSION = "6.0.0"

SUPPORTED_BRANCHED_TRACE_SCHEMA_VERSIONS = {
    "adaexplorebench_branched_trace_v3",
    "adaexplorebench_branched_trace_v4",
    "adaexplorebench_branched_trace_v5",
    "adaexplorebench_branched_trace_v6",
    BRANCHED_TRACE_SCHEMA_VERSION,
}


def _is_dynamic_episode(episode: dict[str, Any]) -> bool:
    return (
        episode.get("dataset_mode") == "dynamic"
        and episode.get("schema_version") in {
            "adaexplorebench_episode_v5",
            "adaexplorebench_episode_v6",
            "adaexplorebench_episode_v7",
            "adaexplorebench_episode_v8",
        }
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def load_episodes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _canonical_sampling_branch(name: str) -> str:
    """Return the stable viewer name for a sampling branch."""

    return (
        "flip_50_eval_case_given"
        if name == "flip_50_full_history"
        else name
    )


def _load_validated_dataset(dataset: Path) -> tuple[list[dict[str, Any]], str]:
    episodes = load_episodes(dataset)
    if not episodes or not _is_dynamic_episode(episodes[0]):
        raise ValueError("viewer data generation requires integrated dynamic episodes")
    for episode in episodes:
        validate_episode(episode, "dynamic")
    return episodes, hashlib.sha256(dataset.read_bytes()).hexdigest()

RUN_SUMMARY_CACHE_NAME = "viewer_summary.json"

RUN_SUMMARY_CACHE_VERSION = 8

_RUN_METADATA_FILES = {"manifest.json", "metrics.json", RUN_SUMMARY_CACHE_NAME}


def _run_artifact_paths(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.glob("*.json"))
        if not path.name.startswith(".")
        and path.name not in _RUN_METADATA_FILES
        and ".interrupted-" not in path.name
    ]


def _log(message: str) -> None:
    print(f"[viewer] {message}", flush=True)


def _load_run_artifacts(
    run_dir: Path, episode_ids: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Load supported branched artifacts, grouped by branch and by episode."""

    artifacts_by_branch: dict[str, list[dict[str, Any]]] = {}
    artifacts_by_episode: dict[str, dict[str, Any]] = {}
    paths = _run_artifact_paths(run_dir)
    _log(f"{run_dir.name}: loading {len(paths)} episode artifacts")
    for index, path in enumerate(paths, start=1):
        if index % 25 == 0 or index == len(paths):
            _log(f"{run_dir.name}: loaded {index}/{len(paths)} artifacts")
        artifact = _load_json(path)
        episode_id = str(artifact.get("episode_id") or "")
        if episode_id not in episode_ids:
            continue
        if (
            artifact.get("trace_schema_version")
            not in SUPPORTED_BRANCHED_TRACE_SCHEMA_VERSIONS
        ):
            continue
        artifacts_by_episode[episode_id] = artifact
        for name, branch in (artifact.get("branches") or {}).items():
            canonical_name = _canonical_sampling_branch(str(name))
            artifacts_by_branch.setdefault(canonical_name, []).append(branch)
    return artifacts_by_branch, artifacts_by_episode


def _load_run_summary_cache(
    run_dir: Path, dataset_sha256: str
) -> dict[str, Any] | None:
    cache = _load_json(run_dir / RUN_SUMMARY_CACHE_NAME)
    if cache.get("cache_version") != RUN_SUMMARY_CACHE_VERSION:
        return None
    if cache.get("viewer_release") != RELEASE_VERSION:
        return None
    if cache.get("dataset_sha256") != dataset_sha256:
        return None
    artifact_paths = _run_artifact_paths(run_dir)
    if cache.get("artifact_count") != len(artifact_paths):
        return None
    latest_mtime = max(
        (path.stat().st_mtime_ns for path in artifact_paths), default=0
    )
    if cache.get("artifact_mtime_ns") != latest_mtime:
        return None
    summary = cache.get("run_summary")
    return summary if isinstance(summary, dict) else None


def _write_run_summary_cache(
    run_dir: Path, dataset_sha256: str, run_summary: dict[str, Any]
) -> None:
    artifact_paths = _run_artifact_paths(run_dir)
    _write_json(run_dir / RUN_SUMMARY_CACHE_NAME, {
        "cache_version": RUN_SUMMARY_CACHE_VERSION,
        "viewer_release": RELEASE_VERSION,
        "dataset_sha256": dataset_sha256,
        "artifact_count": len(artifact_paths),
        "artifact_mtime_ns": max(
            (path.stat().st_mtime_ns for path in artifact_paths), default=0
        ),
        "run_summary": run_summary,
    })

_EFFORTS = ("high", "medium", "low", "minimal", "default")


def _effort_suffix(lower: str) -> str:
    for effort in _EFFORTS:
        if effort in lower:
            return f" {effort.title()}"
    return ""


def _date_suffix(name: str) -> str:
    matches = re.findall(r"(20\d{6})", name)
    return f" {matches[-1][4:]}" if matches else ""


def _run_suffix(name: str) -> str:
    match = re.search(r"-run(\d+)$", name)
    return f" r{match.group(1)}" if match else ""


def _model_display(slug: str) -> str:
    special = {"gpt": "GPT", "deepseek": "DeepSeek"}
    parts = []
    for part in slug.split("-"):
        if part in special:
            parts.append(special[part])
        elif re.fullmatch(r"v\d+(\.\d+)?", part) or re.fullmatch(r"\d+(\.\d+)?b", part):
            parts.append(part.upper())
        else:
            parts.append(part.capitalize())
    if len(parts) > 1 and parts[0] == "GPT" and re.fullmatch(r"\d+(\.\d+)?", parts[1]):
        return "-".join(parts[:2]) + "".join(f" {p}" for p in parts[2:])
    return " ".join(parts)


def _structured_run_label(name: str) -> tuple[str, str] | None:
    # Current convention: <model>-<effort>-<runconfig>-<dataset>-<yyyymmdd>[-runN]
    # (hyphen-separated; legacy underscore names fall through to the substring chain)
    if "_" in name:
        return None
    parts = name.lower().split("-")
    for index, part in enumerate(parts):
        if part in _EFFORTS and index > 0:
            slug = "-".join(parts[:index])
            return (
                f"{_model_display(slug)} {part.title()}{_date_suffix(name)}{_run_suffix(name)}",
                slug.split("-")[0],
            )
    return None


def _run_label(name: str) -> tuple[str, str]:
    structured = _structured_run_label(name)
    if structured is not None:
        return structured
    lower = name.lower()
    date = _date_suffix(name)
    if "deepseek" in lower or "ds_v4" in lower:
        version = (
            "DeepSeek V4.1"
            if "v4.1" in lower or "v4_1" in lower
            else "DeepSeek V4"
        )
        return (
            (f"{version} Flash" if "flash" in lower else f"{version} Pro")
            + _effort_suffix(lower)
            + date,
            "deepseek",
        )
    if "gemini" in lower:
        # Run directories historically used both ``gemini_3_7`` and the
        # compact ``gemini37`` spelling. Keep both forms mapped explicitly so
        # an unrecognized name cannot silently fall back to the 3.5 label.
        if "3_8" in lower or "3.8" in lower or "gemini38" in lower:
            version = "3.8"
        elif "3_7" in lower or "3.7" in lower or "gemini37" in lower:
            version = "3.7"
        elif "3_6" in lower or "3.6" in lower:
            version = "3.6"
        else:
            version = "3.5"
        return (
            f"Gemini {version} Flash{_effort_suffix(lower)}{date}",
            "gemini",
        )
    if (
        "gpt_5_6_terra" in lower
        or "gpt-5.6-terra" in lower
        or "gpt56_terra" in lower
    ):
        return f"GPT-5.6 Terra{_effort_suffix(lower)}{date}", "gpt"
    if (
        "gpt_5_6_luna" in lower
        or "gpt-5.6-luna" in lower
        or "gpt56_luna" in lower
    ):
        return f"GPT-5.6 Luna{_effort_suffix(lower)}{date}", "gpt"
    if "oracle" in lower:
        return "Oracle", "oracle"
    if "random" in lower:
        return "Random", "random"
    return name.replace("_", " ").title(), name.lower()


def _run_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(
        str((manifest.get("run_config") or {}).get("path") or "")
    )
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if yaml is None or not config_path.is_file():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}

SECRET_CONFIG_KEYS = {
    "api_key_env",
    "vertex_project_env",
    "vertex_location_env",
}

MODEL_CONFIG_HYPERPARAMETERS = (
    "model",
    "provider",
    "conversation_protocol",
    "base_url",
)

AGENT_CONFIG_HYPERPARAMETERS = (
    "temperature",
    "max_tokens",
    "output_format",
    "effort",
    "request_timeout_seconds",
    "request_retry_limit",
)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _run_hyperparameters(
    run_dir: Path, artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Flat, secret-free hyperparameter summary for one benchmark run."""
    manifest = _load_json(run_dir / "manifest.json")
    config = _run_config(manifest)
    hyperparameters: dict[str, Any] = {}
    model_path_value = (
        (manifest.get("model_config") or {}).get("path")
        or config.get("model_config")
        or ""
    )
    model_path = Path(str(model_path_value))
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.is_file():
        model_path = run_dir / "config_snapshot" / "model_config.yaml"
    model_config = (
        yaml.safe_load(model_path.read_text()) or {}
        if yaml is not None and model_path.is_file()
        else {}
    )
    for key in MODEL_CONFIG_HYPERPARAMETERS:
        value = model_config.get(key)
        if value is not None and key not in SECRET_CONFIG_KEYS:
            hyperparameters[key] = value
    agent = config.get("agent") or {}
    for key in AGENT_CONFIG_HYPERPARAMETERS:
        value = model_config.get(key)
        if value is None:
            value = agent.get(key)
        if value is not None:
            hyperparameters[key] = value
    interaction_policy = manifest.get("interaction_policy")
    if isinstance(interaction_policy, dict):
        for key, value in interaction_policy.items():
            if key not in SECRET_CONFIG_KEYS and _is_scalar(value):
                hyperparameters[key] = value
    if manifest.get("policy") is not None:
        hyperparameters["policy"] = manifest["policy"]
    identity = manifest.get("run_identity")
    identity = identity if isinstance(identity, dict) else {}
    seed = manifest.get("seed", identity.get("seed"))
    if seed is not None:
        hyperparameters["seed"] = seed
    branches = manifest.get("branches") or identity.get("branches")
    if isinstance(branches, dict):
        enabled = branches.get("enabled") or []
        if enabled:
            hyperparameters["branches"] = ", ".join(str(name) for name in enabled)
        if branches.get("matched_no_change_control") is not None:
            hyperparameters["matched_no_change_control"] = bool(
                branches["matched_no_change_control"]
            )
    if artifacts:
        trace = artifacts[0].get("tool_trace") or []
        if "model" not in hyperparameters:
            model = next(
                (str(event["model"]) for event in trace if event.get("model")),
                None,
            )
            if model:
                hyperparameters["model"] = model
        task_spec = (artifacts[0].get("conversation") or {}).get("task_spec") or {}
        protocol = task_spec.get("interaction_protocol")
        if isinstance(protocol, dict):
            for key, value in protocol.items():
                if (
                    key not in hyperparameters
                    and key not in SECRET_CONFIG_KEYS
                    and _is_scalar(value)
                ):
                    hyperparameters[key] = value
    return hyperparameters


def _run_reasoning_metadata(
    run_dir: Path, artifacts: list[dict[str, Any]], model_key: str
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "manifest.json")
    config = _run_config(manifest)
    model_path_value = (
        (manifest.get("model_config") or {}).get("path")
        or config.get("model_config")
        or ""
    )
    model_path = Path(str(model_path_value))
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.is_file():
        model_path = run_dir / "config_snapshot" / "model_config.yaml"
    model_config = (
        yaml.safe_load(model_path.read_text()) or {}
        if yaml is not None and model_path.is_file()
        else {}
    )
    effort = str(
        model_config.get("effort")
        or (config.get("agent") or {}).get("effort")
        or "unspecified"
    )
    events = [
        event
        for artifact in artifacts
        for event in artifact.get("tool_trace") or []
    ]
    signatures = sum(
        bool(event.get("reasoning_signature_present")) for event in events
    )
    summaries = sum(
        bool(str(event.get("reasoning_content") or "").strip())
        for event in events
    )
    return {
        "parameter": (
            "thinking_level" if model_key == "gemini" else "reasoning_effort"
        ),
        "value": effort,
        "thought_signature_calls": signatures,
        "reasoning_summary_calls": summaries,
        "total_calls": len(events),
    }


def _aggregate_action_batch_accuracy(
    artifacts: list[dict[str, Any]],
    batch_count: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    expected_cases = len(artifacts) * batch_size
    rows = []
    for batch_number in range(1, batch_count + 1):
        results = [
            result
            for artifact in artifacts
            for result in artifact.get("action_results") or []
            if int(result.get("batch_number") or 0) == batch_number
        ]
        correct_cases = sum(bool(result.get("correct")) for result in results)
        answered_cases = sum(not bool(result.get("unanswered")) for result in results)
        rows.append({
            "batch_number": batch_number,
            "accuracy": (
                correct_cases / expected_cases if expected_cases else None
            ),
            "correct_cases": correct_cases,
            "answered_cases": answered_cases,
            "expected_cases": expected_cases,
        })
    return rows


def _aggregate_dynamic_batches(
    artifacts: list[dict[str, Any]],
    batch_count: int,
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for stratum in ("all", "iid", "boundary", "family_diagnostic"):
        rows = []
        for batch_number in range(1, batch_count + 1):
            cases = [
                result
                for artifact in artifacts
                for result in artifact.get("action_results") or []
                if int(result.get("batch_number") or 0) == batch_number
                and (stratum == "all" or result.get("stratum") == stratum)
            ]
            answered = [row for row in cases if not row.get("unanswered")]
            post_answered = [
                row for row in answered
                if row.get("phase") == "post_dynamic"
            ]
            post_cases = [
                row for row in cases
                if row.get("phase") == "post_dynamic"
            ]
            # Keep abstentions in the adaptation denominator; an unanswered
            # post-rule case is a failed adoption/stability decision.
            flipped = [row for row in post_cases if row.get("rule_flip")]
            nonflipped = [row for row in post_cases if not row.get("rule_flip")]
            new_count = sum(
                not row.get("unanswered")
                and row.get("action_id")
                == (row.get("oracle") or {}).get("optimal_action_id")
                for row in flipped
            )
            stale_count = sum(
                row.get("action_id")
                == row.get("stale_rule_optimal_action_id")
                for row in flipped
            )
            batch_partial_credit, batch_partial_credit_baseline = v3_partial_credit(cases)
            rows.append({
                "batch_number": batch_number,
                "case_count": len(cases),
                "answered_cases": len(answered),
                "active_rule_accuracy": (
                    sum(bool(row.get("correct")) for row in cases) / len(cases)
                    if cases else None
                ),
                "partial_credit_accuracy": batch_partial_credit,
                "partial_credit_random_baseline": batch_partial_credit_baseline,
                "old_rule_alignment": (
                    sum(
                        row.get("action_id")
                        == row.get("stale_rule_optimal_action_id")
                        for row in post_answered
                    ) / len(post_answered)
                    if post_answered else None
                ),
                "stale_policy_accuracy": (
                    sum(
                        row.get("stale_rule_optimal_action_id")
                        == (row.get("oracle") or {}).get("optimal_action_id")
                        for row in post_cases
                    ) / len(post_cases)
                    if post_cases else None
                ),
                "flip_case_count": len(flipped),
                "nonflip_case_count": len(nonflipped),
                "flip_adoption_accuracy": (
                    new_count / len(flipped) if flipped else None
                ),
                "nonflip_stability_accuracy": (
                    sum(
                        not row.get("unanswered")
                        and row.get("action_id")
                        == (row.get("oracle") or {}).get("optimal_action_id")
                        for row in nonflipped
                    ) / len(nonflipped)
                    if nonflipped else None
                ),
                "flipped_case_breakdown": {
                    "denominator": len(flipped),
                    "new_rule_adoption": new_count,
                    "stale_rule_adherence": stale_count,
                    "neither_rule": len(flipped) - new_count - stale_count,
                },
            })
        output[stratum] = rows
    return output


def _aggregate_adaptation_curve(
    artifacts: list[dict[str, Any]],
    metrics_by_artifact: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate post-rule behavior by relative batch after each episode's change."""
    by_relative_batch: dict[int, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        if artifact.get("condition") != "dynamic":
            continue
        metrics = (
            metrics_by_artifact[id(artifact)]
            if metrics_by_artifact is not None else v3_episode_metrics(artifact)
        )
        fork = int(metrics.get("after_action_batch") or 0)
        for row in metrics.get("batch_metrics") or []:
            batch_number = int(row.get("batch_number") or 0)
            if batch_number <= fork:
                continue
            relative_batch = batch_number - fork
            if row.get("flip_adoption_accuracy") is None and row.get("nonflip_stability_accuracy") is None:
                continue
            by_relative_batch.setdefault(relative_batch, []).append(row)
    return [
        {
            "batch_number": relative_batch,
            "flip_case_count": sum(int(row.get("flip_case_count") or 0) for row in rows),
            "nonflip_case_count": sum(int(row.get("nonflip_case_count") or 0) for row in rows),
            "flip_adoption_accuracy": (
                mean(float(row["flip_adoption_accuracy"]) for row in rows if row.get("flip_adoption_accuracy") is not None)
                if any(row.get("flip_adoption_accuracy") is not None for row in rows) else None
            ),
            "nonflip_stability_accuracy": (
                mean(float(row["nonflip_stability_accuracy"]) for row in rows if row.get("nonflip_stability_accuracy") is not None)
                if any(row.get("nonflip_stability_accuracy") is not None for row in rows) else None
            ),
            "flipped_case_breakdown": {
                "denominator": sum(int((row.get("flipped_case_breakdown") or {}).get("denominator") or 0) for row in rows),
                "new_rule_adoption": sum(int((row.get("flipped_case_breakdown") or {}).get("new_rule_adoption") or 0) for row in rows),
                "stale_rule_adherence": sum(int((row.get("flipped_case_breakdown") or {}).get("stale_rule_adherence") or 0) for row in rows),
                "neither_rule": sum(int((row.get("flipped_case_breakdown") or {}).get("neither_rule") or 0) for row in rows),
            },
        }
        for relative_batch, rows in sorted(by_relative_batch.items())
    ]


def _aggregate_probe_allocation(
    artifacts: list[dict[str, Any]], batch_count: int,
    metrics_by_artifact: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Macro-averaged probe allocation across the action-batch calendar.

    Each per-batch row pairs the probe cases queried in the round preceding
    that batch with cumulative action utility before that batch. Pre/post-
    dynamic totals split at each artifact's own change point.
    """
    cases_by_batch: dict[int, list[float]] = {
        number: [] for number in range(1, batch_count + 1)
    }
    action_utility_by_batch: dict[int, list[float]] = {
        number: [] for number in range(1, batch_count + 1)
    }
    pre_dynamic: list[float] = []
    post_dynamic: list[float] = []
    for artifact in artifacts:
        if artifact.get("trace_schema_version") not in SEQUENTIAL_TRACE_SCHEMA_VERSIONS:
            continue
        metrics = (
            metrics_by_artifact[id(artifact)]
            if metrics_by_artifact is not None else v3_episode_metrics(artifact)
        )
        probe_cases = {
            int(row["batch_number"]): float(row.get("probe_cases") or 0.0)
            for row in (metrics.get("probe_metrics") or {}).get("by_batch") or []
            if row.get("batch_number") is not None
        }
        batch_metrics = {
            int(row["batch_number"]): row
            for row in metrics.get("batch_metrics") or []
            if row.get("batch_number") is not None
        }
        if not probe_cases or not batch_metrics:
            continue
        fork = int(metrics.get("after_action_batch") or 0)
        pre_dynamic.append(sum(
            value for number, value in probe_cases.items() if 1 <= number <= fork
        ))
        post_dynamic.append(sum(
            value for number, value in probe_cases.items() if number > fork
        ))
        for batch_number, probe_case_count in probe_cases.items():
            if not 1 <= batch_number <= batch_count:
                continue
            cases_by_batch[batch_number].append(probe_case_count)
            previous = batch_metrics.get(batch_number - 1)
            action_utility_by_batch[batch_number].append(
                0.0
                if previous is None
                else float(
                    previous.get("action_utility")
                    if previous.get("action_utility") is not None
                    else previous.get("cumulative_utility") or 0.0
                )
            )
    return {
        "probe_by_batch": [
            {
                "batch_number": batch_number,
                "probe_cases": (
                    round(mean(cases_by_batch[batch_number]), 2)
                    if cases_by_batch[batch_number] else None
                ),
                "action_utility_on_hand": (
                    round(mean(action_utility_by_batch[batch_number]), 2)
                    if action_utility_by_batch[batch_number] else None
                ),
            }
            for batch_number in range(1, batch_count + 1)
        ],
        "probe_pre_dynamic_cases": (
            round(mean(pre_dynamic), 2) if pre_dynamic else None
        ),
        "probe_post_dynamic_cases": (
            round(mean(post_dynamic), 2) if post_dynamic else None
        ),
    }


def _aggregate_static_batches(
    artifacts: list[dict[str, Any]], batch_count: int
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for stratum in ("all", "iid", "boundary", "family_diagnostic"):
        rows = []
        for batch_number in range(1, batch_count + 1):
            cases = [
                result
                for artifact in artifacts
                for result in artifact.get("action_results") or []
                if int(result.get("batch_number") or 0) == batch_number
                and (stratum == "all" or result.get("stratum") == stratum)
            ]
            rows.append({
                "batch_number": batch_number,
                "static_accuracy": (
                    sum(bool(row.get("correct")) for row in cases) / len(cases)
                    if cases else None
                ),
                "static_answered_cases": sum(
                    not bool(row.get("unanswered")) for row in cases
                ),
            })
        output[stratum] = rows
    return output


def _completed_relevant_field_response(
    artifact: dict[str, Any], phase: str
) -> dict[str, str] | None:
    assessments = artifact.get("relevant_fields_assessments") or {}
    assessment = assessments.get(phase)
    if not isinstance(assessment, dict) or assessment.get("status") != "completed":
        return None
    response = assessment.get("response") or {}
    fields = response.get("relevant_fields")
    if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
        return None
    directions = response.get("field_directions") or []
    direction_map = {
        str(item.get("field_id")).strip(): str(item.get("direction")).strip().lower()
        for item in directions if isinstance(item, dict) and item.get("field_id")
    }
    return {field.strip(): direction_map.get(field.strip(), "unknown") for field in fields if field.strip()}


def _field_set_scores(predicted: set[str], expected: set[str]) -> dict[str, float]:
    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _pool_field_set(
    pool: dict[str, int], predicted: set[str], expected: set[str]
) -> None:
    """Add one assessment's TP/FP/FN contribution to a micro pool."""

    pool["tp"] += len(predicted & expected)
    pool["fp"] += len(predicted - expected)
    pool["fn"] += len(expected - predicted)


def _micro_pool_f1(pool: dict[str, int]) -> float | None:
    """Micro F1 from pooled TP/FP/FN counts.

    A pool without expected fields is undefined; a pool whose assessments
    reported nothing scores 0 even though its precision is undefined.
    """

    true_positives = pool["tp"]
    predicted_total = true_positives + pool["fp"]
    expected_total = true_positives + pool["fn"]
    if expected_total == 0:
        return None
    if predicted_total == 0:
        return 0.0
    precision = true_positives / predicted_total
    recall = true_positives / expected_total
    return round(2 * precision * recall / (precision + recall), 6)


def _aggregate_relevant_fields_assessments(
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Score reported relevant-field sets against ground truth with micro pools.

    Dynamic artifacts pool the TP/FP/FN of every completed ``pre_change``
    assessment against the pre-rule relevant fields and of every completed
    ``post_change`` assessment against the post-rule fields; the headline
    ``factor_recognition`` is the mean of the defined phase F1s.  Static
    artifacts pool completed ``final`` assessments against the pre-rule
    fields.  An assessment reporting an empty set contributes only its false
    negatives, and failed or missing assessments never enter a pool.  The
    paired per-episode atom F1 over both assessments is reported as
    ``change_recognition`` together with its eligibility rate; both are
    dynamic-only.
    """

    pools: dict[str, dict[str, int]] = {
        "pre_change": {"tp": 0, "fp": 0, "fn": 0},
        "post_change": {"tp": 0, "fp": 0, "fn": 0},
        "final": {"tp": 0, "fp": 0, "fn": 0},
    }
    pool_episode_counts = {"pre_change": 0, "post_change": 0, "final": 0}
    dynamic_episode_count = 0
    eligible_episode_count = 0
    paired_episode_count = 0
    changed_episode_count = 0
    changed_outcomes = {
        "both_correct": 0,
        "pre_correct_final_incorrect": 0,
        "pre_incorrect_final_correct": 0,
        "both_incorrect": 0,
    }
    unchanged_outcomes = {
        "both_correct": 0,
        "pre_correct_final_incorrect": 0,
        "pre_incorrect_final_correct": 0,
        "both_incorrect": 0,
    }
    change_scores: list[float] = []
    change_precisions: list[float] = []
    change_recalls: list[float] = []
    change_atom_detection = {
        "field_replacement_atoms_detected": 0,
        "field_replacement_atoms_total": 0,
        "direction_reversal_atoms_detected": 0,
        "direction_reversal_atoms_total": 0,
    }
    change_detection_outcomes = {
        "both_detected": 0,
        "field_replacement_only": 0,
        "direction_reversal_only": 0,
        "neither_detected": 0,
        "episode_count": 0,
    }

    for artifact in artifacts:
        transition = ((artifact.get("evaluation") or {}).get("transition") or {})
        pre_rule = transition.get("pre_rule") or {}
        post_rule = transition.get("post_rule") or {}
        pre_expected = {
            str(field).strip()
            for field in pre_rule.get("relevant_fields") or []
            if str(field).strip()
        }
        post_expected = {
            str(field).strip()
            for field in post_rule.get("relevant_fields") or []
            if str(field).strip()
        }
        is_dynamic = str(artifact.get("condition") or "dynamic") == "dynamic"

        pre_predicted = _completed_relevant_field_response(
            artifact, "pre_change"
        )
        post_predicted = _completed_relevant_field_response(
            artifact, "post_change"
        )
        if is_dynamic:
            dynamic_episode_count += 1
            if pre_predicted is not None and pre_expected:
                _pool_field_set(
                    pools["pre_change"], set(pre_predicted), pre_expected
                )
                pool_episode_counts["pre_change"] += 1
            if post_predicted is not None and post_expected:
                _pool_field_set(
                    pools["post_change"], set(post_predicted), post_expected
                )
                pool_episode_counts["post_change"] += 1
        else:
            final_predicted = _completed_relevant_field_response(
                artifact, "final"
            )
            if final_predicted is not None and pre_expected:
                _pool_field_set(
                    pools["final"], set(final_predicted), pre_expected
                )
                pool_episode_counts["final"] += 1
        if not is_dynamic:
            continue
        if (
            pre_predicted is None
            or post_predicted is None
            or not pre_expected
            or not post_expected
        ):
            continue

        eligible_episode_count += 1
        expected_pre = {field: "positive" for field in pre_expected}
        expected_post = {field: "positive" for field in post_expected}
        for rule, target in ((pre_rule, expected_pre), (post_rule, expected_post)):
            target.update({str(field): ("positive" if float((rule.get("field_directions") or {}).get(field, 1)) >= 0 else "negative") for field in target})
        true_change = {(field, expected_pre.get(field)) for field in set(expected_pre) - set(expected_post)} | {(field, expected_post.get(field)) for field in set(expected_post) - set(expected_pre)} | {(field, expected_pre.get(field), expected_post.get(field)) for field in set(expected_pre) & set(expected_post) if expected_pre.get(field) != expected_post.get(field)}
        predicted_change = {(field, pre_predicted.get(field)) for field in set(pre_predicted) - set(post_predicted)} | {(field, post_predicted.get(field)) for field in set(post_predicted) - set(pre_predicted)} | {(field, pre_predicted.get(field), post_predicted.get(field)) for field in set(pre_predicted) & set(post_predicted) if pre_predicted.get(field) != post_predicted.get(field)}
        true_change = {tuple(item) for item in true_change}
        predicted_change = {tuple(item) for item in predicted_change}
        tp = len(true_change & predicted_change)
        change_precision = tp / len(predicted_change) if predicted_change else 0.0
        change_recall = tp / len(true_change) if true_change else 1.0
        change_f1 = 2 * change_precision * change_recall / (change_precision + change_recall) if change_precision + change_recall else 0.0
        change_scores.append(change_f1)
        change_precisions.append(change_precision)
        change_recalls.append(change_recall)
        # Change atoms come in two shapes: 2-tuples for a field leaving or
        # entering the relevant set (field replacement) and 3-tuples for a
        # direction flip on a retained field (direction reversal).
        true_replacement = [item for item in true_change if len(item) == 2]
        true_reversal = [item for item in true_change if len(item) == 3]
        tp_atoms = true_change & predicted_change
        tp_replacement = sum(1 for item in tp_atoms if len(item) == 2)
        tp_reversal = sum(1 for item in tp_atoms if len(item) == 3)
        change_atom_detection["field_replacement_atoms_total"] += len(true_replacement)
        change_atom_detection["field_replacement_atoms_detected"] += tp_replacement
        change_atom_detection["direction_reversal_atoms_total"] += len(true_reversal)
        change_atom_detection["direction_reversal_atoms_detected"] += tp_reversal
        if true_replacement and true_reversal:
            # A change type counts as detected only when every atom of that
            # type matches; partial detection shows up in the atom counts.
            replacement_detected = tp_replacement == len(true_replacement)
            reversal_detected = tp_reversal == len(true_reversal)
            change_detection_outcomes["episode_count"] += 1
            if replacement_detected and reversal_detected:
                change_detection_outcomes["both_detected"] += 1
            elif replacement_detected:
                change_detection_outcomes["field_replacement_only"] += 1
            elif reversal_detected:
                change_detection_outcomes["direction_reversal_only"] += 1
            else:
                change_detection_outcomes["neither_detected"] += 1
        paired_episode_count += 1
        pre_correct = set(pre_predicted) == pre_expected
        final_correct = set(post_predicted) == post_expected
        outcome = (
            "both_correct"
            if pre_correct and final_correct
            else "pre_correct_final_incorrect"
            if pre_correct
            else "pre_incorrect_final_correct"
            if final_correct
            else "both_incorrect"
        )
        if set(pre_predicted) == set(post_predicted):
            unchanged_outcomes[outcome] += 1
            continue
        changed_episode_count += 1
        changed_outcomes[outcome] += 1

    if dynamic_episode_count == 0 and not any(pool_episode_counts.values()):
        return None

    def phase_summary(phase: str) -> dict[str, Any]:
        pool = pools[phase]
        predicted_total = pool["tp"] + pool["fp"]
        expected_total = pool["tp"] + pool["fn"]
        return {
            "episode_count": pool_episode_counts[phase],
            "precision": (
                pool["tp"] / predicted_total if predicted_total else None
            ),
            "recall": pool["tp"] / expected_total if expected_total else None,
            "f1": _micro_pool_f1(pool),
        }

    pre_change_f1 = _micro_pool_f1(pools["pre_change"])
    post_change_f1 = _micro_pool_f1(pools["post_change"])
    final_f1 = _micro_pool_f1(pools["final"])
    if pre_change_f1 is not None or post_change_f1 is not None:
        defined_phase_f1s = [
            f1 for f1 in (pre_change_f1, post_change_f1) if f1 is not None
        ]
        factor_recognition = (
            round(mean(defined_phase_f1s), 6) if defined_phase_f1s else None
        )
    else:
        factor_recognition = final_f1

    return {
        "pre_change": phase_summary("pre_change"),
        "post_change": phase_summary("post_change"),
        "final": phase_summary("final"),
        "paired_episode_count": paired_episode_count,
        "changed_episode_count": changed_episode_count,
        "unchanged_episode_count": paired_episode_count - changed_episode_count,
        "changed_outcomes": changed_outcomes,
        "unchanged_outcomes": unchanged_outcomes,
        "factor_recognition": factor_recognition,
        "factor_recognition_pre": pre_change_f1,
        "factor_recognition_post": post_change_f1,
        "factor_recognition_final": final_f1,
        "change_recognition": (
            round(mean(change_scores), 6) if change_scores else None
        ),
        "change_recognition_eligibility": (
            {
                "eligible": eligible_episode_count,
                "total": dynamic_episode_count,
                "rate": (
                    round(eligible_episode_count / dynamic_episode_count, 6)
                    if dynamic_episode_count
                    else None
                ),
            }
            if dynamic_episode_count
            else None
        ),
        "adaptation_episode_count": len(change_scores),
        "adaptation_precision": (
            mean(change_precisions) if change_precisions else None
        ),
        "adaptation_recall": mean(change_recalls) if change_recalls else None,
        "change_atom_detection": change_atom_detection,
        "change_detection_outcomes": change_detection_outcomes,
    }


def _run_summary_view(
    dynamic_artifacts: list[dict[str, Any]],
    static_artifacts: list[dict[str, Any]],
    *,
    batch_count: int,
    batch_size: int,
    episodes: dict[str, Any] | None = None,
    metrics_by_artifact: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics = (
        aggregate_episode_metrics([
            metrics_by_artifact[id(artifact)] for artifact in dynamic_artifacts
        ])
        if metrics_by_artifact is not None
        else v3_aggregate_metrics(dynamic_artifacts, episodes=episodes)
    )
    overall = dict(metrics.get("overall") or {})
    token_metrics = overall.get("token_metrics") or {}
    overall["total_tokens"] = token_metrics.get("total_tokens")
    overall["total_tokens_all_episodes"] = (
        (overall.get("token_totals") or {}).get("total_tokens")
    )
    overall["latency_seconds"] = (
        (token_metrics.get("latency_seconds") or {}).get("total")
    )
    dynamic_metrics = metrics.get("dynamic") or {}
    for key in (
        "flip_adoption_accuracy",
        "nonflip_stability_accuracy",
    ):
        overall[key] = dynamic_metrics.get(key)
    first_learn = first_learn_rule_comparison(
        [
            item for item in dynamic_artifacts
            if item.get("condition") == "dynamic"
        ],
        static_artifacts,
    )
    relevant_fields_summary = _aggregate_relevant_fields_assessments(dynamic_artifacts)
    overall.update({
        "first_learn_accuracy": first_learn["dynamic_post_rule_accuracy"],
        "first_learn_rule_accuracy": first_learn["static_first_learn_accuracy"],
        "first_learn_static_accuracy": first_learn["static_first_learn_accuracy"],
        "first_learn_accuracy_delta": first_learn["accuracy_delta"],
        "first_learn_comparison_batch_count": (
            round(
                mean(
                    float(item["comparison_batch_count"])
                    for item in first_learn["comparisons"]
                    if item.get("comparison_batch_count") is not None
                ),
                6,
            )
            if first_learn["comparisons"]
            else None
        ),
        "factor_recognition": (
            relevant_fields_summary.get("factor_recognition")
            if relevant_fields_summary else None
        ),
        "change_recognition": (
            relevant_fields_summary.get("change_recognition")
            if relevant_fields_summary else None
        ),
    })
    # Mean Unique Probes and Mean Re-probes after Change are recomputed here
    # from the branch artifacts and the validated dataset so runs published
    # before the episode-level probe fields existed still report them.
    phase_point_sets = [
        probe_phase_point_sets(
            artifact,
            (episodes or {}).get(str(artifact.get("episode_id") or "")),
        )
        for artifact in dynamic_artifacts
    ]
    unique_probe_values = [
        len(sets["pre"]) + len(sets["post"]) if sets["has_change"] else len(sets["pre"])
        for sets in phase_point_sets
    ]
    reprobe_values = [
        len(sets["post"] & sets["pre"]) if sets["has_change"] else None
        for sets in phase_point_sets
    ]
    overall["mean_unique_probes"] = (
        round(mean(unique_probe_values), 6) if unique_probe_values else None
    )
    overall["mean_reprobes_after_change"] = (
        round(mean(float(value) for value in reprobe_values if value is not None), 6)
        if any(value is not None for value in reprobe_values) else None
    )
    batch_dynamic = _aggregate_dynamic_batches(dynamic_artifacts, batch_count)
    adaptation_curve = _aggregate_adaptation_curve(dynamic_artifacts, metrics_by_artifact)
    if static_artifacts:
        static_batches = _aggregate_static_batches(static_artifacts, batch_count)
        for stratum, rows in batch_dynamic.items():
            for row, static_row in zip(rows, static_batches[stratum]):
                row.update(static_row)
    return {
        "episode_count": len(dynamic_artifacts),
        "metrics": overall,
        "relevant_fields": relevant_fields_summary,
        "first_learn_comparison": first_learn,
        "batch_accuracy": _aggregate_action_batch_accuracy(
            dynamic_artifacts, batch_count, batch_size
        ),
        "batch_dynamic": batch_dynamic,
        "adaptation_curve": adaptation_curve,
        **_aggregate_probe_allocation(dynamic_artifacts, batch_count, metrics_by_artifact),
        "breakdowns": {
            "by_family": metrics.get("by_family") or {},
            "by_domain": metrics.get("by_domain") or {},
        },
    }


def _aggregate_accuracy_breakdowns(
    artifacts_by_branch: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Aggregate family/domain accuracy over the intended four trajectories.

    Static contributes its full (pre-rule) trajectory. Dynamic branches
    contribute only post-rule cases; each branch is then equally weighted.
    """

    branch_names = ("static_iid", "flip_25", "flip_50", "flip_75")
    values: dict[str, dict[str, list[float]]] = {
        "by_family": defaultdict(lambda: []),
        "by_domain": defaultdict(lambda: []),
    }
    case_counts: dict[str, dict[str, int]] = {
        "by_family": defaultdict(int),
        "by_domain": defaultdict(int),
    }
    branch_counts: dict[str, dict[str, int]] = {
        "by_family": defaultdict(int),
        "by_domain": defaultdict(int),
    }
    for branch_name in branch_names:
        for artifact in artifacts_by_branch.get(branch_name, []):
            results = list(artifact.get("action_results") or [])
            if branch_name != "static_iid":
                results = [
                    row for row in results if row.get("phase") == "post_dynamic"
                ]
            if not results:
                continue
            metadata = {
                "by_family": str(
                    (artifact.get("evaluation") or {}).get("rule_family")
                    or "unknown"
                ),
                "by_domain": str(
                    (artifact.get("provenance") or {}).get("domain")
                    or "unknown"
                ),
            }
            accuracy = mean(
                1.0
                if not row.get("unanswered") and row.get("correct")
                else 0.0
                for row in results
            )
            for dimension, key in metadata.items():
                values[dimension][key].append(accuracy)
                case_counts[dimension][key] += len(results)
                branch_counts[dimension][key] += 1
    return {
        dimension: {
            key: {
                "action_accuracy": round(mean(branch_values), 6),
                "case_count": case_counts[dimension][key],
                "branch_count": branch_counts[dimension][key],
            }
            for key, branch_values in sorted(dimension_values.items())
        }
        for dimension, dimension_values in values.items()
    }


def _aggregate_utility_breakdowns(
    artifacts_by_branch: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Aggregate utility over static full and dynamic post-rule trajectories."""

    branch_names = ("static_iid", "flip_25", "flip_50", "flip_75")
    grouped: dict[str, dict[str, list[dict[str, float]]]] = {
        "by_family": defaultdict(lambda: []),
        "by_domain": defaultdict(lambda: []),
    }
    for branch_name in branch_names:
        for artifact in artifacts_by_branch.get(branch_name, []):
            results = list(artifact.get("action_results") or [])
            if branch_name != "static_iid":
                results = [
                    row for row in results if row.get("phase") == "post_dynamic"
                ]
            if not results:
                continue
            cumulative = sum(float(row.get("utility") or 0.0) for row in results)
            regret = sum(float(row.get("regret") or 0.0) for row in results)
            grouped_values = {
                "action_accuracy": mean(
                    1.0
                    if not row.get("unanswered") and row.get("correct")
                    else 0.0
                    for row in results
                ),
                "cumulative_utility": cumulative,
                "action_utility": cumulative,
                "cumulative_regret": regret,
                "case_count": float(len(results)),
            }
            metadata = {
                "by_family": str(
                    (artifact.get("evaluation") or {}).get("rule_family")
                    or "unknown"
                ),
                "by_domain": str(
                    (artifact.get("provenance") or {}).get("domain")
                    or "unknown"
                ),
            }
            for dimension, key in metadata.items():
                grouped[dimension][key].append(grouped_values)
    return {
        dimension: {
            key: {
                metric: round(mean(float(item[metric]) for item in rows), 6)
                for metric in (
                    "action_accuracy",
                    "cumulative_utility",
                    "action_utility",
                    "cumulative_regret",
                    "case_count",
                )
            }
            for key, rows in sorted(dimension_values.items())
        }
        for dimension, dimension_values in grouped.items()
    }


def _build_run_summary(
    run_dir: Path,
    run_manifest: dict[str, Any],
    artifacts_by_branch: dict[str, list[dict[str, Any]]],
    episode_ids: list[str],
    episodes_by_id: dict[str, dict[str, Any]],
    batch_count: int,
    batch_size: int,
) -> dict[str, Any]:
    """Aggregate one run's branched artifacts into the viewer run summary."""

    _log(f"{run_dir.name}: aggregating run summary")
    # Key by object identity: one episode has several distinct branch artifacts.
    # Keep this cache local to the build so changed runs never reuse stale rows.
    metrics_by_artifact = {
        id(artifact): v3_episode_metrics(
            artifact, episode=episodes_by_id.get(str(artifact.get("episode_id") or ""))
        )
        for artifacts in artifacts_by_branch.values()
        for artifact in artifacts
    }
    label, model_key = _run_label(run_dir.name)
    views = {}
    for scope in (
        "all",
        "field_replacement",
        "direction_reversal",
        "compound",
    ):
        scoped_by_branch = {
            name: [
                item
                for item in branch_artifacts
                if scope == "all"
                or str(
                    ((item.get("evaluation") or {}).get("transition") or {}).get(
                        "type"
                    )
                )
                == scope
            ]
            for name, branch_artifacts in artifacts_by_branch.items()
        }
        scoped_artifacts = [
            item
            for branch_artifacts in scoped_by_branch.values()
            for item in branch_artifacts
        ]
        view = _run_summary_view(
            scoped_artifacts,
            scoped_by_branch.get("static_iid", []),
            batch_count=batch_count,
            batch_size=batch_size,
            episodes=episodes_by_id,
            metrics_by_artifact=metrics_by_artifact,
        )
        view["accuracy_breakdowns"] = _aggregate_accuracy_breakdowns(
            scoped_by_branch
        )
        view["utility_breakdowns"] = _aggregate_utility_breakdowns(
            scoped_by_branch
        )
        view["branches"] = {
            name: _run_summary_view(
                branch_artifacts,
                scoped_by_branch.get("static_iid", []),
                batch_count=batch_count,
                batch_size=batch_size,
                episodes=episodes_by_id,
                metrics_by_artifact=metrics_by_artifact,
            )
            for name, branch_artifacts in scoped_by_branch.items()
        }
        views[scope] = view
    all_view = views["all"]
    all_artifacts = [
        item
        for branch_artifacts in artifacts_by_branch.values()
        for item in branch_artifacts
    ]
    return {
        "run_id": run_dir.name,
        "label": label,
        "model_key": model_key,
        "simulation_scenario": run_manifest.get("simulation_scenario"),
        "simulation_description": run_manifest.get("simulation_description"),
        "episode_ids": episode_ids,
        "metrics": all_view["metrics"],
        "batch_accuracy": all_view["batch_accuracy"],
        "batch_dynamic": all_view["batch_dynamic"],
        "paired_static": any(
            name.endswith("_no_change_control")
            for name in artifacts_by_branch
        ),
        "reasoning": _run_reasoning_metadata(run_dir, all_artifacts, model_key),
        "hyperparameters": _run_hyperparameters(run_dir, all_artifacts),
        "breakdowns": all_view["breakdowns"],
        "views": views,
    }


def precompute_summaries(dataset: Path, run_dirs: list[Path]) -> None:
    """Refresh summary caches independently of viewer payload generation."""
    episodes, dataset_sha256 = _load_validated_dataset(dataset.resolve())
    episodes_by_id = {str(ep["episode_id"]): ep for ep in episodes}
    protocol = episodes[0]["agent_task"]["interaction_protocol"]
    batch_count = int(protocol["action_batch_count"])
    batch_size = int(protocol["action_batch_size"])
    for run_dir in (path.resolve() for path in run_dirs):
        started = perf_counter()
        run_manifest = _load_json(run_dir / "manifest.json")
        run_dataset_sha = (
            (run_manifest.get("run_identity") or {}).get("dataset_sha256")
            or (run_manifest.get("dataset") or {}).get("sha256")
        )
        if run_dataset_sha != dataset_sha256:
            raise ValueError(
                f"benchmark run does not belong to the viewer dataset: {run_dir}"
            )
        artifacts_by_branch, artifacts_by_episode = _load_run_artifacts(
            run_dir, set(episodes_by_id)
        )
        loaded = perf_counter()
        run_summary = _build_run_summary(
            run_dir,
            run_manifest,
            artifacts_by_branch,
            sorted(artifacts_by_episode),
            episodes_by_id,
            batch_count,
            batch_size,
        )
        _write_run_summary_cache(run_dir, dataset_sha256, run_summary)
        aggregated = perf_counter()
        episode_count = len(artifacts_by_episode)
        # Release the previous run before parsing the next run's artifacts.
        del artifacts_by_branch, artifacts_by_episode, run_summary
        finished = perf_counter()
        print(
            f"Wrote {run_dir / RUN_SUMMARY_CACHE_NAME} "
            f"({episode_count} episodes; "
            f"load {loaded - started:.2f}s, "
            f"aggregate/write {aggregated - loaded:.2f}s, "
            f"cleanup {finished - aggregated:.2f}s, "
            f"total {finished - started:.2f}s)",
            flush=True,
        )
