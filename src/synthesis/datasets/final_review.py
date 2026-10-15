"""Independent final LLM scoring for complete dynamic datasets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from scripts._utils import build_generator, merge_model_config

from .contracts import validate_episode


SCHEMA_VERSION = "adaexplorebench_dataset_llm_review_v1"
DIMENSIONS = (
    "field_realism",
    "rule_change_realism",
    "distractor_quality",
)


@dataclass(frozen=True)
class FinalReviewConfig:
    path: Path
    model_config: Path
    generation: dict[str, Any]
    concurrency: int
    request_attempt_limit: int
    cache_directory: Path
    cache_enabled: bool
    output_filename: str


def _strict_mapping(value: Any, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must define exactly {sorted(keys)}")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    return next(
        (parent for parent in resolved.parents if (parent / "pyproject.toml").is_file()),
        Path.cwd(),
    )


def load_final_review_config(path: Path) -> FinalReviewConfig:
    resolved = path.resolve()
    root = _repository_root(resolved)
    value = _strict_mapping(
        yaml.safe_load(resolved.read_text()) or {},
        "final review config",
        {
            "model_config",
            "generation",
            "concurrency",
            "request_attempt_limit",
            "cache",
            "output_filename",
        },
    )
    cache = _strict_mapping(value["cache"], "final review cache", {"directory", "enabled"})
    if not isinstance(value["generation"], dict) or not isinstance(cache["enabled"], bool):
        raise ValueError("final review generation/cache configuration is invalid")
    model_config = (root / str(value["model_config"])).resolve()
    if not model_config.is_file():
        raise ValueError(f"final review model config does not exist: {model_config}")
    output_filename = str(value["output_filename"])
    if not output_filename or Path(output_filename).name != output_filename:
        raise ValueError("final review output_filename must be a plain filename")
    return FinalReviewConfig(
        resolved,
        model_config,
        dict(value["generation"]),
        _positive(value["concurrency"], "final review concurrency"),
        _positive(value["request_attempt_limit"], "final review request_attempt_limit"),
        (root / str(cache["directory"])).resolve(),
        cache["enabled"],
        output_filename,
    )


def episode_jsonl_bytes(episode: dict[str, Any]) -> bytes:
    return (json.dumps(episode, ensure_ascii=False) + "\n").encode()


def dataset_sha256(episodes: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode_jsonl_bytes(episode))
    return digest.hexdigest()


def load_dynamic_episodes(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            episode = json.loads(line)
            try:
                validate_episode(episode, "dynamic")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            episodes.append(episode)
    if not episodes:
        raise ValueError("final review requires at least one dynamic episode")
    return episodes


def _field_summary(field: dict[str, Any]) -> dict[str, Any]:
    return {
        key: field[key]
        for key in ("display_name", "description", "unit")
    }


def _rule_summary(rule: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "relevant_fields": list(rule["relevant_fields"]),
        "field_directions": dict(rule["field_directions"]),
    }
    if rule.get("priority_order"):
        summary["priority_order"] = list(rule["priority_order"])
    return summary


def build_group_payload(
    domain: str, rule_family: str, episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    templates: dict[str, dict[str, Any]] = {}
    rendered = []
    for episode in episodes:
        task = episode["agent_task"]
        transition = episode["evaluation"]["transition"]
        template_id = str(episode["provenance"]["template_id"])
        template = templates.setdefault(
            template_id,
            {
                "scenario": task["scenario_description"],
                "agent_role": task["agent_role"],
                "actions": task["actions"],
                "fields": {},
            },
        )
        template["fields"].update(
            {
                field_id: _field_summary(field)
                for field_id, field in task["fields"].items()
            }
        )
        pre_rule = transition["pre_rule"]
        post_rule = transition["post_rule"]
        core_fields = set(pre_rule["relevant_fields"]) | set(post_rule["relevant_fields"])
        rendered.append(
            {
                "episode_id": episode["episode_id"],
                "template_id": template_id,
                "dynamic_type": transition["type"],
                "pre_rule": _rule_summary(pre_rule),
                "post_rule": _rule_summary(post_rule),
                "distractor_fields": [
                    field_id for field_id in task["fields"] if field_id not in core_fields
                ],
                "operations": transition["operations"],
                "semantic_change": transition["semantic_change"],
            }
        )
    return {
        "domain": domain,
        "rule_family": rule_family,
        "templates": templates,
        "episodes": rendered,
    }


def _score_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "reason"],
        "properties": {
            "score": {"type": "number"},
            "reason": {"type": "string"},
        },
    }


def _response_schema(episode_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": len(episode_ids),
                "maxItems": len(episode_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["episode_id", "scores"],
                    "properties": {
                        "episode_id": {"type": "string", "enum": episode_ids},
                        "scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(DIMENSIONS),
                            "properties": {
                                dimension: _score_schema() for dimension in DIMENSIONS
                            },
                        },
                    },
                },
            }
        },
    }


def _validated_reviews(value: Any, episode_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"reviews"} or not isinstance(value["reviews"], list):
        raise ValueError("final review response has an invalid root shape")
    expected = set(episode_ids)
    reviews: dict[str, dict[str, Any]] = {}
    for review in value["reviews"]:
        if not isinstance(review, dict) or set(review) != {"episode_id", "scores"}:
            raise ValueError("final review row has an invalid shape")
        episode_id = str(review["episode_id"])
        if episode_id not in expected or episode_id in reviews:
            raise ValueError("final review has an unknown or duplicate episode")
        scores = review["scores"]
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
            raise ValueError("final review row has invalid score dimensions")
        normalized: dict[str, dict[str, Any]] = {}
        for dimension in DIMENSIONS:
            item = scores[dimension]
            if not isinstance(item, dict) or set(item) != {"score", "reason"}:
                raise ValueError("final review score has an invalid shape")
            raw_score = item["score"]
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError("final review score must be numeric")
            score = float(raw_score)
            if not math.isfinite(score) or not 0 <= score <= 5:
                raise ValueError("final review score must lie within 0.0 and 5.0")
            if not math.isclose(score * 10, round(score * 10), abs_tol=1e-8):
                raise ValueError("final review score must use at most one decimal place")
            if not isinstance(item["reason"], str):
                raise ValueError("final review reason must be a string")
            reason = item["reason"].strip()
            if not reason or len(reason) > 240:
                raise ValueError("final review reason must contain 1-240 characters")
            normalized[dimension] = {"score": round(score, 1), "reason": reason}
        reviews[episode_id] = {"episode_id": episode_id, "scores": normalized}
    missing = expected - set(reviews)
    if missing:
        raise ValueError(f"final review omitted episodes: {sorted(missing)}")
    return reviews


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


async def _review_group(
    client: Any,
    semaphore: asyncio.Semaphore,
    prompt: str,
    payload: dict[str, Any],
    config: FinalReviewConfig,
    identity_material: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    episode_ids = [str(item["episode_id"]) for item in payload["episodes"]]
    request_material = {
        **identity_material,
        "payload": payload,
    }
    request_id = hashlib.sha256(
        json.dumps(request_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_path = config.cache_directory / f"{request_id}.json"
    if config.cache_enabled and cache_path.is_file():
        try:
            return _validated_reviews(json.loads(cache_path.read_text()), episode_ids)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    errors = []
    async with semaphore:
        for attempt in range(1, config.request_attempt_limit + 1):
            try:
                response = await client.generate(
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        },
                    ],
                    schema=_response_schema(episode_ids),
                    trace_context={
                        "stage": "final_dataset_review",
                        "domain": payload["domain"],
                        "rule_family": payload["rule_family"],
                        "episode_ids": episode_ids,
                    },
                )
                parsed = json.loads(response.content)
                reviews = _validated_reviews(parsed, episode_ids)
                if config.cache_enabled:
                    _atomic_write(
                        cache_path,
                        json.dumps(parsed, ensure_ascii=False, sort_keys=True) + "\n",
                    )
                return reviews
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"final dataset review failed for {payload['domain']}/{payload['rule_family']}: "
        + "; ".join(errors)
    )


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return round(sum(rows) / len(rows), 2)


async def review_dynamic_dataset(
    episodes: list[dict[str, Any]],
    config: FinalReviewConfig,
    *,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("final review requires at least one episode")
    for episode in episodes:
        validate_episode(episode, "dynamic")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[
            (
                str(episode["provenance"]["domain"]),
                str(episode["evaluation"]["rule_family"]),
            )
        ].append(episode)
    prompt_path = (
        Path(__file__).resolve().parents[3]
        / "prompts/synthesis/datasets/final_dataset_review_system.txt"
    )
    prompt = prompt_path.read_text()
    raw_model = yaml.safe_load(config.model_config.read_text()) or {}
    merged_model = merge_model_config(
        raw_model,
        {**config.generation, "batch_size": config.concurrency},
    )
    client = build_generator(merged_model)
    identity_material = {
        "schema_version": SCHEMA_VERSION,
        "model_config_sha256": hashlib.sha256(config.model_config.read_bytes()).hexdigest(),
        "review_config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    }
    semaphore = asyncio.Semaphore(config.concurrency)
    keys = sorted(grouped)
    payloads = [build_group_payload(*key, grouped[key]) for key in keys]
    reviewed_groups = await asyncio.gather(
        *(
            _review_group(
                client, semaphore, prompt, payload, config, identity_material
            )
            for payload in payloads
        )
    )
    groups = []
    all_rows = []
    for key, source_episodes, reviews in zip(keys, (grouped[key] for key in keys), reviewed_groups):
        rows = []
        template_by_id = {
            str(episode["episode_id"]): str(episode["provenance"]["template_id"])
            for episode in source_episodes
        }
        for episode_id, review in reviews.items():
            scores = review["scores"]
            overall = round(
                sum(scores[dimension]["score"] for dimension in DIMENSIONS) / len(DIMENSIONS),
                2,
            )
            rows.append(
                {
                    "episode_id": episode_id,
                    "template_id": template_by_id[episode_id],
                    "overall_score": overall,
                    "scores": scores,
                }
            )
        rows.sort(
            key=lambda row: (
                -row["overall_score"],
                *(-row["scores"][dimension]["score"] for dimension in DIMENSIONS),
                row["episode_id"],
            )
        )
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        dimension_means = {
            dimension: _mean(row["scores"][dimension]["score"] for row in rows)
            for dimension in DIMENSIONS
        }
        groups.append(
            {
                "domain": key[0],
                "rule_family": key[1],
                "episode_count": len(rows),
                "dimension_means": dimension_means,
                "overall_mean": _mean(row["overall_score"] for row in rows),
                "episodes": rows,
            }
        )
        all_rows.extend(rows)
    dataset_value = dict(dataset)
    dataset_value.update(
        {
            "mode": "dynamic",
            "sha256": dataset_sha256(episodes),
            "episode_count": len(episodes),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_value,
        "reviewer": {
            "model": str(merged_model.get("model") or ""),
            "effort": merged_model.get("effort"),
            **identity_material,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        },
        "dimensions": {
            "field_realism": "Field combination realism and scenario fit.",
            "rule_change_realism": "Trigger/reason and dynamic rule-change realism.",
            "distractor_quality": "Plausible relevance without dominating or leaking the rule core.",
        },
        "summary": {
            **{
                f"{dimension}_mean": _mean(
                    row["scores"][dimension]["score"] for row in all_rows
                )
                for dimension in DIMENSIONS
            },
            "overall_mean": _mean(row["overall_score"] for row in all_rows),
        },
        "groups": groups,
    }


def run_final_review(
    episodes: list[dict[str, Any]],
    config: FinalReviewConfig,
    *,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    return asyncio.run(review_dynamic_dataset(episodes, config, dataset=dataset))


def serialized_review(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + "\n"


def review_manifest_entry(
    result: dict[str, Any], filename: str
) -> dict[str, Any]:
    content = serialized_review(result).encode()
    return {
        "file": filename,
        "schema_version": SCHEMA_VERSION,
        "sha256": hashlib.sha256(content).hexdigest(),
        "reviewed_dataset_sha256": result["dataset"]["sha256"],
    }


def write_review_result(path: Path, result: dict[str, Any]) -> None:
    _atomic_write(path, serialized_review(result))


def attach_review_to_manifest(
    manifest_path: Path,
    result: dict[str, Any],
    review_path: Path,
) -> None:
    if review_path.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError("published review and manifest must share one directory")
    manifest = json.loads(manifest_path.read_text())
    dataset = manifest.get("dataset") or {}
    if dataset.get("sha256") != result["dataset"]["sha256"]:
        raise ValueError("manifest dataset hash differs from the reviewed dataset")
    manifest["llm_review"] = review_manifest_entry(result, review_path.name)
    _atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
