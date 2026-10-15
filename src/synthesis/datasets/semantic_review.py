"""Shared structured semantic review for dataset candidates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts._utils import build_generator, merge_model_config
from .config import DatasetGenerationConfig
from .template_pool import TemplateSelection


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate["core_id"])


def _field_payload(selection: TemplateSelection, field_id: str) -> dict[str, Any]:
    field = selection.fields[field_id]
    return {
        "field_id": field_id,
        "display_name": field["display_name"],
        "description": field["description"],
        "unit": field["unit"],
    }


def _schema(candidate_ids: list[str], *, compound: bool) -> dict[str, Any]:
    nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    properties: dict[str, Any] = {
        "candidate_id": {"type": "string", "enum": candidate_ids},
        "outcome": {
            "type": "string",
            "enum": ["reject_core", "reject_distractors", "accept"],
        },
        "rejection_reason": nullable,
    }
    required = list(properties)
    if compound:
        properties.update({"trigger_event": nullable, "change_reason": nullable})
        required.extend(("trigger_event", "change_reason"))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": required,
                    "properties": properties,
                },
            }
        },
    }


def _roles(candidate: dict[str, Any]) -> dict[str, Any]:
    transition = candidate.get("transition") or {}
    rule = candidate.get("rule") or {}
    return {
        "reversal_field": transition.get("reversal_target")
        or transition.get("target_field"),
        "stable_field": transition.get("stable_relevant"),
        "replacement_outgoing": transition.get("outgoing_field"),
        "replacement_incoming": transition.get("incoming_field"),
        "relevant_fields": transition.get("relevant_fields")
        or rule.get("relevant_fields")
        or candidate.get("_core", {}).get("relevant_fields"),
        "distractor_fields": transition.get("distractor_fields", []),
    }


def _request_payload(
    grouped: list[tuple[TemplateSelection, list[dict[str, Any]]]],
    *,
    compound: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    candidates: dict[str, dict[str, Any]] = {}
    groups = []
    for selection, rows in grouped:
        rendered = []
        for candidate in rows:
            if (candidate.get("dynamic_type") == "compound") != compound:
                continue
            candidate_id = _candidate_id(candidate)
            if candidate_id in candidates:
                raise ValueError("semantic review candidate identities must be unique")
            candidates[candidate_id] = candidate
            rendered.append(
                {
                    "candidate_id": candidate_id,
                    "mode": candidate.get("mode", "dynamic"),
                    "dynamic_type": candidate.get("dynamic_type"),
                    "prior_core_review": candidate.get("semantic_review"),
                    "roles": _roles(candidate),
                    "fields": [
                        _field_payload(selection, field_id)
                        for field_id in candidate["visible_fields"]
                    ],
                }
            )
        if rendered:
            groups.append(
                {
                    "template_id": selection.template_id,
                    "scenario": selection.template["description"],
                    "agent_role": selection.template["agent_role"],
                    "actions": selection.template["actions"],
                    "candidates": rendered,
                }
            )
    return {"template_groups": groups}, candidates


def _validate_review(
    review: Any,
    candidates: dict[str, dict[str, Any]],
    *,
    compound: bool,
) -> tuple[str, dict[str, Any]]:
    keys = {"candidate_id", "outcome", "rejection_reason"}
    if compound:
        keys |= {"trigger_event", "change_reason"}
    if not isinstance(review, dict) or set(review) != keys:
        raise ValueError("review has an invalid shape")
    candidate_id = str(review["candidate_id"])
    outcome = review["outcome"]
    if candidate_id not in candidates or outcome not in {
        "reject_core",
        "reject_distractors",
        "accept",
    }:
        raise ValueError("review has an invalid candidate or outcome")
    candidate = candidates[candidate_id]
    prior = candidate.get("semantic_review")
    if prior and outcome == "reject_core":
        raise ValueError("a frozen approved core cannot be rejected during distractor retry")
    if compound and outcome in {"accept", "reject_distractors"}:
        if not str(review["trigger_event"] or "").strip() or not str(
            review["change_reason"] or ""
        ).strip():
            raise ValueError("approved compound core review lacks trigger/reason")
        if prior and any(
            review[key] != prior[key] for key in ("trigger_event", "change_reason")
        ):
            raise ValueError("a frozen compound rationale changed during distractor retry")
    if outcome == "accept" and review["rejection_reason"] is not None:
        raise ValueError("accepted review has a rejection reason")
    if outcome != "accept" and not str(review["rejection_reason"] or "").strip():
        raise ValueError("rejected review lacks a rejection reason")
    return candidate_id, review


def _error_record(
    batch_id: str,
    request_identity: str,
    candidates: dict[str, dict[str, Any]],
    error_type: str,
    message: str,
    attempts: int,
    candidate_ids: list[str],
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "request_identity": request_identity,
        "candidate_core_ids": candidate_ids,
        "candidate_visible_fields": {
            candidate_id: list(candidates[candidate_id]["visible_fields"])
            for candidate_id in candidate_ids
        },
        "error_type": error_type,
        "message": message,
        "attempts": attempts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _validated_response(
    parsed: Any,
    candidates: dict[str, dict[str, Any]],
    *,
    compound: bool,
) -> dict[str, dict[str, Any]]:
    if not isinstance(parsed, dict) or set(parsed) != {"reviews"} or not isinstance(parsed["reviews"], list):
        raise ValueError("review response has an invalid root shape")
    reviews: dict[str, dict[str, Any]] = {}
    for item in parsed["reviews"]:
        candidate_id, review = _validate_review(
            item, candidates, compound=compound
        )
        if candidate_id in reviews:
            raise ValueError("duplicate review candidate")
        reviews[candidate_id] = review
    missing = set(candidates) - set(reviews)
    if missing:
        raise ValueError(f"review response omitted candidates: {sorted(missing)}")
    return reviews


async def _review_branch(
    payload: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    config: DatasetGenerationConfig,
    *,
    compound: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    review_config = config.semantic_review
    if not candidates:
        return {}, []
    if not review_config.enabled:
        reviews = {}
        for candidate_id in candidates:
            review = {
                "candidate_id": candidate_id,
                "outcome": "accept",
                "rejection_reason": None,
            }
            if compound:
                review.update(
                    {
                        "trigger_event": "Configured deterministic review event.",
                        "change_reason": "Configured deterministic compound acceptance.",
                    }
                )
            reviews[candidate_id] = review
        return reviews, []

    prompt_name = (
        "compound_semantic_review_system.txt"
        if compound
        else "semantic_review_system.txt"
    )
    prompt = (
        Path(__file__).resolve().parents[3]
        / "prompts/synthesis/datasets"
        / prompt_name
    )
    raw_model = yaml.safe_load(review_config.model_config.read_text()) or {}
    client = build_generator(
        merge_model_config(
            raw_model,
            {
                **review_config.generation,
                "batch_size": review_config.concurrency,
                "request_retry_limit": review_config.request_attempt_limit,
            },
        )
    )
    cache_identity = json.dumps(
        {
            "payload": payload,
            "model": hashlib.sha256(review_config.model_config.read_bytes()).hexdigest(),
            "prompt": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    request_identity = hashlib.sha256(cache_identity.encode()).hexdigest()
    batch_id = f"semantic_{request_identity[:20]}"
    cache_path = config.cache.directory / "semantic_reviews" / f"{request_identity}.json"
    cached = None
    if config.cache.enabled and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    if cached is not None:
        try:
            return _validated_response(
                cached, candidates, compound=compound
            ), []
        except ValueError:
            pass

    protocol_errors: list[str] = []
    for attempt in range(1, review_config.request_attempt_limit + 1):
        try:
            response = await client.generate(
                messages=[
                    {"role": "system", "content": prompt.read_text()},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    },
                ],
                schema=_schema(list(candidates), compound=compound),
                trace_context={
                    "stage": "compound_semantic_review"
                    if compound
                    else "semantic_review",
                    "candidate_ids": list(candidates),
                },
            )
            parsed = json.loads(response.content)
            reviews = _validated_response(
                parsed, candidates, compound=compound
            )
            if config.cache.enabled:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(parsed, ensure_ascii=False, sort_keys=True) + "\n"
                )
            return reviews, []
        except (json.JSONDecodeError, ValueError) as exc:
            protocol_errors.append(f"attempt {attempt}: {exc}")
            continue
        except Exception as exc:
            return {}, [
                _error_record(
                    batch_id,
                    request_identity,
                    candidates,
                    type(exc).__name__,
                    str(exc),
                    review_config.request_attempt_limit,
                    list(candidates),
                )
            ]
    return {}, [
        _error_record(
            batch_id,
            request_identity,
            candidates,
            "SemanticReviewProtocolError",
            "; ".join(protocol_errors),
            review_config.request_attempt_limit,
            list(candidates),
        )
    ]


async def review_candidate_batch(
    grouped: list[tuple[TemplateSelection, list[dict[str, Any]]]],
    config: DatasetGenerationConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    count = sum(len(rows) for _, rows in grouped)
    if count > config.semantic_review.batch_size:
        raise ValueError("semantic review batch exceeds configured size")
    reviews: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    for compound in (False, True):
        payload, branch_candidates = _request_payload(grouped, compound=compound)
        branch_reviews, branch_errors = await _review_branch(
            payload, branch_candidates, config, compound=compound
        )
        candidates.update(branch_candidates)
        reviews.update(branch_reviews)
        errors.extend(branch_errors)

    accepted, retryable, audited = [], [], []
    for candidate_id, candidate in candidates.items():
        review = reviews.get(candidate_id)
        if review is None:
            continue
        row = {**candidate, "semantic_review": review}
        audited.append(row)
        if review["outcome"] == "accept":
            accepted.append(row)
        elif review["outcome"] == "reject_distractors":
            retryable.append(row)
    return accepted, retryable, audited, errors
