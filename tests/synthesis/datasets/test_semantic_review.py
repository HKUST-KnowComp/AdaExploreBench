from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from synthesis.datasets import semantic_review
from synthesis.datasets.config import load_config


def _candidate(selection, index: int, dynamic_type: str = "compound") -> dict:
    visible = list(selection.fields)[:8]
    return {
        "core_id": f"core-{index}",
        "mode": "dynamic",
        "dynamic_type": dynamic_type,
        "visible_fields": visible,
        "transition": {
            "reversal_target": visible[0],
            "stable_relevant": visible[1],
            "outgoing_field": visible[2],
            "incoming_field": visible[3],
            "distractor_fields": visible[4:],
        },
    }


def _config(tmp_path):
    config = load_config(__import__("pathlib").Path("configs/synthesis/dynamic.yaml"))
    return replace(config, cache=replace(config.cache, enabled=False, directory=tmp_path))


@pytest.mark.asyncio
async def test_review_routes_all_three_outcomes(monkeypatch, tmp_path, dynamic_selection):
    rows = [_candidate(dynamic_selection, index) for index in range(3)]
    payload = {
        "reviews": [
            {"candidate_id": "core-0", "outcome": "accept", "rejection_reason": None, "trigger_event": "event", "change_reason": "reason"},
            {"candidate_id": "core-1", "outcome": "reject_distractors", "rejection_reason": "fields conflict", "trigger_event": "event", "change_reason": "reason"},
            {"candidate_id": "core-2", "outcome": "reject_core", "rejection_reason": "core conflict", "trigger_event": None, "change_reason": None},
        ]
    }

    class Client:
        async def generate(self, **kwargs):
            return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(semantic_review, "build_generator", lambda config: Client())
    accepted, retryable, audited, errors = await semantic_review.review_candidate_batch(
        [(dynamic_selection, rows)], _config(tmp_path)
    )
    assert [row["core_id"] for row in accepted] == ["core-0"]
    assert [row["core_id"] for row in retryable] == ["core-1"]
    assert len(audited) == 3 and errors == []


@pytest.mark.asyncio
async def test_partial_review_retries_then_discards_the_entire_batch(monkeypatch, tmp_path, dynamic_selection):
    rows = [_candidate(dynamic_selection, index) for index in range(2)]
    calls = []

    class Client:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=json.dumps({"reviews": [{"candidate_id": "core-0", "outcome": "accept", "rejection_reason": None, "trigger_event": "event", "change_reason": "reason"}]}))

    monkeypatch.setattr(semantic_review, "build_generator", lambda config: Client())
    accepted, retryable, audited, errors = await semantic_review.review_candidate_batch(
        [(dynamic_selection, rows)], _config(tmp_path)
    )
    assert accepted == retryable == audited == []
    assert len(calls) == 3
    assert len(errors) == 1
    assert errors[0]["attempts"] == 3
    assert errors[0]["candidate_core_ids"] == ["core-0", "core-1"]


@pytest.mark.asyncio
async def test_protocol_error_retries_the_same_batch_until_fully_valid(monkeypatch, tmp_path, dynamic_selection):
    rows = [_candidate(dynamic_selection, index) for index in range(2)]
    calls = 0

    class Client:
        async def generate(self, **kwargs):
            nonlocal calls
            calls += 1
            reviews = [
                {"candidate_id": "core-0", "outcome": "accept", "rejection_reason": None, "trigger_event": "event", "change_reason": "reason"},
            ]
            if calls > 1:
                reviews.append(
                    {"candidate_id": "core-1", "outcome": "accept", "rejection_reason": None, "trigger_event": "event", "change_reason": "reason"}
                )
            return SimpleNamespace(content=json.dumps({"reviews": reviews}))

    monkeypatch.setattr(semantic_review, "build_generator", lambda config: Client())
    accepted, retryable, audited, errors = await semantic_review.review_candidate_batch(
        [(dynamic_selection, rows)], _config(tmp_path)
    )
    assert calls == 2
    assert [row["core_id"] for row in accepted] == ["core-0", "core-1"]
    assert retryable == [] and len(audited) == 2 and errors == []


@pytest.mark.asyncio
async def test_api_failure_records_one_batch_error_and_skips_candidates(monkeypatch, tmp_path, dynamic_selection):
    rows = [_candidate(dynamic_selection, index) for index in range(2)]

    class Client:
        async def generate(self, **kwargs):
            raise RuntimeError("offline failure")

    monkeypatch.setattr(semantic_review, "build_generator", lambda config: Client())
    accepted, retryable, audited, errors = await semantic_review.review_candidate_batch(
        [(dynamic_selection, rows)], _config(tmp_path)
    )
    assert accepted == retryable == audited == []
    assert len(errors) == 1
    assert errors[0]["attempts"] == 3
    assert errors[0]["candidate_core_ids"] == ["core-0", "core-1"]


@pytest.mark.asyncio
async def test_compound_and_coexistence_reviews_use_distinct_prompts_and_schemas(
    monkeypatch, tmp_path, dynamic_selection
):
    rows = [
        _candidate(dynamic_selection, 0, "field_replacement"),
        _candidate(dynamic_selection, 1, "compound"),
    ]
    calls = []

    class Client:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            required = kwargs["schema"]["properties"]["reviews"]["items"][
                "required"
            ]
            candidate_id = kwargs["trace_context"]["candidate_ids"][0]
            review = {
                "candidate_id": candidate_id,
                "outcome": "accept",
                "rejection_reason": None,
            }
            if "trigger_event" in required:
                review.update(
                    {"trigger_event": "one event", "change_reason": "one reason"}
                )
            return SimpleNamespace(content=json.dumps({"reviews": [review]}))

    monkeypatch.setattr(semantic_review, "build_generator", lambda config: Client())
    accepted, retryable, audited, errors = await semantic_review.review_candidate_batch(
        [(dynamic_selection, rows)], _config(tmp_path)
    )
    assert [row["core_id"] for row in accepted] == ["core-0", "core-1"]
    assert retryable == [] and len(audited) == 2 and errors == []
    assert len(calls) == 2
    coexistence, compound = calls
    coexistence_required = coexistence["schema"]["properties"]["reviews"][
        "items"
    ]["required"]
    compound_required = compound["schema"]["properties"]["reviews"]["items"][
        "required"
    ]
    assert set(coexistence_required) == {
        "candidate_id",
        "outcome",
        "rejection_reason",
    }
    assert set(compound_required) == {
        "candidate_id",
        "outcome",
        "rejection_reason",
        "trigger_event",
        "change_reason",
    }
    assert "do not invent an event" in coexistence["messages"][0]["content"]
    assert "one unified real-world transition" in " ".join(
        compound["messages"][0]["content"].split()
    )
