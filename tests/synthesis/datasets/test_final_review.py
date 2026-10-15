from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthesis.datasets import final_review
from synthesis.datasets import selection
from synthesis.datasets.config import load_config
from synthesis.datasets.contracts import GeneratedDataset
from synthesis.datasets.output import inspect_existing_dataset, write_dataset


def _episodes(dynamic_episode: dict, count: int = 2) -> list[dict]:
    rows = []
    for index in range(count):
        row = deepcopy(dynamic_episode)
        row["episode_id"] = f"review-episode-{index}"
        row["provenance"]["sequence"] = index + 1
        rows.append(row)
    return rows


def _config(tmp_path: Path, *, cache: bool = False):
    config = final_review.load_final_review_config(
        Path("configs/synthesis/final_review.yaml")
    )
    return replace(config, cache_enabled=cache, cache_directory=tmp_path / "cache")


def _response(episode_ids: list[str]) -> dict:
    return {
        "reviews": [
            {
                "episode_id": episode_id,
                "scores": {
                    dimension: {
                        "score": 4.0 + index,
                        "reason": f"Concise {dimension} reason.",
                    }
                    for dimension in final_review.DIMENSIONS
                },
            }
            for index, episode_id in enumerate(episode_ids)
        ]
    }


def test_payload_groups_complete_domain_rule_without_runtime_batches(dynamic_episode):
    rows = _episodes(dynamic_episode, 3)
    payload = final_review.build_group_payload(
        rows[0]["provenance"]["domain"],
        rows[0]["evaluation"]["rule_family"],
        rows,
    )

    assert len(payload["episodes"]) == 3
    assert set(payload) == {"domain", "rule_family", "templates", "episodes"}
    rendered = json.dumps(payload)
    assert "shared_prefix" not in rendered
    assert "post_branches" not in rendered
    assert "population_reference" not in rendered
    assert "action_transition_matrix" not in rendered
    assert all(len(row["distractor_fields"]) == 4 for row in payload["episodes"])


def test_review_scores_sorts_within_group_and_reuses_group_cache(
    monkeypatch, tmp_path: Path, dynamic_episode
):
    rows = _episodes(dynamic_episode)
    calls = 0

    class Client:
        async def generate(self, **kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                content=json.dumps(_response(kwargs["trace_context"]["episode_ids"]))
            )

    monkeypatch.setattr(final_review, "build_generator", lambda config: Client())
    config = _config(tmp_path, cache=True)
    dataset = {"id": "test", "version": "1", "file": "dynamic.jsonl"}
    first = asyncio.run(
        final_review.review_dynamic_dataset(rows, config, dataset=dataset)
    )
    second = asyncio.run(
        final_review.review_dynamic_dataset(rows, config, dataset=dataset)
    )

    assert calls == 1
    assert first["dataset"]["episode_count"] == 2
    assert first["groups"][0]["episodes"][0]["episode_id"] == "review-episode-1"
    assert [row["rank"] for row in first["groups"][0]["episodes"]] == [1, 2]
    assert first["summary"] == second["summary"]


def test_review_rejects_scores_with_more_than_one_decimal_place():
    payload = _response(["episode"])
    payload["reviews"][0]["scores"]["field_realism"]["score"] = 4.25

    with pytest.raises(ValueError, match="one decimal"):
        final_review._validated_reviews(payload, ["episode"])


def test_output_publishes_generic_review_sidecar_bound_to_dataset(
    tmp_path: Path, dynamic_episode
):
    episode = _episodes(dynamic_episode, 1)[0]
    dataset_sha256 = final_review.dataset_sha256([episode])
    review = {
        "schema_version": final_review.SCHEMA_VERSION,
        "dataset": {"sha256": dataset_sha256},
    }
    content = final_review.serialized_review(review)
    entry = final_review.review_manifest_entry(review, "llm_review.json")
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            output_dir=tmp_path / "dataset",
            artifacts_dir=tmp_path / "artifacts",
            overwrite=True,
        ),
    )

    write_dataset(
        GeneratedDataset("dynamic", iter([episode]), episode_count=1),
        config,
        inspect_existing_dataset(config),
        run_id="reviewed",
        publication_files={"llm_review.json": content},
        manifest_extensions={"llm_review": entry},
        expected_dataset_sha256=dataset_sha256,
    )

    assert (config.dataset.output_dir / "llm_review.json").read_text() == content
    manifest = json.loads((config.dataset.output_dir / "manifest.json").read_text())
    assert manifest["llm_review"] == entry
    assert (config.dataset.artifacts_dir / "reviewed/llm_review.json").is_file()


def test_selection_applies_dimension_cutoff_and_weakest_group_limit(
    tmp_path: Path, dynamic_episode
):
    rows = _episodes(dynamic_episode, 4)
    for row in rows[2:]:
        row["provenance"]["domain"] = "second-domain"
    groups = []
    scores = (5.0, 2.9, 4.5, 4.0)
    for domain, indexes in ((rows[0]["provenance"]["domain"], (0, 1)), ("second-domain", (2, 3))):
        group_rows = []
        for index in indexes:
            score = scores[index]
            group_rows.append(
                {
                    "episode_id": rows[index]["episode_id"],
                    "template_id": rows[index]["provenance"]["template_id"],
                    "overall_score": score,
                    "rank": index + 1,
                    "scores": {
                        dimension: {"score": score, "reason": "Reason."}
                        for dimension in final_review.DIMENSIONS
                    },
                }
            )
        groups.append(
            {
                "domain": domain,
                "rule_family": rows[0]["evaluation"]["rule_family"],
                "episodes": group_rows,
            }
        )
    review = {
        "schema_version": final_review.SCHEMA_VERSION,
        "dataset": {"sha256": final_review.dataset_sha256(rows)},
        "groups": groups,
    }
    config = replace(
        selection.load_selection_config(Path("configs/synthesis/dataset_selection.yaml")),
        target_episode_count=4,
    )

    result = selection.select_episodes(rows, review, config)

    assert len(result.episodes) == 2
    assert result.report["policy"]["requested_per_group"] == 2
    assert result.report["policy"]["effective_per_group"] == 1
    assert {row["episode_id"] for row in result.episodes} == {
        "review-episode-0",
        "review-episode-2",
    }
