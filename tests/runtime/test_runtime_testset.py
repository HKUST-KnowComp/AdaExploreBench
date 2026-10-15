from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from synthesis.datasets.contracts import validate_episode


TESTSET = Path("datasets/test-16/dynamic_episodes.jsonl")
MANIFEST = TESTSET.parent / "manifest.json"


def test_runtime_default_testset_is_balanced_and_diverse():
    episodes = [
        json.loads(line) for line in TESTSET.read_text().splitlines() if line.strip()
    ]
    manifest = json.loads(MANIFEST.read_text())

    assert len(episodes) == 16
    for episode in episodes:
        validate_episode(episode, "dynamic")

    cells = Counter(
        (
            episode["provenance"]["domain"],
            episode["evaluation"]["rule_family"],
        )
        for episode in episodes
    )
    assert len(cells) == 16
    assert set(cells.values()) == {1}
    assert len({episode["provenance"]["template_id"] for episode in episodes}) == 16
    assert len(
        {
            field
            for episode in episodes
            for field in episode["agent_task"]["fields"]
        }
    ) == 128

    observed_sha = hashlib.sha256(TESTSET.read_bytes()).hexdigest()
    assert manifest["dataset"]["sha256"] == observed_sha
    assert manifest["selection"]["task_coverage"] == 16
    assert sorted(manifest["dynamic_after_action_batches"].values()) == [5, 5, 6]


def test_run_and_model_configs_are_decoupled():
    runtime = yaml.safe_load(
        Path("configs/runtime/full.yaml").read_text()
    )
    model = yaml.safe_load(
        Path("configs/models/gpt-5.6-terra.yaml").read_text()
    )

    assert set(runtime) == {"branches", "interaction"}
    assert runtime["branches"] == {
        "static_iid": {
            "sampling": "static_iid",
            "rule_change": False,
            "history_policy": "eval_case_hidden",
            "enable_probe_skip": True,
            "least_probe_count": None,
        },
        "flip_25": {
            "sampling": "flip_25",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
            "enable_probe_skip": True,
            "least_probe_count": None,
        },
        "flip_50": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
            "enable_probe_skip": True,
            "least_probe_count": None,
        },
        "flip_50_no_change_control": {
            "sampling": "flip_50",
            "rule_change": False,
            "history_policy": "eval_case_hidden",
            "enable_probe_skip": True,
            "least_probe_count": None,
        },
        "flip_75": {
            "sampling": "flip_75",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
            "enable_probe_skip": True,
            "least_probe_count": None,
        },
        "flip_50_eval_case_given": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_given",
            "enable_probe_skip": True,
            "least_probe_count": None,
        },
        "flip_50_probe_required": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
            "enable_probe_skip": False,
            "least_probe_count": 4,
        },
    }
    assert model["model"] == "gpt-5.6-terra"
    assert model["concurrency"] == 40
    assert model["effort"] == "high"
    assert model["max_completion_tokens"] == 32768
    assert model["base_url_env"] == "CODEX_FOR_BASE_URL"
    assert model["api_key_env"] == "CODEX_FOR_API_KEY"


def test_all_runtime_and_model_profiles_respect_the_config_boundary():
    runtime_paths = sorted(Path("configs/runtime").glob("*.yaml"))
    assert [path.name for path in runtime_paths] == [
        "all.yaml",
        "default.yaml",
        "diagnostics.yaml",
        "eval-given-only.yaml",
        "full.yaml",
        "probe-required-only.yaml",
    ]
    for path in runtime_paths:
        runtime = yaml.safe_load(path.read_text())
        assert set(runtime) == {"branches", "interaction"}, path
        assert runtime["branches"], path
        for branch in runtime["branches"].values():
            assert set(branch) == {
                "sampling",
                "rule_change",
                "history_policy",
                "enable_probe_skip",
                "least_probe_count",
            }, path
            assert isinstance(branch["rule_change"], bool), path
            assert branch["history_policy"] in {
                "eval_case_hidden",
                "eval_case_given",
            }, path
            assert isinstance(branch["enable_probe_skip"], bool), path
            if branch["least_probe_count"] is not None:
                assert (
                    isinstance(branch["least_probe_count"], int)
                    and not isinstance(branch["least_probe_count"], bool)
                    and branch["least_probe_count"] > 0
                ), path
            if not branch["enable_probe_skip"]:
                assert branch["least_probe_count"] == 4, path

    default_runtime = yaml.safe_load(Path("configs/runtime/default.yaml").read_text())
    assert list(default_runtime["branches"]) == ["flip_50"]

    probe_required_runtime = yaml.safe_load(
        Path("configs/runtime/probe-required-only.yaml").read_text()
    )
    assert list(probe_required_runtime["branches"]) == [
        "flip_50_probe_required_least4"
    ]

    configured_models = []
    for path in Path("configs/models").glob("*.yaml"):
        model = yaml.safe_load(path.read_text())
        configured_models.append(model["model"])
        assert model["concurrency"] >= 1, path
        assert model["effort"], path
        assert model.get("max_tokens") or model.get("max_completion_tokens"), path
        assert model["request_timeout_seconds"] > 0, path
        assert model["request_retry_limit"] >= 0, path
    assert len(configured_models) == len(set(configured_models))
