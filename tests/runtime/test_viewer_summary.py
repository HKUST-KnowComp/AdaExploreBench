from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from runtime.runner import run_branched_baseline_episode
from viewer import run_summary


def test_precompute_reuses_branch_metrics_and_preserves_all_views(
    tmp_path, dynamic_episode, runtime_interaction_policy, monkeypatch
):
    artifact = run_branched_baseline_episode(
        dynamic_episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        branches={
            "static_iid": {
                "sampling": "static_iid", "rule_change": False,
                "history_policy": "eval_case_hidden",
            },
            "flip_50": {
                "sampling": "flip_50", "rule_change": True,
                "history_policy": "eval_case_hidden",
            },
        },
        seed=11,
    )
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(dynamic_episode) + "\n")
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "dataset": {"sha256": dataset_sha},
    }))
    episode_id = dynamic_episode["episode_id"]
    (run_dir / f"{episode_id}.json").write_text(json.dumps(artifact))

    real_metrics = run_summary.v3_episode_metrics
    calls = []

    def counted_metrics(artifact, episode=None):
        calls.append(id(artifact))
        return real_metrics(artifact, episode=episode)

    monkeypatch.setattr(run_summary, "v3_episode_metrics", counted_metrics)
    run_summary.precompute_summaries(dataset, [run_dir])
    assert len(calls) == len(artifact["branches"])
    assert len(set(calls)) == len(calls)
    summary = run_summary._load_run_summary_cache(run_dir, dataset_sha)
    assert summary["episode_ids"] == [episode_id]
    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json", f"{episode_id}.json", "viewer_summary.json",
    }

    # Compare every scope and branch with the uncached calculation, including
    # empty transition scopes and different branches sharing one episode ID.
    monkeypatch.setattr(run_summary, "v3_episode_metrics", real_metrics)
    protocol = dynamic_episode["agent_task"]["interaction_protocol"]
    for scope, view in summary["views"].items():
        branches = {
            name: [branch] if scope == "all" or (
                branch["evaluation"]["transition"]["type"] == scope
            ) else []
            for name, branch in artifact["branches"].items()
        }
        for name in [None, *branches]:
            expected = run_summary._run_summary_view(
                [item for items in branches.values() for item in items]
                if name is None else branches[name],
                branches["static_iid"],
                batch_count=protocol["action_batch_count"],
                batch_size=protocol["action_batch_size"],
                episodes={episode_id: dynamic_episode},
            )
            actual = view if name is None else view["branches"][name]
            assert {key: actual[key] for key in expected} == expected

    from viewer.generate_data import generate
    from viewer.validate_public_data import validate

    publication = tmp_path / "publication"
    manifest = generate(dataset, publication, [run_dir], light=True)
    assert manifest["episode_count"] == 1
    assert len(manifest["runs"]) == 1
    branch = manifest["runs"][0]["views"]["all"]["branches"]["flip_50"]
    # The viewer no longer publishes an exploration view or informativeness
    # aggregates anywhere in the run summary.
    assert "exploration" not in branch
    assert "exploration_by_rule_family" not in branch["metrics"]
    assert "exploration_performance" not in branch["metrics"]
    assert branch["metrics"]["mean_unique_probes"] is not None
    assert branch["metrics"]["mean_reprobes_after_change"] is not None
    # The oracle baseline collects no out-of-band assessments, so the
    # assessment-based headlines stay empty while eligibility still reports
    # the zero success rate.
    assert branch["metrics"]["factor_recognition"] is None
    assert branch["metrics"]["change_recognition"] is None
    assert branch["relevant_fields"]["change_recognition_eligibility"] == {
        "eligible": 0,
        "total": 1,
        "rate": 0.0,
    }
    static_branch = manifest["runs"][0]["views"]["all"]["branches"]["static_iid"]
    assert static_branch["metrics"]["change_recognition"] is None
    assert static_branch["metrics"]["mean_reprobes_after_change"] is None
    assert static_branch["metrics"]["mean_unique_probes"] is not None
    # The static branch has no completed assessments, so the whole
    # assessment summary stays null.
    assert static_branch["relevant_fields"] is None
    validate(publication, compressed=False)


def test_summary_cache_rejects_previous_metric_contract(tmp_path):
    (tmp_path / "viewer_summary.json").write_text(json.dumps({
        "cache_version": run_summary.RUN_SUMMARY_CACHE_VERSION - 1,
        "viewer_release": run_summary.RELEASE_VERSION,
        "dataset_sha256": "dataset",
        "artifact_count": 0,
        "artifact_mtime_ns": 0,
        "run_summary": {"views": {}},
    }))
    assert run_summary._load_run_summary_cache(tmp_path, "dataset") is None


def test_run_artifact_paths_ignore_mac_metadata_files(tmp_path):
    (tmp_path / "episode.json").write_text(json.dumps({"episode_id": "episode"}))
    (tmp_path / "._episode.json").write_bytes(b"\x00\xb0not json")

    assert [path.name for path in run_summary._run_artifact_paths(tmp_path)] == [
        "episode.json"
    ]
    _, artifacts = run_summary._load_run_artifacts(tmp_path, {"episode"})
    assert artifacts == {}


def test_published_examples_keep_full_run_summary(
    tmp_path, dynamic_episode, runtime_interaction_policy
):
    from viewer.generate_data import generate

    second_episode = deepcopy(dynamic_episode)
    second_episode["episode_id"] = f"{dynamic_episode['episode_id']}_second"
    episodes = [dynamic_episode, second_episode]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(item) + "\n" for item in episodes))
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "dataset": {"sha256": dataset_sha},
    }))
    branches = {
        "static_iid": {
            "sampling": "static_iid", "rule_change": False,
            "history_policy": "eval_case_hidden",
        },
        "flip_50": {
            "sampling": "flip_50", "rule_change": True,
            "history_policy": "eval_case_hidden",
        },
    }
    for episode in episodes:
        artifact = run_branched_baseline_episode(
            episode, "oracle", interaction_policy=runtime_interaction_policy,
            branches=branches, seed=11,
        )
        (run_dir / f"{episode['episode_id']}.json").write_text(json.dumps(artifact))

    publication = tmp_path / "publication"
    manifest = generate(
        dataset, publication, [run_dir],
        example_episode_ids={dynamic_episode["episode_id"]},
    )
    assert manifest["episode_count"] == 2
    assert [item["episode_id"] for item in manifest["episodes"]] == [dynamic_episode["episode_id"]]
    assert manifest["runs"][0]["views"]["all"]["branches"]["flip_50"]["episode_count"] == 2
    assert manifest["runs"][0]["episode_ids"] == [dynamic_episode["episode_id"]]
    assert len(list((publication / "runs" / "run").glob("*.json"))) == 1


def test_direction_adaptation_preserves_answer_set_diagnostic():
    def assessment(direction):
        return {"status": "completed", "response": {
            "relevant_fields": ["a"],
            "field_directions": [{"field_id": "a", "direction": direction}],
        }}

    summary = run_summary._aggregate_relevant_fields_assessments([{
        "condition": "dynamic",
        "evaluation": {"transition": {
            "pre_rule": {"relevant_fields": ["a"], "field_directions": {"a": 1}},
            "post_rule": {"relevant_fields": ["a"], "field_directions": {"a": -1}},
        }},
        "relevant_fields_assessments": {
            "pre_change": assessment("positive"),
            "post_change": assessment("negative"),
        },
    }])
    assert summary["change_recognition"] == 1
    assert summary["factor_recognition"] == 1
    assert summary["changed_episode_count"] == 0
    assert summary["unchanged_episode_count"] == 1
    assert summary["unchanged_outcomes"]["both_correct"] == 1
    assert summary["change_atom_detection"]["direction_reversal_atoms_detected"] == 1
    assert summary["change_atom_detection"]["direction_reversal_atoms_total"] == 1
    # A direction-only episode has no replacement atoms, so it does not enter
    # the 2x2 change-detection grid.
    assert summary["change_detection_outcomes"]["episode_count"] == 0
    assert summary["change_recognition_eligibility"] == {
        "eligible": 1,
        "total": 1,
        "rate": 1.0,
    }
