from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_benchmark import (
    _artifact_covers_requested_branches,
    _interaction_progress,
    _progress_cell_color,
    _progress_snapshot,
    _progress_source,
    _resume_branch_change_is_compatible,
    _resolve_input_paths,
    _snapshot_configs,
)
from runtime.schema import BRANCHED_TRACE_SCHEMA_VERSION


def test_interaction_progress_counts_protocol_milestones_but_not_repair_errors():
    artifact = {
        "conversation": {
            "events": [
                {"type": "environment_turn", "stage": "control"},
                {"type": "environment_turn", "stage": "reveal"},
                {
                    "type": "environment_turn",
                    "stage": "action_solving",
                    "error": "retry",
                },
                {
                    "type": "environment_turn",
                    "stage": "completion",
                    "error": "repair limit exhausted",
                },
                {"type": "environment_turn", "stage": "reveal"},
            ]
        },
        "tool_trace": [],
    }

    position, error_slots = _interaction_progress(artifact, 24)

    assert position == 3
    assert error_slots == set()


def test_interaction_progress_marks_only_exhausted_provider_errors():
    artifact = {
        "conversation": {
            "events": [
                {"type": "environment_turn", "stage": "reveal"},
                {"type": "environment_turn", "stage": "completion"},
            ]
        },
        "tool_trace": [{"status": "provider_error"}],
    }

    position, error_slots = _interaction_progress(artifact, 24)

    assert position == 2
    assert error_slots == {2}


@pytest.mark.parametrize(
    ("reached", "expected"),
    [
        (0, None),
        (1, "gray"),
        (8, "gray"),
        (9, "yellow"),
        (15, "yellow"),
        (16, "green"),
    ],
)
def test_progress_cell_color_shows_aggregate_reach(reached, expected):
    assert _progress_cell_color(reached, 16) == expected


def test_progress_cell_color_reserves_red_for_provider_error():
    assert _progress_cell_color(16, 16, provider_error=True) == "red"


def test_config_snapshots_preserve_inputs_and_are_verified_on_resume(tmp_path: Path):
    run_config = tmp_path / "source-run.yaml"
    model_config = tmp_path / "source-model.yaml"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    run_config.write_text("branches: {}\n")
    model_config.write_text("model: example\n")

    snapshots = _snapshot_configs(
        output_dir,
        run_config=run_config,
        model_config=model_config,
        require_existing=False,
    )

    assert (output_dir / "config_snapshot/run_config.yaml").read_text() == (
        "branches: {}\n"
    )
    assert (output_dir / "config_snapshot/model_config.yaml").read_text() == (
        "model: example\n"
    )
    assert set(snapshots) == {"run_config", "model_config"}

    model_config.write_text("model: changed\n")
    with pytest.raises(ValueError, match="does not match"):
        _snapshot_configs(
            output_dir,
            run_config=run_config,
            model_config=model_config,
            require_existing=True,
        )

    updated = _snapshot_configs(
        output_dir,
        run_config=run_config,
        model_config=model_config,
        require_existing=False,
    )
    archived = Path(updated["model_config"]["archived_previous_path"])
    assert archived.read_text() == "model: example\n"
    assert (output_dir / "config_snapshot/model_config.yaml").read_text() == (
        "model: changed\n"
    )


def test_resume_inputs_prefer_explicit_values_then_snapshots(tmp_path: Path):
    output_dir = tmp_path / "output"
    snapshot_dir = output_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True)
    dataset = tmp_path / "dataset.jsonl"
    recorded_run = snapshot_dir / "run_config.yaml"
    recorded_model = snapshot_dir / "model_config.yaml"
    explicit_run = tmp_path / "explicit-run.yaml"
    explicit_model = tmp_path / "explicit-model.yaml"
    for path in (
        dataset,
        recorded_run,
        recorded_model,
        explicit_run,
        explicit_model,
    ):
        path.write_text("value\n")
    manifest = {"dataset": {"path": str(dataset)}}

    recorded_args = Namespace(
        resume=True,
        output_dir=output_dir,
        dataset=None,
        run_config=None,
        model_config=None,
    )
    assert _resolve_input_paths(recorded_args, manifest) == (
        dataset,
        recorded_run,
        recorded_model,
    )

    explicit_args = Namespace(
        resume=True,
        output_dir=output_dir,
        dataset=dataset,
        run_config=explicit_run,
        model_config=explicit_model,
    )
    assert _resolve_input_paths(explicit_args, manifest) == (
        dataset,
        explicit_run,
        explicit_model,
    )


def test_progress_snapshot_records_each_branch_at_a_config_change():
    episode_id = "episode-1"
    episodes = {
        episode_id: {
            "agent_task": {
                "interaction_protocol": {"action_batch_count": 12}
            }
        }
    }
    branches = {
        "flip_50": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
        },
        "flip_50_eval_case_given": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_given",
        },
    }
    artifacts = {
        episode_id: {
            "trace_schema_version": BRANCHED_TRACE_SCHEMA_VERSION,
            "shared_prefixes": {
                "eval_case_hidden": {
                    "conversation": {
                        "events": [
                            {"type": "environment_turn", "stage": "reveal"},
                            {
                                "type": "environment_turn",
                                "stage": "completion",
                            },
                        ]
                    }
                },
                "eval_case_given": {
                    "conversation": {"events": []},
                },
            },
        }
    }

    progress = _progress_snapshot(branches, episodes, artifacts)

    assert progress["interactions_per_episode"] == 24
    assert progress["branches"]["flip_50"]["episodes"] == {
        episode_id: 2
    }
    assert progress["branches"]["flip_50_eval_case_given"]["episodes"] == {
        episode_id: 0
    }


def test_progress_source_uses_only_the_branch_history_policy_before_fork():
    hidden_prefix = {"history_policy": "eval_case_hidden"}
    given_prefix = {"history_policy": "eval_case_given"}
    artifact = {
        "trace_schema_version": BRANCHED_TRACE_SCHEMA_VERSION,
        "shared_prefixes": {
            "eval_case_hidden": hidden_prefix,
            "eval_case_given": given_prefix,
        },
        "branches": {},
    }

    assert (
        _progress_source(
            artifact,
            "flip_25",
            "eval_case_hidden",
        )
        is hidden_prefix
    )
    assert (
        _progress_source(
            artifact,
            "flip_50_eval_case_given",
            "eval_case_given",
        )
        is given_prefix
    )


def test_dynamic_resume_allows_adding_branches_but_pins_existing_definitions():
    hidden = {
        "sampling": "flip_50",
        "rule_change": True,
        "history_policy": "eval_case_hidden",
    }
    given = {
        "sampling": "flip_50",
        "rule_change": True,
        "history_policy": "eval_case_given",
    }
    manifest = {
        "run_identity": {
            "branches": {"flip_50": hidden},
            "history": {
                "policies": ["eval_case_hidden"],
                "version": "history-v1",
            },
        }
    }
    requested = {"flip_50": hidden, "flip_50_eval_case_given": given}
    requested_history = {
        "policies": ["eval_case_given", "eval_case_hidden"],
        "version": "history-v1",
    }

    assert _resume_branch_change_is_compatible(
        manifest,
        requested,
        dataset_mode="dynamic",
        requested_history=requested_history,
    )
    assert not _resume_branch_change_is_compatible(
        manifest,
        {"flip_50_eval_case_given": given},
        dataset_mode="dynamic",
        requested_history=requested_history,
    )
    changed = {**hidden, "rule_change": False}
    assert not _resume_branch_change_is_compatible(
        manifest,
        {"flip_50": changed, "flip_50_eval_case_given": given},
        dataset_mode="dynamic",
        requested_history=requested_history,
    )


def test_artifact_with_missing_dynamic_branches_is_resumed_for_extension():
    artifact = {
        "run_status": "completed",
        "branches": {"flip_50": {}},
    }
    assert not _artifact_covers_requested_branches(
        artifact,
        {"flip_50", "flip_50_eval_case_given"},
        dataset_mode="dynamic",
    )
    assert _artifact_covers_requested_branches(
        artifact,
        {"flip_50"},
        dataset_mode="dynamic",
    )
    assert _artifact_covers_requested_branches(
        {"run_status": "completed"},
        {"unused": "branch"},
        dataset_mode="static",
    )
