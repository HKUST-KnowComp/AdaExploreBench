from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthesis.datasets.config import load_config
from synthesis.datasets.dynamic.pipeline import run_dynamic_cell_stage
from synthesis.datasets.pipeline import (
    _priority_cells,
    _run_static_cell_stage,
)
from synthesis.datasets.planning import GenerationCell
from synthesis.datasets.state import DatasetRunState
from synthesis.datasets.template_pool import load_template_pool, select_template


TEMPLATE_POOL = Path("datasets/templates/domain_templates.json")


def _worker_smoke_config(path: Path):
    config = load_config(path)
    return replace(
        config,
        resources=replace(
            config.resources,
            max_core_visits_per_cell=20,
        ),
        search=replace(
            config.search,
            calibration_sample_count=128,
            parameter_screening_sample_count=64,
            evaluation_sample_count=128,
            held_out_pool_size=512,
            coupling_grid=(0.5,),
            reversal_additive_grid=(1.0,),
            population_flip_range=(0.0, 1.0),
            action_share_range=(0.0, 1.0),
            early_rejection_confidence=0.9,
            min_transition_probability=0.0,
            achieved_correlation_range=(0.0, 1.0),
            minimum_mean_normalized_divergence=0.0,
            beta_lookup_knots=128,
            conditional_integration_points=128,
        ),
        semantic_review=replace(config.semantic_review, batch_size=1),
    )


def _finance_selection():
    pool = load_template_pool(TEMPLATE_POOL)
    return select_template(pool, "finance", "finance_card_authorization")


def _state(tmp_path: Path) -> DatasetRunState:
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(config, dataset=replace(config.dataset, artifacts_dir=tmp_path))
    return DatasetRunState(config, "run", resume=False)


def _candidate() -> dict:
    records = [
        {"pool_index": 0, "raw_values": {"x": 0.25, "y": 2}, "operational_profile": "low", "old_action": 1, "new_action": 2},
        {"pool_index": 1, "raw_values": {"x": 0.25, "y": 2}, "operational_profile": "low", "old_action": 1, "new_action": 2},
    ]
    return {
        "core_id": "core-a",
        "distractor_trial": 0,
        "certification": {
            "inventory": {
                "records": records,
                "buckets": {"1,2": [0, 1]},
            }
        },
    }


def test_candidate_checkpoint_is_columnar_on_disk_and_expands_losslessly(tmp_path):
    state = _state(tmp_path)
    candidate = _candidate()
    cell = ("finance", "template", "additive_aggregation", "dynamic", "compound")
    relative = state.write_candidate(cell, candidate)
    assert relative.startswith("cells/") and "/candidates/" in relative
    assert "finance--template--additive_aggregation--dynamic--compound" in relative
    raw = __import__("json").loads((state.run_dir / relative).read_text())
    assert "record_columns" in raw["certification"]["inventory"]
    restored = state.read_candidate(relative)
    assert restored == candidate
    assert len(restored["certification"]["inventory"]["records"]) == 2


def test_cell_checkpoint_preserves_current_core_cursor_and_candidate_reference(tmp_path):
    state = _state(tmp_path)
    cell = ("finance", "template", "additive_aggregation", "dynamic", "compound")
    relative = state.write_candidate(cell, _candidate())
    payload = {
        "traversal": {"cursor": 7},
        "current_core": {"core_space_index": 11},
        "candidate_files": [relative],
        "surplus_files": [],
    }
    state.save_cell_checkpoint(cell, "awaiting_review", payload)
    restored = state.cell_checkpoint(cell)
    assert restored["traversal"]["cursor"] == 7
    assert restored["current_core"]["core_space_index"] == 11
    assert state.read_candidate(restored["candidate_files"][0])["core_id"] == "core-a"


def test_group_cleanup_can_resume_after_completion_marker(tmp_path):
    state = _state(tmp_path)
    episode = SimpleNamespace(
        episode={"episode_id": "episode-a"}, diagnostics={"episode_id": "episode-a"}
    )
    candidate = _candidate()
    state.record_episode(cell := ("finance", "template", "additive_aggregation", "dynamic", "compound"), candidate, episode)
    resource = state.cell_order_path(cell)
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_bytes(b"order")
    group = ("finance", "additive_aggregation", "dynamic", "compound")
    relative = str(resource.relative_to(state.run_dir))
    state.mark_group_complete(group, ["episode-a"], [relative])
    assert resource.is_file()
    state.clean_group_resources(group)
    assert not resource.exists()
    entry = state.value["groups"][state._key(group)]
    assert entry["resources_cleaned"] is True
    assert entry["resource_paths"] == []


def test_resume_recovers_episode_written_before_run_index(tmp_path):
    state = _state(tmp_path)
    cell = ("finance", "template", "additive_aggregation", "dynamic", "compound")
    build = SimpleNamespace(
        episode={"episode_id": "episode-orphan"},
        diagnostics={"episode_id": "episode-orphan"},
    )
    state.record_episode(cell, _candidate(), build)
    state.value["episodes"] = {}
    state.value["candidate_commits"] = {}
    state._write()
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(config, dataset=replace(config.dataset, artifacts_dir=tmp_path))
    resumed = DatasetRunState(config, "run", resume=True)
    assert resumed.episode("episode-orphan")["episode"]["episode_id"] == "episode-orphan"
    assert resumed.candidate_committed(cell, _candidate())


def test_review_error_identity_is_idempotent_across_resume(tmp_path):
    state = _state(tmp_path)
    error = {"batch_id": "batch-a", "error_type": "failure"}
    state.record_review_error(error)
    state.record_review_error(error)
    assert state.value["review_errors"] == [error]


def test_priority_queue_prefers_callbacks_then_idle_cells_then_balances_workers():
    cells = [
        GenerationCell(index, "finance", f"template-{index}", "additive_aggregation", "dynamic", "compound", 1)
        for index in range(4)
    ]
    group = cells[0].group_key
    ready = _priority_cells(
        cells,
        {group: 4},
        {group: 0},
        {cell.key: 0 for cell in cells},
        {cell.key: {"stage": "ready"} for cell in cells},
        {cells[0].key: 3, cells[1].key: 1},
        set(),
        {cells[0].key},
    )
    ordered = [__import__("heapq").heappop(ready)[-1].priority for _ in range(len(ready))]
    assert ordered == [0, 2, 3, 1]


def test_static_worker_runs_a_complete_core_to_an_eight_field_candidate(tmp_path):
    config = _worker_smoke_config(Path("configs/synthesis/static.yaml"))
    selection = _finance_selection()
    cell = GenerationCell(
        0,
        "finance",
        selection.template_id,
        "additive_aggregation",
        "static",
        None,
        1,
    )
    result = _run_static_cell_stage(
        selection,
        cell,
        config,
        {},
        order_path=tmp_path / "static-order.bin",
        run_dir=tmp_path,
    )
    assert result["stage"] == "awaiting_review"
    assert result["traversal"]["cursor"] <= 20
    assert len(result["candidate_files"]) == 1
    assert (tmp_path / result["candidate_files"][0]).is_file()


@pytest.mark.parametrize(
    "dynamic_type",
    ["field_replacement", "direction_reversal", "compound"],
)
def test_dynamic_worker_runs_a_complete_core_to_a_certified_candidate(
    tmp_path, dynamic_type
):
    config = _worker_smoke_config(Path("configs/synthesis/dynamic.yaml"))
    selection = _finance_selection()
    cell = GenerationCell(
        0,
        "finance",
        selection.template_id,
        "additive_aggregation",
        "dynamic",
        dynamic_type,
        1,
    )
    result = run_dynamic_cell_stage(
        selection,
        cell,
        config,
        {},
        visit_limit=20,
        materialize_threshold=20,
        review_batch_size=1,
        visible_field_count=8,
        order_path=tmp_path / f"{dynamic_type}-order.bin",
        run_dir=tmp_path,
        checkpoint_interval=1,
    )
    assert result["stage"] == "awaiting_review"
    assert result["traversal"]["cursor"] <= 20
    assert len(result["candidate_files"]) == 1
    candidate = __import__("json").loads((tmp_path / result["candidate_files"][0]).read_text())
    assert len(candidate["visible_fields"]) == 8
    assert len(candidate["certification"]["inventory"]["record_columns"]["raw_values"]) == 512


def test_dynamic_worker_records_error_and_continues_with_next_core(tmp_path, monkeypatch):
    from synthesis.datasets.dynamic import certification

    config = _worker_smoke_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(config, dataset=replace(config.dataset, artifacts_dir=tmp_path))
    state = DatasetRunState(config, "crash", resume=False)
    selection = _finance_selection()
    cell = GenerationCell(0, "finance", selection.template_id, "additive_aggregation", "dynamic", "compound", 1)
    state.save_cell_checkpoint(cell.key, "ready", {"candidate_files": [], "surplus_files": []})

    def fail_core(*_args, **_kwargs):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(certification, "evaluate_core", fail_core)
    result = run_dynamic_cell_stage(
        selection, cell, config, {}, visit_limit=20,
        materialize_threshold=20, review_batch_size=1,
        visible_field_count=8, order_path=state.cell_order_path(cell.key),
        run_dir=state.run_dir, checkpoint_interval=1,
    )
    checkpoint = state.cell_checkpoint(cell.key)
    assert result["stage"] == checkpoint["stage"] == "exhausted"
    assert checkpoint["traversal"]["cursor"] == 20
    assert checkpoint["current_core"] is None
    assert checkpoint["counts"] == {
        "visited": 20,
        "certified": 0,
        "rejected": 20,
        "errors": 20,
        "rejection_stages": {},
    }
    assert checkpoint["worker_errors"][0]["type"] == "RuntimeError"
    assert checkpoint["worker_errors"][0]["message"] == "worker crashed"
    assert checkpoint["worker_errors"][0]["count"] == 20


def test_static_worker_records_error_and_continues_with_next_core(tmp_path, monkeypatch):
    from synthesis.datasets import pipeline

    config = _worker_smoke_config(Path("configs/synthesis/static.yaml"))
    selection = _finance_selection()
    cell = GenerationCell(
        0, "finance", selection.template_id,
        "additive_aggregation", "static", None, 1,
    )

    def fail_core(*_args, **_kwargs):
        raise RuntimeError("static worker crashed")

    monkeypatch.setattr(pipeline, "compile_core_constraints", fail_core)
    result = _run_static_cell_stage(
        selection, cell, config, {},
        order_path=tmp_path / "static-order.bin", run_dir=tmp_path,
    )
    assert result["stage"] == "exhausted"
    assert result["traversal"]["cursor"] == 20
    assert result["counts"]["rejected"] == result["counts"]["errors"] == 20
    assert result["worker_errors"][0]["message"] == "static worker crashed"
    assert result["worker_errors"][0]["count"] == 20


def test_dynamic_worker_resumes_persisted_candidate_without_recertifying(tmp_path, monkeypatch):
    from synthesis.datasets.dynamic import certification

    config = _worker_smoke_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(config, dataset=replace(config.dataset, artifacts_dir=tmp_path))
    state = DatasetRunState(config, "candidate-resume", resume=False)
    selection = _finance_selection()
    cell = GenerationCell(0, "finance", selection.template_id, "additive_aggregation", "dynamic", "compound", 1)
    first = run_dynamic_cell_stage(
        selection, cell, config, {}, visit_limit=20, materialize_threshold=20,
        review_batch_size=1, visible_field_count=8,
        order_path=state.cell_order_path(cell.key), run_dir=state.run_dir,
        checkpoint_interval=1,
    )
    loaded = [state.read_candidate(path) for path in first["candidate_files"]]

    def fail_core(*_args, **_kwargs):
        raise AssertionError("persisted candidate was recertified")

    monkeypatch.setattr(certification, "evaluate_core", fail_core)
    resumed = run_dynamic_cell_stage(
        selection, cell, config, {**first, "candidates": loaded}, visit_limit=20,
        materialize_threshold=20, review_batch_size=1, visible_field_count=8,
        order_path=state.cell_order_path(cell.key), run_dir=state.run_dir,
        checkpoint_interval=1,
    )
    assert resumed["candidate_files"] == first["candidate_files"]
