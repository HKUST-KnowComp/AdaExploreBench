from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path

import pytest

from synthesis.datasets.config import load_config
from synthesis.datasets.contracts import GeneratedDataset
from synthesis.datasets.output import inspect_existing_dataset, write_dataset
from synthesis.datasets.state import DatasetRunState


def test_fresh_refuses_nonempty_output_without_overwrite(tmp_path: Path):
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve")
    config = replace(config, dataset=replace(config.dataset, output_dir=output))
    with pytest.raises(ValueError, match="output is not empty"):
        inspect_existing_dataset(config)


def test_append_requires_manifest_and_mode_specific_episode_file(tmp_path: Path):
    config = load_config(Path("configs/synthesis/static.yaml"))
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            operation="append",
            output_dir=tmp_path / "missing",
        ),
    )
    with pytest.raises(ValueError, match="append requires"):
        inspect_existing_dataset(config)


def test_resume_rejects_experiment_changes_and_allows_execution_changes(tmp_path: Path):
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, artifacts_dir=tmp_path),
    )
    DatasetRunState(config, "run", resume=False)
    assert config.dynamic is not None
    changed = replace(
        config,
        search=replace(config.search, q_flip_targets=(0.2, 0.5, 0.8)),
    )
    with pytest.raises(ValueError, match="pinned run identity"):
        DatasetRunState(changed, "run", resume=True)
    execution_change = replace(
        config,
        resources=replace(config.resources, cpu_worker_count=1),
    )
    resumed = DatasetRunState(execution_change, "run", resume=True)
    assert resumed.value["status"] == "running"


def test_failed_run_state_returns_to_running_on_resume(tmp_path: Path):
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, artifacts_dir=tmp_path),
    )
    state = DatasetRunState(config, "run", resume=False)
    state.fail(ValueError("review failed"))
    assert state.value["status"] == "failed"
    assert state.value["failure"]["type"] == "ValueError"

    resumed = DatasetRunState(config, "run", resume=True)
    assert resumed.value["status"] == "running"


def test_streaming_append_keeps_duplicate_content_with_new_identity(
    tmp_path: Path, dynamic_episode: dict
):
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
    first = deepcopy(dynamic_episode)
    write_dataset(
        GeneratedDataset("dynamic", iter([first]), episode_count=1),
        config,
        inspect_existing_dataset(config),
        run_id="first",
    )
    append = replace(
        config,
        dataset=replace(config.dataset, operation="append", overwrite=False),
    )
    second = deepcopy(first)
    second["episode_id"] = "same-content-new-identity"
    second["provenance"]["sequence"] = 2
    write_dataset(
        GeneratedDataset("dynamic", iter([second]), episode_count=1),
        append,
        inspect_existing_dataset(append),
        run_id="second",
    )
    lines = [json.loads(line) for line in (config.dataset.output_dir / "dynamic_episodes.jsonl").read_text().splitlines()]
    assert [row["episode_id"] for row in lines] == [first["episode_id"], second["episode_id"]]
    manifest = json.loads((config.dataset.output_dir / "manifest.json").read_text())
    assert [row["operation"] for row in manifest["generations"]] == ["fresh", "append"]
    assert all("template_sha256" in row and "generation_contract_sha256" in row for row in manifest["generations"])


def test_completed_run_resume_is_marked_published_and_does_not_reopen(tmp_path: Path):
    config = load_config(Path("configs/synthesis/dynamic.yaml"))
    config = replace(config, dataset=replace(config.dataset, artifacts_dir=tmp_path))
    state = DatasetRunState(config, "run", resume=False)
    state.complete()
    resumed = DatasetRunState(config, "run", resume=True)
    assert resumed.published is True
    assert resumed.value["status"] == "completed"
