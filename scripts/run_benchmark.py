#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

from scripts._utils import build_generator, load_repo_env
from runtime.metrics import (
    aggregate_branched_metrics,
    aggregate_static_metrics,
)
from synthesis.datasets.contracts import validate_episode
from protocol.provenance import code_identity
from runtime.runner import (
    _branch_prefix_key,
    _prefix_group_labels,
    assessment_is_terminal,
    resolve_runtime_branches,
    run_agent_episode,
    run_branched_agent_episode,
)
from runtime.history import HISTORY_POLICY_VERSION
from runtime.schema import (
    BRANCHED_TRACE_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    sequential_trace_schema_version,
)


def _dataset_provenance(dataset: Path) -> dict:
    manifest_path = dataset.parent / "manifest.json"
    observed_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )
    claimed = (manifest or {}).get("dataset") or {}
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path) if manifest is not None else None,
        "claimed_sha256": claimed.get("sha256"),
        "claimed_file": claimed.get("file"),
        "manifest_consistent": bool(
            manifest is not None
            and claimed.get("sha256") == observed_sha
            and claimed.get("file") == dataset.name
        ),
    }


def load_cases(
    path: Path,
    limit: int | None,
    episode_ids: list[str] | None = None,
) -> list[dict]:
    cases = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if episode_ids:
        indexed = {case["episode_id"]: case for case in cases}
        missing = [item for item in episode_ids if item not in indexed]
        if missing:
            raise ValueError(f"episode IDs not found in dataset: {missing}")
        cases = [indexed[item] for item in episode_ids]
    return cases[:limit] if limit else cases


def _file_identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _snapshot_configs(
    output_dir: Path,
    *,
    run_config: Path,
    model_config: Path,
    require_existing: bool,
) -> dict[str, dict[str, str]]:
    snapshot_dir = output_dir / "config_snapshot"
    if not require_existing:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, dict[str, str]] = {}
    for name, source in {
        "run_config": run_config,
        "model_config": model_config,
    }.items():
        payload = source.read_bytes()
        destination = snapshot_dir / f"{name}.yaml"
        archived_path: Path | None = None
        if require_existing:
            if not destination.is_file():
                raise ValueError(f"resume config snapshot is missing: {destination}")
            if destination.read_bytes() != payload:
                raise ValueError(
                    f"resume config snapshot does not match {source}"
                )
        else:
            if destination.is_file() and destination.read_bytes() != payload:
                previous = destination.read_bytes()
                previous_sha = hashlib.sha256(previous).hexdigest()
                archive_dir = snapshot_dir / "history"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archived_path = archive_dir / f"{name}-{previous_sha[:12]}.yaml"
                if not archived_path.exists():
                    archived_path.write_bytes(previous)
            temporary = destination.with_suffix(".yaml.tmp")
            temporary.write_bytes(payload)
            temporary.replace(destination)
        snapshots[name] = {
            "source_path": str(source.resolve()),
            "snapshot_path": str(destination.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if archived_path is not None:
            snapshots[name]["archived_previous_path"] = str(
                archived_path.resolve()
            )
    return snapshots


def _recorded_path(manifest: dict[str, Any], key: str) -> Path | None:
    value = manifest.get(key)
    if not isinstance(value, dict):
        return None
    path = value.get("path")
    return Path(path) if isinstance(path, str) and path else None


def _resolve_input_paths(
    args: argparse.Namespace,
    manifest: dict[str, Any] | None,
) -> tuple[Path, Path, Path]:
    if not args.resume:
        missing = [
            option
            for option, value in {
                "--dataset": args.dataset,
                "--run-config": args.run_config,
                "--model-config": args.model_config,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "new benchmark runs require " + ", ".join(missing)
            )
        return args.dataset, args.run_config, args.model_config

    if manifest is None:
        raise ValueError("resume requires an existing run manifest")
    dataset = args.dataset or _recorded_path(manifest, "dataset")
    snapshot_dir = args.output_dir / "config_snapshot"
    run_snapshot = snapshot_dir / "run_config.yaml"
    model_snapshot = snapshot_dir / "model_config.yaml"
    snapshot_metadata = manifest.get("config_snapshot") or {}

    def latest_snapshot(key: str, fallback: Path) -> Path | None:
        recorded = snapshot_metadata.get(key) or {}
        recorded_path = recorded.get("snapshot_path")
        if isinstance(recorded_path, str) and Path(recorded_path).is_file():
            return Path(recorded_path)
        return fallback if fallback.is_file() else None

    run_config = args.run_config or latest_snapshot("run_config", run_snapshot)
    model_config = args.model_config or latest_snapshot(
        "model_config", model_snapshot
    )
    missing = [
        option
        for option, value in {
            "--dataset": dataset,
            "--run-config": run_config,
            "--model-config": model_config,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "resume output has no recorded input for "
            + ", ".join(missing)
            + "; provide the missing fallback explicitly"
        )
    return dataset, run_config, model_config


def _write_artifact(output_dir: Path, artifact: dict) -> Path:
    path = output_dir / f"{artifact['episode_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    return path


def _json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _interaction_progress(
    artifact: dict[str, Any] | None,
    target: int,
) -> tuple[int, set[int]]:
    """Return protocol milestones and exhausted-provider error slots.

    Environment validation and response-format errors are part of the normal
    repair loop, so they do not make a progress cell red.  ``provider_error``
    records are written only after the provider retry budget is exhausted; the
    returned slots drive the red cells in the TTY progress view, while the
    plain progress line reports per-branch failed counts instead.
    """

    if not artifact:
        return 0, set()
    position = 0
    error_slots: set[int] = set()
    conversation = artifact.get("conversation") or {}
    for event in conversation.get("events") or []:
        if event.get("type") != "environment_turn":
            continue
        is_milestone = event.get("stage") in {"reveal", "completion"}
        if is_milestone:
            position = min(position + 1, target)
    if not conversation:
        state = artifact.get("private_state") or {}
        completed = int(state.get("completed_batch_count", 0))
        position = min(
            completed + len(artifact.get("action_spans") or []),
            target,
        )
    if any(
        record.get("status") == "provider_error"
        for record in artifact.get("tool_trace") or []
    ):
        error_slots.add(min(position, target - 1))
    return position, error_slots


def _progress_cell_color(
    reached: int,
    episode_count: int,
    *,
    provider_error: bool = False,
) -> str | None:
    """Classify one aggregate milestone cell for terminal rendering."""

    if provider_error:
        return "red"
    if reached <= 0:
        return None
    if reached >= episode_count:
        return "green"
    if reached * 2 > episode_count:
        return "yellow"
    return "gray"


def _progress_source(
    artifact: dict[str, Any] | None,
    branch_name: str,
    history_policy: str,
    *,
    branch_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not artifact:
        return None
    branch = (artifact.get("branches") or {}).get(branch_name)
    if branch is not None:
        return branch
    if artifact.get("trace_schema_version") != BRANCHED_TRACE_SCHEMA_VERSION:
        return artifact
    prefix_key = history_policy
    if branch_specs and branch_name in branch_specs:
        groups = {
            _branch_prefix_key(spec): {}
            for spec in branch_specs.values()
        }
        prefix_key = _prefix_group_labels(groups)[
            _branch_prefix_key(branch_specs[branch_name])
        ]
    prefix = (artifact.get("shared_prefixes") or {}).get(prefix_key)
    if prefix is not None:
        return prefix
    if prefix_key != history_policy:
        # Artifacts written before probe-policy grouping used the history
        # policy as the prefix key.  That legacy prefix is reusable only for
        # the default probe policy; non-default settings must start fresh.
        requested = branch_specs.get(branch_name) if branch_specs else None
        requested_policy = (
            bool(requested.get("enable_probe_skip", True)),
            requested.get("least_probe_count"),
        ) if requested else None
        recorded_policy = (
            bool(artifact.get("enable_probe_skip", True)),
            artifact.get("least_probe_count"),
        )
        if requested_policy != recorded_policy:
            return None
    if artifact.get("history_policy") == history_policy:
        return artifact.get("shared_prefix")
    return None


class _TerminalProgress:
    def __init__(
        self,
        branches: dict[str, dict[str, Any]],
        episodes: dict[str, dict],
    ) -> None:
        self.branches = branches
        self.episodes = episodes
        self.artifacts: dict[str, dict] = {}
        self._rendered_lines = 0
        self._last_plain_signature: tuple[Any, ...] | None = None

    def update(self, artifact: dict) -> None:
        self.artifacts[str(artifact["episode_id"])] = artifact
        self.render()

    def render(self) -> None:
        targets = {
            episode_id: 2
            * int(episode["agent_task"]["interaction_protocol"]["action_batch_count"])
            for episode_id, episode in self.episodes.items()
        }
        if set(targets.values()) != {24}:
            raise ValueError("runtime progress requires exactly 24 protocol milestones")
        rows: list[tuple[str, list[int], set[int], int, int, int]] = []
        for name, spec in self.branches.items():
            positions: list[int] = []
            errors: set[int] = set()
            finished = 0
            failed = 0
            for episode_id, target in targets.items():
                parent = self.artifacts.get(episode_id)
                source = _progress_source(
                    parent,
                    name,
                    spec["history_policy"],
                    branch_specs=self.branches,
                )
                position, source_errors = _interaction_progress(source, target)
                positions.append(position)
                errors.update(source_errors)
                if source and source.get("run_status") in {"completed", "failed"}:
                    finished += 1
                if source and source.get("run_status") == "failed":
                    failed += 1
            rows.append((name, positions, errors, finished, failed, sum(targets.values())))

        signature = tuple(
            (name, tuple(positions), tuple(sorted(errors)), finished, failed)
            for name, positions, errors, finished, failed, _ in rows
        )
        is_tty = sys.stdout.isatty()
        if not is_tty:
            if signature == self._last_plain_signature:
                return
            self._last_plain_signature = signature
            summary = " ".join(
                f"{name}={sum(positions)}/{total}"
                + (f" errors={failed}" if failed else "")
                for name, positions, errors, _, failed, total in rows
            )
            print(f"[progress] {summary}", flush=True)
            return

        colors = {
            "green": "\033[32m",
            "yellow": "\033[33m",
            "red": "\033[31m",
            "gray": "\033[90m",
        }
        reset = "\033[0m"
        lines = [f"{len(self.branches)} branches · 24 interactions each"]
        width = max(len(name) for name in self.branches)
        episode_count = len(self.episodes)
        for name, positions, errors, finished, _, total in rows:
            cells = []
            for slot in range(24):
                reached = sum(position > slot for position in positions)
                color = _progress_cell_color(
                    reached,
                    episode_count,
                    provider_error=slot in errors,
                )
                cells.append(
                    " "
                    if color is None
                    else f"{colors[color]}│{reset}"
                )
            lines.append(
                f"{name:<{width}}  {''.join(cells)} "
                f"{sum(positions):>4}/{total:<4} episodes {finished}/{episode_count}"
            )
        if self._rendered_lines:
            sys.stdout.write(f"\033[{self._rendered_lines}F")
        for line in lines:
            sys.stdout.write(f"\033[2K{line}\n")
        sys.stdout.flush()
        self._rendered_lines = len(lines)


def _progress_snapshot(
    branches: dict[str, dict[str, Any]],
    episodes: dict[str, dict],
    artifacts: dict[str, dict],
) -> dict[str, Any]:
    output: dict[str, Any] = {"interactions_per_episode": 24, "branches": {}}
    for branch_name, spec in branches.items():
        positions: dict[str, int] = {}
        errors: dict[str, list[int]] = {}
        for episode_id, episode in episodes.items():
            target = 2 * int(
                episode["agent_task"]["interaction_protocol"][
                    "action_batch_count"
                ]
            )
            if target != 24:
                raise ValueError(
                    "runtime progress requires exactly 24 protocol milestones"
                )
            source = _progress_source(
                artifacts.get(episode_id),
                branch_name,
                spec["history_policy"],
                branch_specs=branches,
            )
            position, error_slots = _interaction_progress(source, target)
            positions[episode_id] = position
            if error_slots:
                errors[episode_id] = sorted(error_slots)
        output["branches"][branch_name] = {
            "episodes": positions,
            "errors": errors,
        }
    return output


def _resume_branch_change_is_compatible(
    manifest: dict[str, Any],
    requested_branches: dict[str, dict[str, Any]],
    *,
    dataset_mode: str,
    requested_history: dict[str, Any],
) -> bool:
    """Validate the branch change allowed by an additive resume.

    Dynamic runs may add branches to an existing output.  Existing branch
    definitions remain pinned so that an output cannot silently mix results
    from two meanings of the same setting.  Static runs still require the
    original branch mapping because their single trajectory depends on the
    selected history policy.
    """

    previous_identity = manifest.get("run_identity") or {}
    previous_branches = previous_identity.get("branches")
    if previous_branches is None:
        previous_branches = manifest.get("branches")
    if not isinstance(previous_branches, dict):
        return False

    if dataset_mode != "dynamic":
        if previous_branches != requested_branches:
            return False
    else:
        previous_names = set(previous_branches)
        requested_names = set(requested_branches)
        if not previous_names <= requested_names:
            return False
        if any(
            requested_branches.get(name) != previous_branches[name]
            for name in previous_names
        ):
            return False

    previous_history = previous_identity.get("history")
    if previous_history is None:
        previous_history = manifest.get("history")
    if previous_history is not None:
        if not isinstance(previous_history, dict):
            return False
        if previous_history.get("version") != requested_history.get("version"):
            return False
        previous_policies = set(previous_history.get("policies") or [])
        requested_policies = set(requested_history.get("policies") or [])
        if not previous_policies <= requested_policies:
            return False
    return True


def _artifact_covers_requested_branches(
    artifact: dict[str, Any],
    requested_branch_names: set[str],
    *,
    dataset_mode: str,
    branch_specs: dict[str, dict[str, Any]] | None = None,
    collect_relevant_fields: bool = False,
) -> bool:
    """Return whether an artifact can be reused without running anything."""

    if artifact.get("run_status") != "completed":
        return False
    if dataset_mode != "dynamic":
        if not collect_relevant_fields:
            return True
        assessment = artifact.get("relevant_fields_assessments") or {}
        if not isinstance(assessment, dict):
            return False
        return assessment_is_terminal(assessment.get("final"))
    branches = artifact.get("branches")
    if not isinstance(branches, dict) or set(branches) != requested_branch_names:
        return False
    if not collect_relevant_fields:
        return True
    if not isinstance(branch_specs, dict):
        return False
    for name in requested_branch_names:
        branch = branches.get(name) or {}
        assessments = branch.get("relevant_fields_assessments") or {}
        required = (
            {"pre_change", "post_change"}
            if bool((branch_specs.get(name) or {}).get("rule_change"))
            else {"final"}
        )
        if not isinstance(assessments, dict):
            return False
        if any(
            not assessment_is_terminal(assessments.get(phase))
            for phase in required
        ):
            return False
    return True


def _resumable_artifacts(
    output_dir: Path, episodes: dict[str, dict]
) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for path in output_dir.glob("*.json"):
        if path.name in {"manifest.json", "metrics.json"}:
            continue
        try:
            artifact = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        episode_id = artifact.get("episode_id")
        if episode_id not in episodes:
            continue
        expected = episodes[str(episode_id)]
        expected_trace_schema = (
            BRANCHED_TRACE_SCHEMA_VERSION
            if expected["dataset_mode"] == "dynamic"
            else sequential_trace_schema_version(expected["dataset_mode"])
        )
        if artifact.get("trace_schema_version") != expected_trace_schema:
            raise ValueError(
                f"completed artifact {episode_id} uses an incompatible runtime "
                "trace schema; use a new output directory"
            )
        if artifact.get("episode_schema_version") != expected["schema_version"]:
            raise ValueError(
                f"completed artifact {episode_id} uses an incompatible episode "
                "schema; use a new output directory"
            )
        if artifact.get("episode_sha256") != _json_sha256(expected):
            raise ValueError(
                f"completed artifact {episode_id} does not match the current "
                "dataset; use a new output directory"
            )
        artifacts[str(episode_id)] = artifact
    return artifacts


async def main(args: argparse.Namespace) -> None:
    # Pin provenance before loading cases. A long benchmark may overlap with a
    # dataset/config regeneration at the same path; hashing again at the end
    # would then describe different bytes than the run actually consumed.
    manifest_path = args.output_dir / "manifest.json"
    existing_run_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )
    dataset, run_config, model_config = _resolve_input_paths(
        args, existing_run_manifest
    )
    cfg = yaml.safe_load(run_config.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError("run config must be a mapping")
    dataset_provenance = _dataset_provenance(dataset)
    dataset_identity = _file_identity(dataset)
    run_config_identity = _file_identity(run_config)
    model_config_identity = _file_identity(model_config)
    launch = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    recorded_identity = (existing_run_manifest or {}).get("run_identity") or {}
    selected_episode_ids = (
        list(recorded_identity.get("episode_ids") or [])
        if args.resume
        else args.episode_id
    )
    seed = int(recorded_identity.get("seed", args.seed)) if args.resume else args.seed
    cases = load_cases(
        dataset,
        None if args.resume else args.max_cases,
        selected_episode_ids,
    )
    if not cases:
        raise ValueError("benchmark dataset contains no selected episodes")
    modes = {str(case.get("dataset_mode") or "") for case in cases}
    if len(modes) != 1 or next(iter(modes)) not in {"static", "dynamic"}:
        raise ValueError("benchmark dataset must contain one recognized dataset mode")
    dataset_mode = next(iter(modes))
    for case in cases:
        validate_episode(case, dataset_mode)
    interaction_policy = cfg.get("interaction")
    if not isinstance(interaction_policy, dict):
        raise ValueError("runtime config requires an interaction mapping")
    runtime_branches = resolve_runtime_branches(cfg.get("branches"))
    history_policies = sorted({
        spec["history_policy"] for spec in runtime_branches.values()
    })
    history_policy = next(iter(runtime_branches.values()))["history_policy"]
    if dataset_mode == "dynamic":
        available = set(cases[0]["post_branches"])
        invalid_sampling_branches = sorted({
            spec["sampling"]
            for spec in runtime_branches.values()
            if spec["sampling"] not in available
        })
        if invalid_sampling_branches:
            raise ValueError(
                "runtime branches select sampling branches absent from the dataset: "
                f"{invalid_sampling_branches}"
            )
    requested_branch_names = list(runtime_branches)
    episode_hashes = {
        str(case["episode_id"]): _json_sha256(case) for case in cases
    }
    run_identity = {
        "dataset_sha256": dataset_identity["sha256"],
        "episode_schema_versions": sorted(
            {str(case["schema_version"]) for case in cases}
        ),
        "run_config_sha256": run_config_identity["sha256"],
        "model_config_sha256": model_config_identity["sha256"],
        "policy": "agent",
        "seed": seed,
        "episode_ids": [str(case["episode_id"]) for case in cases],
        "episode_sha256": episode_hashes,
        "branches": runtime_branches,
        "history": {
            "policies": history_policies,
            "version": HISTORY_POLICY_VERSION,
        },
        "interaction_policy": interaction_policy,
    }
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise ValueError(
            f"output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_config_snapshot = (
        (existing_run_manifest or {}).get("config_snapshot") or {}
    )
    if args.resume:
        if not existing_run_manifest:
            raise ValueError("resume requires an existing run manifest")
        previous_identity = existing_run_manifest.get("run_identity") or {}
        invariant_keys = {
            "dataset_sha256",
            "episode_schema_versions",
            "policy",
            "seed",
            "episode_ids",
            "episode_sha256",
        }
        identity_mismatch = any(
            previous_identity.get(key) != run_identity.get(key)
            for key in invariant_keys
        )
        if not _resume_branch_change_is_compatible(
            existing_run_manifest,
            runtime_branches,
            dataset_mode=dataset_mode,
            requested_history=run_identity["history"],
        ):
            identity_mismatch = True
        previous_interaction = previous_identity.get(
            "interaction_policy",
            existing_run_manifest.get("interaction_policy"),
        )
        if previous_interaction is not None and previous_interaction != interaction_policy:
            identity_mismatch = True
        if identity_mismatch:
            raise ValueError(
                "resume inputs change the pinned dataset, branches, or "
                "interaction protocol"
            )
        config_snapshot = _snapshot_configs(
            args.output_dir,
            run_config=run_config,
            model_config=model_config,
            require_existing=False,
        )
        launch_history = list(
            existing_run_manifest.get("launch_history") or []
        )
        if not launch_history and existing_run_manifest.get("launch"):
            launch_history.append({
                "launch": existing_run_manifest["launch"],
                "config_snapshot": previous_config_snapshot,
            })
        launch_history.append({
            "launch": launch,
            "config_snapshot": config_snapshot,
        })
        config_changes = list(
            existing_run_manifest.get("config_changes") or []
        )
    else:
        config_snapshot = _snapshot_configs(
            args.output_dir,
            run_config=run_config,
            model_config=model_config,
            require_existing=False,
        )
        launch_history = [
            {"launch": launch, "config_snapshot": config_snapshot}
        ]
        config_changes = []
        initial_manifest = {
            "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "status": "running",
            "run_identity": run_identity,
            "dataset": dataset_identity,
            "run_config": run_config_identity,
            "model_config": model_config_identity,
            "config_snapshot": config_snapshot,
            "launch": launch,
            "launch_history": launch_history,
            "config_changes": config_changes,
            "branches": runtime_branches,
            "interaction_policy": interaction_policy,
            "history": run_identity["history"],
            "code": code_identity(),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(initial_manifest, ensure_ascii=False, indent=2) + "\n"
        )
    episode_index = {
        str(case["episode_id"]): case for case in cases
    }
    existing_artifacts = (
        _resumable_artifacts(
            args.output_dir,
            episode_index,
        )
        if args.resume
        else {}
    )
    if args.resume:
        previous_identity = existing_run_manifest.get("run_identity") or {}
        previous_hashes = {
            "run_config": (
                (existing_run_manifest.get("run_config") or {}).get("sha256")
                or previous_identity.get("run_config_sha256")
            ),
            "model_config": (
                (existing_run_manifest.get("model_config") or {}).get("sha256")
                or previous_identity.get("model_config_sha256")
            ),
        }
        current_hashes = {
            "run_config": run_config_identity["sha256"],
            "model_config": model_config_identity["sha256"],
        }
        changed = [
            name
            for name in ("run_config", "model_config")
            if previous_hashes[name] != current_hashes[name]
        ]
        if changed:
            previous_configs: dict[str, dict[str, Any]] = {}
            for name in ("run_config", "model_config"):
                previous = dict(previous_config_snapshot.get(name) or {})
                archived = config_snapshot[name].get(
                    "archived_previous_path"
                )
                if archived:
                    previous["snapshot_path"] = archived
                previous["sha256"] = previous_hashes[name]
                previous_configs[name] = previous
            config_changes.append({
                "changed_at": launch["started_at"],
                "changed": changed,
                "progress": _progress_snapshot(
                    runtime_branches,
                    episode_index,
                    existing_artifacts,
                ),
                "from": previous_configs,
                "to": config_snapshot,
                "launch": launch,
            })
        running_manifest = {
            **existing_run_manifest,
            "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "status": "running",
            "run_identity": run_identity,
            "dataset": dataset_identity,
            "run_config": run_config_identity,
            "model_config": model_config_identity,
            "config_snapshot": config_snapshot,
            "launch": launch,
            "launch_history": launch_history,
            "config_changes": config_changes,
            "branches": runtime_branches,
            "interaction_policy": interaction_policy,
            "history": run_identity["history"],
            "code": code_identity(),
        }
        manifest_path.write_text(
            json.dumps(running_manifest, ensure_ascii=False, indent=2) + "\n"
        )
    requested_branches = set(requested_branch_names)
    for episode_id, artifact in existing_artifacts.items():
        branches = artifact.get("branches")
        if dataset_mode == "dynamic" and branches is not None and not isinstance(
            branches, dict
        ):
            raise ValueError(
                f"resume artifact has an invalid branch map: {episode_id}"
            )
        observed = set(branches) if isinstance(branches, dict) else set()
        if not observed <= requested_branches:
            raise ValueError(
                f"resume artifact has the wrong branch set: {episode_id}"
            )
    resumed = {
        episode_id: artifact
        for episode_id, artifact in existing_artifacts.items()
        if _artifact_covers_requested_branches(
            artifact,
            requested_branches,
            dataset_mode=dataset_mode,
            branch_specs=runtime_branches,
            collect_relevant_fields=True,
        )
    }
    # A completed dynamic artifact that predates newly requested branches is
    # resumable too: pass it through to the branched runner so it can retain
    # the completed branches/prefix and execute only the missing branches.
    partial = {
        episode_id: artifact
        for episode_id, artifact in existing_artifacts.items()
        if episode_id not in resumed
    }
    pending_cases = [case for case in cases if case["episode_id"] not in resumed]
    if resumed:
        print(f"Resuming with {len(resumed)} completed episodes", flush=True)
    extending = [
        episode_id
        for episode_id in partial
        if existing_artifacts[episode_id].get("run_status") == "completed"
    ]
    if extending:
        print(
            f"Extending {len(extending)} completed episodes with new branches",
            flush=True,
        )
    load_repo_env(model_config.parent)
    model_cfg = yaml.safe_load(model_config.read_text()) or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping")
    concurrency = int(model_cfg.get("concurrency", 1))
    if concurrency < 1:
        raise ValueError("model config concurrency must be at least 1")
    client = build_generator(model_cfg)
    semaphore = asyncio.Semaphore(min(concurrency, 20))
    request_semaphore = asyncio.Semaphore(concurrency)

    class RequestLimitedClient:
        config = client.config

        async def generate(self, **kwargs):
            async with request_semaphore:
                return await client.generate(**kwargs)

    limited_client = RequestLimitedClient()
    progress_lock = asyncio.Lock()
    terminal_progress = _TerminalProgress(runtime_branches, episode_index)
    terminal_progress.artifacts.update(existing_artifacts)
    terminal_progress.render()

    async def report_progress(artifact: dict) -> None:
        async with progress_lock:
            artifact["episode_sha256"] = episode_hashes[artifact["episode_id"]]
            _write_artifact(args.output_dir, artifact)
            terminal_progress.update(artifact)

    async def bounded(episode: dict) -> dict:
        async with semaphore:
            if dataset_mode == "dynamic":
                artifact = await run_branched_agent_episode(
                    episode,
                    limited_client,
                    interaction_policy=interaction_policy,
                    branches=runtime_branches,
                    checkpoint=report_progress,
                    resume_artifact=partial.get(str(episode["episode_id"])),
                    collect_relevant_fields=True,
                )
            else:
                artifact = await run_agent_episode(
                    episode,
                    limited_client,
                    interaction_policy=interaction_policy,
                    checkpoint=report_progress,
                    condition="static",
                    resume_artifact=partial.get(str(episode["episode_id"])),
                    history_policy=history_policy,
                    enable_probe_skip=runtime_branches[requested_branch_names[0]].get(
                        "enable_probe_skip", True
                    ),
                    least_probe_count=runtime_branches[requested_branch_names[0]].get(
                        "least_probe_count"
                    ),
                    collect_relevant_fields=True,
                )
            artifact["episode_sha256"] = episode_hashes[episode["episode_id"]]
            _write_artifact(args.output_dir, artifact)
            return artifact

    new_artifacts = list(
        await asyncio.gather(*(bounded(item) for item in pending_cases))
    )
    indexed_artifacts = {
        **resumed,
        **{item["episode_id"]: item for item in new_artifacts},
    }
    artifacts = [indexed_artifacts[case["episode_id"]] for case in cases]
    episodes_by_id = {case["episode_id"]: case for case in cases}
    artifact_bytes: dict[str, int] = {}
    for artifact in artifacts:
        artifact_path = _write_artifact(args.output_dir, artifact)
        artifact_bytes[artifact["episode_id"]] = artifact_path.stat().st_size
    metrics = (
        aggregate_static_metrics(artifacts)
        if dataset_mode == "static"
        else (
            aggregate_branched_metrics(artifacts, episodes=episodes_by_id)
        )
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    failed_ids = [
        item["episode_id"]
        for item in artifacts
        if item.get("run_status") != "completed"
    ]
    if not artifacts:
        run_status = "failed"
    elif failed_ids:
        run_status = "completed_with_failures"
    else:
        run_status = "completed"
    manifest = {
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "release": (dataset_provenance["manifest"] or {}).get("release"),
        "dataset_manifest_provenance": dataset_provenance,
        "status": run_status,
        "run_identity": run_identity,
        "dataset": dataset_identity,
        "run_config": run_config_identity,
        "model_config": model_config_identity,
        "config_snapshot": config_snapshot,
        "launch": launch,
        "launch_history": launch_history,
        "config_changes": config_changes,
        "code": code_identity(),
        "policy": "agent",
        "evaluation_mode": "branched" if dataset_mode == "dynamic" else "static",
        "branches": runtime_branches,
        "interaction_policy": interaction_policy,
        "history": {
            "policies": history_policies,
            "version": HISTORY_POLICY_VERSION,
        },
        "episode_count": len(artifacts),
        "episode_ids": [item["episode_id"] for item in artifacts],
        "failed_episode_ids": failed_ids,
        "artifact_bytes": artifact_bytes,
        "artifact_bytes_total": sum(artifact_bytes.values()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"Wrote {len(artifacts)} {dataset_mode} episodes "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a AdaExploreBench static or dynamic sequential dataset"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="episode JSONL path; required for a new run",
    )
    parser.add_argument(
        "--run-config",
        dest="run_config",
        type=Path,
        help="run config; required for a new run",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="model/provider config; required for a new run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse completed artifacts; dynamic runs may execute only newly "
            "added branches"
        ),
    )
    parsed = parser.parse_args()
    if not parsed.resume:
        missing = [
            option
            for option, value in {
                "--dataset": parsed.dataset,
                "--run-config": parsed.run_config,
                "--model-config": parsed.model_config,
            }.items()
            if value is None
        ]
        if missing:
            parser.error("new runs require " + ", ".join(missing))
    asyncio.run(main(parsed))
