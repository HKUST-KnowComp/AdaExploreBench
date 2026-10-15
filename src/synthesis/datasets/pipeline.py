"""Single incremental coordinator for static and dynamic synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import signal
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from protocol.constraints import compile_core_constraints
from protocol.mechanism import validate_mechanism
from .config import DatasetGenerationConfig
from .contracts import GeneratedDataset
from .dynamic.episodes import make_dynamic_episode
from .dynamic.pipeline import initialize_dynamic_worker, run_dynamic_cell_stage, run_dynamic_core_task
from .dynamic.structures import DynamicCoreSpace
from .evaluation import (
    CalibrationDegeneracyError,
    fit_frozen_pre_calibration,
    normalized_state_arrays,
)
from .final_review import (
    load_dynamic_episodes,
    load_final_review_config,
    review_manifest_entry,
    run_final_review,
    serialized_review,
)
from .output import ExistingDataset, inspect_existing_dataset, write_dataset
from .selection import (
    load_selection_config,
    published_dataset_manifest_entry,
    select_episodes,
    selection_manifest_entry,
    serialized_selected_episodes,
    serialized_selection_report,
)
from .planning import BoundedPermutation, GenerationCell, episode_spec_for_accept, generation_cells, quota_targets, stable_seed
from .population import PopulationBudgetExhausted, prepare_base_proposals, sample_states
from .semantic_review import review_candidate_batch
from .state import DatasetRunState, capture_worker_error, write_worker_candidate, write_worker_cell_checkpoint
from .static.episodes import make_static_episode
from .static.rules import sample_rule
from .static.structures import StaticCoreSpace, materialize_distractors
from .template_pool import load_template_pool, select_template


def _emit_progress(event: str, **values: Any) -> None:
    fields = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {event} {fields}".rstrip(), flush=True)


def _advance_heartbeat_deadline(
    now: float, deadline: float, interval: float
) -> tuple[bool, float]:
    if now < deadline:
        return False, deadline
    while deadline <= now:
        deadline += interval
    return True, deadline


def _initialize_worker(config: DatasetGenerationConfig, pool: dict[str, Any]) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    initialize_dynamic_worker(config, pool)


def _priority_cells(
    cells: list[GenerationCell],
    targets: dict[tuple[str, ...], int],
    accepted_by_group: dict[tuple[str, ...], int],
    accepted_by_cell: dict[tuple[str, ...], int],
    checkpoints: dict[tuple[str, ...], dict[str, Any]],
    active_counts: dict[tuple[str, ...], int],
    reviewing: set[tuple[str, ...]],
    callbacks: set[tuple[str, ...]],
) -> list[tuple[int, int, int, int, GenerationCell]]:
    """Build the live CPU queue; callbacks and idle cells outrank shared cells."""

    scheduled: list[tuple[int, int, int, int, GenerationCell]] = []
    for group, target in targets.items():
        if accepted_by_group[group] >= target:
            continue
        group_cells = [cell for cell in cells if cell.group_key == group]
        unmet = [
            cell
            for cell in group_cells
            if accepted_by_cell[cell.key] < cell.initial_quota
            and checkpoints.get(cell.key, {}).get("stage") != "exhausted"
        ]
        choices = unmet
        if not choices:
            choice = min(
                (
                    cell
                    for cell in group_cells
                    if checkpoints.get(cell.key, {}).get("stage") != "exhausted"
                ),
                key=lambda cell: (accepted_by_cell[cell.key], cell.priority),
                default=None,
            )
            choices = [choice] if choice is not None else []
        for cell in choices:
            if cell.key in reviewing:
                continue
            active = active_counts.get(cell.key, 0)
            tier = 0 if cell.key in callbacks else 1 if active == 0 else 2
            scheduled.append(
                (tier, active, accepted_by_cell[cell.key], cell.priority, cell)
            )
    heapq.heapify(scheduled)
    return scheduled


def _static_core_id(template_id: str, family: str, core: dict[str, Any]) -> str:
    payload = json.dumps([template_id, family, core["relevant_fields"]], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _run_static_cell_stage(
    selection: Any,
    cell: GenerationCell,
    config: DatasetGenerationConfig,
    checkpoint: dict[str, Any],
    *,
    order_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    space = StaticCoreSpace(selection, cell.rule_family)
    traversal = BoundedPermutation(space.count, stable_seed(config.dataset.seed, *cell.key, "core-order"), config.resources.max_core_visits_per_cell, materialize_threshold=config.resources.max_core_visits_per_cell, order_path=order_path, state=checkpoint.get("traversal"))
    candidates = list(checkpoint.get("candidates") or [])
    candidate_files = list(checkpoint.get("candidate_files") or [])
    current = checkpoint.get("current_core")
    counts = dict(checkpoint.get("counts") or {})
    counts.setdefault("visited", int((checkpoint.get("traversal") or {}).get("cursor", 0)))
    counts.setdefault("certified", len(candidates))
    counts.setdefault("rejected", 0)
    counts.setdefault("errors", 0)
    worker_errors = list(checkpoint.get("worker_errors") or [])

    def persist(stage: str) -> None:
        write_worker_cell_checkpoint(run_dir, cell.key, stage, {
            "traversal": traversal.snapshot(), "current_core": current,
            "candidate_files": candidate_files,
            "surplus_files": list(checkpoint.get("surplus_files") or []),
            "space_count": space.count, "counts": counts,
            "worker_errors": worker_errors,
            "visit_budget_truncated": space.count > traversal.limit and traversal.cursor >= traversal.limit,
        })

    while len(candidates) < config.semantic_review.batch_size:
        if current is None:
            try:
                current = space.at(traversal.next())
            except StopIteration:
                break
        persist("running")
        certified_before = len(candidates)
        failed = False
        try:
            fields = list(map(str, current["relevant_fields"]))
            compiled = compile_core_constraints(selection, fields, saved_decimal_places=config.search.saved_decimal_places, solver_timeout_ms=config.search.solver_timeout_ms, conditional_branch_limit=config.search.conditional_branch_limit)
            if compiled["status"] == "feasible":
                rule = sample_rule(selection, current, cell.rule_family, stable_seed(config.dataset.seed, *cell.key, current["core_space_index"], "rule"))
                validate_mechanism(rule, selection.fields)
                base = prepare_base_proposals(selection, fields, stable_seed(config.dataset.seed, *cell.key, current["core_space_index"], "calibration"))
                try:
                    states = sample_states(selection, fields, base, compiled, replacement_group_id=None, effective_coupling=None, config=config.search, count=config.search.calibration_sample_count)
                    rule = fit_frozen_pre_calibration(rule, normalized_state_arrays(selection, fields, states))
                except (PopulationBudgetExhausted, CalibrationDegeneracyError):
                    states = []
                if states:
                    transition = materialize_distractors(selection, current, stable_seed(config.dataset.seed, *cell.key, current["core_space_index"], "distractors"), config.static.visible_field_count, 0)
                    candidates.append({"core_id": _static_core_id(selection.template_id, cell.rule_family, current), "template_id": selection.template_id, "catalog_seed": stable_seed(config.dataset.seed, *cell.key, current["core_space_index"], "numeric"), "mode": "static", "dynamic_type": None, "_core": current, "rule": rule, "compiled_constraints": compiled, "transition": transition, "visible_fields": [*fields, *transition["distractor_fields"]], "distractor_source_ids": list(transition["distractor_fields"]), "distractor_trial": 0})
                    candidate_files.append(write_worker_candidate(run_dir, cell.key, candidates[-1]))
                    counts["certified"] += 1
            elif compiled["status"] == "unknown":
                raise RuntimeError(f"constraint solver did not establish feasibility: {compiled}")
        except Exception as error:
            capture_worker_error(cell.key, current, error, worker_errors)
            counts["errors"] += 1
            failed = True
        if len(candidates) == certified_before:
            counts["rejected"] += 1
        current = None
        counts["visited"] = traversal.cursor
        if failed or candidates or traversal.cursor % config.resources.checkpoint_interval == 0:
            persist("running")
    result = {"stage": "awaiting_review" if candidates else "exhausted", "traversal": traversal.snapshot(), "current_core": current, "candidate_files": candidate_files, "surplus_files": list(checkpoint.get("surplus_files") or []), "space_count": space.count, "counts": counts, "worker_errors": worker_errors, "visit_budget_truncated": space.count > traversal.limit and traversal.cursor >= traversal.limit}
    write_worker_cell_checkpoint(run_dir, cell.key, result["stage"], result)
    return result


def _retry_distractors(selection: Any, cell: GenerationCell, candidate: dict[str, Any], config: DatasetGenerationConfig) -> dict[str, Any] | None:
    trial = int(candidate["distractor_trial"]) + 1
    if trial >= config.semantic_review.max_distractor_trials_per_core:
        return None
    try:
        if cell.mode == "dynamic":
            from .dynamic.structures import materialize_distractors as materialize_dynamic, visible_fields
            transition = materialize_dynamic(selection, candidate["_core"], str(cell.dynamic_type), stable_seed(config.dataset.seed, *cell.key, candidate["_core"]["core_space_index"], "distractors"), config.dynamic.dataset_design.visible_field_count, trial)
            visible = visible_fields(transition, str(cell.dynamic_type))
        else:
            transition = materialize_distractors(selection, candidate["_core"], stable_seed(config.dataset.seed, *cell.key, candidate["_core"]["core_space_index"], "distractors"), config.static.visible_field_count, trial)
            visible = [*candidate["_core"]["relevant_fields"], *transition["distractor_fields"]]
    except (IndexError, ValueError):
        return None
    return {**candidate, "transition": transition, "visible_fields": visible, "distractor_trial": trial}


async def _review_cell_result(
    cell: GenerationCell,
    result: dict[str, Any],
    selection: Any,
    config: DatasetGenerationConfig,
    state: DatasetRunState,
    review_semaphore: asyncio.Semaphore,
) -> tuple[
    GenerationCell,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    files = list(result.get("candidate_files") or [])
    candidates = [state.read_candidate(path) for path in files]
    accepted: list[dict[str, Any]] = []
    retryable: list[dict[str, Any]] = []
    audited: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if candidates:
        batch_identity = [
            (candidate["core_id"], candidate.get("distractor_trial", 0))
            for candidate in candidates
        ]
        batch_id = hashlib.sha256(
            json.dumps([cell.key, batch_identity], separators=(",", ":")).encode()
        ).hexdigest()[:20]
        saved = next(
            (
                batch for batch in state.value.get("review_batches", [])
                if batch.get("batch_id") == batch_id
            ),
            None,
        )
        if saved is None:
            async with review_semaphore:
                accepted, retryable, audited, errors = await review_candidate_batch(
                    [(selection, candidates)], config
                )
        else:
            by_identity = {
                (str(candidate["core_id"]), int(candidate.get("distractor_trial", 0))): candidate
                for candidate in candidates
            }
            for item in saved.get("candidates", []):
                candidate = by_identity[(str(item["core_id"]), int(item["distractor_trial"]))]
                reviewed = {**candidate, "semantic_review": item["review"]}
                audited.append(reviewed)
                if item["review"]["outcome"] == "accept":
                    accepted.append(reviewed)
                elif item["review"]["outcome"] == "reject_distractors":
                    retryable.append(reviewed)
            errors = list(saved.get("errors", []))
        state.record_review_batch({
            "batch_id": batch_id,
            "cell": list(cell.key),
            "candidates": [
                {
                    "core_id": candidate["core_id"],
                    "distractor_trial": candidate.get("distractor_trial", 0),
                    "visible_fields": list(candidate["visible_fields"]),
                    "review": candidate["semantic_review"],
                }
                for candidate in audited
            ],
            "errors": [{**error, "cell": list(cell.key)} for error in errors],
        })
    pending = [
        retried
        for candidate in retryable
        if (retried := _retry_distractors(selection, cell, candidate, config))
        is not None
    ]
    return (
        cell,
        result,
        accepted,
        pending,
        files,
        [{**error, "cell": list(cell.key)} for error in errors],
    )


def _episode_group(episode: dict[str, Any], mode: str) -> tuple[str, ...]:
    dynamic_type = (
        (episode.get("evaluation", {}).get("transition") or {}).get("type")
        or "static"
    )
    return (
        str(episode["provenance"]["domain"]),
        str(episode["evaluation"]["rule_family"]),
        mode,
        str(dynamic_type),
    )


def _complete_filled_groups(
    state: DatasetRunState,
    cells: list[GenerationCell],
    targets: dict[tuple[str, ...], int],
    accepted_by_group: dict[tuple[str, ...], int],
) -> None:
    episodes_by_group: dict[tuple[str, ...], list[str]] = {
        key: [] for key in targets
    }
    for episode_id in state.value["episodes"]:
        episode = state.episode(episode_id)["episode"]
        key = _episode_group(episode, episode["dataset_mode"])
        if key in episodes_by_group:
            episodes_by_group[key].append(episode_id)
    for group, target in targets.items():
        state_key = state._key(group)
        existing_group = state.value["groups"].get(state_key)
        if existing_group:
            if not existing_group.get("resources_cleaned"):
                state.clean_group_resources(group)
            continue
        if accepted_by_group[group] < target:
            continue
        resources = []
        for cell in cells:
            if cell.group_key != group:
                continue
            checkpoint = state.cell_checkpoint(cell.key) or {}
            resources.extend(checkpoint.get("candidate_files", []))
            resources.extend(checkpoint.get("prefetched_candidate_files", []))
            resources.extend(checkpoint.get("review_candidate_files", []))
            resources.extend(checkpoint.get("surplus_files", []))
            resources.append(str(state.cell_order_path(cell.key).relative_to(state.run_dir)))
            state.save_cell_checkpoint(
                cell.key,
                "completed",
                {
                    **{
                        key: value for key, value in checkpoint.items()
                        if key not in {
                            "stage",
                            "candidate_files",
                            "prefetched_candidate_files",
                            "review_candidate_files",
                            "surplus_files",
                        }
                    },
                    "candidate_files": [],
                    "prefetched_candidate_files": [],
                    "review_candidate_files": [],
                    "surplus_files": [],
                },
            )
        state.mark_group_complete(group, episodes_by_group[group], resources)
        state.clean_group_resources(group)


async def _generate_dataset(config: DatasetGenerationConfig, existing: ExistingDataset, state: DatasetRunState, pool: dict[str, Any]) -> GeneratedDataset:
    cells = list(generation_cells(config, pool))
    targets = quota_targets(config, cells)
    accepted_by_group = {key: 0 for key in targets}
    accepted_by_cell = {cell.key: 0 for cell in cells}
    for checkpoint in state.value.get("episodes", {}):
        episode = state.episode(checkpoint)["episode"]
        key = _episode_group(episode, config.dataset.mode)
        if key in accepted_by_group:
            accepted_by_group[key] += 1
        cell_key = (
            str(episode["provenance"]["domain"]),
            str(episode["provenance"]["template_id"]),
            str(episode["evaluation"]["rule_family"]),
            config.dataset.mode,
            key[-1],
        )
        if cell_key in accepted_by_cell:
            accepted_by_cell[cell_key] += 1
    sequence = existing.starting_sequence + len(state.value.get("episodes", {}))
    diagnostics: list[dict[str, Any]] = []
    worker_count = min(config.resources.cpu_worker_count, max(1, len(cells)))
    review_semaphore = asyncio.Semaphore(config.semantic_review.concurrency)
    checkpoints: dict[tuple[str, ...], dict[str, Any]] = {}
    for cell in cells:
        checkpoint = state.cell_checkpoint(cell.key)
        if checkpoint is None:
            state.save_cell_checkpoint(cell.key, "ready", {
                "candidate_files": [], "surplus_files": [],
                "counts": {"visited": 0, "certified": 0, "rejected": 0},
            })
            checkpoint = state.cell_checkpoint(cell.key)
        checkpoints[cell.key] = checkpoint or {}
    _emit_progress("generation_started", cells=len(cells), workers=worker_count, resumed=len(state.value.get("episodes", {})))

    def save_checkpoint(cell: GenerationCell, checkpoint: dict[str, Any]) -> None:
        stage = str(checkpoint.get("stage") or "ready")
        checkpoint = {**checkpoint, "stage": stage}
        state.save_cell_checkpoint(
            cell.key,
            stage,
            {key: value for key, value in checkpoint.items() if key != "stage"},
        )
        checkpoints[cell.key] = checkpoint

    def commit_candidate(cell: GenerationCell, candidate: dict[str, Any]) -> None:
        nonlocal sequence
        if accepted_by_group[cell.group_key] >= targets[cell.group_key]:
            raise RuntimeError(
                f"episode quota already filled for group: {cell.group_key}"
            )
        group_slot = accepted_by_group[cell.group_key]
        spec = episode_spec_for_accept(
            config, cell, candidate, sequence=sequence, group_slot=group_slot
        )
        build = (
            make_dynamic_episode(pool, spec, config, candidate)
            if cell.mode == "dynamic"
            else make_static_episode(pool, spec, config, candidate)
        )
        state.record_episode(cell.key, candidate, build)
        diagnostics.append(build.diagnostics)
        accepted_by_group[cell.group_key] += 1
        accepted_by_cell[cell.key] += 1
        sequence += 1
        _emit_progress("episode_committed", cell=cell.priority, episode=spec.episode_id, total=len(state.value["episodes"]))

    def consume_surplus() -> bool:
        consumed = False
        for cell in cells:
            checkpoint = checkpoints[cell.key]
            surplus = list(checkpoint.get("surplus_files", []))
            while (
                surplus
                and accepted_by_cell[cell.key] < cell.initial_quota
                and accepted_by_group[cell.group_key] < targets[cell.group_key]
            ):
                path = surplus.pop(0)
                candidate = state.read_candidate(path)
                if not state.candidate_committed(cell.key, candidate):
                    commit_candidate(cell, candidate)
                save_checkpoint(cell, {**checkpoint, "surplus_files": surplus})
                state.delete_candidate(path)
                consumed = True
        for group, target in targets.items():
            group_cells = [cell for cell in cells if cell.group_key == group]
            initial_resolved = all(
                accepted_by_cell[cell.key] >= cell.initial_quota
                or (
                    checkpoints[cell.key].get("stage") == "exhausted"
                    and not checkpoints[cell.key].get("candidate_files")
                    and not checkpoints[cell.key].get("prefetched_candidate_files")
                    and not checkpoints[cell.key].get("review_candidate_files")
                )
                for cell in group_cells
            )
            while initial_resolved and accepted_by_group[group] < target:
                choices = []
                for cell in group_cells:
                    checkpoint = checkpoints[cell.key]
                    if checkpoint.get("surplus_files"):
                        choices.append((accepted_by_cell[cell.key], cell.priority, cell, checkpoint))
                if not choices:
                    break
                _, _, cell, checkpoint = min(choices)
                surplus = list(checkpoint["surplus_files"])
                path = surplus.pop(0)
                candidate = state.read_candidate(path)
                if not state.candidate_committed(cell.key, candidate):
                    commit_candidate(cell, candidate)
                save_checkpoint(cell, {**checkpoint, "surplus_files": surplus})
                state.delete_candidate(path)
                consumed = True
        return consumed

    callback_cells: set[tuple[str, ...]] = set()

    def apply_review(reviewed: tuple[Any, ...]) -> None:
        cell, result, accepted, pending, files, errors = reviewed
        for error in errors:
            state.record_review_error(error)
        pending_files = [state.write_candidate(cell.key, candidate) for candidate in pending]
        surplus_files = [state.write_candidate(cell.key, candidate) for candidate in accepted]
        latest = checkpoints[cell.key]
        traversal = latest["traversal"]
        exhausted = (
            int(traversal["cursor"]) >= int(traversal["limit"])
            and not latest.get("inflight_core_indices")
        )
        queued = [
            *latest.get("candidate_files", []),
            *latest.get("prefetched_candidate_files", []),
        ]
        next_files = [*pending_files, *queued]
        awaiting_review = len(next_files) >= config.semantic_review.batch_size or (
            exhausted and bool(next_files)
        )
        checkpoint = {
            **latest,
            "stage": (
                "awaiting_review" if awaiting_review
                else "ready" if next_files or not exhausted
                else "exhausted"
            ),
            "candidate_files": next_files,
            "prefetched_candidate_files": [],
            "review_candidate_files": [],
            "surplus_files": [
                *latest.get("surplus_files", []),
                *surplus_files,
            ],
        }
        save_checkpoint(cell, checkpoint)
        if checkpoint["stage"] == "ready":
            callback_cells.add(cell.key)
        for path in files:
            if path not in pending_files and path not in surplus_files:
                state.delete_candidate(path)
        _emit_progress("review_completed", cell=cell.priority, accepted=len(accepted), retry=len(pending), errors=len(errors))

    dynamic_traversals: dict[tuple[str, ...], BoundedPermutation] = {}
    live_dynamic_claims: set[tuple[tuple[str, ...], int]] = set()
    dirty_dynamic_cells: set[tuple[str, ...]] = set()

    def dynamic_traversal(cell: GenerationCell, checkpoint: dict[str, Any]) -> BoundedPermutation:
        traversal = dynamic_traversals.get(cell.key)
        if traversal is not None:
            return traversal
        selection = select_template(pool, cell.domain, cell.template_id)
        space = DynamicCoreSpace(selection, cell.rule_family, str(cell.dynamic_type))
        traversal = BoundedPermutation(
            space.count,
            stable_seed(config.dataset.seed, *cell.key, "core-order"),
            config.resources.max_core_visits_per_cell,
            materialize_threshold=config.resources.max_core_visits_per_cell,
            order_path=state.cell_order_path(cell.key),
            state=checkpoint.get("traversal"),
        )
        dynamic_traversals[cell.key] = traversal
        return traversal

    def write_dynamic_checkpoint(
        cell: GenerationCell, checkpoint: dict[str, Any], *, force: bool = False
    ) -> None:
        checkpoints[cell.key] = checkpoint
        dirty_dynamic_cells.add(cell.key)
        if force:
            flush_dynamic_checkpoints({cell.key})

    def flush_dynamic_checkpoints(
        selected: set[tuple[str, ...]] | None = None,
    ) -> None:
        keys = dirty_dynamic_cells if selected is None else dirty_dynamic_cells & selected
        for key in list(keys):
            checkpoint = checkpoints[key]
            write_worker_cell_checkpoint(
                state.run_dir,
                key,
                str(checkpoint.get("stage") or "running"),
                {
                    name: value
                    for name, value in checkpoint.items()
                    if name not in {"cell", "stage"}
                },
            )
            dirty_dynamic_cells.discard(key)

    def claim_dynamic_core(cell: GenerationCell) -> int | None:
        checkpoint = checkpoints[cell.key]
        inflight = list(map(int, checkpoint.get("inflight_core_indices") or []))
        current = checkpoint.get("current_core")
        if current is not None:
            index = int(current["core_space_index"])
            if index not in inflight:
                inflight.append(index)
            checkpoint["current_core"] = None
        index = next(
            (item for item in inflight if (cell.key, item) not in live_dynamic_claims),
            None,
        )
        traversal = dynamic_traversal(cell, checkpoint)
        if index is None:
            try:
                index = traversal.next()
            except StopIteration:
                return None
            inflight.append(index)
        live_dynamic_claims.add((cell.key, index))
        checkpoint.update({
            "stage": "running",
            "traversal": traversal.snapshot(),
            "current_core": None,
            "inflight_core_indices": inflight,
        })
        write_dynamic_checkpoint(cell, checkpoint)
        return index

    def complete_dynamic_core(cell: GenerationCell, result: dict[str, Any]) -> dict[str, Any]:
        index = int(result["core_space_index"])
        live_dynamic_claims.discard((cell.key, index))
        checkpoint = checkpoints[cell.key]
        inflight = list(map(int, checkpoint.get("inflight_core_indices") or []))
        if index in inflight:
            inflight.remove(index)
        reviewing = checkpoint.get("stage") == "reviewing"
        file_key = "prefetched_candidate_files" if reviewing else "candidate_files"
        files = list(checkpoint.get(file_key) or [])
        candidate_file = result.get("candidate_file")
        if candidate_file and candidate_file not in files:
            files.append(candidate_file)
        counts = dict(checkpoint.get("counts") or {})
        counts["visited"] = int(counts.get("visited", 0)) + 1
        counts["certified"] = int(counts.get("certified", 0)) + int(candidate_file is not None)
        counts["rejected"] = int(counts.get("rejected", 0)) + int(candidate_file is None)
        errors = list(result.get("worker_errors") or [])
        counts["errors"] = int(counts.get("errors", 0)) + int(bool(errors))
        stages = counts.setdefault("rejection_stages", {})
        for stage, amount in (result.get("rejection_stages") or {}).items():
            stages[stage] = int(stages.get(stage, 0)) + int(amount)
        worker_errors = list(checkpoint.get("worker_errors") or [])
        worker_errors.extend(errors)
        traversal = dynamic_traversal(cell, checkpoint)
        exhausted = traversal.cursor >= traversal.limit and not inflight
        ready_for_review = not reviewing and len(files) >= config.semantic_review.batch_size
        stage = (
            "reviewing" if reviewing
            else "awaiting_review" if ready_for_review
            else "exhausted" if exhausted
            else "running"
        )
        prefetched: list[str] = []
        if ready_for_review:
            files, prefetched = files[: config.semantic_review.batch_size], files[config.semantic_review.batch_size :]
        checkpoint.update({
            "stage": stage,
            "traversal": traversal.snapshot(),
            "current_core": None,
            "inflight_core_indices": inflight,
            "candidate_files": checkpoint.get("candidate_files", []) if reviewing else files,
            "prefetched_candidate_files": [
                *(files if reviewing else checkpoint.get("prefetched_candidate_files", [])),
                *prefetched,
            ],
            "counts": counts,
            "worker_errors": worker_errors,
        })
        write_dynamic_checkpoint(cell, checkpoint)
        return checkpoint

    with ProcessPoolExecutor(
        max_workers=worker_count, initializer=_initialize_worker, initargs=(config, pool)
    ) as executor:
        cpu_futures: dict[asyncio.Future[Any], tuple[GenerationCell, int | None]] = {}
        review_tasks: dict[asyncio.Task[Any], GenerationCell] = {}
        loop = asyncio.get_running_loop()
        heartbeat_interval = config.resources.progress_interval_seconds
        next_heartbeat = loop.time() + heartbeat_interval
        try:
            while (
                any(accepted_by_group[key] < target for key, target in targets.items())
                or cpu_futures
                or review_tasks
            ):
                progress = consume_surplus()

                reviewing = {cell.key for cell in review_tasks.values()}
                for cell in cells:
                    checkpoint = checkpoints[cell.key]
                    files = list(checkpoint.get("candidate_files") or [])
                    resumed_review_files = list(
                        checkpoint.get("review_candidate_files") or []
                    )
                    if cell.key in reviewing:
                        continue
                    if checkpoint.get("stage") == "reviewing":
                        if not resumed_review_files:
                            raise RuntimeError(
                                f"reviewing cell has no durable batch: {cell.key}"
                            )
                        batch_files, remaining = resumed_review_files, files
                    elif (
                        files
                        and checkpoint.get("stage")
                        in {"awaiting_review", "exhausted"}
                    ):
                        batch_size = config.semantic_review.batch_size
                        batch_files, remaining = files[:batch_size], files[batch_size:]
                    else:
                        continue
                    review_input = {**checkpoint, "candidate_files": batch_files}
                    checkpoint = {
                        **checkpoint,
                        "stage": "reviewing",
                        "candidate_files": [],
                        "review_candidate_files": batch_files,
                        "prefetched_candidate_files": [
                            *checkpoint.get("prefetched_candidate_files", []),
                            *remaining,
                        ],
                    }
                    if cell.mode == "dynamic":
                        write_dynamic_checkpoint(cell, checkpoint, force=True)
                    else:
                        save_checkpoint(cell, checkpoint)
                    selection = select_template(pool, cell.domain, cell.template_id)
                    review = asyncio.create_task(
                        _review_cell_result(
                            cell,
                            review_input,
                            selection,
                            config,
                            state,
                            review_semaphore,
                        )
                    )
                    review_tasks[review] = cell
                    reviewing.add(cell.key)
                    _emit_progress(
                        "candidates_ready",
                        cell=cell.priority,
                        count=len(batch_files),
                        cursor=checkpoint["traversal"]["cursor"],
                    )

                active_counts: dict[tuple[str, ...], int] = {}
                for active_cell, _ in cpu_futures.values():
                    active_counts[active_cell.key] = active_counts.get(active_cell.key, 0) + 1
                dispatch_blocked: set[tuple[str, ...]] = set()
                while len(cpu_futures) < worker_count:
                    blocked = reviewing | dispatch_blocked
                    if config.dataset.mode != "dynamic":
                        blocked = blocked | set(active_counts)
                    ready = _priority_cells(
                        cells,
                        targets,
                        accepted_by_group,
                        accepted_by_cell,
                        checkpoints,
                        active_counts,
                        blocked,
                        callback_cells,
                    )
                    if not ready:
                        break
                    *_, cell = heapq.heappop(ready)
                    checkpoint = checkpoints[cell.key]
                    was_idle = active_counts.get(cell.key, 0) == 0
                    callback_cells.discard(cell.key)
                    if cell.mode == "dynamic":
                        core_index = claim_dynamic_core(cell)
                        if core_index is None:
                            dispatch_blocked.add(cell.key)
                            stage = (
                                "running" if checkpoint.get("inflight_core_indices")
                                else "awaiting_review" if checkpoint.get("candidate_files")
                                else "exhausted"
                            )
                            write_dynamic_checkpoint(
                                cell,
                                {**checkpoint, "stage": stage},
                                force=not checkpoint.get("inflight_core_indices"),
                            )
                            continue
                        task = partial(
                            run_dynamic_core_task,
                            cell,
                            core_index,
                            state.run_dir,
                        )
                    else:
                        loaded = [
                            state.read_candidate(path)
                            for path in checkpoint.get("candidate_files", [])
                        ]
                        runtime_checkpoint = {**checkpoint, "candidates": loaded}
                        save_checkpoint(cell, {**checkpoint, "stage": "running"})
                        task = partial(
                            _run_static_cell_stage,
                            select_template(pool, cell.domain, cell.template_id),
                            cell,
                            config,
                            runtime_checkpoint,
                            order_path=state.cell_order_path(cell.key),
                            run_dir=state.run_dir,
                        )
                        core_index = None
                    future = loop.run_in_executor(executor, task)
                    cpu_futures[future] = (cell, core_index)
                    active_counts[cell.key] = active_counts.get(cell.key, 0) + 1
                    if was_idle:
                        _emit_progress(
                            "cell_leased",
                            cell=cell.priority,
                            domain=cell.domain,
                            template=cell.template_id,
                            rule=cell.rule_family,
                        )

                waiters = [*cpu_futures, *review_tasks]
                if not waiters:
                    if not any(
                        accepted_by_group[key] < target
                        for key, target in targets.items()
                    ):
                        break
                    if progress:
                        continue
                    raise RuntimeError("all cells exhausted before quota groups were filled")
                timeout = max(0.0, next_heartbeat - loop.time())
                done, _ = await asyncio.wait(
                    waiters,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for future in sorted(
                    (item for item in done if item in cpu_futures),
                    key=lambda item: cpu_futures[item][0].priority,
                ):
                    cell, core_index = cpu_futures.pop(future)
                    result = future.result()
                    if core_index is not None:
                        complete_dynamic_core(cell, result)
                    else:
                        save_checkpoint(cell, result)
                for review in sorted(
                    (item for item in done if item in review_tasks),
                    key=lambda item: review_tasks[item].priority,
                ):
                    review_tasks.pop(review)
                    apply_review(review.result())

                now = loop.time()
                heartbeat_due, next_heartbeat = _advance_heartbeat_deadline(
                    now, next_heartbeat, heartbeat_interval
                )
                if heartbeat_due:
                    flush_dynamic_checkpoints()
                    snapshots = list(checkpoints.values())
                    _emit_progress(
                        "heartbeat",
                        cpu=min(worker_count, len(cpu_futures)),
                        llm=len(review_tasks),
                        visited=sum(
                            int((item.get("counts") or {}).get("visited", 0))
                            for item in snapshots
                        ),
                        errors=sum(
                            int((item.get("counts") or {}).get("errors", 0))
                            for item in snapshots
                        ),
                        candidates=sum(
                            len(item.get("candidate_files") or [])
                            + len(item.get("prefetched_candidate_files") or [])
                            + len(item.get("review_candidate_files") or [])
                            + len(item.get("surplus_files") or [])
                            for item in snapshots
                        ),
                        episodes=len(state.value["episodes"]),
                    )
                if not progress and not cpu_futures and not review_tasks and all(
                    checkpoints[cell.key].get("stage") == "exhausted"
                    and not checkpoints[cell.key].get("surplus_files")
                    and not checkpoints[cell.key].get("candidate_files")
                    and not checkpoints[cell.key].get("prefetched_candidate_files")
                    and not checkpoints[cell.key].get("review_candidate_files")
                    for cell in cells
                    if accepted_by_group[cell.group_key] < targets[cell.group_key]
                ):
                    raise RuntimeError(
                        "quota groups remain incomplete after all legal cores were visited"
                    )
        finally:
            flush_dynamic_checkpoints()
        _complete_filled_groups(state, cells, targets, accepted_by_group)
    episode_ids = tuple(state.value["episodes"])
    return GeneratedDataset(
        mode=config.dataset.mode,
        episodes=(state.episode(episode_id)["episode"] for episode_id in episode_ids),
        diagnostics=(state.episode(episode_id)["diagnostics"] for episode_id in episode_ids),
        episode_count=len(episode_ids),
        review_errors=tuple(state.value.get("review_errors") or ()),
    )


def generate_dataset(config: DatasetGenerationConfig, existing: ExistingDataset, state: DatasetRunState, pool: dict[str, Any]) -> GeneratedDataset:
    return asyncio.run(_generate_dataset(config, existing, state, pool))


def run_generation(config: DatasetGenerationConfig, *, run_id: str, resume: bool = False) -> GeneratedDataset:
    pool = load_template_pool(config.dataset.template_pool, manifest_path=config.dataset.template_manifest, audit_path=config.dataset.template_audit)
    state = DatasetRunState(config, run_id, resume=resume)
    if state.published:
        return GeneratedDataset(mode=config.dataset.mode, episodes=(), episode_count=0)
    existing = inspect_existing_dataset(config)
    try:
        generated = generate_dataset(config, existing, state, pool)
        publication_files: dict[str, str] = {}
        manifest_extensions: dict[str, Any] = {}
        reviewed_dataset_sha256 = None
        review = None
        combined: list[dict[str, Any]] | None = None
        if config.final_review.enabled:
            if config.dataset.mode != "dynamic":
                raise ValueError("final dataset review currently supports dynamic datasets only")
            generated_episodes = list(generated.episodes)
            generated = GeneratedDataset(
                mode=generated.mode,
                episodes=iter(generated_episodes),
                diagnostics=generated.diagnostics,
                episode_count=generated.episode_count,
                review_errors=generated.review_errors,
            )
            combined = [
                *(load_dynamic_episodes(existing.path) if existing.path is not None else []),
                *generated_episodes,
            ]
            _emit_progress("final_review_started", episodes=len(combined))
            review_config = load_final_review_config(config.final_review.config)
            review = run_final_review(
                combined,
                review_config,
                dataset={
                    "id": config.dataset.dataset_id,
                    "version": config.dataset.version,
                    "file": f"{config.dataset.mode}_episodes.jsonl",
                },
            )
            review_text = serialized_review(review)
            publication_files[review_config.output_filename] = review_text
            manifest_extensions["llm_review"] = review_manifest_entry(
                review, review_config.output_filename
            )
            reviewed_dataset_sha256 = str(review["dataset"]["sha256"])
            _emit_progress(
                "final_review_completed",
                episodes=len(combined),
                groups=len(review["groups"]),
                overall=review["summary"]["overall_mean"],
            )
        if config.dataset_selection.enabled:
            if review is None or combined is None:
                raise ValueError("dataset selection requires final_review.enabled")
            selection_config = load_selection_config(config.dataset_selection.config)
            selection = select_episodes(combined, review, selection_config)
            publication_files[selection_config.output_filename] = serialized_selected_episodes(selection)
            publication_files[selection_config.report_filename] = serialized_selection_report(selection.report)
            manifest_extensions["published_dataset"] = published_dataset_manifest_entry(
                selection, selection_config.output_filename
            )
            manifest_extensions["dataset_selection"] = selection_manifest_entry(
                selection.report, selection_config.report_filename
            )
            _emit_progress(
                "dataset_selection_completed",
                source_episodes=len(combined),
                published_episodes=len(selection.episodes),
                per_group=selection.report["policy"]["effective_per_group"],
            )
        write_dataset(
            generated,
            config,
            existing,
            run_id=run_id,
            publication_files=publication_files,
            manifest_extensions=manifest_extensions,
            expected_dataset_sha256=reviewed_dataset_sha256,
        )
        state.complete()
        return generated
    except BaseException as exc:
        state.fail(exc)
        _emit_progress("generation_failed", error=type(exc).__name__, message=str(exc))
        raise
