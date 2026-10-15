"""CPU worker stage for one dynamic cell; orchestration lives in the public pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DatasetGenerationConfig
from ..planning import BoundedPermutation, GenerationCell, stable_seed
from ..population import configure_population_cache
from ..state import capture_worker_error, write_worker_candidate, write_worker_cell_checkpoint
from ..template_pool import TemplateSelection, select_template
from .structures import DynamicCoreSpace, materialize_distractors, visible_fields


_WORKER_CONFIG: DatasetGenerationConfig | None = None
_WORKER_POOL: dict[str, Any] | None = None
_WORKER_SPACES: dict[tuple[str, str, str], tuple[TemplateSelection, DynamicCoreSpace]] = {}


def initialize_dynamic_worker(config: DatasetGenerationConfig, pool: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_POOL
    _WORKER_CONFIG, _WORKER_POOL = config, pool
    _WORKER_SPACES.clear()
    configure_population_cache(config.cache.directory if config.cache.enabled else None)


def run_dynamic_core_task(
    cell: GenerationCell,
    core_space_index: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Evaluate one coordinator-claimed core without mutating shared cell state."""

    if _WORKER_CONFIG is None or _WORKER_POOL is None or cell.dynamic_type is None:
        raise RuntimeError("dynamic worker context is not initialized")
    key = (cell.domain, cell.template_id, cell.rule_family)
    cached = _WORKER_SPACES.get(key)
    if cached is None:
        selection = select_template(_WORKER_POOL, cell.domain, cell.template_id)
        cached = (selection, DynamicCoreSpace(selection, cell.rule_family, cell.dynamic_type))
        _WORKER_SPACES[key] = cached
    selection, space = cached
    core = space.at(core_space_index)
    rejection_stages: dict[str, int] = {}
    worker_errors: list[dict[str, Any]] = []
    try:
        from .certification import evaluate_core

        evaluated = evaluate_core(
            selection,
            core,
            cell.rule_family,
            cell.dynamic_type,
            _WORKER_CONFIG,
            seed=stable_seed(
                _WORKER_CONFIG.dataset.seed,
                *cell.key,
                core_space_index,
                "numeric",
            ),
            rejection_counts=rejection_stages,
        )
        candidate_file = None
        if evaluated is not None:
            transition = materialize_distractors(
                selection,
                core,
                cell.dynamic_type,
                stable_seed(
                    _WORKER_CONFIG.dataset.seed,
                    *cell.key,
                    core_space_index,
                    "distractors",
                ),
                _WORKER_CONFIG.dynamic.dataset_design.visible_field_count,
                trial=0,
            )
            candidate = {
                **evaluated,
                "mode": "dynamic",
                "dynamic_type": cell.dynamic_type,
                "_core": core,
                "transition": transition,
                "visible_fields": visible_fields(transition, cell.dynamic_type),
                "distractor_trial": 0,
            }
            candidate_file = write_worker_candidate(run_dir, cell.key, candidate)
        return {
            "core_space_index": core_space_index,
            "candidate_file": candidate_file,
            "rejection_stages": rejection_stages,
            "worker_errors": worker_errors,
        }
    except Exception as error:
        capture_worker_error(cell.key, core, error, worker_errors)
        return {
            "core_space_index": core_space_index,
            "candidate_file": None,
            "rejection_stages": rejection_stages,
            "worker_errors": worker_errors,
        }


def run_dynamic_cell_stage(
    selection: TemplateSelection,
    cell: GenerationCell,
    config: DatasetGenerationConfig,
    checkpoint: dict[str, Any],
    *,
    visit_limit: int,
    materialize_threshold: int,
    review_batch_size: int,
    visible_field_count: int,
    order_path: Path,
    run_dir: Path,
    checkpoint_interval: int,
) -> dict[str, Any]:
    """Advance one leased cell until an LLM batch or core exhaustion."""

    if cell.dynamic_type is None:
        raise ValueError("dynamic cell worker requires a dynamic type")
    from .certification import evaluate_core

    space = DynamicCoreSpace(selection, cell.rule_family, cell.dynamic_type)
    traversal = BoundedPermutation(
        space.count,
        stable_seed(config.dataset.seed, *cell.key, "core-order"),
        visit_limit,
        materialize_threshold=materialize_threshold,
        order_path=order_path,
        state=checkpoint.get("traversal"),
    )
    candidates = list(checkpoint.get("candidates") or [])
    candidate_files = list(checkpoint.get("candidate_files") or [])
    current = checkpoint.get("current_core")
    counts = dict(checkpoint.get("counts") or {})
    counts.setdefault("visited", int((checkpoint.get("traversal") or {}).get("cursor", 0)))
    counts.setdefault("certified", len(candidates))
    counts.setdefault("rejected", 0)
    counts.setdefault("errors", 0)
    rejection_stages = counts.setdefault("rejection_stages", {})
    worker_errors = list(checkpoint.get("worker_errors") or [])

    def persist(stage: str) -> None:
        write_worker_cell_checkpoint(
            run_dir,
            cell.key,
            stage,
            {
                "traversal": traversal.snapshot(),
                "current_core": current,
                "candidate_files": candidate_files,
                "surplus_files": list(checkpoint.get("surplus_files") or []),
                "space_count": space.count,
                "counts": counts,
                "worker_errors": worker_errors,
                "visit_budget_truncated": space.count > visit_limit
                and traversal.cursor >= traversal.limit,
            },
        )

    while len(candidates) < review_batch_size:
        if current is None:
            try:
                current = space.at(traversal.next())
            except StopIteration:
                break
        persist("running")
        evaluated = None
        failed = False
        try:
            evaluated = evaluate_core(
                selection,
                current,
                cell.rule_family,
                cell.dynamic_type,
                config,
                seed=stable_seed(
                    config.dataset.seed,
                    *cell.key,
                    current["core_space_index"],
                    "numeric",
                ),
                rejection_counts=rejection_stages,
            )
            if evaluated is not None:
                transition = materialize_distractors(
                    selection,
                    current,
                    cell.dynamic_type,
                    stable_seed(
                        config.dataset.seed,
                        *cell.key,
                        current["core_space_index"],
                        "distractors",
                    ),
                    visible_field_count,
                    trial=0,
                )
                candidates.append(
                    {
                        **evaluated,
                        "mode": "dynamic",
                        "dynamic_type": cell.dynamic_type,
                        "_core": current,
                        "transition": transition,
                        "visible_fields": visible_fields(transition, cell.dynamic_type),
                        "distractor_trial": 0,
                    }
                )
                candidate_files.append(write_worker_candidate(run_dir, cell.key, candidates[-1]))
                counts["certified"] += 1
            else:
                counts["rejected"] += 1
        except Exception as error:
            capture_worker_error(cell.key, current, error, worker_errors)
            counts["errors"] += 1
            counts["rejected"] += 1
            failed = True
        current = None
        counts["visited"] = traversal.cursor
        if failed or evaluated is not None or traversal.cursor % checkpoint_interval == 0:
            persist("running")
    result = {
        "stage": "awaiting_review" if candidates else "exhausted",
        "traversal": traversal.snapshot(),
        "current_core": current,
        "candidate_files": candidate_files,
        "surplus_files": list(checkpoint.get("surplus_files") or []),
        "space_count": space.count,
        "counts": counts,
        "worker_errors": worker_errors,
        "visit_budget_truncated": space.count > visit_limit
        and traversal.cursor >= traversal.limit,
    }
    write_worker_cell_checkpoint(run_dir, cell.key, result["stage"], result)
    return result
