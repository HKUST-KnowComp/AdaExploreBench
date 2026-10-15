"""Strict configuration for the unified dataset generator."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

DatasetMode = Literal["static", "dynamic"]
WriteMode = Literal["fresh", "append"]


@dataclass(frozen=True)
class DatasetIdentity:
    name: str
    slug: str
    dataset_id: str
    version: str
    mode: DatasetMode
    operation: WriteMode
    template_pool: Path
    template_audit: Path
    template_manifest: Path
    output_dir: Path
    artifacts_dir: Path
    seed: int
    overwrite: bool


@dataclass(frozen=True)
class TemplateAssignment:
    strategy: Literal["seeded_balanced"]
    seed: int


@dataclass(frozen=True)
class EpisodeLayout:
    action_case_count: int
    action_batch_count: int
    probe_budget_per_batch: int
    max_probe_cases_per_batch: int
    max_probe_budget: int

    @property
    def action_batch_size(self) -> int:
        return self.action_case_count // self.action_batch_count

    @property
    def case_counts(self) -> dict[str, int]:
        return {"iid": self.action_case_count}

    @property
    def case_counts_per_batch(self) -> dict[str, int]:
        return {"iid": self.action_batch_size}


@dataclass(frozen=True)
class CacheConfig:
    directory: Path
    enabled: bool


@dataclass(frozen=True)
class ResourceConfig:
    cpu_worker_count: int
    max_core_visits_per_cell: int
    proposal_block_size: int
    checkpoint_interval: int
    progress_interval_seconds: int


@dataclass(frozen=True)
class SearchConfig:
    calibration_sample_count: int
    parameter_screening_sample_count: int
    evaluation_sample_count: int
    refinement_coupling_count: int
    refinement_settings_per_coupling: int
    held_out_pool_size: int
    coupling_grid: tuple[float, ...]
    coupling_baseline: float
    reversal_additive_grid: tuple[float, ...]
    reversal_additive_baseline: float
    reversal_threshold_grid: tuple[float, ...]
    reversal_threshold_baseline: float
    population_flip_range: tuple[float, float]
    action_share_range: tuple[float, float]
    early_rejection_confidence: float
    min_transition_probability: float
    q_flip_targets: tuple[float, ...]
    q_action_share_range: tuple[float, float]
    q_tolerance: float
    achieved_correlation_range: tuple[float, float]
    minimum_mean_normalized_divergence: float
    fallback_raw_draw_multiplier: int
    conditional_branch_limit: int
    conditional_integration_points: int
    beta_lookup_knots: int
    saved_decimal_places: int
    solver_timeout_ms: int


@dataclass(frozen=True)
class SemanticReviewConfig:
    model_config: Path
    generation: dict[str, Any]
    batch_size: int
    concurrency: int
    request_attempt_limit: int
    max_distractor_trials_per_core: int
    diagnostic_sample_limit: int
    enabled: bool


@dataclass(frozen=True)
class FinalReviewHook:
    enabled: bool
    config: Path


@dataclass(frozen=True)
class DatasetSelectionHook:
    enabled: bool
    config: Path


@dataclass(frozen=True)
class DynamicDatasetDesign:
    episode_groups_per_domain_rule: int
    visible_field_count: int
    require_exact_quota: bool
    pooling_unit: Literal["domain_rule"]
    template_strategy: Literal["balanced_round_robin"]
    require_unique_core: bool


@dataclass(frozen=True)
class DynamicConfig:
    enabled_types: tuple[str, ...]
    type_assignment_strategy: Literal["seeded_random", "cross_product"]
    type_assignment_seed: int
    change_after_batches: tuple[int, ...]
    change_point_seed: int
    replacement_coupling_scope: Literal["active_group_only"]
    dataset_design: DynamicDatasetDesign
    replacement_role_strength_jitter_range: tuple[float, float]
    replacement_threshold_quantile_jitter_range: tuple[float, float]


@dataclass(frozen=True)
class StaticConfig:
    episodes_per_cell: int
    visible_field_count: int


@dataclass(frozen=True)
class DatasetGenerationConfig:
    dataset: DatasetIdentity
    template_assignment: TemplateAssignment
    episode_layout: EpisodeLayout
    cache: CacheConfig
    resources: ResourceConfig
    search: SearchConfig
    semantic_review: SemanticReviewConfig
    final_review: FinalReviewHook
    dataset_selection: DatasetSelectionHook
    dynamic: DynamicConfig | None
    static: StaticConfig | None

    def to_manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("template_pool", "template_audit", "template_manifest", "output_dir", "artifacts_dir"):
            value["dataset"][key] = str(value["dataset"][key])
        value["cache"]["directory"] = str(value["cache"]["directory"])
        value["semantic_review"]["model_config"] = str(value["semantic_review"]["model_config"])
        value["final_review"]["config"] = str(value["final_review"]["config"])
        value["dataset_selection"]["config"] = str(value["dataset_selection"]["config"])
        return value

    def generation_contract_hash(self) -> str:
        value = self.to_manifest_dict()
        for key in ("operation", "output_dir", "artifacts_dir", "overwrite"):
            value["dataset"].pop(key, None)
        value.pop("resources", None)
        value.pop("cache", None)
        value.pop("final_review", None)
        value.pop("dataset_selection", None)
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def candidate_search_contract_hash(self) -> str:
        value = self.to_manifest_dict()
        payload = {"search": value["search"], "dynamic": value["dynamic"], "static": value["static"]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _mapping(value: Any, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must define exactly {sorted(keys)}")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _probability(value: Any, label: str, *, strict: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or not (0 < number < 1 if strict else 0 <= number <= 1):
        raise ValueError(f"{label} must be a probability")
    return number


def _interval(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two numbers")
    low, high = (_finite(item, label) for item in value)
    if not low < high:
        raise ValueError(f"{label} must be increasing")
    return low, high


def _numbers(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = tuple(_finite(item, label) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _parse_identity(raw: Any, root: Path) -> DatasetIdentity:
    keys = {"name", "slug", "dataset_id", "version", "mode", "operation", "template_pool", "template_audit", "template_manifest", "output_dir", "artifacts_dir", "seed", "overwrite"}
    value = _mapping(raw, "dataset", keys)
    if value["mode"] not in {"static", "dynamic"} or value["operation"] not in {"fresh", "append"}:
        raise ValueError("dataset mode or operation is invalid")
    if not isinstance(value["overwrite"], bool) or value["operation"] == "append" and value["overwrite"]:
        raise ValueError("dataset overwrite is invalid")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ValueError("dataset seed must be an integer")
    paths = tuple(
        (root / str(value[key])).resolve()
        for key in (
            "template_pool", "template_audit", "template_manifest",
            "output_dir", "artifacts_dir",
        )
    )
    return DatasetIdentity(
        str(value["name"]), str(value["slug"]), str(value["dataset_id"]),
        str(value["version"]), value["mode"], value["operation"], *paths,
        value["seed"], value["overwrite"],
    )


def _parse_layout(raw: Any) -> EpisodeLayout:
    value = _mapping(raw, "episode_layout", {"action_case_count", "action_batch_count", "probe_budget_per_batch", "max_probe_cases_per_batch", "max_probe_budget"})
    action_count, batch_count = _positive(value["action_case_count"], "action_case_count"), _positive(value["action_batch_count"], "action_batch_count")
    if action_count % batch_count:
        raise ValueError("action cases must divide evenly across batches")
    probe_budget = _positive(value["probe_budget_per_batch"], "probe_budget_per_batch")
    maximum_cases = _positive(value["max_probe_cases_per_batch"], "max_probe_cases_per_batch")
    maximum_budget = _positive(value["max_probe_budget"], "max_probe_budget")
    if probe_budget > maximum_budget:
        raise ValueError("probe_budget_per_batch cannot exceed max_probe_budget")
    return EpisodeLayout(
        action_count, batch_count, probe_budget, maximum_cases, maximum_budget
    )


def _parse_sampling(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _positive(raw[key], key)
        for key in (
            "fallback_raw_draw_multiplier",
            "conditional_branch_limit",
            "conditional_integration_points",
            "beta_lookup_knots",
        )
    } | {
        "saved_decimal_places": _non_negative(
            raw["saved_decimal_places"], "saved_decimal_places"
        )
    }


def _parse_search(raw: Any) -> SearchConfig:
    keys = set(SearchConfig.__dataclass_fields__)
    value = _mapping(raw, "search", keys)
    sampling = _parse_sampling(value)
    search = SearchConfig(
        *(_positive(value[key], key) for key in (
            "calibration_sample_count",
            "parameter_screening_sample_count",
            "evaluation_sample_count",
            "refinement_coupling_count",
            "refinement_settings_per_coupling",
            "held_out_pool_size",
        )),
        _numbers(value["coupling_grid"], "coupling_grid"), _finite(value["coupling_baseline"], "coupling_baseline"),
        _numbers(value["reversal_additive_grid"], "reversal_additive_grid"), _finite(value["reversal_additive_baseline"], "reversal_additive_baseline"),
        _numbers(value["reversal_threshold_grid"], "reversal_threshold_grid"), _finite(value["reversal_threshold_baseline"], "reversal_threshold_baseline"),
        _interval(value["population_flip_range"], "population_flip_range"), _interval(value["action_share_range"], "action_share_range"),
        _probability(value["early_rejection_confidence"], "early_rejection_confidence", strict=True), _probability(value["min_transition_probability"], "min_transition_probability", strict=True),
        _numbers(value["q_flip_targets"], "q_flip_targets"), _interval(value["q_action_share_range"], "q_action_share_range"), _finite(value["q_tolerance"], "q_tolerance"),
        _interval(value["achieved_correlation_range"], "achieved_correlation_range"), _finite(value["minimum_mean_normalized_divergence"], "minimum_mean_normalized_divergence"),
        *(sampling[key] for key in ("fallback_raw_draw_multiplier", "conditional_branch_limit", "conditional_integration_points", "beta_lookup_knots")),
        sampling["saved_decimal_places"], _positive(value["solver_timeout_ms"], "solver_timeout_ms"),
    )
    if not search.parameter_screening_sample_count < search.evaluation_sample_count <= search.held_out_pool_size:
        raise ValueError("search pool sizes must satisfy screening < evaluation <= held-out")
    if any(len(grid) != 9 for grid in (search.coupling_grid, search.reversal_additive_grid, search.reversal_threshold_grid)):
        raise ValueError("each configured parameter grid must contain exactly nine values")
    if search.refinement_coupling_count > len(search.coupling_grid) or search.refinement_settings_per_coupling > min(len(search.reversal_additive_grid), len(search.reversal_threshold_grid)):
        raise ValueError("refinement dimensions cannot exceed the parameter grids")
    if search.coupling_baseline not in search.coupling_grid or search.reversal_additive_baseline not in search.reversal_additive_grid or search.reversal_threshold_baseline not in search.reversal_threshold_grid:
        raise ValueError("each parameter baseline must occur in its configured grid")
    if search.q_flip_targets != (0.25, 0.5, 0.75):
        raise ValueError("q_flip_targets must be exactly 0.25, 0.5, and 0.75")
    if search.q_tolerance <= 0 or search.minimum_mean_normalized_divergence <= 0:
        raise ValueError("Q tolerance and minimum divergence must be positive")
    for label, interval in (("population_flip_range", search.population_flip_range), ("action_share_range", search.action_share_range), ("q_action_share_range", search.q_action_share_range)):
        if interval[0] < 0 or interval[1] > 1:
            raise ValueError(f"{label} must lie within [0, 1]")
    if search.achieved_correlation_range[0] < 0 or search.achieved_correlation_range[1] > 1:
        raise ValueError("achieved_correlation_range must lie within [0, 1]")
    return search


def _parse_semantic_review(path: Path, root: Path) -> SemanticReviewConfig:
    if not path.is_file():
        raise ValueError(f"semantic review config does not exist: {path}")
    value = _mapping(yaml.safe_load(path.read_text()) or {}, "semantic_review", {"enabled", "model_config", "generation", "batch_size", "concurrency", "request_attempt_limit", "max_distractor_trials_per_core", "diagnostic_sample_limit"})
    if not isinstance(value["enabled"], bool) or not isinstance(value["generation"], dict):
        raise ValueError("semantic review enabled/generation is invalid")
    model_config = (root / str(value["model_config"])).resolve()
    if not model_config.is_file():
        raise ValueError(f"semantic review model config does not exist: {model_config}")
    return SemanticReviewConfig(model_config, dict(value["generation"]), *(_positive(value[key], key) for key in ("batch_size", "concurrency", "request_attempt_limit", "max_distractor_trials_per_core")), _non_negative(value["diagnostic_sample_limit"], "diagnostic_sample_limit"), value["enabled"])


def _parse_template_assignment(raw: Any) -> TemplateAssignment:
    value = _mapping(raw, "template_assignment", {"strategy", "seed"})
    if value["strategy"] != "seeded_balanced":
        raise ValueError("template assignment strategy is invalid")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ValueError("template assignment seed must be an integer")
    return TemplateAssignment("seeded_balanced", value["seed"])


def _parse_cache(raw: Any, root: Path) -> CacheConfig:
    value = _mapping(raw, "cache", {"directory", "enabled"})
    if not isinstance(value["enabled"], bool):
        raise ValueError("cache.enabled must be boolean")
    return CacheConfig((root / str(value["directory"])).resolve(), value["enabled"])


def _parse_dynamic_controls(raw: dict[str, Any]) -> tuple[str, tuple[float, float], tuple[float, float]]:
    scope = raw["replacement_coupling_scope"]
    if scope != "active_group_only":
        raise ValueError("replacement_coupling_scope must be active_group_only")
    return (
        scope,
        _interval(raw["replacement_role_strength_jitter_range"], "replacement jitter"),
        _interval(raw["replacement_threshold_quantile_jitter_range"], "threshold jitter"),
    )


def _parse_dataset_design(raw: Any) -> DynamicDatasetDesign:
    value = _mapping(raw, "dynamic.dataset_design", set(DynamicDatasetDesign.__dataclass_fields__))
    if value["pooling_unit"] != "domain_rule" or value["template_strategy"] != "balanced_round_robin":
        raise ValueError("dynamic dataset design strategy is invalid")
    if not isinstance(value["require_exact_quota"], bool) or not isinstance(value["require_unique_core"], bool):
        raise ValueError("dynamic dataset design flags must be boolean")
    return DynamicDatasetDesign(
        _positive(value["episode_groups_per_domain_rule"], "episode_groups_per_domain_rule"),
        _positive(value["visible_field_count"], "dynamic.visible_field_count"),
        value["require_exact_quota"], value["pooling_unit"],
        value["template_strategy"], value["require_unique_core"],
    )


def _parse_dynamic(raw: Any, layout: EpisodeLayout) -> DynamicConfig:
    item = _mapping(raw, "dynamic", {"enabled_types", "type_assignment", "change_point", "replacement_coupling_scope", "dataset_design", "replacement_role_strength_jitter_range", "replacement_threshold_quantile_jitter_range"})
    assignment = _mapping(item["type_assignment"], "dynamic.type_assignment", {"strategy", "seed"})
    change = _mapping(item["change_point"], "dynamic.change_point", {"after_action_batches", "seed"})
    enabled_types = tuple(map(str, item["enabled_types"])) if isinstance(item["enabled_types"], list) else ()
    if not enabled_types or len(enabled_types) != len(set(enabled_types)) or not set(enabled_types) <= {"field_replacement", "direction_reversal", "compound"}:
        raise ValueError("dynamic.enabled_types must be a unique non-empty supported list")
    if assignment["strategy"] not in {"seeded_random", "cross_product"}:
        raise ValueError("dynamic type assignment strategy is invalid")
    if any(isinstance(source["seed"], bool) or not isinstance(source["seed"], int) for source in (assignment, change)):
        raise ValueError("dynamic assignment/change seeds must be integers")
    after_batches = tuple(change["after_action_batches"]) if isinstance(change["after_action_batches"], list) else ()
    if not after_batches or any(isinstance(number, bool) or not isinstance(number, int) or not 0 < number < layout.action_batch_count for number in after_batches):
        raise ValueError("dynamic change points must fall between action batches")
    if len(after_batches) != len(set(after_batches)):
        raise ValueError("dynamic change points must be unique")
    scope, role_jitter, threshold_jitter = _parse_dynamic_controls(item)
    return DynamicConfig(
        enabled_types, assignment["strategy"], assignment["seed"],
        after_batches, change["seed"], scope,
        _parse_dataset_design(item["dataset_design"]), role_jitter, threshold_jitter,
    )


def _parse_static(raw: Any) -> StaticConfig:
    item = _mapping(raw, "static", {"episodes_per_cell", "visible_field_count"})
    return StaticConfig(
        _positive(item["episodes_per_cell"], "episodes_per_cell"),
        _positive(item["visible_field_count"], "static.visible_field_count"),
    )


def load_config(path: Path) -> DatasetGenerationConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    keys = {"dataset", "template_assignment", "episode_layout", "cache", "resources", "search", "semantic_review_config", "final_review", "dataset_selection", "dynamic", "static"}
    value = _mapping(raw, "dataset generation config", keys)
    resolved = path.resolve()
    root = next((parent for parent in resolved.parents if (parent / "pyproject.toml").is_file()), Path.cwd())
    dataset = _parse_identity(value["dataset"], root)
    layout = _parse_layout(value["episode_layout"])
    resources = _mapping(value["resources"], "resources", set(ResourceConfig.__dataclass_fields__))
    final_review = _mapping(value["final_review"], "final_review", {"enabled", "config"})
    if not isinstance(final_review["enabled"], bool):
        raise ValueError("final_review.enabled must be boolean")
    final_review_path = (root / str(final_review["config"])).resolve()
    if not final_review_path.is_file():
        raise ValueError(f"final review config does not exist: {final_review_path}")
    dataset_selection = _mapping(value["dataset_selection"], "dataset_selection", {"enabled", "config"})
    if not isinstance(dataset_selection["enabled"], bool):
        raise ValueError("dataset_selection.enabled must be boolean")
    dataset_selection_path = (root / str(dataset_selection["config"])).resolve()
    if not dataset_selection_path.is_file():
        raise ValueError(f"dataset selection config does not exist: {dataset_selection_path}")
    dynamic = None
    static = None
    if dataset.mode == "dynamic":
        if value["static"] is not None:
            raise ValueError("inactive static config must be null")
        dynamic = _parse_dynamic(value["dynamic"], layout)
    else:
        if value["dynamic"] is not None:
            raise ValueError("inactive dynamic config must be null")
        static = _parse_static(value["static"])
    for source in (dataset.template_pool, dataset.template_audit, dataset.template_manifest):
        if not source.is_file():
            raise ValueError(f"required dataset input does not exist: {source}")
    return DatasetGenerationConfig(
        dataset,
        _parse_template_assignment(value["template_assignment"]),
        layout,
        _parse_cache(value["cache"], root),
        ResourceConfig(*(_positive(resources[key], key) for key in ResourceConfig.__dataclass_fields__)),
        _parse_search(value["search"]),
        _parse_semantic_review((root / str(value["semantic_review_config"])).resolve(), root),
        FinalReviewHook(final_review["enabled"], final_review_path),
        DatasetSelectionHook(dataset_selection["enabled"], dataset_selection_path),
        dynamic,
        static,
    )
