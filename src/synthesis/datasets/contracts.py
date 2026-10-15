"""Stable contracts passed between dataset-generation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal


STATIC_EPISODE_SCHEMA_VERSION = "adaexplorebench_episode_v7"
DYNAMIC_EPISODE_SCHEMA_VERSION = "adaexplorebench_episode_v8"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    seed: int
    domain: str
    template_id: str
    rule_family: str
    sequence: int
    dynamic_type: str | None = None
    after_action_batch: int | None = None
    catalog_seed: int | None = None
    candidate_slot: int | None = None


@dataclass(frozen=True)
class EpisodeBuild:
    episode: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class GeneratedDataset:
    mode: Literal["static", "dynamic"]
    episodes: Iterable[dict[str, Any]]
    diagnostics: Iterable[dict[str, Any]] = ()
    episode_count: int = 0
    review_errors: tuple[dict[str, Any], ...] = ()


_COMMON_TOP_LEVEL_KEYS = {
    "schema_version",
    "dataset_mode",
    "episode_id",
    "seed",
    "provenance",
    "agent_task",
    "evaluation",
}


def _require_exact_keys(label: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} must define exactly {sorted(expected)}; observed {observed}")
    return value


def validate_episode(episode: dict[str, Any], mode: str) -> None:
    expected_top = _COMMON_TOP_LEVEL_KEYS | (
        {"shared_prefix", "post_branches"}
        if mode == "dynamic"
        else {"cases", "batches"}
    )
    observed_keys = set(episode)
    if frozenset(observed_keys) not in {
        frozenset(expected_top),
        frozenset(expected_top | {"episode_fingerprint"}),
    }:
        raise ValueError(
            f"episode has invalid top-level keys: {sorted(observed_keys)}"
        )
    if "episode_fingerprint" in episode and not str(episode["episode_fingerprint"]):
        raise ValueError("episode_fingerprint must be non-empty")
    expected_schema = (
        STATIC_EPISODE_SCHEMA_VERSION
        if mode == "static"
        else DYNAMIC_EPISODE_SCHEMA_VERSION
    )
    if episode["schema_version"] != expected_schema:
        raise ValueError("episode has the wrong schema version")
    if episode["dataset_mode"] != mode:
        raise ValueError("episode has the wrong dataset mode")
    if not str(episode["episode_id"] or ""):
        raise ValueError("episode_id must be non-empty")
    provenance = _require_exact_keys(
        "provenance", episode["provenance"], {"domain", "template_id", "sequence"}
    )
    if not all(str(provenance[key] or "") for key in ("domain", "template_id")):
        raise ValueError("episode provenance is incomplete")
    task = _require_exact_keys(
        "agent_task",
        episode["agent_task"],
        {
            "agent_role",
            "scenario_description",
            "actions",
            "fields",
            "feasibility_constraints",
            "interaction_protocol",
        },
    )
    expected_field_count = 8
    if len(task["actions"]) != 4 or len(task["fields"]) != expected_field_count:
        raise ValueError(
            "agent_task requires exactly four actions and "
            f"{expected_field_count} fields"
        )
    for action in task["actions"]:
        _require_exact_keys(
            "action", action, {"action_id", "display_name", "description"}
        )
    field_keys = {
        "display_name",
        "description",
        "type",
        "unit",
        "minimum",
        "maximum",
        "typical_range",
    }
    for field_id, field in task["fields"].items():
        _require_exact_keys("agent field", field, field_keys)
        if field["type"] not in {"integer", "decimal"}:
            raise ValueError(
                f"agent field {field_id} type must be integer or decimal"
            )
    interaction = _require_exact_keys(
        "interaction_protocol",
        task["interaction_protocol"],
        {
            "action_batch_count",
            "action_batch_size",
            "probe_budget_per_batch",
            "max_probe_cases_per_batch",
            "max_probe_budget",
        },
    )
    evaluation = episode["evaluation"]
    expected_evaluation = {"rule_family", "rule", "batch_contract"}
    if mode == "dynamic":
        expected_evaluation = {"rule_family", "transition", "batch_contract"}
    _require_exact_keys("evaluation", evaluation, expected_evaluation)
    if mode == "static":
        if evaluation["rule"].get("family") != evaluation["rule_family"]:
            raise ValueError("static rule family does not match evaluation")
    else:
        transition = _require_exact_keys(
            "transition",
            evaluation["transition"],
            {
                "type",
                "after_action_batch",
                "pre_rule",
                "post_rule",
                "operations",
                "population_reference",
                "sampling_design",
                "semantic_change",
            },
        )
        if transition["type"] not in {
            "field_replacement",
            "direction_reversal",
            "compound",
        }:
            raise ValueError("transition has an unsupported dynamic type")
        if any(
            rule.get("family") != evaluation["rule_family"]
            for rule in (transition["pre_rule"], transition["post_rule"])
        ):
            raise ValueError("dynamic rule family does not match evaluation")
        operation_types = [item.get("type") for item in transition["operations"]]
        expected_operations = {
            "field_replacement": ["field_replacement"],
            "direction_reversal": ["direction_reversal"],
            "compound": ["field_replacement", "direction_reversal"],
        }[transition["type"]]
        if operation_types != expected_operations:
            raise ValueError("transition operations do not match dynamic type")
        for operation in transition["operations"]:
            expected_keys = (
                {
                    "type",
                    "group_id",
                    "transition_id",
                    "outgoing_field",
                    "incoming_field",
                }
                if operation["type"] == "field_replacement"
                else {"type", "event_id", "target_field"}
            )
            _require_exact_keys("dynamic operation", operation, expected_keys)
        _require_exact_keys(
            "semantic_change",
            transition["semantic_change"],
            {"trigger_event", "change_reason"},
        )
        if not all(str(value or "").strip() for value in transition["semantic_change"].values()):
            raise ValueError("dynamic semantic change must be complete")
        _require_exact_keys(
            "sampling_design",
            transition["sampling_design"],
            {"candidate_slot_id", "available_post_branches"},
        )
        if not str(transition["sampling_design"]["candidate_slot_id"]):
            raise ValueError("dynamic candidate_slot_id must be non-empty")
        available = transition["sampling_design"]["available_post_branches"]
        if not isinstance(available, list) or not available or len(available) != len(set(available)):
            raise ValueError("dynamic available_post_branches must be a unique non-empty list")
        _require_exact_keys(
            "population_reference",
            transition["population_reference"],
            {
                "flip_rate",
                "old_action_marginal",
                "new_action_marginal",
                "action_transition_matrix",
                "mean_score_shift",
                "mean_absolute_score_shift",
            },
        )
    strata = _require_exact_keys(
        "batch_contract",
        evaluation["batch_contract"],
        {"strata_per_batch"},
    )["strata_per_batch"]
    _require_exact_keys("strata_per_batch", strata, {"iid"})
    if mode == "static":
        _validate_timeline(
            episode["cases"], episode["batches"], task, interaction, strata,
            start_batch=1, expected_phase="static",
        )
        return
    transition_batch = int(evaluation["transition"]["after_action_batch"])
    shared = _require_exact_keys(
        "shared_prefix", episode["shared_prefix"], {"cases", "batches"}
    )
    _validate_timeline(
        shared["cases"], shared["batches"], task, interaction, strata,
        start_batch=1, expected_phase="pre_dynamic",
        expected_batch_count=transition_batch,
    )
    branches = episode["post_branches"]
    if not isinstance(branches, dict) or set(branches) != set(available):
        raise ValueError("post branches do not match sampling design")
    expected_post_count = int(interaction["action_batch_count"]) - transition_batch
    for branch_name, branch in branches.items():
        branch = _require_exact_keys(
            f"post branch {branch_name}",
            branch,
            {"condition", "sampling_design", "cases", "batches"},
        )
        if branch["condition"] not in {"static", "dynamic"}:
            raise ValueError("post branch condition is invalid")
        sampling = _require_exact_keys(
            f"post branch {branch_name} sampling",
            branch["sampling_design"],
            {"mode", "target_flip_rate"},
        )
        target = sampling["target_flip_rate"]
        if branch_name == "static_iid":
            if branch["condition"] != "static" or sampling != {
                "mode": "natural_iid", "target_flip_rate": None
            }:
                raise ValueError("static_iid branch contract is invalid")
        elif (
            branch["condition"] != "dynamic"
            or sampling["mode"] != "target_flip"
            or target is None
            or not 0.0 < float(target) < 1.0
        ):
            raise ValueError("target-flip branch contract is invalid")
        _validate_timeline(
            branch["cases"], branch["batches"], task, interaction, strata,
            start_batch=transition_batch + 1,
            expected_phase="post_dynamic",
            expected_batch_count=expected_post_count,
        )


def _validate_timeline(
    cases: Any,
    batches: Any,
    task: dict[str, Any],
    interaction: dict[str, Any],
    strata: dict[str, int],
    *,
    start_batch: int,
    expected_phase: str,
    expected_batch_count: int | None = None,
) -> None:
    if not isinstance(cases, list) or not isinstance(batches, list):
        raise ValueError("timeline cases and batches must be lists")
    batch_count = (
        int(interaction["action_batch_count"])
        if expected_batch_count is None
        else expected_batch_count
    )
    if len(batches) != batch_count:
        raise ValueError("timeline has the wrong action batch count")
    expected_case_count = batch_count * int(interaction["action_batch_size"])
    if len(cases) != expected_case_count:
        raise ValueError("timeline has the wrong case count")
    context_ids: list[str] = []
    case_ids: set[str] = set()
    case_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        _require_exact_keys("case", case, {"context_id", "raw_values", "stratum"})
        context_id = str(case["context_id"])
        if context_id in case_ids or set(case["raw_values"]) != set(task["fields"]):
            raise ValueError("cases must have unique IDs and exactly the task fields")
        if case["stratum"] not in strata:
            raise ValueError("case has an unsupported stratum")
        case_ids.add(context_id)
        case_by_id[context_id] = case
    for number, batch in enumerate(batches, start=start_batch):
        _require_exact_keys(
            "batch", batch, {"batch_id", "batch_number", "phase", "context_ids"}
        )
        if int(batch["batch_number"]) != number:
            raise ValueError("batches must be consecutively numbered")
        if len(batch["context_ids"]) != int(interaction["action_batch_size"]):
            raise ValueError("batch has the wrong number of contexts")
        observed_strata = {
            stratum: sum(
                case_by_id[str(context_id)]["stratum"] == stratum
                for context_id in batch["context_ids"]
            )
            for stratum in strata
        }
        if observed_strata != strata:
            raise ValueError("batch does not match the declared strata contract")
        if batch["phase"] != expected_phase:
            raise ValueError("batch phase does not match the rule schedule")
        context_ids.extend(map(str, batch["context_ids"]))
    if len(context_ids) != len(set(context_ids)) or set(context_ids) != case_ids:
        raise ValueError("batches must cover every case exactly once")


def validate_generated_dataset(dataset: GeneratedDataset) -> None:
    ids: set[str] = set()
    for episode in dataset.episodes:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id or episode_id in ids:
            raise ValueError("generated episode IDs must be non-empty and unique")
        ids.add(episode_id)
        validate_episode(episode, dataset.mode)
