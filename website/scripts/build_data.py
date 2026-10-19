#!/usr/bin/env python3
"""Build the static data bundle for the AdaExploreBench project page.

The script reads one dynamic episode per domain from the selected-256 test
release, computes reference action utilities from each episode's hidden
pre-change rule with the same deterministic mechanism used by the runtime,
and writes both JSON bundles consumed by app.js.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SITE_ROOT.parent
OUTPUT_DIR = SITE_ROOT / "assets" / "data"

RERULE_DEFAULT = Path(
    "/Volumes/RoryCat/projects/ReRule/datasets/selected-256/dynamic_episodes.jsonl"
)
DEFAULT_SOURCE = (
    RERULE_DEFAULT
    if RERULE_DEFAULT.exists()
    else REPO_ROOT / "datasets" / "selected-256" / "dynamic_episodes.jsonl"
)

SCHEMA_VERSION = "adaexplorebench_site_data_v1"
SAMPLE_BRANCH = "flip_50"

RULE_FAMILY_LABELS = {
    "additive_aggregation": "Additive aggregation",
    "conjunctive_gating": "Conjunctive gating",
    "disjunctive_sufficiency": "Disjunctive sufficiency",
    "lexicographic_priority": "Lexicographic priority",
}

DOMAIN_LABELS = {
    "finance": "Finance",
    "healthcare": "Healthcare",
    "cybersecurity": "Cybersecurity",
    "manufacturing": "Manufacturing",
}

# One episode per domain, deliberately spanning all four rule families.
SAMPLE_EPISODES = {
    "finance": "conjunctive_finance_f18e17ea_compound_slot01_001",
    "healthcare": "additive_healthcare_779f3f3b_compound_slot01_037",
    "cybersecurity": "disjunctive_cybersecurity_e4d6ced6_compound_slot01_319",
    "manufacturing": "lexicographic_manufacturing_8ec8802a_compound_slot04_256",
}

ACTION_ANCHORS = (-0.75, -0.25, 0.25, 0.75)
UTILITY_BETA = 8.0


# ---------------------------------------------------------------------------
# Mechanism scoring
# ---------------------------------------------------------------------------
# Prefer the repository implementation so the site data matches the runtime
# exactly. The compact fallback keeps the script usable if only website/ is
# copied to another checkout.


def _load_mechanism():
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from protocol.mechanism import (  # type: ignore
            hydrate_rule,
            intervention_score,
            utilities_from_score,
        )

        return hydrate_rule, intervention_score, utilities_from_score
    except Exception:
        return _fallback_hydrate_rule, _fallback_intervention_score, _fallback_utilities


def _fallback_hydrate_rule(rule: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    hydrated = json.loads(json.dumps(rule))
    hydrated["action_levels"] = list(ACTION_ANCHORS)
    hydrated["field_ranges"] = {
        key: {
            "minimum": fields[key]["minimum"],
            "maximum": fields[key]["maximum"],
        }
        for key in hydrated["relevant_fields"]
    }
    return hydrated


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _normalize(mechanism: dict[str, Any], field: str, raw: float) -> float:
    bounds = mechanism["field_ranges"][field]
    low = float(bounds["minimum"])
    high = float(bounds["maximum"])
    return _clip(2.0 * (float(raw) - low) / (high - low) - 1.0)


def _oriented(mechanism: dict[str, Any], field: str, raw: float) -> float:
    return float(mechanism["field_directions"][field]) * _normalize(mechanism, field, raw)


def _margin(mechanism: dict[str, Any], field: str, values: dict[str, float]) -> float:
    return _oriented(mechanism, field, values[field]) - _oriented(
        mechanism, field, mechanism["thresholds"][field]
    )


def _fallback_raw_evidence(mechanism: dict[str, Any], values: dict[str, float]) -> float:
    family = str(mechanism["family"])
    fields = list(mechanism["relevant_fields"])

    if family == "additive_aggregation":
        contributions = {
            field: float(mechanism["weights"][field])
            * _normalize(mechanism, field, values[field])
            for field in fields
        }
        return float(mechanism["bias"]) + sum(contributions.values())

    margins = {field: _margin(mechanism, field, values) for field in fields}
    if family == "conjunctive_gating":
        controller = min(fields, key=lambda field: (margins[field], fields.index(field)))
        return margins[controller]
    if family == "disjunctive_sufficiency":
        controller = max(fields, key=lambda field: (margins[field], -fields.index(field)))
        return margins[controller]
    if family == "lexicographic_priority":
        order = list(mechanism["priority_order"])
        band = float(mechanism["indifference_band"])
        decisive = next((field for field in order if abs(margins[field]) > band), order[-1])
        return margins[decisive]
    raise ValueError(f"unsupported mechanism family: {family}")


def _fallback_intervention_score(
    mechanism: dict[str, Any], values: dict[str, float]
) -> float:
    evidence = _fallback_raw_evidence(mechanism, values)
    calibration = mechanism.get("score_calibration") or {}
    if calibration.get("method") == "frozen_pre_quartile_piecewise":
        knots = [float(knot) for knot in calibration["evidence_knots"]]
        targets = [float(target) for target in calibration["score_knots"]]
        if evidence <= knots[1]:
            slope = (targets[1] - targets[0]) / (knots[1] - knots[0])
            return _clip(targets[0] + (evidence - knots[0]) * slope)
        slope = (targets[2] - targets[1]) / (knots[2] - knots[1])
        return _clip(targets[1] + (evidence - knots[1]) * slope)
    negative_anchor = abs(float(calibration.get("negative_anchor", -1.0)))
    positive_anchor = float(calibration.get("positive_anchor", 1.0))
    if evidence < 0:
        return max(-1.0, evidence / max(negative_anchor, 1e-9))
    if evidence > 0:
        return min(1.0, evidence / max(positive_anchor, 1e-9))
    return 0.0


def _fallback_utilities(score: float) -> list[float]:
    logits = [-UTILITY_BETA * (float(score) - anchor) ** 2 for anchor in ACTION_ANCHORS]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [
        (len(ACTION_ANCHORS) * weight / total - 1.0) / (len(ACTION_ANCHORS) - 1.0)
        for weight in weights
    ]


hydrate_rule, intervention_score, utilities_from_score = _load_mechanism()


# ---------------------------------------------------------------------------
# Episode extraction
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    episodes = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return episodes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_catalog(episode: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = []
    for key, meta in episode["agent_task"]["fields"].items():
        catalog.append(
            {
                "key": key,
                "display_name": meta.get("display_name", key),
                "description": meta.get("description", ""),
                "type": meta.get("type", ""),
                "unit": meta.get("unit", ""),
                "minimum": meta.get("minimum"),
                "maximum": meta.get("maximum"),
                "typical_range": meta.get("typical_range", []),
            }
        )
    return catalog


def rule_summary(rule: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    relevant = []
    for key in rule["relevant_fields"]:
        direction = int(rule["field_directions"][key])
        relevant.append(
            {
                "key": key,
                "display_name": fields[key]["display_name"],
                "direction": direction,
                "direction_label": "positive" if direction > 0 else "negative",
            }
        )
    return {
        "family": rule["family"],
        "family_label": RULE_FAMILY_LABELS.get(rule["family"], rule["family"]),
        "relevant_fields": relevant,
        "field_directions": {
            key: int(value) for key, value in rule["field_directions"].items()
        },
    }


def transition_summary(episode: dict[str, Any]) -> dict[str, Any]:
    transition = episode["evaluation"]["transition"]
    fields = episode["agent_task"]["fields"]
    operations = []

    for operation in transition.get("operations", []):
        kind = operation.get("type")
        if kind == "field_replacement":
            outgoing = operation["outgoing_field"]
            incoming = operation["incoming_field"]
            operations.append(
                {
                    "type": kind,
                    "outgoing_field": outgoing,
                    "outgoing_display_name": fields[outgoing]["display_name"],
                    "incoming_field": incoming,
                    "incoming_display_name": fields[incoming]["display_name"],
                }
            )
        elif kind == "direction_reversal":
            target = operation["target_field"]
            pre_direction = int(transition["pre_rule"]["field_directions"][target])
            post_direction = int(transition["post_rule"]["field_directions"][target])
            operations.append(
                {
                    "type": kind,
                    "target_field": target,
                    "target_display_name": fields[target]["display_name"],
                    "pre_direction": pre_direction,
                    "post_direction": post_direction,
                    "pre_direction_label": "positive" if pre_direction > 0 else "negative",
                    "post_direction_label": "positive" if post_direction > 0 else "negative",
                }
            )

    return {
        "type": transition.get("type", "compound"),
        "after_action_batch": int(transition["after_action_batch"]),
        "pre_rule": rule_summary(transition["pre_rule"], fields),
        "post_rule": rule_summary(transition["post_rule"], fields),
        "operations": operations,
        "semantic_change": transition.get("semantic_change", {}),
    }


def evaluate_case(
    episode: dict[str, Any],
    hydrated_rule: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    utilities = utilities_from_score(
        intervention_score(hydrated_rule, case["raw_values"])
    )
    rounded = [round(value, 2) for value in utilities]
    optimal_index = max(range(len(utilities)), key=utilities.__getitem__)
    field_order = list(episode["agent_task"]["fields"].keys())
    raw_values = {
        key: case["raw_values"][key]
        for key in field_order
        if key in case["raw_values"]
    }
    return {
        "context_id": case["context_id"],
        "stratum": case.get("stratum", ""),
        "raw_values": raw_values,
        "reference_utilities": {
            f"action_{index}": value for index, value in enumerate(rounded)
        },
        "reference_optimal_action": f"action_{optimal_index}",
    }


def choose_cases(
    episode: dict[str, Any],
    hydrated_rule: dict[str, Any],
    count: int = 3,
) -> list[dict[str, Any]]:
    """Pick representative pre-change cases covering distinct optimal actions."""

    evaluated = []
    for case in episode["shared_prefix"]["cases"]:
        utilities = utilities_from_score(
            intervention_score(hydrated_rule, case["raw_values"])
        )
        optimal = max(range(len(utilities)), key=utilities.__getitem__)
        evaluated.append(
            {
                "case": case,
                "optimal": optimal,
                "score": utilities[optimal],
            }
        )

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in evaluated:
        groups[item["optimal"]].append(item)

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered_groups = sorted(groups.items(), key=lambda entry: (-len(entry[1]), entry[0]))
    for _, items in ordered_groups:
        items.sort(key=lambda item: item["score"])
        item = items[len(items) // 2]
        context_id = item["case"]["context_id"]
        if context_id not in seen:
            chosen.append(item)
            seen.add(context_id)
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        ordered_cases = sorted(evaluated, key=lambda item: item["score"])
        targets = [0, len(ordered_cases) // 2, len(ordered_cases) - 1]
        for index in targets:
            item = ordered_cases[index]
            context_id = item["case"]["context_id"]
            if context_id not in seen:
                chosen.append(item)
                seen.add(context_id)
            if len(chosen) >= count:
                break

    chosen.sort(key=lambda item: item["score"])
    return [evaluate_case(episode, hydrated_rule, item["case"]) for item in chosen]


def build_episode(episode: dict[str, Any]) -> dict[str, Any]:
    domain = episode["provenance"]["domain"]
    family = episode["evaluation"]["rule_family"]
    fields = episode["agent_task"]["fields"]
    transition = episode["evaluation"]["transition"]
    hydrated_pre = hydrate_rule(transition["pre_rule"], fields)

    cases = choose_cases(episode, hydrated_pre)
    actions = []
    for index, action in enumerate(episode["agent_task"]["actions"]):
        actions.append(
            {
                "action_id": action["action_id"],
                "display_name": action["display_name"],
                "description": action["description"],
                "strength_rank": index + 1,
                "strength_label": (
                    "Least invasive"
                    if index == 0
                    else "Most invasive"
                    if index == len(episode["agent_task"]["actions"]) - 1
                    else f"Intervention level {index + 1}"
                ),
            }
        )

    return {
        "episode_id": episode["episode_id"],
        "domain": domain,
        "domain_label": DOMAIN_LABELS.get(domain, domain.title()),
        "rule_family": family,
        "rule_family_label": RULE_FAMILY_LABELS.get(family, family),
        "template_id": episode["provenance"].get("template_id", ""),
        "sequence": episode["provenance"].get("sequence"),
        "agent_role": episode["agent_task"]["agent_role"],
        "scenario_description": episode["agent_task"]["scenario_description"],
        "interaction_protocol": episode["agent_task"]["interaction_protocol"],
        "branch": SAMPLE_BRANCH,
        "post_branch_available": SAMPLE_BRANCH in episode.get("post_branches", {}),
        "actions": actions,
        "fields": field_catalog(episode),
        "cases": cases,
        "transition": transition_summary(episode),
    }


def build_episodes(source: Path) -> dict[str, Any]:
    episodes = read_jsonl(source)
    by_id = {episode["episode_id"]: episode for episode in episodes}

    missing = [episode_id for episode_id in SAMPLE_EPISODES.values() if episode_id not in by_id]
    if missing:
        raise SystemExit(
            "sample episodes not found in source: " + ", ".join(missing)
        )

    selected = []
    for domain in ("finance", "healthcare", "cybersecurity", "manufacturing"):
        episode = by_id[SAMPLE_EPISODES[domain]]
        if episode["provenance"]["domain"] != domain:
            raise SystemExit(
                f"{SAMPLE_EPISODES[domain]} is not a {domain} episode"
            )
        selected.append(build_episode(episode))

    family_counts = Counter(episode["rule_family"] for episode in selected)
    if len(family_counts) != 4:
        raise SystemExit(
            "sample episodes must cover four distinct rule families; got "
            + ", ".join(sorted(family_counts))
        )

    source_label = f"{source.parent.name}/{source.name}"
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": source_label,
            "sha256": sha256(source),
            "episode_count": len(episodes),
            "selection": "one episode per domain, flip_50 branch",
        },
        "utility_note": (
            "The released episode JSON does not store per-case action utilities. "
            "Reference utilities shown here are computed from the pre-change hidden "
            "rule with the same deterministic mechanism used by the runtime and are "
            "rounded to two decimals."
        ),
        "episodes": selected,
    }


# ---------------------------------------------------------------------------
# Hand-authored site data
# ---------------------------------------------------------------------------

STATS = [
    {
        "value": "256",
        "label": "Test episodes",
        "sub": "selected-256 dynamic split",
    },
    {
        "value": "1,024",
        "label": "Training episodes",
        "sub": "same construction pipeline",
    },
    {
        "value": "4",
        "label": "Domains",
        "sub": "finance · healthcare · cyber · manufacturing",
    },
    {
        "value": "4",
        "label": "Rule families",
        "sub": "additive · conjunctive · disjunctive · lexicographic",
    },
    {
        "value": "8",
        "label": "Models evaluated",
        "sub": "proprietary and open-source",
    },
]

MODEL_KINDS = {
    "Llama-3.1-8B": "open",
    "Qwen3.5-9B": "open",
    "Qwen3.8-27B": "open",
    "Gemma-4-12B": "open",
    "DeepSeek-V4.1-Flash": "open",
    "GPT-6-Luna": "proprietary",
    "GPT-6-Sol": "proprietary",
    "Gemini-3.8-Flash": "proprietary",
}


def row(
    model: str,
    ea: float,
    ca: float,
    probes: float,
    reprobes: float | None = None,
    factor_rec: float | None = None,
    change_rec: float | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "kind": MODEL_KINDS[model],
        "ea": ea,
        "ca": ca,
        "probes": probes,
        "reprobes": reprobes,
        "factor_rec": factor_rec,
        "change_rec": change_rec,
    }


LEADERBOARD_SETTINGS = [
    {
        "id": "default",
        "label": "Default",
        "description": (
            "Rule drift occurs during the interaction, evaluation batches are "
            "masked, and the agent decides whether and how much to probe."
        ),
        "note": "8 models · main setting",
        "rows": [
            row("Llama-3.1-8B", 30.67, 8.13, 4.77, 0.24, 52.54, 6.30),
            row("Qwen3.5-9B", 32.17, 12.31, 7.81, 1.04, 39.40, 13.80),
            row("Qwen3.8-27B", 45.95, 37.73, 42.20, 5.46, 65.87, 25.93),
            row("Gemma-4-12B", 37.98, 20.59, 12.52, 0.13, 53.21, 10.34),
            row("DeepSeek-V4.1-Flash", 58.01, 52.87, 113.42, 16.04, 78.07, 44.52),
            row("GPT-6-Luna", 46.50, 35.10, 22.84, 0.73, 56.66, 9.96),
            row("GPT-6-Sol", 65.69, 61.87, 99.03, 6.95, 73.79, 47.56),
            row("Gemini-3.8-Flash", 69.47, 67.25, 88.12, 10.25, 78.30, 63.68),
        ],
    },
    {
        "id": "stationary",
        "label": "Stationary",
        "description": (
            "Rule drift is removed while all other default conditions hold. "
            "Re-probes and recognition metrics are undefined."
        ),
        "note": "6 models · Gemini-3.8-Flash has no diagnostic runs",
        "rows": [
            row("DeepSeek-V4.1-Flash", 69.19, 67.60, 85.36),
            row("Qwen3.8-27B", 54.74, 50.57, 33.31),
            row("GPT-6-Luna", 50.26, 41.14, 21.11),
            row("Gemma-4-12B", 39.37, 22.78, 12.28),
            row("Qwen3.5-9B", 32.15, 12.41, 6.76),
            row("Llama-3.1-8B", 30.90, 8.74, 4.53),
        ],
    },
    {
        "id": "evidence-given",
        "label": "Evidence Given",
        "description": (
            "Evaluation batches are unmasked, so evidence accumulates passively "
            "across the interaction."
        ),
        "note": "7 models · Gemini-3.8-Flash omitted",
        "rows": [
            row("Llama-3.1-8B", 34.13, 13.78, 4.87, 0.41, 52.55, 5.32),
            row("Qwen3.5-9B", 43.30, 28.18, 4.72, 0.29, 40.91, 15.39),
            row("Qwen3.8-27B", 35.73, 33.78, 17.36, 0.08, 62.65, 13.85),
            row("Gemma-4-12B", 48.10, 36.52, 7.00, 0.00, 50.34, 12.10),
            row("DeepSeek-V4.1-Flash", 71.51, 70.34, 119.28, 6.38, 79.92, 43.65),
            row("GPT-6-Luna", 56.93, 51.53, 23.92, 0.34, 59.73, 9.91),
            row("GPT-6-Sol", 68.51, 66.57, 79.08, 2.55, 76.41, 44.40),
        ],
    },
    {
        "id": "forced-probe",
        "label": "Forced-Probe",
        "description": (
            "Probing is mandatory in every exploration window, with a minimum "
            "number of probe cases required."
        ),
        "note": "7 models · Gemini-3.8-Flash omitted",
        "rows": [
            row("Llama-3.1-8B", 27.83, 4.07, 19.81, 2.05, 51.10, 7.74),
            row("Qwen3.5-9B", 33.59, 14.32, 17.46, 4.41, 38.66, 12.76),
            row("Qwen3.8-27B", 47.98, 38.69, 58.09, 8.01, 66.34, 38.70),
            row("Gemma-4-12B", 38.96, 22.02, 26.25, 5.37, 52.10, 9.59),
            row("DeepSeek-V4.1-Flash", 57.98, 53.27, 133.05, 18.68, 78.54, 48.91),
            row("GPT-6-Luna", 44.77, 32.27, 68.04, 2.31, 59.60, 14.01),
            row("GPT-6-Sol", 59.89, 53.99, 120.44, 5.02, 73.61, 50.66),
        ],
    },
]

for _setting in LEADERBOARD_SETTINGS:
    _setting["rows"].sort(key=lambda row: (-row["ca"], row["model"]))

TAKEAWAYS = [
    {
        "group": "results",
        "statement": "Proactive and adaptive exploration remains highly challenging for most models.",
        "summary": (
            "All eight models finish below 70% calibrated accuracy. Gemini-3.8-Flash "
            "leads at 67.25%, and the gap between exact and calibrated accuracy widens "
            "from roughly 2–5 points for the top three models to 8–22 points for the rest, "
            "so weaker agents also make more costly mistakes."
        ),
    },
    {
        "group": "results",
        "statement": (
            "Greater exploration and verification effort do not necessarily translate "
            "into better end-to-end performance across models."
        ),
        "summary": (
            "DeepSeek-V4.1-Flash issues about 24 times as many probes as Llama-3.1-8B, "
            "yet its calibrated accuracy trails Gemini-3.8-Flash and GPT-6-Sol. It also "
            "performs more post-change re-probes than either model without closing the gap."
        ),
    },
    {
        "group": "results",
        "statement": "Adaptation ability distinguishes effective agents more clearly than exploration efforts.",
        "summary": (
            "The four strongest models by calibrated accuracy rank in exactly the same "
            "order on Change Recognition, while exploration effort does not separate "
            "performance as consistently."
        ),
    },
    {
        "group": "analysis",
        "statement": "Removing unannounced rule drift significantly recovers agents' end-to-end outcome.",
        "summary": (
            "The Stationary setting raises calibrated accuracy by as much as 14.7 points "
            "for DeepSeek-V4.1-Flash and by 6.2 points on average. The weakest models gain "
            "less than one point, so removing drift alone cannot compensate for weak "
            "rule learning."
        ),
    },
    {
        "group": "analysis",
        "statement": "Agents exploit effectively when sufficient evidence is provided externally.",
        "summary": (
            "Six of seven models improve under Evidence Given, four by more than 15 "
            "calibrated-accuracy points. The gains identify proactive evidence acquisition, "
            "rather than reasoning from available evidence, as a key bottleneck."
        ),
    },
    {
        "group": "analysis",
        "statement": "Effective exploration depends more on explored evidence quality than quantity.",
        "summary": (
            "Forced-Probe lifts mean unique probes from 33.9 to 53.8 and increases "
            "re-probing, yet mean calibrated accuracy falls by 2.0 points. More probes "
            "do not help when the selected cases are uninformative."
        ),
    },
    {
        "group": "error",
        "statement": "Rule Drift causes an immediate performance drop, and no model recovers at last.",
        "summary": (
            "Every model above the random floor loses on average 26.9 calibrated-accuracy "
            "points at the first post-change batch. Averaged over the final two batches, "
            "all six remain 6.9–19.2 points below their own pre-change plateau."
        ),
    },
    {
        "group": "error",
        "statement": (
            "Re-probing provides additional evidence, but successful adaptation also "
            "depends on how effectively a model exploit evidence."
        ),
        "summary": (
            "Re-probing correlates with Change Recognition within each model, but the "
            "cross-model correlation disappears after normalizing by pre-change exploration. "
            "Agents differ in how well they interpret before-and-after observations and "
            "revise their hypotheses."
        ),
    },
    {
        "group": "error",
        "statement": "Agents tend to settle on incomplete representations of environmental dynamics.",
        "summary": (
            "Precision exceeds recall for both Factor and Change Recognition, and in 36.3% "
            "of eligible episodes agents report identical pre- and post-change assessments "
            "despite an actual change. Plausible but incomplete rules often go unchallenged."
        ),
    },
    {
        "group": "error",
        "statement": "Agents struggle to recognize changes in how existing factors influence outcomes.",
        "summary": (
            "Agents identify both replacements but miss the direction reversal in 17.7% of "
            "episodes, compared with only 4.1% for the converse. Updating which factors "
            "matter is easier than revising how an existing factor affects the outcome."
        ),
    },
]

SITE_DATA = {
    "schema_version": SCHEMA_VERSION,
    "stats": STATS,
    "models": MODEL_KINDS,
    "leaderboard": {
        "metrics": [
            {
                "key": "ea",
                "label": "EA (%)",
                "full_label": "Exact Accuracy",
                "better": "high",
                "group": "End-to-end Outcome",
                "highlight": True,
            },
            {
                "key": "ca",
                "label": "CA (%)",
                "full_label": "Calibrated Accuracy",
                "better": "high",
                "group": "End-to-end Outcome",
                "highlight": True,
            },
            {
                "key": "probes",
                "label": "Mean Uni. Probes",
                "full_label": "Mean Unique Probes",
                "better": None,
                "group": "Exploration Behavior",
                "highlight": False,
            },
            {
                "key": "reprobes",
                "label": "Mean Re-probes",
                "full_label": "Mean Re-probes after Change",
                "better": None,
                "group": "Exploration Behavior",
                "highlight": False,
            },
            {
                "key": "factor_rec",
                "label": "Factor Rec. (%)",
                "full_label": "Factor Recognition",
                "better": "high",
                "group": "Adaptation Performance",
                "highlight": True,
            },
            {
                "key": "change_rec",
                "label": "Change Rec. (%)",
                "full_label": "Change Recognition",
                "better": "high",
                "group": "Adaptation Performance",
                "highlight": True,
            },
        ],
        "settings": LEADERBOARD_SETTINGS,
    },
    "takeaways": TAKEAWAYS,
}


def write_json(path: Path, payload: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="path to dynamic_episodes.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for site-data.json and episodes.json",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"episode source not found: {source}")

    episodes_payload = build_episodes(source)
    site_data_path = args.output_dir / "site-data.json"
    episodes_path = args.output_dir / "episodes.json"
    site_bytes = write_json(site_data_path, SITE_DATA)
    episode_bytes = write_json(episodes_path, episodes_payload)

    print(f"source: {source}")
    print(f"wrote {site_data_path} ({site_bytes:,} bytes)")
    print(f"wrote {episodes_path} ({episode_bytes:,} bytes)")
    for episode in episodes_payload["episodes"]:
        print(
            "  {domain:14s} {family:24s} {episode_id}".format(
                domain=episode["domain"],
                family=episode["rule_family"],
                episode_id=episode["episode_id"],
            )
        )
        print(
            "    cases: "
            + ", ".join(
                f"{case['context_id']} -> {case['reference_optimal_action']}"
                for case in episode["cases"]
            )
        )


if __name__ == "__main__":
    main()
