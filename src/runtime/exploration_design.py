"""Family-specific diagnostic geometry for exploration analysis."""
from __future__ import annotations

from itertools import product
from statistics import mean
from typing import Any

from protocol.mechanism import FAMILIES

DESIGN_VERSION = 3


def _covariance_determinant(matrix: list[list[float]]) -> float:
    """Pivoted elimination with a relative tolerance for numerical rank."""
    work = [row[:] for row in matrix]
    scale = max(abs(value) for row in work for value in row)
    if scale == 0:
        return 0.0
    determinant = 1.0
    for j in range(len(work)):
        pivot = max(range(j, len(work)), key=lambda i: abs(work[i][j]))
        if abs(work[pivot][j]) <= scale * 1e-12:
            return 0.0
        if pivot != j:
            work[j], work[pivot] = work[pivot], work[j]
            determinant *= -1
        value = work[j][j]
        determinant *= value
        for i in range(j + 1, len(work)):
            factor = work[i][j] / value
            for k in range(j + 1, len(work)):
                work[i][k] -= factor * work[j][k]
    return max(0.0, determinant)


def _phase_design(
    points: list[dict[str, Any]], rule: dict[str, Any], fields: dict[str, Any]
) -> dict[str, Any]:
    relevant = list(rule["relevant_fields"])
    family = rule["family"]
    d = len(relevant)
    if not d or family not in FAMILIES:
        raise ValueError("exploration geometry requires a supported rule and relevant fields")
    # Duplicating an input or changing only distractors contributes no extra
    # weight. The effective n is the number of unique relevant-field vectors.
    rows = sorted({
        tuple(float(point["normalized_values"][field]) for field in relevant)
        for point in points
    })
    n = len(rows)
    details: dict[str, Any] = {}
    if family == "additive_aggregation":
        centers = [mean(row[j] for row in rows) for j in range(d)] if n else [0.0] * d
        covariance = [[
            mean((row[i] - centers[i]) * (row[j] - centers[j]) for row in rows)
            if n else 0.0
            for j in range(d)
        ] for i in range(d)]
        # Population covariance (divide by n), not sample covariance. Each
        # normalized coordinate has variance <= 0.25.
        determinant = _covariance_determinant(covariance) if n > d else 0.0
        score = min(1.0, 4.0 * determinant ** (1.0 / d))
        components = {"independent_effect_coverage": score}
        details = {"covariance": covariance, "covariance_determinant": determinant}
    else:
        thresholds = {}
        for field in relevant:
            low, high = float(fields[field]["minimum"]), float(fields[field]["maximum"])
            if high <= low:
                raise ValueError("relevant field range must have positive width")
            thresholds[field] = (float(rule["thresholds"][field]) - low) / (high - low)
        # Match the evaluator's [-1,1] coordinate system, including the scale
        # of indifference_band. Equality at a threshold is fail/non-trigger.
        margins = [{
            field: 2.0 * float(rule["field_directions"][field]) * (row[j] - thresholds[field])
            for j, field in enumerate(relevant)
        } for row in rows]
        if family in {"conjunctive_gating", "disjunctive_sufficiency"}:
            regions = tuple(product((0, 1), repeat=d))
            visited = {tuple(int(m[field] > 0) for field in relevant) for m in margins}

            def weight(region: tuple[int, ...]) -> float:
                count = d - sum(region) if family == "conjunctive_gating" else sum(region)
                return 1.0 / max(1, count)

            covered_weight = sum(weight(region) for region in sorted(visited))
            total_weight = sum(weight(region) for region in regions)
            score = covered_weight / total_weight
            components = {"diagnostic_region_coverage": score}
            details = {
                "visited_regions": ["".join(map(str, region)) for region in sorted(visited)],
                "total_regions": len(regions),
                "covered_weight": covered_weight,
                "total_weight": total_weight,
            }
        else:
            order = list(rule["priority_order"])
            band = float(rule["indifference_band"])
            signed_regions: set[tuple[int, str]] = set()
            for margin in margins:
                for k, field in enumerate(order):
                    if abs(margin[field]) <= band:
                        continue
                    # This is the first decisive cue; all higher cues are
                    # neutral. Every non-final level requires a decisive
                    # lower-priority cue with the opposite sign.
                    conflict = any(
                        abs(margin[lower]) > band and margin[field] * margin[lower] < 0
                        for lower in order[k + 1:]
                    )
                    if k == d - 1 or conflict:
                        signed_regions.add((k + 1, "+" if margin[field] > 0 else "-"))
                    break
            score = len(signed_regions) / (2 * d)
            components = {"priority_conflict_region_coverage": score}
            details = {
                "visited_regions": [f"D_{k},{sign}" for k, sign in sorted(signed_regions)],
                "total_regions": 2 * d,
            }
    # A phase is assessable only with enough distinct relevant-field vectors:
    # region families need at least one visited region, while a d-dimensional
    # covariance volume has rank < d (determinant zero) with at most d points.
    min_unique = d + 1 if family == "additive_aggregation" else 1
    return {"score": score, "components": components, "unique_probes": n,
            "valid": n >= min_unique, **details}


def exploration_design(points: list[dict[str, Any]], episode: dict[str, Any], *,
                       condition: str, change_after_batch: int) -> dict[str, Any]:
    transition = episode["evaluation"]["transition"]
    pre, post = transition["pre_rule"], transition["post_rule"]
    fields = episode["agent_task"]["fields"]
    if condition == "dynamic":
        phases = {
            "pre": _phase_design([p for p in points if p["window"] < change_after_batch], pre, fields),
            "post": _phase_design([p for p in points if p["window"] >= change_after_batch], post, fields),
        }
    else:
        phases = {"static": _phase_design(points, pre, fields)}
    valid_scores = [phase["score"] for phase in phases.values() if phase["valid"]]
    return {"version": DESIGN_VERSION,
            "score": mean(valid_scores) if valid_scores else None,
            "rule_family": pre["family"], "phases": phases, "probe_count": len(points)}


def aggregate_exploration_design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    for row in rows:
        design = ((row.get("probe_metrics") or {}).get("geometry") or {}).get("design")
        if design and design.get("version") == DESIGN_VERSION and design.get("rule_family") in grouped:
            grouped[design["rule_family"]].append(design)

    def phase_stats(designs: list[dict[str, Any]], phase: str) -> tuple[float | None, int]:
        scores = [
            float(design["phases"][phase]["score"])
            for design in designs
            if (design.get("phases") or {}).get(phase, {}).get("valid")
        ]
        return (round(mean(scores), 6) if scores else None, len(scores))

    by_family: dict[str, Any] = {}
    for family, designs in grouped.items():
        if not designs:
            by_family[family] = None
            continue
        valid_scores = [float(d["score"]) for d in designs if d.get("score") is not None]
        unique_counts = [
            sum(int(phase.get("unique_probes") or 0) for phase in (design.get("phases") or {}).values())
            for design in designs
        ]
        pre_score, pre_valid = phase_stats(designs, "pre")
        post_score, post_valid = phase_stats(designs, "post")
        by_family[family] = {
            "score": round(mean(valid_scores), 6) if valid_scores else None,
            "valid_episodes": len(valid_scores),
            "episode_count": len(designs),
            "pre_score": pre_score,
            "pre_valid_episodes": pre_valid,
            "post_score": post_score,
            "post_valid_episodes": post_valid,
            "mean_unique_probes": round(mean(unique_counts), 6) if unique_counts else None,
        }
    present = [entry["score"] for entry in by_family.values() if entry and entry["score"] is not None]
    return {"exploration_performance": round(mean(present), 6) if present else None,
            "exploration_by_rule_family": by_family}
