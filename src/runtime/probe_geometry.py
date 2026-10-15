from __future__ import annotations

from statistics import mean
from typing import Any

from protocol.precision import round_field_values

# Episodes with fewer than 2 probe points count as exploration failure:
# variance is undefined below two points, and a lone point cannot localize
# the rule. Geometry leaves stay None so they are excluded from aggregation;
# probe_points itself is still reported as the exploration-willingness
# quantity.
MIN_GEOMETRY_PROBE_POINTS = 2
# A phase window with a single point yields degenerate share values
# (zero variance); require two points per phase.
MIN_PHASE_PROBE_POINTS = 2


def _normalize(value: float, meta: dict[str, Any]) -> float:
    low, high = float(meta["minimum"]), float(meta["maximum"])
    return (float(value) - low) / (high - low) if high > low else 0.0


def _resolve_probe_values(
    probe: Any, contexts: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if not isinstance(probe, dict):
        return None
    if isinstance(probe.get("values"), dict):
        return dict(probe["values"])
    context_id = str(probe.get("base_context_id") or "")
    if context_id not in contexts or not isinstance(probe.get("changes"), dict):
        return None
    values = dict(contexts[context_id]["raw_values"])
    values.update(probe["changes"])
    return values


def _initial_contexts(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for case in (episode.get("shared_prefix") or {}).get("cases") or []:
        if isinstance(case, dict) and case.get("context_id"):
            contexts[str(case["context_id"])] = case
    for branch in (episode.get("post_branches") or {}).values():
        for case in (branch or {}).get("cases") or []:
            if isinstance(case, dict) and case.get("context_id"):
                contexts[str(case["context_id"])] = case
    return contexts


def _runtime_case_contexts(
    artifact: dict[str, Any], episode: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Map anonymized runtime case ids (``case_001`` …) to dataset cases.

    The runtime replaces release context ids with sequential aliases over the
    concatenated prefix + selected-post-branch case list, so probes that modify
    an observed case reference ids that do not exist in the dataset.
    """

    cases = list((episode.get("shared_prefix") or {}).get("cases") or [])
    post = (episode.get("post_branches") or {}).get(
        str(artifact.get("sampling_branch") or "")
    ) or {}
    cases.extend(post.get("cases") or [])
    contexts: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, start=1):
        if isinstance(case, dict) and isinstance(case.get("raw_values"), dict):
            contexts[f"case_{index:03d}"] = case
    return contexts


def materialize_probe_points(
    artifact: dict[str, Any], episode: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """Resolve probe submissions into normalized 8D points with their windows.

    Values come from the tool trace (new-context ``values`` or
    existing-context ``base_context_id`` + ``changes`` resolved against dataset
    cases and already-materialized probe contexts).  The window is the 0-based
    ``batch_number`` of the matching private-history probe event.
    """

    fields = (episode.get("agent_task") or {}).get("fields") or {}
    contexts = _initial_contexts(episode)
    contexts.update(_runtime_case_contexts(artifact, episode))
    private_probes = {
        int(item.get("interaction") or 0): item
        for item in artifact.get("private_history") or []
        if item.get("type") == "probe_batch"
    }
    points: list[dict[str, Any]] = []
    unresolved = 0
    probe_attempt = 0
    for event in artifact.get("tool_trace") or []:
        response = event.get("response") or {}
        if response.get("tool") != "probe_batch":
            continue
        event_result = event.get("result") or {}
        if event.get("valid") is False or not isinstance(
            event_result.get("results"), list
        ):
            continue
        probe_attempt += 1
        private_probe = private_probes.get(int(event.get("interaction") or 0), {})
        window = int(private_probe.get("batch_number", probe_attempt - 1))
        submitted = (response.get("arguments") or {}).get("probes") or []
        results = event_result["results"]
        for index, probe in enumerate(submitted):
            values = _resolve_probe_values(probe, contexts)
            if values is None:
                unresolved += 1
                continue
            values = round_field_values(values, fields)
            result = results[index] if index < len(results) else {}
            points.append({
                "sequence": len(points),
                "window": window,
                "normalized_values": {
                    field: _normalize(values[field], fields[field])
                    for field in fields
                },
                "utilities": [float(value) for value in (result.get("utilities") or {}).values()],
            })
            probe_context_id = str(result.get("context_id") or "")
            requested_base_id = str((probe or {}).get("base_context_id") or "")
            if probe_context_id and probe_context_id != requested_base_id:
                contexts[probe_context_id] = {"raw_values": dict(values)}
    return points, unresolved


def probe_phase_point_sets(
    artifact: dict[str, Any], episode: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Deduplicate materialized probe points into pre/post-change phase sets.

    Each point becomes a tuple of its normalized values over the episode
    fields, so equality is exact on the agent-facing value grid.  A probe
    window (0-based) belongs to the post-change phase iff
    ``window >= evaluation.transition.after_action_batch``; branches without
    a rule change keep every point in ``pre`` and report
    ``has_change == False``.  Branch artifacts do not embed
    ``agent_task.fields`` or dataset case values, so materialization still
    requires the dataset episode; with ``episode=None`` both sets come back
    empty.
    """

    transition = (artifact.get("evaluation") or {}).get("transition") or {}
    raw_change = transition.get("after_action_batch")
    has_transition = (
        not isinstance(raw_change, bool) and isinstance(raw_change, int) and raw_change > 0
    )
    has_change = (
        str(artifact.get("condition") or "dynamic") == "dynamic" and has_transition
    )
    if episode is None:
        return {"pre": set(), "post": set(), "has_change": has_change}
    fields = (episode.get("agent_task") or {}).get("fields") or {}
    points, _unresolved = materialize_probe_points(artifact, episode)

    def key(point: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(point["normalized_values"][field] for field in fields)

    if has_change:
        pre = {
            key(point) for point in points if point["window"] < raw_change
        }
        post = {
            key(point) for point in points if point["window"] >= raw_change
        }
    else:
        pre = {key(point) for point in points}
        post = set()
    return {"pre": pre, "post": post, "has_change": has_change}


def _axis_variances(
    points: list[dict[str, Any]], fields: dict[str, Any]
) -> list[float] | None:
    if len(points) < 2:
        return None
    variances = []
    for field in fields:
        values = [point["normalized_values"][field] for point in points]
        center = mean(values)
        variances.append(
            sum((value - center) ** 2 for value in values) / (len(values) - 1)
        )
    return variances


def _total_variance(
    points: list[dict[str, Any]], fields: dict[str, Any]
) -> float | None:
    variances = _axis_variances(points, fields)
    if variances is None:
        return None
    return round(sum(variances), 6)


def _focus_share(
    variances: list[float] | None,
    relevant: list[str],
    fields: dict[str, Any],
) -> float | None:
    if variances is None:
        return None
    total = sum(variances)
    if total == 0:
        return None
    field_names = list(fields)
    focused = sum(
        variances[field_names.index(field)]
        for field in relevant
        if field in field_names
    )
    return round(focused / total, 6)


def compute_geometry(
    points: list[dict[str, Any]],
    episode: dict[str, Any],
    *,
    condition: str,
    change_after_batch: int,
) -> dict[str, Any]:
    """Compute the exploration-geometry block for one episode/branch artifact."""

    fields = (episode.get("agent_task") or {}).get("fields") or {}
    evaluation = episode.get("evaluation") or {}
    transition = evaluation.get("transition") or {}
    pre_rule = transition.get("pre_rule") or {}
    post_rule = transition.get("post_rule") or {}
    pre_points = [
        point for point in points if point["window"] < change_after_batch
    ]
    post_points = [
        point for point in points if point["window"] >= change_after_batch
    ]
    post_rule_active = post_rule if condition == "dynamic" else pre_rule
    sufficient = len(points) >= MIN_GEOMETRY_PROBE_POINTS
    pre_relevant = [str(field) for field in (pre_rule.get("relevant_fields") or [])]
    post_relevant = [str(field) for field in (post_rule.get("relevant_fields") or [])]
    incoming_fields = sorted(set(post_relevant) - set(pre_relevant))
    outgoing_fields = sorted(set(pre_relevant) - set(post_relevant))
    swap_fields = condition == "dynamic" and bool(incoming_fields) and bool(
        outgoing_fields
    )

    def phase_block(
        phase_points: list[dict[str, Any]], active_rule: dict[str, Any]
    ) -> dict[str, Any]:
        if not sufficient or len(phase_points) < MIN_PHASE_PROBE_POINTS:
            return {
                "focus_variance_share": None,
                "incoming_field_share": None,
                "outgoing_field_share": None,
            }
        relevant = [str(field) for field in (active_rule.get("relevant_fields") or [])]
        variances = _axis_variances(phase_points, fields)
        return {
            "focus_variance_share": _focus_share(variances, relevant, fields),
            "incoming_field_share": (
                _focus_share(variances, incoming_fields, fields)
                if swap_fields
                else None
            ),
            "outgoing_field_share": (
                _focus_share(variances, outgoing_fields, fields)
                if swap_fields
                else None
            ),
        }

    return {
        "probe_points": len(points),
        "unresolved_probes": 0,
        "spread": {
            "total_variance": (
                _total_variance(points, fields) if sufficient else None
            ),
        },
        "pre": phase_block(pre_points, pre_rule),
        "post": phase_block(post_points, post_rule_active),
    }
