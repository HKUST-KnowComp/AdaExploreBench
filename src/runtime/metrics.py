from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from runtime.probe_geometry import probe_phase_point_sets


def _change_after_batch(artifact: dict[str, Any]) -> int:
    value = (
        ((artifact.get("evaluation") or {}).get("transition") or {}).get(
            "after_action_batch"
        )
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("dynamic artifact lacks a valid change batch")
    return value


def _first_learn_rows(
    rows: list[dict[str, Any]],
    *,
    start_batch: int,
    batch_count: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return rows from a contiguous learning window and its batch count.

    Accuracy uses every row in the selected batches, including unanswered
    cases (which have ``correct == False``).  Keeping the window batch-based
    makes the static reference comparable when episodes have different change
    points and therefore different numbers of post-change batches.
    """

    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_number = row.get("batch_number")
        if isinstance(raw_number, bool):
            continue
        try:
            number = int(raw_number)
        except (TypeError, ValueError):
            continue
        by_batch[number].append(row)
    available = sorted(number for number in by_batch if number >= start_batch)
    if batch_count is not None:
        available = available[: max(0, int(batch_count))]
    return [row for number in available for row in by_batch[number]], len(available)


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    return mean(1.0 if row.get("correct") else 0.0 for row in rows) if rows else None


def _first_learn_comparison(
    dynamic_artifact: dict[str, Any],
    static_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Compare dynamic post-change learning with a same-length static window."""

    try:
        fork = _change_after_batch(dynamic_artifact)
    except ValueError:
        return {
            "dynamic_post_rule_accuracy": None,
            "static_first_learn_accuracy": None,
            "accuracy_delta": None,
            "post_change_batch_count": 0,
            "static_reference_batch_count": 0,
            "comparison_batch_count": 0,
        }
    dynamic_rows, post_count = _first_learn_rows(
        list(dynamic_artifact.get("action_results") or []),
        start_batch=fork + 1,
    )
    # The dynamic post window defines the horizon.  Static starts at batch 1,
    # so a 7-batch post window is compared with static batches 1..7, while a
    # 5-batch post window is compared with static batches 1..5.
    static_rows, static_count = _first_learn_rows(
        list(static_artifact.get("action_results") or []),
        start_batch=1,
        batch_count=post_count,
    )
    comparison_count = min(post_count, static_count)
    if comparison_count <= 0:
        return {
            "dynamic_post_rule_accuracy": None,
            "static_first_learn_accuracy": None,
            "accuracy_delta": None,
            "post_change_batch_count": post_count,
            "static_reference_batch_count": static_count,
            "comparison_batch_count": 0,
        }
    # Re-slice both sides to the common count when a failed/incomplete static
    # artifact has fewer batches than expected.
    dynamic_rows, _ = _first_learn_rows(
        list(dynamic_artifact.get("action_results") or []),
        start_batch=fork + 1,
        batch_count=comparison_count,
    )
    static_rows, _ = _first_learn_rows(
        list(static_artifact.get("action_results") or []),
        start_batch=1,
        batch_count=comparison_count,
    )
    dynamic_accuracy = _accuracy(dynamic_rows)
    static_accuracy = _accuracy(static_rows)
    return {
        "dynamic_post_rule_accuracy": dynamic_accuracy,
        "static_first_learn_accuracy": static_accuracy,
        "accuracy_delta": (
            dynamic_accuracy - static_accuracy
            if dynamic_accuracy is not None and static_accuracy is not None
            else None
        ),
        "post_change_batch_count": post_count,
        "static_reference_batch_count": static_count,
        "comparison_batch_count": comparison_count,
    }


def first_learn_rule_comparison(
    dynamic_artifacts: list[dict[str, Any]],
    static_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the matched first-rule-learning accuracy comparison.

    Artifacts are paired by episode id.  The returned accuracies are macro
    means over paired episodes, while ``comparisons`` retains the per-episode
    horizon and values for auditability.
    """

    static_by_episode = {
        str(item.get("episode_id")): item for item in static_artifacts
    }
    comparisons = []
    for dynamic in dynamic_artifacts:
        static = static_by_episode.get(str(dynamic.get("episode_id")))
        if static is None:
            continue
        comparison = _first_learn_comparison(dynamic, static)
        comparisons.append({
            "episode_id": dynamic.get("episode_id"),
            **comparison,
        })

    def average(key: str) -> float | None:
        values = [
            float(item[key])
            for item in comparisons
            if item.get(key) is not None
        ]
        return round(mean(values), 6) if values else None

    return {
        "episode_count": len(comparisons),
        "dynamic_post_rule_accuracy": average("dynamic_post_rule_accuracy"),
        "static_first_learn_accuracy": average("static_first_learn_accuracy"),
        "accuracy_delta": average("accuracy_delta"),
        "comparisons": comparisons,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _rounded_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    value = _mean(rows, key)
    return round(value, 2) if value is not None else None


def _performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "accuracy": (
            mean(1.0 if row.get("correct") else 0.0 for row in rows)
            if rows
            else None
        ),
        "mean_utility": _rounded_mean(rows, "utility"),
        "cumulative_utility": round(
            sum(float(row.get("utility") or 0.0) for row in rows), 2
        ),
        "mean_regret": _rounded_mean(rows, "regret"),
        "cumulative_regret": round(
            sum(float(row.get("regret") or 0.0) for row in rows), 2
        ),
    }


def partial_credit(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Min-max normalized utility per case, plus the uniform-random baseline.

    Each case scores (chosen_utility - min) / (max - min) over the case's oracle
    utility vector; the baseline is the same transform averaged over all actions,
    i.e. the expected score of a uniformly random policy.
    """
    scores: list[float] = []
    baselines: list[float] = []
    for row in rows:
        utilities = (row.get("oracle") or {}).get("utilities") or {}
        values = [float(value) for value in utilities.values()]
        if len(values) < 2:
            continue
        low, high = min(values), max(values)
        if high == low:
            scores.append(1.0)
            baselines.append(1.0)
            continue
        scores.append((float(row.get("utility") or 0.0) - low) / (high - low))
        baselines.append(mean((value - low) / (high - low) for value in values))
    if not scores:
        return None, None
    return mean(scores), mean(baselines)


def _calibrated_accuracy(
    partial_credit: float | None, baseline: float | None
) -> float | None:
    """Chance-correct partial credit: random policy -> 0, oracle -> 1."""
    if partial_credit is None or baseline is None or baseline >= 1.0:
        return None
    return (partial_credit - baseline) / (1.0 - baseline)


def _dynamic_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [row for row in rows if not row.get("unanswered")]
    post = [row for row in observed if row.get("phase") == "post_dynamic"]
    flipped = [row for row in post if row.get("rule_flip")]
    new_adoption = 0
    stale_adherence = 0
    neither = 0
    for row in flipped:
        selected = row.get("action_id")
        active = (row.get("oracle") or {}).get("optimal_action_id")
        stale = row.get("stale_rule_optimal_action_id")
        if selected == active:
            new_adoption += 1
        elif selected == stale:
            stale_adherence += 1
        else:
            neither += 1
    return {
        "case_count": len(observed),
        "active_rule_accuracy": (
            mean(1.0 if row.get("correct") else 0.0 for row in observed)
            if observed
            else None
        ),
        "old_rule_alignment": (
            mean(
                1.0
                if row.get("action_id")
                == row.get("stale_rule_optimal_action_id")
                else 0.0
                for row in post
            )
            if post
            else None
        ),
        "stale_policy_accuracy": (
            mean(
                1.0
                if row.get("stale_rule_optimal_action_id")
                == (row.get("oracle") or {}).get("optimal_action_id")
                else 0.0
                for row in post
            )
            if post
            else None
        ),
        "flipped_case_breakdown": {
            "denominator": len(flipped),
            "new_rule_adoption": new_adoption,
            "stale_rule_adherence": stale_adherence,
            "neither_rule": neither,
        },
    }


def _adaptation_batch_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure post-change adoption and stability for one action batch.

    Unanswered cases remain in the denominator and count as incorrect. This
    prevents abstention from inflating either adaptation or stability.
    """

    post = [row for row in rows if row.get("phase") == "post_dynamic"]
    flipped = [row for row in post if row.get("rule_flip")]
    nonflipped = [row for row in post if not row.get("rule_flip")]

    def matches_post(row: dict[str, Any]) -> bool:
        return (
            not row.get("unanswered")
            and row.get("action_id")
            == (row.get("oracle") or {}).get("optimal_action_id")
        )

    return {
        "flip_case_count": len(flipped),
        "nonflip_case_count": len(nonflipped),
        "flip_adoption_accuracy": (
            sum(matches_post(row) for row in flipped) / len(flipped)
            if flipped
            else None
        ),
        "nonflip_stability_accuracy": (
            sum(matches_post(row) for row in nonflipped) / len(nonflipped)
            if nonflipped
            else None
        ),
    }


def _stale_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stale_rows = []
    for row in rows:
        stale_action = row.get("stale_rule_optimal_action_id")
        utilities = (row.get("oracle") or {}).get("utilities") or {}
        if stale_action not in utilities:
            continue
        oracle_utility = max(float(value) for value in utilities.values())
        stale_utility = float(utilities[stale_action])
        stale_rows.append(
            {
                "utility": stale_utility,
                "regret": oracle_utility - stale_utility,
                "correct": stale_action
                == (row.get("oracle") or {}).get("optimal_action_id"),
            }
        )
    return _performance(stale_rows)


def _counterfactual_policy(rows: list[dict[str, Any]], action_id: str) -> dict[str, Any]:
    outcomes = []
    for row in rows:
        utilities = (row.get("oracle") or {}).get("utilities") or {}
        if action_id not in utilities:
            continue
        utility = float(utilities[action_id])
        optimal = max(float(value) for value in utilities.values())
        outcomes.append(
            {
                "utility": utility,
                "regret": optimal - utility,
                "correct": action_id
                == (row.get("oracle") or {}).get("optimal_action_id"),
            }
        )
    return {"action_id": action_id, **_performance(outcomes)}


def _batch_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = Counter(
        str(row.get("action_id")) for row in rows if row.get("action_id") is not None
    )
    oracle = Counter(
        str((row.get("oracle") or {}).get("optimal_action_id"))
        for row in rows
        if (row.get("oracle") or {}).get("optimal_action_id") is not None
    )
    action_ids = sorted(
        {
            str(action_id)
            for row in rows
            for action_id in ((row.get("oracle") or {}).get("utilities") or {})
        }
    )
    fixed = [_counterfactual_policy(rows, action_id) for action_id in action_ids]
    best_accuracy = max(
        fixed,
        key=lambda item: (float(item["accuracy"] or 0.0), item["action_id"]),
        default=None,
    )
    best_utility = max(
        fixed,
        key=lambda item: (float(item["mean_utility"] or 0.0), item["action_id"]),
        default=None,
    )
    margins = []
    random_utilities = []
    random_regrets = []
    for row in rows:
        utilities = [
            float(value)
            for value in ((row.get("oracle") or {}).get("utilities") or {}).values()
        ]
        if not utilities:
            continue
        ordered = sorted(utilities, reverse=True)
        margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else 0.0)
        expected = mean(utilities)
        random_utilities.append(expected)
        random_regrets.append(max(utilities) - expected)
    return {
        "selected_action_distribution": dict(sorted(selected.items())),
        "oracle_action_distribution": dict(sorted(oracle.items())),
        "oracle_majority_share": max(oracle.values()) / len(rows) if rows and oracle else None,
        "best_constant_accuracy_policy": best_accuracy,
        "best_constant_utility_policy": best_utility,
        "uniform_random_policy": {
            "expected_accuracy": 1.0 / len(action_ids) if action_ids else None,
            "expected_mean_utility": round(mean(random_utilities), 2)
            if random_utilities
            else None,
            "expected_mean_regret": round(mean(random_regrets), 2)
            if random_regrets
            else None,
        },
        "oracle_utility_margin": {
            "minimum": round(min(margins), 2) if margins else None,
            "median": round(median(margins), 2) if margins else None,
            "mean": round(mean(margins), 2) if margins else None,
            "maximum": round(max(margins), 2) if margins else None,
        },
    }


def _token_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize provider usage without assuming one vendor's field names."""
    totals = {
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    finish_reasons: dict[str, int] = defaultdict(int)
    invalid_reasons: dict[str, int] = defaultdict(int)
    prompt_characters = 0
    wire_characters = 0
    latencies = []
    usage_reported_request_count = 0
    for record in trace:
        usage = record.get("usage") or {}
        prompt_details = (
            usage.get("prompt_tokens_details")
            or usage.get("input_tokens_details")
            or {}
        )
        completion_details = (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or {}
        )
        usage_reported_request_count += int(bool(usage))
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        cached = int(
            usage.get("prompt_cache_hit_tokens")
            or usage.get("cached_prompt_tokens")
            or usage.get("cache_read_input_tokens")
            or prompt_details.get("cached_tokens")
            or 0
        )
        completion = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        reasoning = int(
            usage.get("reasoning_tokens")
            or usage.get("thoughts_tokens")
            or completion_details.get("reasoning_tokens")
            or 0
        )
        total = int(
            usage.get("total_tokens") or prompt + completion + reasoning
        )
        totals["prompt_tokens"] += prompt
        totals["cached_prompt_tokens"] += cached
        totals["completion_tokens"] += completion
        totals["reasoning_tokens"] += reasoning
        totals["total_tokens"] += total
        prompt_characters += int(record.get("request_prompt_characters") or 0)
        wire_characters += int(record.get("provider_wire_request_characters") or 0)
        reason = str(record.get("finish_reason") or "")
        if reason:
            finish_reasons[reason] += 1
        invalid_reason = str(record.get("error_type") or "")
        if invalid_reason:
            invalid_reasons[invalid_reason] += 1
        if record.get("latency_seconds") is not None:
            latencies.append(float(record["latency_seconds"]))
    return {
        **totals,
        "uncached_prompt_tokens": max(
            0, totals["prompt_tokens"] - totals["cached_prompt_tokens"]
        ),
        "request_count": len(trace),
        "usage_reported_request_count": usage_reported_request_count,
        "missing_usage_request_count": len(trace) - usage_reported_request_count,
        "max_request_prompt_tokens": max(
            (
                int(
                    (record.get("usage") or {}).get("prompt_tokens")
                    or (record.get("usage") or {}).get("input_tokens")
                    or 0
                )
                for record in trace
            ),
            default=0,
        ),
        "max_request_prompt_characters": max(
            (int(record.get("request_prompt_characters") or 0) for record in trace),
            default=0,
        ),
        "total_request_prompt_characters": prompt_characters,
        "max_provider_wire_request_characters": max(
            (int(record.get("provider_wire_request_characters") or 0) for record in trace),
            default=0,
        ),
        "total_provider_wire_request_characters": wire_characters,
        "max_canonical_visible_history_characters": max(
            (
                int(record.get("canonical_visible_history_characters") or 0)
                for record in trace
            ),
            default=0,
        ),
        "max_history_event_count": max(
            (int(record.get("history_event_count") or 0) for record in trace),
            default=0,
        ),
        "provider_continuation_response_count": sum(
            bool(record.get("provider_continuation_present")) for record in trace
        ),
        "reasoning_signature_response_count": sum(
            bool(record.get("reasoning_signature_present")) for record in trace
        ),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "invalid_call_reasons": dict(sorted(invalid_reasons.items())),
        "latency_seconds": {
            "total": round(sum(latencies), 6),
            "mean": round(mean(latencies), 6) if latencies else None,
            "maximum": round(max(latencies), 6) if latencies else None,
        },
        "invalid_response_count": sum(
            record.get("status") == "invalid_response" for record in trace
        ),
    }


def _probe_adaptation_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    trace_by_interaction = {
        int(record.get("interaction") or 0): record
        for record in artifact.get("tool_trace") or []
        if (record.get("response") or {}).get("tool") == "probe_batch"
    }
    event = ((artifact.get("evaluation") or {}).get("transition") or {})
    change_after_batch = _change_after_batch(artifact)
    operations = event.get("operations") or []
    replacement = next(
        (item for item in operations if item.get("type") == "field_replacement"),
        {},
    )
    reversal = next(
        (item for item in operations if item.get("type") == "direction_reversal"),
        {},
    )
    outgoing = replacement.get("outgoing_field")
    incoming = replacement.get("incoming_field")
    reversal_target = reversal.get("target_field")
    prior_utilities: dict[str, tuple[tuple[str, float], ...]] = {}
    selected_feedback: dict[str, tuple[str, float]] = {}
    for row in artifact.get("action_results") or []:
        if not row.get("context_id") or row.get("action_id") is None:
            continue
        feedback = (str(row.get("action_id")), float(row.get("utility") or 0.0))
        selected_feedback[str(row["context_id"])] = feedback
        # Runtime ids are opaque and branch-local; private records retain the
        # source id for artifact joins.  Index both namespaces for adaptation
        # metrics so probes submitted with model-visible ids still match.
        runtime_id = row.get("runtime_context_id")
        if runtime_id:
            selected_feedback[str(runtime_id)] = feedback
    counts = {
        "post_change_identity_replays": 0,
        "full_vector_replay_comparisons": 0,
        "full_vector_change_detections": 0,
        "selected_utility_replay_comparisons": 0,
        "selected_utility_change_detections": 0,
        "outgoing_field_single_edits": 0,
        "incoming_field_single_edits": 0,
        "reversal_target_single_edits": 0,
    }
    strategy_totals: Counter[str] = Counter()
    strategy_by_window: dict[int, Counter[str]] = defaultdict(Counter)
    base_source_totals: Counter[str] = Counter()
    edited_fields: Counter[str] = Counter()
    unique_bases: set[str] = set()
    identity_bases: set[str] = set()
    post_identity_bases: set[str] = set()
    for history in artifact.get("private_history") or []:
        if history.get("type") != "probe_batch":
            continue
        trace = trace_by_interaction.get(int(history.get("interaction") or 0), {})
        requests = (
            ((trace.get("response") or {}).get("arguments") or {}).get("probes")
            or []
        )
        results = history.get("results") or []
        batch_number = int(history.get("batch_number") or 0)
        post_change = batch_number >= change_after_batch
        for request, result in zip(requests, results):
            base = request.get("base_context_id")
            changes = request.get("changes")
            utilities = tuple(
                sorted(
                    (str(key), float(value))
                    for key, value in (result.get("utilities") or {}).items()
                )
            )
            if not base:
                strategy = "new_context"
            elif changes == {}:
                strategy = "identity_replay"
            elif isinstance(changes, dict) and len(changes) == 1:
                strategy = "single_field_edit"
            else:
                strategy = "multi_field_edit"
            strategy_totals[strategy] += 1
            strategy_by_window[batch_number][strategy] += 1
            if base:
                base = str(base)
                unique_bases.add(base)
                base_source_totals[
                    "probe_context" if base.startswith("probe_") else "action_context"
                ] += 1
                if changes == {}:
                    identity_bases.add(base)
                    if post_change:
                        post_identity_bases.add(base)
            if isinstance(changes, dict):
                edited_fields.update(str(field) for field in changes)
            if post_change and base and changes == {}:
                counts["post_change_identity_replays"] += 1
                if base in prior_utilities:
                    counts["full_vector_replay_comparisons"] += 1
                    if prior_utilities[base] != utilities:
                        counts["full_vector_change_detections"] += 1
                if base in selected_feedback:
                    action_id, old_utility = selected_feedback[base]
                    current = dict(utilities).get(action_id)
                    if current is not None:
                        counts["selected_utility_replay_comparisons"] += 1
                        if current != old_utility:
                            counts["selected_utility_change_detections"] += 1
            if post_change and isinstance(changes, dict) and len(changes) == 1:
                field = next(iter(changes))
                if field == outgoing:
                    counts["outgoing_field_single_edits"] += 1
                if field == incoming:
                    counts["incoming_field_single_edits"] += 1
                if field == reversal_target:
                    counts["reversal_target_single_edits"] += 1
            if not post_change:
                context_id = result.get("context_id")
                if context_id:
                    prior_utilities[str(context_id)] = utilities
    return {
        **counts,
        "strategy_totals": dict(sorted(strategy_totals.items())),
        "base_source_totals": dict(sorted(base_source_totals.items())),
        "edited_field_distribution": dict(sorted(edited_fields.items())),
        "unique_base_contexts": len(unique_bases),
        "identity_replay_unique_bases": len(identity_bases),
        "post_change_identity_replay_unique_bases": len(post_identity_bases),
        "strategy_by_window": [
            {
                "window_number": window,
                **dict(sorted(window_counts.items())),
            }
            for window, window_counts in sorted(strategy_by_window.items())
        ],
    }


def episode_metrics(
    artifact: dict[str, Any], episode: dict[str, Any] | None = None
) -> dict[str, Any]:
    results = list(artifact.get("action_results") or [])
    answered = [item for item in results if not item.get("unanswered")]
    private_history = list(artifact.get("private_history") or [])
    submit_events = [
        event for event in private_history if event.get("type") == "submit_action_batch"
    ]
    state = dict(artifact.get("private_state") or {})
    partial_credit_score, partial_credit_baseline = partial_credit(results)
    calibrated_accuracy = _calibrated_accuracy(partial_credit_score, partial_credit_baseline)
    change_after_batch = _change_after_batch(artifact)
    rounds = {
        int(key): int(value)
        for key, value in (state.get("probe_batches_by_batch") or {}).items()
    }
    cases = {
        int(key): int(value)
        for key, value in (state.get("probe_cases_by_batch") or {}).items()
    }
    batch_rows = []
    adaptation_batch_rows = []
    cumulative_utility = 0.0
    cumulative_regret = 0.0
    cumulative_correct = 0
    cumulative_cases = 0
    cumulative_probes = 0
    for index, event in enumerate(submit_events):
        items = list(event.get("results") or [])
        cumulative_probes += cases.get(index, 0)
        cumulative_cases += len(items)
        cumulative_correct += sum(bool(item.get("correct")) for item in items)
        utility = sum(float(item.get("utility") or 0.0) for item in items)
        regret = sum(float(item.get("regret") or 0.0) for item in items)
        cumulative_utility += utility
        cumulative_regret += regret
        oracle_actions = Counter(
            str((item.get("oracle") or {}).get("optimal_action_id"))
            for item in items
            if (item.get("oracle") or {}).get("optimal_action_id") is not None
        )
        stratum_counts = Counter(str(item.get("stratum")) for item in items)
        flip_count = sum(bool(item.get("rule_flip")) for item in items)
        behavior = _dynamic_behavior(items)
        diagnostics = _batch_diagnostics(items)
        stratum_metrics = {
            stratum: _dynamic_behavior(
                [item for item in items if item.get("stratum") == stratum]
            )
            for stratum in ("iid", "boundary", "family_diagnostic")
        }
        adaptation_batch = _adaptation_batch_metrics(items)
        if int(event.get("batch_number") or 0) > change_after_batch:
            adaptation_batch_rows.append({
                "batch_number": event.get("batch_number"),
                **adaptation_batch,
            })
        batch_rows.append(
            {
                "batch_id": event.get("batch_id"),
                "batch_number": event.get("batch_number"),
                "phase": (
                    items[0].get("phase")
                    if items
                    else (
                        "pre_dynamic"
                        if int(event.get("batch_number") or 0) <= change_after_batch
                        else "post_dynamic"
                    )
                ),
                "case_count": len(items),
                "abstained_count": sum(
                    bool(item.get("abstained")) for item in items
                ),
                "flip_count": flip_count,
                "flip_rate": flip_count / len(items) if items else None,
                "oracle_action_distribution": dict(sorted(oracle_actions.items())),
                **diagnostics,
                "stratum_counts": dict(sorted(stratum_counts.items())),
                **behavior,
                "stratum_metrics": stratum_metrics,
                "accuracy": (
                    mean(1.0 if item.get("correct") else 0.0 for item in items)
                    if items
                    else None
                ),
                "mean_utility": _rounded_mean(items, "utility"),
                "cumulative_utility": round(cumulative_utility, 2),
                "action_utility": round(cumulative_utility, 2),
                "probe_count": cumulative_probes,
                "mean_regret": _rounded_mean(items, "regret"),
                "cumulative_regret": round(cumulative_regret, 2),
                "cumulative_accuracy": (
                    cumulative_correct / cumulative_cases
                    if cumulative_cases
                    else None
                ),
                "probe_batches_after_preceding_batch": rounds.get(index, 0),
                "probe_cases_after_preceding_batch": cases.get(index, 0),
                "cumulative_probe_cases_before_batch": cumulative_probes,
                **adaptation_batch,
            }
        )
    total_probe_rounds = sum(rounds.values())  # probe batch calls
    total_probe_cases = sum(cases.values())  # probe cases used
    unique_probe_count: int | None = None
    reprobes_after_change: int | None = None
    if episode is not None:
        phase_sets = probe_phase_point_sets(artifact, episode)
        unique_probe_count = len(phase_sets["pre"]) + len(phase_sets["post"])
        reprobes_after_change = (
            len(phase_sets["post"] & phase_sets["pre"])
            if phase_sets["has_change"]
            else None
        )
    total = len(results)
    stratum_accuracy = {
        f"{stratum}_accuracy": (
            mean(
                1.0 if item.get("correct") else 0.0
                for item in results
                if item.get("stratum") == stratum
            )
            if any(item.get("stratum") == stratum for item in results)
            else None
        )
        for stratum in ("iid", "boundary", "family_diagnostic")
    }
    pre_rows = [
        item for item in results
        if int(item.get("batch_number") or 0) <= change_after_batch
    ]
    post_rows = [
        item for item in results
        if int(item.get("batch_number") or 0) > change_after_batch
    ]
    pre_performance = _performance(pre_rows)
    post_performance = _performance(post_rows)

    def abstention_rate(rows: list[dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return sum(bool(item.get("abstained")) for item in rows) / len(rows)

    abstention = {
        "abstained_count": sum(bool(item.get("abstained")) for item in results),
        "abstention_rate": abstention_rate(results),
        "pre_dynamic_abstention_rate": abstention_rate(pre_rows),
        "post_dynamic_abstention_rate": abstention_rate(post_rows),
        "mean_utility_answered": _rounded_mean(answered, "utility"),
    }
    adaptation_by_stratum = {}
    for stratum in ("iid", "boundary", "family_diagnostic"):
        stratum_pre = [
            row for row in pre_rows if row.get("stratum") == stratum
        ]
        stratum_post = [
            row for row in post_rows if row.get("stratum") == stratum
        ]
        adaptation_by_stratum[stratum] = {
            "pre_dynamic": _performance(stratum_pre),
            "post_dynamic": _performance(stratum_post),
            "post_behavior": _dynamic_behavior(stratum_post),
            "stale_rule_baseline": _stale_performance(stratum_post),
        }
    population = (
        ((artifact.get("evaluation") or {}).get("transition") or {}).get(
            "population_reference", {}
        )
    )
    flip_by_phase_stratum: dict[str, dict[str, dict[str, float | int]]] = {}
    for phase in ("pre_dynamic", "post_dynamic"):
        phase_rows = [item for item in results if item.get("phase") == phase]
        flip_by_phase_stratum[phase] = {}
        for stratum in ("iid", "boundary", "family_diagnostic"):
            stratum_rows = [
                item for item in phase_rows if item.get("stratum") == stratum
            ]
            flip_by_phase_stratum[phase][stratum] = {
                "case_count": len(stratum_rows),
                "flip_count": sum(bool(item.get("rule_flip")) for item in stratum_rows),
                "flip_rate": (
                    sum(bool(item.get("rule_flip")) for item in stratum_rows)
                    / len(stratum_rows)
                    if stratum_rows
                    else None
                ),
            }
    post_behavior = _dynamic_behavior(post_rows)
    token_metrics = _token_metrics(list(artifact.get("tool_trace") or []))
    cold_start_rows = list(submit_events[0].get("results") or []) if submit_events else []
    learned_pre_rows = [
        item
        for event in submit_events[1:change_after_batch]
        for item in (event.get("results") or [])
    ]
    last_pre_change_rows = (
        list(submit_events[change_after_batch - 1].get("results") or [])
        if len(submit_events) >= change_after_batch
        else []
    )
    first_post_change_rows = (
        list(submit_events[change_after_batch].get("results") or [])
        if len(submit_events) > change_after_batch
        else []
    )
    later_post_rows = [
        item
        for event in submit_events[change_after_batch + 1 :]
        for item in (event.get("results") or [])
    ]
    cold_start = _performance(cold_start_rows)
    learned_pre = _performance(learned_pre_rows)
    last_pre_change = _performance(last_pre_change_rows)
    first_post_change = _performance(first_post_change_rows)
    later_post = _performance(later_post_rows)

    def delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
        if left.get(key) is None or right.get(key) is None:
            return None
        return round(float(left[key]) - float(right[key]), 6)

    return {
        "episode_id": artifact.get("episode_id"),
        "setting_id": artifact.get("setting_id"),
        "sampling_branch": artifact.get("sampling_branch"),
        "history_policy": artifact.get("history_policy", "eval_case_given"),
        "condition": artifact.get("condition", "dynamic"),
        "run_success": bool(artifact.get("run_success")),
        "domain": (artifact.get("provenance") or {}).get("domain"),
        "rule_family": (artifact.get("evaluation") or {}).get("rule_family"),
        "dynamic_type": (
            ((artifact.get("evaluation") or {}).get("transition") or {}).get("type")
        ),
        "after_action_batch": change_after_batch,
        "action_case_count": total,
        "completed_count": len(answered),
        "completion_rate": len(answered) / total if total else 0.0,
        "action_accuracy": (
            mean(1.0 if item.get("correct") else 0.0 for item in results)
            if results
            else None
        ),
        "partial_credit_accuracy": partial_credit_score,
        "partial_credit_random_baseline": partial_credit_baseline,
        "calibrated_accuracy": calibrated_accuracy,
        "dynamic_overall_accuracy": (
            mean(1.0 if item.get("correct") else 0.0 for item in results)
            if artifact.get("condition") == "dynamic" and results
            else None
        ),
        "static_overall_accuracy": (
            mean(1.0 if item.get("correct") else 0.0 for item in results)
            if artifact.get("condition") == "static" and results
            else None
        ),
        "dynamic_post_rule_accuracy": (
            post_performance["accuracy"]
            if artifact.get("condition") == "dynamic"
            else None
        ),
        "mean_utility": _rounded_mean(results, "utility"),
        "cumulative_utility": round(
            sum(float(item.get("utility") or 0.0) for item in results), 2
        ),
        "action_utility": round(
            sum(float(item.get("utility") or 0.0) for item in results), 2
        ),
        "probe_count": total_probe_cases,
        "unique_probe_count": unique_probe_count,
        "reprobes_after_change": reprobes_after_change,
        "abstention": abstention,
        "abstention_rate": abstention["abstention_rate"],
        "mean_regret": _rounded_mean(results, "regret"),
        "cumulative_regret": round(
            sum(float(item.get("regret") or 0.0) for item in results), 2
        ),
        "action_batches": len(submit_events),
        "batch_metrics": batch_rows,
        "learning_diagnostics": {
            "cold_start_batch_1": {
                **cold_start,
                **_batch_diagnostics(cold_start_rows),
                "preceding_probe_batches": rounds.get(0, 0),
                "preceding_probe_cases": cases.get(0, 0),
            },
            "learned_pre_change_batches": {
                **learned_pre,
                "start_batch": 2,
                "end_batch": change_after_batch,
                "accuracy_delta_from_cold_start": delta(
                    learned_pre, cold_start, "accuracy"
                ),
                "mean_utility_delta_from_cold_start": delta(
                    learned_pre, cold_start, "mean_utility"
                ),
                "mean_regret_delta_from_cold_start": delta(
                    learned_pre, cold_start, "mean_regret"
                ),
            },
            "dynamic_transition": {
                "last_pre_change_batch_number": change_after_batch,
                "first_post_change_batch_number": change_after_batch + 1,
                "last_pre_change_batch": last_pre_change,
                "first_post_change_batch": first_post_change,
                "first_post_change_accuracy_delta_from_last_pre_change": delta(
                    first_post_change, last_pre_change, "accuracy"
                ),
                "first_post_change_mean_utility_delta_from_last_pre_change": delta(
                    first_post_change, last_pre_change, "mean_utility"
                ),
                "later_post_change_batches": later_post,
                "later_post_change_accuracy_delta_from_first_post_change": delta(
                    later_post, first_post_change, "accuracy"
                ),
                "later_post_change_mean_utility_delta_from_first_post_change": delta(
                    later_post, first_post_change, "mean_utility"
                ),
            },
        },
        "pre_dynamic": pre_performance,
        "post_dynamic": post_performance,
        "adaptation": {
            "accuracy_delta": (
                post_performance["accuracy"] - pre_performance["accuracy"]
                if post_performance["accuracy"] is not None
                and pre_performance["accuracy"] is not None
                else None
            ),
            "utility_delta": (
                post_performance["mean_utility"] - pre_performance["mean_utility"]
                if post_performance["mean_utility"] is not None
                and pre_performance["mean_utility"] is not None
                else None
            ),
            "post_dynamic_batch_curve": [
                {
                    "batch_number": row["batch_number"],
                    "accuracy": row["accuracy"],
                    "mean_utility": row["mean_utility"],
                    "mean_regret": row["mean_regret"],
                }
                for row in batch_rows
                if row["phase"] == "post_dynamic"
            ],
            "adaptation_curve": adaptation_batch_rows,
            "by_stratum": adaptation_by_stratum,
        },
        "dynamic_change": {
            "dynamic_type": (
                ((artifact.get("evaluation") or {}).get("transition") or {}).get(
                    "type"
                )
            ),
            "population_flip_rate": population.get("flip_rate"),
            "population_action_transition_matrix": population.get(
                "action_transition_matrix"
            ),
            "population_mean_score_shift": population.get("mean_score_shift"),
            "population_mean_absolute_score_shift": population.get(
                "mean_absolute_score_shift"
            ),
            "action_case_realized_flip_rate": (
                mean(1.0 if item.get("rule_flip") else 0.0 for item in results)
                if results
                else None
            ),
            "batch_flip_counts": [row["flip_count"] for row in batch_rows],
            "batch_flip_rates": [row["flip_rate"] for row in batch_rows],
            "flip_by_phase_stratum": flip_by_phase_stratum,
            "post_flipped_case_behavior": {
                **post_behavior["flipped_case_breakdown"],
            },
        },
        "stale_rule_baseline": _stale_performance(post_rows),
        "probe_metrics": {
            "batches": total_probe_rounds,
            "cases": total_probe_cases,
            "invalid_calls": int(state.get("invalid_probe_calls") or 0),
            "by_batch": [
                {
                    "batch_number": index + 1,
                    "probe_batches": rounds.get(index, 0),
                    "probe_cases": cases.get(index, 0),
                }
                for index in range(len(batch_rows))
            ],
            "cumulative_cases_before_each_action_batch": [
                row["cumulative_probe_cases_before_batch"] for row in batch_rows
            ],
            "post_change_adaptation": _probe_adaptation_metrics(artifact),
        },
        "interactions_used": int(state.get("interactions_used") or 0),
        "invalid_calls": len(state.get("invalid_calls") or []),
        "calculator_metrics": {
            "calls": int(state.get("calculator_calls") or 0),
            "expressions": int(state.get("calculator_expressions") or 0),
            "invalid_expressions": int(
                state.get("calculator_expression_errors") or 0
            ),
        },
        "token_metrics": token_metrics,
        **stratum_accuracy,
    }


def aggregate_metrics(
    artifacts: list[dict[str, Any]],
    episodes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    episode_by_id = episodes or {}
    rows = [
        episode_metrics(
            artifact,
            episode=episode_by_id.get(str(artifact.get("episode_id") or "")),
        )
        for artifact in artifacts
    ]
    return aggregate_episode_metrics(rows)


def aggregate_episode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate precomputed episode metrics, preserving row order and weights."""

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        keys = (
            "completion_rate",
            "action_accuracy",
            "partial_credit_accuracy",
            "partial_credit_random_baseline",
            "calibrated_accuracy",
            "dynamic_overall_accuracy",
            "static_overall_accuracy",
            "dynamic_post_rule_accuracy",
            "mean_utility",
            "cumulative_utility",
            "action_utility",
            "abstention_rate",
            "mean_regret",
            "cumulative_regret",
            "action_batches",
            "interactions_used",
            "invalid_calls",
            "iid_accuracy",
            "boundary_accuracy",
            "family_diagnostic_accuracy",
        )
        return {
            "episode_count": len(group),
            **{
                key: (
                    round(
                        mean(
                            float(row[key])
                            for row in group
                            if row.get(key) is not None
                        ),
                        2
                        if key
                        in {
                            "mean_utility",
                            "cumulative_utility",
                            "action_utility",
                            "mean_regret",
                            "cumulative_regret",
                        }
                        else 6,
                    )
                    if any(row.get(key) is not None for row in group)
                    else None
                )
                for key in keys
            },
            "mean_unique_probes": (
                round(
                    mean(
                        float(row["unique_probe_count"])
                        for row in group
                        if row.get("unique_probe_count") is not None
                    ),
                    6,
                )
                if any(row.get("unique_probe_count") is not None for row in group)
                else None
            ),
            "mean_reprobes_after_change": (
                round(
                    mean(
                        float(row["reprobes_after_change"])
                        for row in group
                        if row.get("reprobes_after_change") is not None
                    ),
                    6,
                )
                if any(row.get("reprobes_after_change") is not None for row in group)
                else None
            ),
            "probe_batches": (
                mean(row["probe_metrics"]["batches"] for row in group)
                if group
                else None
            ),
            "probe_cases": (
                mean(row["probe_metrics"]["cases"] for row in group)
                if group
                else None
            ),
            "calculator_metrics": {
                key: (
                    mean(row["calculator_metrics"][key] for row in group)
                    if group
                    else None
                )
                for key in ("calls", "expressions", "invalid_expressions")
            },
            "calculator_totals": {
                key: sum(row["calculator_metrics"][key] for row in group)
                for key in ("calls", "expressions", "invalid_expressions")
            },
            "token_metrics": {
                key: (
                    round(mean(float(row["token_metrics"][key]) for row in group), 2)
                    if group
                    else 0
                )
                for key in (
                    "prompt_tokens",
                    "cached_prompt_tokens",
                    "uncached_prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                    "request_count",
                    "usage_reported_request_count",
                    "missing_usage_request_count",
                    "max_request_prompt_tokens",
                    "max_request_prompt_characters",
                    "total_request_prompt_characters",
                    "max_provider_wire_request_characters",
                    "total_provider_wire_request_characters",
                    "max_canonical_visible_history_characters",
                    "max_history_event_count",
                    "provider_continuation_response_count",
                    "reasoning_signature_response_count",
                    "invalid_response_count",
                )
            }
            | {
                "latency_seconds": {
                    key: (
                        round(
                            mean(
                                float(row["token_metrics"]["latency_seconds"][key])
                                for row in group
                                if row["token_metrics"]["latency_seconds"][key]
                                is not None
                            ),
                            6,
                        )
                        if any(
                            row["token_metrics"]["latency_seconds"][key] is not None
                            for row in group
                        )
                        else None
                    )
                    for key in ("total", "mean", "maximum")
                }
            },
            "token_totals": {
                key: sum(float(row["token_metrics"][key]) for row in group)
                for key in (
                    "prompt_tokens",
                    "cached_prompt_tokens",
                    "uncached_prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                    "request_count",
                    "usage_reported_request_count",
                    "missing_usage_request_count",
                )
            },
        }

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dynamic_types: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[str(row["rule_family"])].append(row)
        domains[str(row["domain"])].append(row)
        dynamic_types[str(row.get("dynamic_type") or "field_replacement")].append(row)

    def nested_mean(path: tuple[str, ...]) -> float | None:
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        return round(mean(values), 6) if values else None

    return {
        "episode_count": len(rows),
        "overall": summarize(rows),
        "dynamic": {
            "pre_dynamic_accuracy": nested_mean(("pre_dynamic", "accuracy")),
            "post_dynamic_accuracy": nested_mean(("post_dynamic", "accuracy")),
            "pre_dynamic_mean_utility": nested_mean(
                ("pre_dynamic", "mean_utility")
            ),
            "post_dynamic_mean_utility": nested_mean(
                ("post_dynamic", "mean_utility")
            ),
            "adaptation_accuracy_delta": nested_mean(
                ("adaptation", "accuracy_delta")
            ),
            "population_flip_rate": nested_mean(
                ("dynamic_change", "population_flip_rate")
            ),
            "action_case_realized_flip_rate": nested_mean(
                ("dynamic_change", "action_case_realized_flip_rate")
            ),
            "stale_policy_accuracy": nested_mean(
                ("stale_rule_baseline", "accuracy")
            ),
            "stale_policy_mean_utility": nested_mean(
                ("stale_rule_baseline", "mean_utility")
            ),
        },
        "learning_diagnostics": {
            "cold_start_batch_1_accuracy": nested_mean(
                ("learning_diagnostics", "cold_start_batch_1", "accuracy")
            ),
            "cold_start_batch_1_mean_utility": nested_mean(
                ("learning_diagnostics", "cold_start_batch_1", "mean_utility")
            ),
            "cold_start_batch_1_mean_regret": nested_mean(
                ("learning_diagnostics", "cold_start_batch_1", "mean_regret")
            ),
            "cold_start_best_constant_accuracy": nested_mean(
                (
                    "learning_diagnostics",
                    "cold_start_batch_1",
                    "best_constant_accuracy_policy",
                    "accuracy",
                )
            ),
            "learned_pre_change_accuracy": nested_mean(
                (
                    "learning_diagnostics",
                    "learned_pre_change_batches",
                    "accuracy",
                )
            ),
            "learned_pre_change_accuracy_delta_from_cold_start": nested_mean(
                (
                    "learning_diagnostics",
                    "learned_pre_change_batches",
                    "accuracy_delta_from_cold_start",
                )
            ),
            "learned_pre_change_mean_utility": nested_mean(
                (
                    "learning_diagnostics",
                    "learned_pre_change_batches",
                    "mean_utility",
                )
            ),
            "learned_pre_change_mean_utility_delta_from_cold_start": nested_mean(
                (
                    "learning_diagnostics",
                    "learned_pre_change_batches",
                    "mean_utility_delta_from_cold_start",
                )
            ),
            "learned_pre_change_mean_regret_delta_from_cold_start": nested_mean(
                (
                    "learning_diagnostics",
                    "learned_pre_change_batches",
                    "mean_regret_delta_from_cold_start",
                )
            ),
            "first_post_change_accuracy_delta_from_last_pre_change": nested_mean(
                (
                    "learning_diagnostics",
                    "dynamic_transition",
                    "first_post_change_accuracy_delta_from_last_pre_change",
                )
            ),
            "later_post_change_accuracy_delta_from_first_post_change": nested_mean(
                (
                    "learning_diagnostics",
                    "dynamic_transition",
                    "later_post_change_accuracy_delta_from_first_post_change",
                )
            ),
            "first_post_change_mean_utility_delta_from_last_pre_change": nested_mean(
                (
                    "learning_diagnostics",
                    "dynamic_transition",
                    "first_post_change_mean_utility_delta_from_last_pre_change",
                )
            ),
            "later_post_change_mean_utility_delta_from_first_post_change": nested_mean(
                (
                    "learning_diagnostics",
                    "dynamic_transition",
                    "later_post_change_mean_utility_delta_from_first_post_change",
                )
            ),
        },
        "by_family": {
            key: summarize(value) for key, value in sorted(families.items())
        },
        "by_domain": {
            key: summarize(value) for key, value in sorted(domains.items())
        },
        "by_dynamic_type": {
            key: summarize(value)
            for key, value in sorted(dynamic_types.items())
        },
        "episodes": rows,
    }


def aggregate_branched_metrics(
    artifacts: list[dict[str, Any]],
    episodes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate arbitrary post branches and exact-case no-change controls."""

    names = sorted({
        name
        for artifact in artifacts
        for name in (artifact.get("branches") or {})
    })
    by_branch = {
        name: aggregate_metrics([
            artifact["branches"][name]
            for artifact in artifacts
            if name in (artifact.get("branches") or {})
        ], episodes=episodes)
        for name in names
    }
    static_artifacts = [
        (artifact.get("branches") or {}).get("static_iid")
        for artifact in artifacts
        if (artifact.get("branches") or {}).get("static_iid") is not None
    ]
    first_learn_by_branch = {
        name: first_learn_rule_comparison(
            [
                (artifact.get("branches") or {}).get(name)
                for artifact in artifacts
                if (artifact.get("branches") or {}).get(name) is not None
                and (artifact.get("branches") or {}).get(name, {}).get("condition")
                != "static"
            ],
            static_artifacts,
        )
        for name in names
        if name != "static_iid" and not name.endswith("_no_change_control")
    }
    for name, comparison in first_learn_by_branch.items():
        # Keep the comparison alongside the normal per-branch aggregate so
        # metrics.json consumers do not need to reconstruct branch pairing.
        by_branch[name]["first_learn_rule_accuracy"] = comparison
        by_branch[name]["overall"].update({
            "first_learn_accuracy": comparison["dynamic_post_rule_accuracy"],
            "first_learn_rule_accuracy": comparison["static_first_learn_accuracy"],
            "first_learn_static_accuracy": comparison["static_first_learn_accuracy"],
            "first_learn_accuracy_delta": comparison["accuracy_delta"],
        })
    comparisons = []
    for artifact in artifacts:
        branches = artifact.get("branches") or {}
        for control_name, control in branches.items():
            suffix = "_no_change_control"
            if not control_name.endswith(suffix):
                continue
            dynamic_name = control_name[: -len(suffix)]
            dynamic = branches.get(dynamic_name)
            if not dynamic:
                continue
            dynamic_rows = {
                str(row["context_id"]): row
                for row in dynamic.get("action_results") or []
            }
            control_rows = {
                str(row["context_id"]): row
                for row in control.get("action_results") or []
            }
            post_ids = [
                context_id
                for context_id in sorted(set(dynamic_rows) & set(control_rows))
                if int(dynamic_rows[context_id].get("batch_number") or 0)
                > int(artifact.get("fork_after_batch") or 0)
            ]
            comparisons.append({
                "episode_id": artifact.get("episode_id"),
                "dynamic_branch": dynamic_name,
                "control_branch": control_name,
                "post_case_count": len(post_ids),
                "dynamic_minus_control_accuracy": (
                    mean(float(bool(dynamic_rows[key].get("correct"))) for key in post_ids)
                    - mean(float(bool(control_rows[key].get("correct"))) for key in post_ids)
                    if post_ids else None
                ),
                "dynamic_minus_control_utility": (
                    mean(float(dynamic_rows[key].get("utility") or 0.0) for key in post_ids)
                    - mean(float(control_rows[key].get("utility") or 0.0) for key in post_ids)
                    if post_ids else None
                ),
            })
    return {
        "metrics_schema_version": "adaexplorebench_branched_metrics_v4",
        "episode_count": len(artifacts),
        "branch_names": names,
        "by_branch": by_branch,
        "leaderboard_setting_id": "flip_50",
        "leaderboard": by_branch.get("flip_50"),
        "first_learn_rule_accuracy": first_learn_by_branch,
        # Descriptive alias retained alongside the compact field name for
        # consumers that treat this as a comparison object rather than a
        # scalar accuracy metric.
        "first_learn_comparison": first_learn_by_branch,
        "matched_no_change_comparisons": comparisons,
    }


def static_episode_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    """Summarize a genuinely static dataset episode without adaptation fields."""

    rows = list(artifact.get("action_results") or [])
    state = dict(artifact.get("private_state") or {})
    probe_cases_by_batch = {
        int(key): int(value)
        for key, value in (state.get("probe_cases_by_batch") or {}).items()
    }
    performance = _performance(rows)
    partial_credit_score, partial_credit_baseline = partial_credit(rows)
    calibrated_accuracy = _calibrated_accuracy(partial_credit_score, partial_credit_baseline)
    batches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        batch_number = row.get("batch_number")
        if isinstance(batch_number, int):
            batches[batch_number].append(row)
    return {
        "episode_id": artifact.get("episode_id"),
        "condition": "static",
        "domain": (artifact.get("provenance") or {}).get("domain"),
        "rule_family": (artifact.get("evaluation") or {}).get("rule_family"),
        "run_success": artifact.get("run_success"),
        "performance": performance,
        "static_overall_accuracy": performance["accuracy"],
        "partial_credit_accuracy": partial_credit_score,
        "partial_credit_random_baseline": partial_credit_baseline,
        "calibrated_accuracy": calibrated_accuracy,
        "action_utility": round(float(performance["cumulative_utility"]), 2),
        "probe_count": sum(probe_cases_by_batch.values()),
        "abstained_count": sum(bool(row.get("abstained")) for row in rows),
        "batch_curve": [
            {"batch_number": number, **_performance(batch_rows)}
            for number, batch_rows in sorted(batches.items())
        ],
        "probe_round_count": sum(
            1
            for row in artifact.get("private_history") or []
            if row.get("type") == "probe_batch"
        ),
        "interaction_count": int(
            (artifact.get("private_state") or {}).get("interactions_used") or 0
        ),
    }


def aggregate_static_metrics(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate static runs independently of dynamic adaptation metrics."""

    episodes = [static_episode_metrics(item) for item in artifacts]
    completed = [item for item in episodes if item.get("run_success") is True]

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        performances = [item["performance"] for item in rows]
        return {
            "episode_count": len(rows),
            "mean_accuracy": (
                mean(float(item["accuracy"]) for item in performances)
                if performances
                else None
            ),
            "static_overall_accuracy": (
                mean(float(item["static_overall_accuracy"]) for item in rows)
                if rows
                else None
            ),
            "partial_credit_accuracy": (
                mean(
                    float(item["partial_credit_accuracy"])
                    for item in rows
                    if item.get("partial_credit_accuracy") is not None
                )
                if any(item.get("partial_credit_accuracy") is not None for item in rows)
                else None
            ),
            "partial_credit_random_baseline": (
                mean(
                    float(item["partial_credit_random_baseline"])
                    for item in rows
                    if item.get("partial_credit_random_baseline") is not None
                )
                if any(
                    item.get("partial_credit_random_baseline") is not None
                    for item in rows
                )
                else None
            ),
            "calibrated_accuracy": (
                mean(
                    float(item["calibrated_accuracy"])
                    for item in rows
                    if item.get("calibrated_accuracy") is not None
                )
                if any(item.get("calibrated_accuracy") is not None for item in rows)
                else None
            ),
            "mean_utility": (
                mean(float(item["mean_utility"]) for item in performances)
                if performances
                else None
            ),
            "mean_regret": (
                mean(float(item["mean_regret"]) for item in performances)
                if performances
                else None
            ),
            "mean_action_utility": (
                mean(float(item["action_utility"]) for item in rows)
                if rows
                else None
            ),
        }

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in completed:
        by_family[str(item.get("rule_family"))].append(item)
        by_domain[str(item.get("domain"))].append(item)
    return {
        "metrics_schema_version": "adaexplorebench_static_metrics_v3",
        "dataset_mode": "static",
        "summary": summary(completed),
        "failed_episode_count": len(episodes) - len(completed),
        "by_rule_family": {
            key: summary(value) for key, value in sorted(by_family.items())
        },
        "by_domain": {
            key: summary(value) for key, value in sorted(by_domain.items())
        },
        "episodes": episodes,
    }
