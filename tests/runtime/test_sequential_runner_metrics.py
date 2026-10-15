from __future__ import annotations

import pytest

from runtime.runner import (
    run_baseline_episode,
    run_branched_baseline_episode,
)
from runtime.metrics import (
    _calibrated_accuracy,
    _token_metrics,
    aggregate_branched_metrics,
    episode_metrics,
    first_learn_rule_comparison,
    partial_credit,
)


def test_token_metrics_reads_nested_openai_responses_usage_details():
    metrics = _token_metrics(
        [
            {
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 85},
                    "output_tokens": 40,
                    "output_tokens_details": {"reasoning_tokens": 30},
                    "total_tokens": 140,
                }
            }
        ]
    )

    assert metrics["prompt_tokens"] == 100
    assert metrics["cached_prompt_tokens"] == 85
    assert metrics["uncached_prompt_tokens"] == 15
    assert metrics["completion_tokens"] == 40
    assert metrics["reasoning_tokens"] == 30
    assert metrics["total_tokens"] == 140


def _branch(
    sampling: str,
    rule_change: bool,
    history_policy: str = "eval_case_hidden",
) -> dict[str, object]:
    return {
        "sampling": sampling,
        "rule_change": rule_change,
        "history_policy": history_policy,
    }


def test_first_learn_rule_comparison_matches_static_horizon_to_post_change_batches():
    # Three cases per batch make the expected values easy to inspect.  The
    # dynamic episode has a 2-batch post-change window, so static uses batches
    # 1..2 rather than its final two batches.
    dynamic = {
        "episode_id": "episode-1",
        "evaluation": {"transition": {"after_action_batch": 3}},
        "action_results": [
            {"batch_number": 4, "correct": True},
            {"batch_number": 4, "correct": False},
            {"batch_number": 5, "correct": False},
            {"batch_number": 5, "correct": False},
        ],
    }
    static = {
        "episode_id": "episode-1",
        "action_results": [
            {"batch_number": 1, "correct": True},
            {"batch_number": 1, "correct": True},
            {"batch_number": 2, "correct": True},
            {"batch_number": 2, "correct": False},
            {"batch_number": 3, "correct": False},
        ],
    }
    comparison = first_learn_rule_comparison([dynamic], [static])
    assert comparison["episode_count"] == 1
    assert comparison["dynamic_post_rule_accuracy"] == 0.25
    assert comparison["static_first_learn_accuracy"] == 0.75
    assert comparison["accuracy_delta"] == -0.5
    assert comparison["comparisons"][0]["comparison_batch_count"] == 2


def test_oracle_baseline_runs_on_new_dynamic_protocol(
    dynamic_episode, runtime_interaction_policy
):
    artifact = run_baseline_episode(
        dynamic_episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        seed=9,
        condition="dynamic",
        sampling_branch="flip_50",
    )
    assert artifact["run_success"] is True
    assert len(artifact["action_results"]) == 96
    assert artifact["evaluation"]["transition"]["type"] == "field_replacement"
    metrics = episode_metrics(artifact)
    assert metrics["probe_metrics"]["cases"] == 12
    assert metrics["probe_count"] == 12
    assert metrics["action_utility"] == metrics["cumulative_utility"]
    assert "probe_spend_utility" not in metrics
    assert "net_utility" not in metrics
    assert metrics["abstention_rate"] == 0.0


def test_integrated_episode_runs_no_change_control_as_a_branch(
    dynamic_episode, runtime_interaction_policy
):
    artifact = run_branched_baseline_episode(
        dynamic_episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        branches={
            "static_iid": _branch("static_iid", False),
            "flip_25": _branch("flip_25", True),
            "flip_50": _branch("flip_50", True),
            "flip_50_no_change_control": _branch("flip_50", False),
            "flip_75": _branch("flip_75", True),
        },
        seed=11,
    )
    assert list(artifact["branches"]) == [
        "static_iid",
        "flip_25",
        "flip_50",
        "flip_50_no_change_control",
        "flip_75",
    ]
    dynamic_post = {
        row["context_id"]
        for row in artifact["branches"]["flip_50"]["action_results"]
        if row["batch_number"] > artifact["fork_after_batch"]
    }
    control_post = {
        row["context_id"]
        for row in artifact["branches"]["flip_50_no_change_control"]["action_results"]
        if row["batch_number"] > artifact["fork_after_batch"]
    }
    assert dynamic_post == control_post
    metrics = aggregate_branched_metrics([artifact])
    assert metrics["branch_names"] == [
        "flip_25",
        "flip_50",
        "flip_50_no_change_control",
        "flip_75",
        "static_iid",
    ]
    static_overall = metrics["by_branch"]["static_iid"]["overall"]
    assert static_overall["action_utility"] == static_overall["cumulative_utility"]
    assert "probe_per_task" not in static_overall
    assert static_overall["mean_unique_probes"] is None
    assert len(metrics["matched_no_change_comparisons"]) == 1
    first_learn = metrics["first_learn_rule_accuracy"]["flip_50"]
    assert first_learn["episode_count"] == 1
    assert first_learn["static_first_learn_accuracy"] == 1.0
    assert first_learn["dynamic_post_rule_accuracy"] == 1.0
    assert first_learn["accuracy_delta"] == 0.0
    assert metrics["by_branch"]["flip_50"]["overall"]["first_learn_rule_accuracy"] == 1.0


def test_integrated_episode_supports_branch_specific_history_policy(
    dynamic_episode, runtime_interaction_policy
):
    artifact = run_branched_baseline_episode(
        dynamic_episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        branches={
            "static_iid": _branch("static_iid", False),
            "flip_50": _branch("flip_50", True),
            "flip_50_eval_case_given": _branch(
                "flip_50", True, "eval_case_given"
            ),
        },
    )

    assert list(artifact["branches"]) == [
        "static_iid",
        "flip_50",
        "flip_50_eval_case_given",
    ]
    assert artifact["branches"]["flip_50"]["history_policy"] == (
        "eval_case_hidden"
    )
    given = artifact["branches"]["flip_50_eval_case_given"]
    assert given["history_policy"] == "eval_case_given"
    assert given["sampling_branch"] == "flip_50"
    assert given["setting_id"] == "flip_50_eval_case_given"


def test_random_baseline_samples_all_four_actions(
    dynamic_episode, runtime_interaction_policy
):
    artifact = run_baseline_episode(
        dynamic_episode,
        "random",
        interaction_policy=runtime_interaction_policy,
        seed=5,
        condition="dynamic",
        sampling_branch="flip_50",
    )
    assert artifact["run_success"] is True
    submitted = {row["action_id"] for row in artifact["action_results"]}
    assert submitted == {"action_0", "action_1", "action_2", "action_3"}
    metrics = episode_metrics(artifact)
    assert all(
        row["uniform_random_policy"]["expected_accuracy"] == 0.25
        for row in metrics["batch_metrics"]
    )


def test_partial_credit_minmax_normalizes_each_case():
    rows = [
        {"utility": 0.9, "oracle": {"utilities": {"a0": -0.5, "a1": 0.9}}},
        {"utility": -0.5, "oracle": {"utilities": {"a0": -0.5, "a1": 0.9}}},
        {"utility": 0.2, "oracle": {"utilities": {"a0": -0.5, "a1": 0.9}}},
    ]
    score, baseline = partial_credit(rows)
    # Per-case scores: 1.0 (best), 0.0 (worst), (0.2 + 0.5) / 1.4 = 0.5.
    assert score == pytest.approx(0.5)
    # Every case's actions sit at normalized positions 0 and 1, so a uniform
    # random policy expects 0.5 on each.
    assert baseline == pytest.approx(0.5)


def test_partial_credit_edge_cases():
    tied = [{"utility": 0.3, "oracle": {"utilities": {"a0": 0.3, "a1": 0.3}}}]
    assert partial_credit(tied) == (1.0, 1.0)

    abstained = [{"utility": None, "oracle": {"utilities": {"a0": -1.0, "a1": 1.0}}}]
    score, baseline = partial_credit(abstained)
    assert score == pytest.approx(0.5)  # unanswered cases earn utility 0.0
    assert baseline == pytest.approx(0.5)

    skipped = [
        {"utility": 1.0},  # no oracle at all
        {"oracle": {"utilities": {"a0": 1.0}}},  # single-action vector
        {"utility": 1.0, "oracle": {"utilities": {"a0": 0.0, "a1": 1.0}}},
    ]
    score, _ = partial_credit(skipped)
    assert score == 1.0

    assert partial_credit([]) == (None, None)
    assert partial_credit([{"utility": 1.0}]) == (None, None)


def test_calibrated_accuracy_anchors_random_to_zero_and_oracle_to_one():
    assert _calibrated_accuracy(1.0, 0.5) == pytest.approx(1.0)
    assert _calibrated_accuracy(0.5, 0.5) == pytest.approx(0.0)
    assert _calibrated_accuracy(0.3, 0.5) == pytest.approx(-0.4)
    assert _calibrated_accuracy(None, 0.5) is None
    assert _calibrated_accuracy(0.5, None) is None
    assert _calibrated_accuracy(0.8, 1.0) is None  # degenerate baseline guard


def test_oracle_episode_scores_fullpartial_credit(
    dynamic_episode, runtime_interaction_policy
):
    artifact = run_baseline_episode(
        dynamic_episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        seed=9,
        condition="dynamic",
        sampling_branch="flip_50",
    )
    metrics = episode_metrics(artifact)
    assert metrics["partial_credit_accuracy"] == pytest.approx(1.0)
    assert metrics["calibrated_accuracy"] == pytest.approx(1.0)
    assert 0.0 < metrics["partial_credit_random_baseline"] < 1.0


def test_branched_aggregation_exposes_partial_credit_on_leaderboard(
    dynamic_episode, runtime_interaction_policy
):
    artifact = run_branched_baseline_episode(
        dynamic_episode,
        "oracle",
        interaction_policy=runtime_interaction_policy,
        branches={
            "static_iid": _branch("static_iid", False),
            "flip_50": _branch("flip_50", True),
        },
        seed=11,
    )
    metrics = aggregate_branched_metrics([artifact])
    flip_50 = metrics["by_branch"]["flip_50"]["overall"]
    assert flip_50["partial_credit_accuracy"] == pytest.approx(1.0)
    assert flip_50["calibrated_accuracy"] == pytest.approx(1.0)
    assert flip_50["partial_credit_random_baseline"] is not None
    assert flip_50["probe_cases"] is not None
    leaderboard = metrics["leaderboard"]["overall"]
    assert leaderboard["partial_credit_accuracy"] == pytest.approx(
        flip_50["partial_credit_accuracy"]
    )
    assert leaderboard["calibrated_accuracy"] == pytest.approx(
        flip_50["calibrated_accuracy"]
    )
    assert "probe_cases" in leaderboard
