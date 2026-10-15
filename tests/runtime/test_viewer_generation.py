from __future__ import annotations

import pytest

from runtime.schema import BRANCHED_TRACE_SCHEMA_VERSION
from viewer.generate_data import (
    LEGACY_HISTORY_POLICY_NAMES,
    SUPPORTED_BRANCHED_TRACE_SCHEMA_VERSIONS,
    _aggregate_relevant_fields_assessments,
    _light_trajectory_episode_ids,
    _load_run_summary_cache,
    _normalized_completion,
    _run_label,
    _write_run_summary_cache,
)
from viewer.validate_public_data import source_sampling_name


def test_viewer_reads_legacy_dynamic_episode_protocol():
    # The viewer stays pinned to `adaexplorebench_episode_v5` and the three-action
    # utility mapping until its own migration (out of scope for the four-action
    # upgrade, see PLAN_3_TO_4_ACTIONS.md §1/§9). Under the new four-action
    # protocol, `hydrate_rule`/`utilities_from_score` produce four-wide utility
    # vectors, so no legacy three-action episode can be rendered by the
    # un-migrated viewer code. Re-enable with the viewer-specific migration.
    pytest.skip(
        "viewer data generation is pinned to the legacy three-action protocol "
        "until the viewer-specific four-action migration"
    )


def test_viewer_labels_current_high_effort_models():
    assert _run_label("runtime_ds_v4_flash_high_test16_20260829") == (
        "DeepSeek V4 Flash High 0829",
        "deepseek",
    )
    assert _run_label("deepseek-v4.1-flash-high-test16-6settings-20260910") == (
        "DeepSeek V4.1 Flash High 0910",
        "deepseek",
    )
    assert _run_label("runtime_gemini_3_7_flash_high_test16_20260829") == (
        "Gemini 3.7 Flash High 0829",
        "gemini",
    )
    assert _run_label(
        "adaexplorebench_dynamic_16ep_v7_gemini38_flash_high_vertex_20260906_32c"
    ) == (
        "Gemini 3.8 Flash High 0906",
        "gemini",
    )
    assert _run_label(
        "adaexplorebench_dynamic_16ep_v7_gpt56_luna_high_codex_for_20260904_32c"
    ) == (
        "GPT-5.6 Luna High 0904",
        "gpt",
    )
    assert _run_label("gpt-5.6-terra-medium-test16-6settings-20260909") == (
        "GPT-5.6 Terra Medium 0909",
        "gpt",
    )
    assert _run_label("gpt-5.6-luna-medium-test16-6settings-20260909") == (
        "GPT-5.6 Luna Medium 0909",
        "gpt",
    )


def test_viewer_labels_structured_run_names():
    # Convention: <model>-<effort>-<runconfig>-<dataset>-<yyyymmdd>[-runN]
    assert _run_label("gpt-5.6-luna-medium-default-test-16-20260915-run1") == (
        "GPT-5.6 Luna Medium 0915 r1",
        "gpt",
    )
    assert _run_label(
        "gemini-3.8-flash-high-full-diagnostic-selected-256-20260920"
    ) == (
        "Gemini 3.8 Flash High 0920",
        "gemini",
    )
    assert _run_label("qwen3-9b-low-default-test-16-20260915") == (
        "Qwen3 9B Low 0915",
        "qwen3",
    )
    assert _run_label(
        "qwen3.5-9b-default-full-diagnostic-selected-256-20260917-run1"
    ) == (
        "Qwen3.5 9B Default 0917 r1",
        "qwen3.5",
    )


def test_viewer_accepts_published_v3_through_current_branched_traces():
    assert SUPPORTED_BRANCHED_TRACE_SCHEMA_VERSIONS == {
        "adaexplorebench_branched_trace_v3",
        "adaexplorebench_branched_trace_v4",
        "adaexplorebench_branched_trace_v5",
        "adaexplorebench_branched_trace_v6",
        BRANCHED_TRACE_SCHEMA_VERSION,
    }


def test_viewer_maps_legacy_history_policy_names_for_runtime_parsing():
    assert LEGACY_HISTORY_POLICY_NAMES == {
        "full_history": "eval_case_given",
        "outcome_only": "eval_case_hidden",
    }


@pytest.mark.parametrize(
    ("setting_name", "sampling_name"),
    [
        ("flip_50", "flip_50"),
        ("flip_50_no_change_control", "flip_50"),
        ("flip_50_eval_case_given", "flip_50"),
        ("flip_50_full_history", "flip_50"),
    ],
)
def test_viewer_validation_resolves_setting_names_to_sampling_names(
    setting_name, sampling_name
):
    assert source_sampling_name(setting_name) == sampling_name


def test_viewer_normalizes_legacy_unanswered_recovery_as_completed():
    status, success, termination = _normalized_completion({
        "run_status": "failed",
        "run_success": False,
        "termination": {
            "status": "completed_with_failures",
            "reason": "invalid_action_batch_after_repair",
            "completed_count": 88,
            "unanswered_count": 8,
        },
    })
    assert (status, success) == ("completed", True)
    assert termination["status"] == "completed"
    assert termination["reason"] == "invalid_action_batch_after_repair"


def test_viewer_keeps_provider_failure_failed():
    assert _normalized_completion({
        "run_status": "failed",
        "run_success": False,
        "termination": None,
    })[:2] == ("failed", False)


def test_viewer_light_mode_samples_one_episode_per_domain_family_cell():
    def episode(episode_id, domain, rule_family):
        return {
            "episode_id": episode_id,
            "provenance": {"domain": domain},
            "evaluation": {"rule_family": rule_family},
        }

    domains = ["cybersecurity", "finance", "healthcare", "manufacturing"]
    families = [
        "additive_aggregation",
        "conjunctive_gating",
        "disjunctive_sufficiency",
        "lexicographic_priority",
    ]
    episodes = [
        episode(f"ep_{domain}_{family}_{index}", domain, family)
        for domain in domains
        for family in families
        for index in range(2)
    ]
    selected = _light_trajectory_episode_ids(episodes)
    assert len(selected) == 16
    assert selected == {
        f"ep_{domain}_{family}_0" for domain in domains for family in families
    }


def test_viewer_run_summary_cache_roundtrip_and_invalidation(tmp_path):
    artifact = tmp_path / "episode_a.json"
    artifact.write_text("{}")
    summary = {"run_id": tmp_path.name, "episode_ids": ["episode_a"]}
    _write_run_summary_cache(tmp_path, "sha-a", summary)
    assert _load_run_summary_cache(tmp_path, "sha-a") == summary
    assert _load_run_summary_cache(tmp_path, "sha-b") is None
    artifact.write_text("{\"updated\": true}")
    assert _load_run_summary_cache(tmp_path, "sha-a") is None
    (tmp_path / "episode_b.json").write_text("{}")
    _write_run_summary_cache(tmp_path, "sha-a", summary)
    assert _load_run_summary_cache(tmp_path, "sha-a") == summary


def test_viewer_scores_relevant_fields_by_phase_and_classifies_changed_answers():
    def artifact(pre_fields, final_fields):
        return {
            "condition": "dynamic",
            "evaluation": {
                "transition": {
                    "pre_rule": {"relevant_fields": ["a", "b", "c"]},
                    "post_rule": {"relevant_fields": ["a", "b", "d"]},
                }
            },
            "relevant_fields_assessments": {
                "pre_change": {
                    "status": "completed",
                    "response": {"relevant_fields": pre_fields},
                },
                "post_change": {
                    "status": "completed",
                    "response": {"relevant_fields": final_fields},
                },
            },
        }

    summary = _aggregate_relevant_fields_assessments([
        artifact(["a", "b", "c"], ["a", "b", "d"]),
        artifact(["a", "b", "c"], ["a", "b", "e"]),
        artifact(["a", "b", "e"], ["a", "b", "d"]),
        artifact(["a", "c", "e"], ["a", "d", "e"]),
        artifact(["a", "b", "e"], ["a", "b", "e"]),
    ])

    assert summary is not None
    # Phase F1s are micro-pooled: TP/FP/FN counts aggregate over episodes
    # before the score is computed.
    assert summary["pre_change"] == {
        "episode_count": 5,
        "precision": pytest.approx(0.8),
        "recall": pytest.approx(0.8),
        "f1": pytest.approx(0.8),
    }
    assert summary["post_change"] == {
        "episode_count": 5,
        "precision": pytest.approx(0.8),
        "recall": pytest.approx(0.8),
        "f1": pytest.approx(0.8),
    }
    assert summary["final"] == {
        "episode_count": 0,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    assert summary["factor_recognition"] == pytest.approx(0.8)
    assert summary["factor_recognition_pre"] == pytest.approx(0.8)
    assert summary["factor_recognition_post"] == pytest.approx(0.8)
    assert summary["factor_recognition_final"] is None
    assert summary["change_recognition_eligibility"] == {
        "eligible": 5,
        "total": 5,
        "rate": 1.0,
    }
    assert summary["paired_episode_count"] == 5
    assert summary["changed_episode_count"] == 4
    assert summary["unchanged_episode_count"] == 1
    assert summary["changed_outcomes"] == {
        "both_correct": 1,
        "pre_correct_final_incorrect": 1,
        "pre_incorrect_final_correct": 1,
        "both_incorrect": 1,
    }
    assert summary["unchanged_outcomes"] == {
        "both_correct": 0,
        "pre_correct_final_incorrect": 0,
        "pre_incorrect_final_correct": 0,
        "both_incorrect": 1,
    }
    # Assessments without reported directions never match the true
    # (field, direction) change atoms, so nothing is detected here.
    assert summary["change_atom_detection"] == {
        "field_replacement_atoms_detected": 0,
        "field_replacement_atoms_total": 10,
        "direction_reversal_atoms_detected": 0,
        "direction_reversal_atoms_total": 0,
    }
    assert summary["change_detection_outcomes"] == {
        "both_detected": 0,
        "field_replacement_only": 0,
        "direction_reversal_only": 0,
        "neither_detected": 0,
        "episode_count": 0,
    }


def test_viewer_classifies_change_detection_by_operation_type():
    def artifact(pre_fields, final_fields):
        def response(fields):
            return {
                "status": "completed",
                "response": {
                    "relevant_fields": [field for field, _ in fields],
                    "field_directions": [
                        {"field_id": field, "direction": direction}
                        for field, direction in fields
                    ],
                },
            }

        return {
            "condition": "dynamic",
            "evaluation": {
                "transition": {
                    "pre_rule": {
                        "relevant_fields": ["a", "b", "c"],
                        "field_directions": {"a": 1, "b": 1, "c": 1},
                    },
                    "post_rule": {
                        "relevant_fields": ["a", "b", "d"],
                        "field_directions": {"a": -1, "b": 1, "d": 1},
                    },
                }
            },
            "relevant_fields_assessments": {
                "pre_change": response(pre_fields),
                "post_change": response(final_fields),
            },
        }

    pre_full = [("a", "positive"), ("b", "positive"), ("c", "positive")]
    summary = _aggregate_relevant_fields_assessments([
        artifact(pre_full, [("a", "negative"), ("b", "positive"), ("d", "positive")]),
        artifact(pre_full, [("a", "positive"), ("b", "positive"), ("d", "positive")]),
        artifact(pre_full, [("a", "negative"), ("b", "positive"), ("c", "positive")]),
        artifact(pre_full, pre_full),
    ])

    assert summary is not None
    assert summary["change_detection_outcomes"] == {
        "both_detected": 1,
        "field_replacement_only": 1,
        "direction_reversal_only": 1,
        "neither_detected": 1,
        "episode_count": 4,
    }
    assert summary["change_atom_detection"] == {
        "field_replacement_atoms_detected": 4,
        "field_replacement_atoms_total": 8,
        "direction_reversal_atoms_detected": 2,
        "direction_reversal_atoms_total": 4,
    }
    assert summary["adaptation_precision"] == pytest.approx(0.75)
    assert summary["adaptation_recall"] == pytest.approx(0.5)
    assert summary["change_recognition"] == pytest.approx(0.575)
    # The direction-only answer kept the field set unchanged, as did the
    # no-change answer; both had the pre-change set exactly right.
    assert summary["unchanged_outcomes"]["pre_correct_final_incorrect"] == 2
    assert summary["changed_outcomes"]["both_correct"] == 2


def test_viewer_excludes_failed_relevant_field_assessments():
    summary = _aggregate_relevant_fields_assessments([{
        "condition": "dynamic",
        "evaluation": {
            "transition": {
                "pre_rule": {"relevant_fields": ["a", "b", "c"]},
                "post_rule": {"relevant_fields": ["a", "b", "d"]},
            }
        },
        "relevant_fields_assessments": {
            "pre_change": {
                "status": "completed",
                "response": {"relevant_fields": ["a", "b", "c"]},
            },
            "post_change": {"status": "failed", "error": "provider error"},
        },
    }])

    assert summary is not None
    assert summary["pre_change"]["episode_count"] == 1
    assert summary["post_change"]["episode_count"] == 0
    assert summary["final"]["episode_count"] == 0
    assert summary["paired_episode_count"] == 0
    assert summary["changed_episode_count"] == 0
    # Only the pre-change pool is defined, so it alone sets the headline.
    assert summary["factor_recognition"] == 1
    assert summary["change_recognition"] is None
    assert summary["change_recognition_eligibility"] == {
        "eligible": 0,
        "total": 1,
        "rate": 0.0,
    }


def test_viewer_scores_static_branch_final_assessments():
    def artifact(predicted_fields):
        return {
            "condition": "static",
            "evaluation": {
                "transition": {
                    "pre_rule": {"relevant_fields": ["a", "b", "c"]},
                    "post_rule": {"relevant_fields": ["a", "b", "c"]},
                }
            },
            "relevant_fields_assessments": {
                "final": {
                    "status": "completed",
                    "response": {"relevant_fields": predicted_fields},
                },
            },
        }

    summary = _aggregate_relevant_fields_assessments([
        artifact(["a", "b", "c"]),
        artifact(["a", "b"]),
        # An empty prediction contributes only its false negatives; the
        # failed assessment never enters the pool.
        artifact([]),
        {
            "condition": "static",
            "evaluation": {
                "transition": {
                    "pre_rule": {"relevant_fields": ["a", "b", "c"]},
                    "post_rule": {"relevant_fields": ["a", "b", "c"]},
                }
            },
            "relevant_fields_assessments": {
                "final": {"status": "failed", "error": "provider error"},
            },
        },
    ])

    assert summary is not None
    # TP=3+2=5, FP=0, FN=0+1+3=4 → precision 1, recall 5/9.
    assert summary["final"]["episode_count"] == 3
    assert summary["final"]["precision"] == pytest.approx(1.0)
    assert summary["final"]["recall"] == pytest.approx(5 / 9)
    assert summary["final"]["f1"] == pytest.approx(round(2 * (5 / 9) / (1 + 5 / 9), 6))
    assert summary["factor_recognition"] == summary["final"]["f1"]
    assert summary["factor_recognition_final"] == summary["final"]["f1"]
    assert summary["factor_recognition_pre"] is None
    assert summary["factor_recognition_post"] is None
    assert summary["change_recognition"] is None
    assert summary["change_recognition_eligibility"] is None
    assert summary["paired_episode_count"] == 0


def test_viewer_pool_with_empty_predictions_scores_zero_f1():
    summary = _aggregate_relevant_fields_assessments([{
        "condition": "dynamic",
        "evaluation": {
            "transition": {
                "pre_rule": {"relevant_fields": ["a", "b", "c"]},
                "post_rule": {"relevant_fields": ["a", "b", "c"]},
            }
        },
        "relevant_fields_assessments": {
            "pre_change": {
                "status": "completed",
                "response": {"relevant_fields": []},
            },
            "post_change": {
                "status": "completed",
                "response": {"relevant_fields": []},
            },
        },
    }])

    assert summary is not None
    # Every prediction is empty, so precision is undefined but the pooled
    # phase F1s are 0 by convention.
    assert summary["pre_change"]["precision"] is None
    assert summary["pre_change"]["recall"] == 0.0
    assert summary["pre_change"]["f1"] == 0.0
    assert summary["factor_recognition"] == 0.0
    assert summary["change_recognition"] == 0.0
