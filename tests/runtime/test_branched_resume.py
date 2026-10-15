from __future__ import annotations

from argparse import Namespace
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from generator import GeneratorResponse

import scripts.run_benchmark as benchmark
from runtime.runner import run_branched_agent_episode


class ScriptedClient:
    config = SimpleNamespace(model="fake", conversation_protocol="deepseek_chat")

    def __init__(self, fail_when=None):
        self.fail_when = fail_when
        self.calls: list[dict] = []

    async def generate(self, *, request, schema, trace_context):
        del schema
        context = deepcopy(trace_context)
        self.calls.append(context)
        if (
            self.fail_when is not None
            and self.fail_when(context)
        ):
            raise RuntimeError("injected provider failure")

        state_text = request.messages[-1]["content"].rsplit(
            "Current state:\n", 1
        )[1]
        state = json.loads(state_text)
        pending = state["pending_action_batch"]
        if pending is None:
            payload = {"tool": "request_next_batch", "arguments": {}}
        else:
            payload = {
                "tool": "submit_action_batch",
                "arguments": {
                    "batch_id": pending["batch_id"],
                    "actions": {
                        case["context_id"]: "action_0"
                        for case in pending["cases"]
                    },
                },
            }
        return GeneratorResponse(
            content=json.dumps(payload),
            model="fake",
            usage={},
            raw={"fake": True},
        )


def _run(
    dynamic_episode,
    interaction_policy,
    client,
    *,
    resume_artifact=None,
    checkpoints=None,
):
    async def checkpoint(artifact):
        if checkpoints is not None:
            checkpoints.append(deepcopy(artifact))

    return asyncio.run(
        run_branched_agent_episode(
            dynamic_episode,
            client,
            interaction_policy=interaction_policy,
            branches={
                "static_iid": {
                    "sampling": "static_iid",
                    "rule_change": False,
                    "history_policy": "eval_case_given",
                },
                "flip_50": {
                    "sampling": "flip_50",
                    "rule_change": True,
                    "history_policy": "eval_case_given",
                },
            },
            checkpoint=checkpoint,
            resume_artifact=resume_artifact,
        )
    )


def test_resume_continues_unforked_trajectory_from_failed_interaction(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("runtime.runner.asyncio.sleep", no_sleep)
    checkpoints = []
    failed_client = ScriptedClient(
        lambda call: call["sampling_branch"] == "static_iid"
        and call["condition"] == "dynamic"
        and call["interaction"] == 3
    )
    partial = _run(
        dynamic_episode,
        runtime_interaction_policy,
        failed_client,
        checkpoints=checkpoints,
    )

    assert partial["run_status"] == "failed"
    assert partial["branches"] == {}
    assert partial["shared_prefix"]["termination"] is None
    assert partial["shared_prefix"]["tool_trace"][-1]["status"] == "provider_error"
    assert any(
        item["shared_prefix"]["tool_trace"][-1]["status"]
        == "awaiting_response"
        for item in checkpoints
    )

    resumed_client = ScriptedClient()
    completed = _run(
        dynamic_episode,
        runtime_interaction_policy,
        resumed_client,
        resume_artifact=partial,
    )

    assert resumed_client.calls[0]["interaction"] == 3
    assert resumed_client.calls[0]["condition"] == "dynamic"
    assert completed["run_status"] == "completed"
    assert set(completed["branches"]) == {"static_iid", "flip_50"}
    assert all(
        record["status"] != "provider_error"
        for branch in completed["branches"].values()
        for record in branch["tool_trace"]
    )


def test_resume_continues_each_forked_trajectory_from_failed_interaction(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("runtime.runner.asyncio.sleep", no_sleep)
    checkpoints = []
    failed_client = ScriptedClient(
        lambda call: call["sampling_branch"] == "flip_50"
        and call["condition"] == "dynamic"
        and call["interaction"] == 14
    )
    partial = _run(
        dynamic_episode,
        runtime_interaction_policy,
        failed_client,
        checkpoints=checkpoints,
    )

    assert partial["run_status"] == "completed_with_failures"
    assert partial["branches"]["static_iid"]["run_status"] == "completed"
    assert partial["branches"]["flip_50"]["run_status"] == "failed"
    assert partial["branches"]["flip_50"]["termination"] is None
    assert partial["branches"]["flip_50"]["tool_trace"][-1]["status"] == (
        "provider_error"
    )
    assert any(
        (item.get("branches") or {}).get("flip_50", {}).get("tool_trace", [{}])[-1]
        .get("status")
        == "completed"
        for item in checkpoints
    )

    completed_static = deepcopy(partial["branches"]["static_iid"])
    resumed_client = ScriptedClient()
    completed = _run(
        dynamic_episode,
        runtime_interaction_policy,
        resumed_client,
        resume_artifact=partial,
    )

    assert resumed_client.calls
    assert {call["sampling_branch"] for call in resumed_client.calls} == {"flip_50"}
    assert resumed_client.calls[0]["interaction"] == 14
    assert completed["run_status"] == "completed"
    assert completed["branches"]["static_iid"] == completed_static
    flip_trace = completed["branches"]["flip_50"]["tool_trace"]
    assert sum(record["interaction"] == 13 for record in flip_trace) == 1
    assert all(record["status"] != "provider_error" for record in flip_trace)


def test_branches_share_one_prefix_per_history_policy(
    dynamic_episode, runtime_interaction_policy
):
    artifact = asyncio.run(
        run_branched_agent_episode(
            dynamic_episode,
            ScriptedClient(),
            interaction_policy=runtime_interaction_policy,
            branches={
                "static_iid": {
                    "sampling": "static_iid",
                    "rule_change": False,
                    "history_policy": "eval_case_hidden",
                },
                "flip_50": {
                    "sampling": "flip_50",
                    "rule_change": True,
                    "history_policy": "eval_case_hidden",
                },
                "static_iid_rule_change": {
                    "sampling": "static_iid",
                    "rule_change": True,
                    "history_policy": "eval_case_hidden",
                },
                "flip_50_no_rule_change": {
                    "sampling": "flip_50",
                    "rule_change": False,
                    "history_policy": "eval_case_hidden",
                },
                "flip_50_eval_case_given": {
                    "sampling": "flip_50",
                    "rule_change": True,
                    "history_policy": "eval_case_given",
                },
            },
        )
    )

    assert artifact["run_status"] == "completed"
    assert set(artifact["branches"]) == {
        "static_iid",
        "flip_50",
        "static_iid_rule_change",
        "flip_50_no_rule_change",
        "flip_50_eval_case_given",
    }
    assert artifact["history_policies"] == ["eval_case_given", "eval_case_hidden"]
    assert set(artifact["shared_prefixes"]) == {
        "eval_case_hidden",
        "eval_case_given",
    }
    assert artifact["branches"]["flip_50"]["history_policy"] == (
        "eval_case_hidden"
    )
    given = artifact["branches"]["flip_50_eval_case_given"]
    assert given["history_policy"] == "eval_case_given"
    assert given["setting_id"] == "flip_50_eval_case_given"
    assert given["shared_prefix_hash"] == artifact["shared_prefix_hashes"][
        "eval_case_given"
    ]
    assert artifact["branches"]["static_iid"]["shared_prefix_hash"] == (
        artifact["branches"]["flip_50"]["shared_prefix_hash"]
    )
    hidden_hashes = {
        branch["shared_prefix_hash"]
        for name, branch in artifact["branches"].items()
        if name != "flip_50_eval_case_given"
    }
    assert hidden_hashes == {artifact["shared_prefix_hashes"]["eval_case_hidden"]}
    assert artifact["branches"]["static_iid_rule_change"]["condition"] == (
        "dynamic"
    )
    assert artifact["branches"]["flip_50_no_rule_change"]["condition"] == (
        "static"
    )


def test_resume_adds_branches_without_rerunning_completed_branch(
    dynamic_episode, runtime_interaction_policy
):
    hidden = {
        "sampling": "flip_50",
        "rule_change": True,
        "history_policy": "eval_case_hidden",
    }
    given = {
        "sampling": "flip_50",
        "rule_change": True,
        "history_policy": "eval_case_given",
    }
    initial_client = ScriptedClient()
    initial = asyncio.run(
        run_branched_agent_episode(
            dynamic_episode,
            initial_client,
            interaction_policy=runtime_interaction_policy,
            branches={"flip_50": hidden},
        )
    )
    assert initial["run_status"] == "completed"

    # Put the newly added history policy first to exercise migration from the
    # legacy single-policy ``shared_prefix`` field as well as branch extension.
    extended_client = ScriptedClient()
    extended = asyncio.run(
        run_branched_agent_episode(
            dynamic_episode,
            extended_client,
            interaction_policy=runtime_interaction_policy,
            branches={
                "flip_50_eval_case_given": given,
                "flip_50": hidden,
                "static_iid": {
                    "sampling": "static_iid",
                    "rule_change": False,
                    "history_policy": "eval_case_hidden",
                },
                "flip_25": {
                    "sampling": "flip_25",
                    "rule_change": True,
                    "history_policy": "eval_case_hidden",
                },
                "flip_50_no_change_control": {
                    "sampling": "flip_50",
                    "rule_change": False,
                    "history_policy": "eval_case_hidden",
                },
                "flip_75": {
                    "sampling": "flip_75",
                    "rule_change": True,
                    "history_policy": "eval_case_hidden",
                },
            },
            resume_artifact=initial,
        )
    )

    assert extended["run_status"] == "completed"
    assert set(extended["branches"]) == {
        "flip_50_eval_case_given",
        "flip_50",
        "static_iid",
        "flip_25",
        "flip_50_no_change_control",
        "flip_75",
    }
    hidden_calls = [
        call
        for call in extended_client.calls
        if call["history_policy"] == "eval_case_hidden"
    ]
    assert hidden_calls
    assert min(call["interaction"] for call in hidden_calls) > int(
        dynamic_episode["evaluation"]["transition"]["after_action_batch"]
    ) * 2
    assert all(
        call["history_policy"] != "eval_case_hidden"
        or call["interaction"] > int(
            dynamic_episode["evaluation"]["transition"]["after_action_batch"]
        )
        * 2
        for call in extended_client.calls
    )
    assert any(
        call["sampling_branch"] == "static_iid"
        and call["history_policy"] == "eval_case_hidden"
        for call in extended_client.calls
    )
    assert any(
        call["sampling_branch"] == "flip_50"
        and call["history_policy"] == "eval_case_given"
        for call in extended_client.calls
    )
    assert extended["branches"]["flip_50"]["tool_trace"] == initial[
        "branches"
    ]["flip_50"]["tool_trace"]


def test_benchmark_resume_extends_default_output_in_place(
    monkeypatch, tmp_path: Path, dynamic_episode, runtime_interaction_policy
):
    dataset = tmp_path / "episodes.jsonl"
    dataset.write_text(json.dumps(dynamic_episode) + "\n")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: fake\nconcurrency: 1\n")
    default_config = tmp_path / "default.yaml"
    full_config = tmp_path / "full.yaml"
    default_branches = {
        "flip_50": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
        }
    }
    full_branches = {
        **default_branches,
        "static_iid": {
            "sampling": "static_iid",
            "rule_change": False,
            "history_policy": "eval_case_hidden",
        },
        "flip_25": {
            "sampling": "flip_25",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
        },
        "flip_50_no_change_control": {
            "sampling": "flip_50",
            "rule_change": False,
            "history_policy": "eval_case_hidden",
        },
        "flip_75": {
            "sampling": "flip_75",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
        },
        "flip_50_eval_case_given": {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_given",
        },
    }
    for path, branches in (
        (default_config, default_branches),
        (full_config, full_branches),
    ):
        path.write_text(
            yaml.safe_dump(
                {"branches": branches, "interaction": runtime_interaction_policy}
            )
        )

    clients: list[ScriptedClient] = []

    def build_client(_config):
        client = ScriptedClient()
        clients.append(client)
        return client

    monkeypatch.setattr(benchmark, "build_generator", build_client)
    monkeypatch.setattr(benchmark, "load_repo_env", lambda _path: None)
    output_dir = tmp_path / "run"
    common = {
        "output_dir": output_dir,
        "dataset": dataset,
        "model_config": model_config,
        "max_cases": 1,
        "episode_id": None,
        "seed": 7,
    }
    asyncio.run(
        benchmark.main(
            Namespace(
                **common,
                run_config=default_config,
                resume=False,
            )
        )
    )
    asyncio.run(
        benchmark.main(
            Namespace(**{
                **common,
                "dataset": None,
                "model_config": None,
                "run_config": full_config,
                "resume": True,
            })
        )
    )

    assert len(clients) == 2
    saved = json.loads(
        (output_dir / f"{dynamic_episode['episode_id']}.json").read_text()
    )
    assert set(saved["branches"]) == set(full_branches)
    fork_interactions = int(
        dynamic_episode["evaluation"]["transition"]["after_action_batch"]
    ) * 2
    assert min(
        call["interaction"]
        for call in clients[1].calls
        if call["history_policy"] == "eval_case_hidden"
    ) > fork_interactions


class TruncatingClient(ScriptedClient):
    def __init__(self, truncate_calls: int = 2):
        super().__init__()
        self.truncate_calls = truncate_calls

    async def generate(self, *, request, schema, trace_context):
        if len(self.calls) < self.truncate_calls:
            self.calls.append(deepcopy(trace_context))
            return GeneratorResponse(
                content="{",
                model="fake",
                usage={},
                raw={"fake": True},
                finish_reason="length",
            )
        return await super().generate(
            request=request, schema=schema, trace_context=trace_context
        )


def test_resume_replays_truncated_window_skip(
    monkeypatch, dynamic_episode, runtime_interaction_policy
):
    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("runtime.runner.asyncio.sleep", no_sleep)
    checkpoints = []
    final = _run(
        dynamic_episode,
        runtime_interaction_policy,
        TruncatingClient(truncate_calls=2),
        checkpoints=checkpoints,
    )
    assert final["run_status"] == "completed"

    def has_skip_then_submit(artifact):
        trace = artifact["shared_prefix"]["tool_trace"]
        for index, record in enumerate(trace):
            if record.get("error_type") != "response_truncated":
                continue
            return any(
                item.get("valid") is True
                and (item.get("response") or {}).get("tool") == "submit_action_batch"
                for item in trace[index + 1 :]
            )
        return False

    mid = next(artifact for artifact in checkpoints if has_skip_then_submit(artifact))
    assert mid["run_status"] == "running"

    resumed_client = ScriptedClient()
    completed = _run(
        dynamic_episode,
        runtime_interaction_policy,
        resumed_client,
        resume_artifact=mid,
    )

    assert completed["run_status"] == "completed"
    assert set(completed["branches"]) == {"static_iid", "flip_50"}
