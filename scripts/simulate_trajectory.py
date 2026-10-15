#!/usr/bin/env python3
"""Run deterministic model-response simulations against the real runtime.

The simulator deliberately uses :func:`run_branched_agent_episode`, so the
JSON artifacts it writes have the same shape as API-backed benchmark runs and
can be passed directly to ``viewer/generate_data.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from generator import GeneratorResponse
from protocol.precision import round_field_values
from protocol.provenance import code_identity
from runtime.metrics import aggregate_branched_metrics
from runtime.runner import resolve_runtime_branches, run_branched_agent_episode
from runtime.history import HISTORY_POLICY_VERSION
from runtime.schema import RUN_MANIFEST_SCHEMA_VERSION
from viewer.generate_data import generate as generate_viewer_data
from viewer.generate_data import write_gzip_mirror


SCENARIOS = (
    "valid",
    "valid_probe",
    "calculator_trajectory",
    "invalid_json",
    "schema_error",
    "combined_tool_schema",
    "combined_environment_rejection",
    "boundary_probe",
    "multiple_boundary_probe",
    "wrong_batch",
    "partial_actions",
    "invalid_action",
    "repeated_environment_rejection",
    "action_repair_exhausted",
    "provider_error",
)

# Probe-policy checks are kept separate from the long-running trajectory
# scenarios so the historical fixture tests continue to exercise their
# original branch matrix.  They are still accepted by the CLI and are useful
# as focused protocol smoke tests.
PROBE_POLICY_SCENARIOS = (
    "probe_skip_disabled",
    "least_probe_count",
)
ALL_SCENARIOS = SCENARIOS + PROBE_POLICY_SCENARIOS

SCENARIO_DESCRIPTIONS = {
    "valid": "Nominal trajectory with valid control and action calls.",
    "valid_probe": "A valid probe round before every action batch.",
    "calculator_trajectory": "Repeated calculator calls across control and action turns.",
    "invalid_json": "Two consecutive malformed JSON responses followed by recovery.",
    "schema_error": "Two consecutive tool-schema errors followed by recovery.",
    "combined_tool_schema": "One response containing multiple tool-schema violations.",
    "combined_environment_rejection": "One action request rejected for multiple environment violations.",
    "boundary_probe": "A probe with a value above the field maximum, followed by repair.",
    "multiple_boundary_probe": "A multi-probe round containing upper- and lower-bound violations.",
    "wrong_batch": "An action submission with the wrong batch identifier, followed by repair.",
    "partial_actions": "An action submission that omits cases and records abstentions.",
    "invalid_action": "An action submission containing an unknown action identifier.",
    "repeated_environment_rejection": "One recoverable environment rejection for every action batch.",
    "action_repair_exhausted": "Repeated invalid actions exhaust repair attempts and skip a batch.",
    "provider_error": "A simulated provider failure on the dynamic path.",
    "probe_skip_disabled": (
        "The probe-required diagnostic rejects a batch-skip attempt, then recovers."
    ),
    "least_probe_count": (
        "The probe-required diagnostic rejects two probes when four are required, then repairs."
    ),
}

ENABLED_BRANCHES = ("static_iid", "flip_25", "flip_50", "flip_75")
MATCHED_NO_CHANGE_CONTROL_BRANCHES = ("flip_50",)
EVAL_CASE_GIVEN_SETTING = "flip_50_eval_case_given"
PROBE_POLICY_SETTING = "flip_50_probe_required_least4"


def _state_from_request(request: Any) -> dict[str, Any]:
    """Extract the canonical state card from a DeepSeek/OpenAI-style request."""

    messages = request.messages or request.contents
    latest = messages[-1]
    if "content" in latest:
        text = str(latest["content"])
    else:
        text = "".join(
            str(part.get("text") or "")
            for part in latest.get("parts") or []
            if isinstance(part, dict)
        )
    marker = "Current state:\n"
    if marker not in text:
        raise ValueError("simulator could not find the current state card")
    return json.loads(text.rsplit(marker, 1)[1])


def _visible_request_text(request: Any) -> str:
    """Flatten provider wire content without JSON-escaping model-visible text."""

    strings: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(request.messages or request.contents)
    return "\n".join(strings)


class SimulatedModel:
    """Small deterministic client with API-compatible ``generate`` output."""

    config = SimpleNamespace(
        model="simulated-boundary-model",
        conversation_protocol="deepseek_chat",
    )

    def __init__(self, episode: dict[str, Any], scenario: str) -> None:
        self.episode = episode
        self.scenario = scenario
        self.calls: list[dict[str, Any]] = []
        self.wire_requests: list[str] = []
        self._scenario_calls: dict[tuple[str, ...], int] = {}
        self._action_attempts: dict[tuple[str, ...], int] = {}
        self._probe_attempts: dict[tuple[str, ...], int] = {}
        self._calculator_blocked_rejections: set[
            tuple[str, ...]
        ] = set()
        self._calculator_pending_calls: set[
            tuple[str, ...]
        ] = set()

    def _key(self, trace_context: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(trace_context.get("sampling_branch") or ""),
            str(trace_context.get("condition") or ""),
            str(trace_context.get("history_policy") or ""),
            str(trace_context.get("enable_probe_skip", True)),
            str(trace_context.get("least_probe_count")),
        )

    def _count(self, trace_context: dict[str, Any]) -> int:
        key = self._key(trace_context)
        self._scenario_calls[key] = self._scenario_calls.get(key, 0) + 1
        return self._scenario_calls[key]

    def _normal_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        pending = state.get("pending_action_batch")
        if pending is None:
            return {"tool": "request_next_batch", "arguments": {}}
        return {
            "tool": "submit_action_batch",
            "arguments": {
                "batch_id": pending["batch_id"],
                "actions": {
                    str(case["context_id"]): "action_0"
                    for case in pending.get("cases") or []
                },
            },
        }

    def _action_attempt(
        self, state: dict[str, Any], trace_context: dict[str, Any]
    ) -> int:
        pending = state.get("pending_action_batch") or {}
        batch_id = str(pending.get("batch_id") or "")
        key = (*self._key(trace_context), batch_id)
        self._action_attempts[key] = self._action_attempts.get(key, 0) + 1
        return self._action_attempts[key]

    def _invalid_action_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = self._normal_payload(state)
        actions = payload["arguments"]["actions"]
        if actions:
            actions[next(iter(actions))] = "action_999"
        return payload

    def _probe_attempt(
        self, state: dict[str, Any], trace_context: dict[str, Any]
    ) -> int:
        remaining = int(state["remaining_action_batches"])
        key = (*self._key(trace_context), remaining)
        self._probe_attempts[key] = self._probe_attempts.get(key, 0) + 1
        return self._probe_attempts[key]

    def _valid_probe_payload(self, count: int = 1) -> dict[str, Any]:
        values = round_field_values(
            self.episode["shared_prefix"]["cases"][0]["raw_values"],
            self.episode["agent_task"]["fields"],
        )
        return {
            "tool": "probe_batch",
            "arguments": {
                "probes": [{"values": dict(values)} for _ in range(count)]
            },
        }

    def _calculator_payload(self, stage: str) -> dict[str, Any]:
        expressions_by_stage = {
            "initial": [
                "X = 2 + 3",
                "Y = X * 4",
                "Z = Y - X",
                "Z + 1",
            ],
            "pending": [
                # A calculator batch is stateless across calls: the X from
                # ``initial`` is not visible here and must be assigned again.
                "X + 1",
                "X = 5",
                "X + 1",
                # Carrying a prior result across calls means spelling out the
                # numeric value, not relying on the old variable binding.
                "5 + 1",
            ],
            "forward_reference": [
                "later + 1",
                "later = 4",
                "later * 2",
            ],
            "post": [
                # X was assigned in an earlier calculator call, not this one.
                # The per-call scope must reject this direct cross-call use.
                "X + 1",
                "A = 6",
                "B = A ** 2",
                "sqrt(B) + A",
                "mean(A, B)",
            ],
        }
        return {
            "tool": "calculator",
            "arguments": {"expressions": expressions_by_stage[stage]},
        }

    def _boundary_probe_payload(self) -> dict[str, Any]:
        payload = self._valid_probe_payload()
        values = payload["arguments"]["probes"][0]["values"]
        field_id = next(iter(self.episode["agent_task"]["fields"]))
        field = self.episode["agent_task"]["fields"][field_id]
        values[field_id] = field["maximum"] + 1
        return payload

    def _multiple_boundary_probe_payload(self) -> dict[str, Any]:
        field_ids = list(self.episode["agent_task"]["fields"])
        probes = []
        for offset in (0, 2):
            values = round_field_values(
                self.episode["shared_prefix"]["cases"][0]["raw_values"],
                self.episode["agent_task"]["fields"],
            )
            upper_id = field_ids[offset]
            lower_id = field_ids[offset + 1]
            values[upper_id] = (
                self.episode["agent_task"]["fields"][upper_id]["maximum"] + 1
            )
            values[lower_id] = (
                self.episode["agent_task"]["fields"][lower_id]["minimum"] - 1
            )
            probes.append({"values": values})
        return {"tool": "probe_batch", "arguments": {"probes": probes}}

    def _payload(self, state: dict[str, Any], call_number: int, trace_context: dict[str, Any]) -> str:
        if self.scenario == "calculator_trajectory":
            pending = state.get("pending_action_batch") or {}
            available = {
                item["tool"]: bool(item["available"])
                for item in state.get("tool_usability") or []
            }
            key = self._key(trace_context)
            if pending:
                batch_number = int(pending["batch_number"])
                pending_key = (*key, str(pending["batch_id"]))
                if batch_number == 1 and available.get("calculator") and pending_key not in self._calculator_pending_calls:
                    self._calculator_pending_calls.add(pending_key)
                    return json.dumps(self._calculator_payload("pending"))
                return json.dumps(self._normal_payload(state))

            remaining = int(state["remaining_action_batches"])
            window_key = (*key, remaining)
            if available.get("calculator") and remaining in {12, 9, 6}:
                stage = "initial" if remaining == 12 else (
                    "forward_reference" if remaining == 9 else "post"
                )
                return json.dumps(self._calculator_payload(stage))
            if not available.get("calculator") and remaining in {12, 9, 6}:
                if window_key not in self._calculator_blocked_rejections:
                    self._calculator_blocked_rejections.add(window_key)
                    return json.dumps(self._calculator_payload("post"))
            return json.dumps(self._normal_payload(state))
        if self.scenario == "valid_probe" and state.get("pending_action_batch") is None:
            return json.dumps(self._valid_probe_payload())
        if (
            self.scenario == "probe_skip_disabled"
            and state.get("pending_action_batch") is None
        ):
            # Deliberately emit the removed tool once.  The runtime should
            # report a protocol-level "tool does not exist" error, after which
            # the simulated model submits a valid probe.
            if self._probe_attempt(state, trace_context) == 1:
                return json.dumps({"tool": "request_next_batch", "arguments": {}})
            required = int(trace_context.get("least_probe_count") or 1)
            return json.dumps(self._valid_probe_payload(count=required))
        if (
            self.scenario == "least_probe_count"
            and state.get("pending_action_batch") is None
        ):
            # The first attempt intentionally violates least_probe_count; the
            # second attempt repairs it with exactly the configured minimum.
            attempt = self._probe_attempt(state, trace_context)
            required = int(trace_context.get("least_probe_count") or 4)
            return json.dumps(
                self._valid_probe_payload(
                    count=max(1, required - 2) if attempt == 1 else required
                )
            )
        if self.scenario == "invalid_json" and call_number <= 2:
            return "not valid json"
        if self.scenario == "schema_error":
            if call_number == 1:
                return json.dumps({"tool": "unknown_tool", "arguments": {}})
            if call_number == 2:
                return json.dumps({"tool": "request_next_batch", "arguments": {"extra": True}})
        if self.scenario == "combined_tool_schema" and call_number == 1:
            return json.dumps({"unexpected": True})
        if self.scenario == "combined_environment_rejection":
            pending = state.get("pending_action_batch") or {}
            if (
                pending
                and int(pending["batch_number"]) == 1
                and self._action_attempt(state, trace_context) == 1
            ):
                return json.dumps(
                    {
                        "tool": "submit_action_batch",
                        "arguments": {
                            "batch_id": pending["batch_id"],
                            "unexpected": True,
                        },
                    }
                )
        if self.scenario == "boundary_probe" and state.get("pending_action_batch") is None:
            if self._probe_attempt(state, trace_context) == 1:
                return json.dumps(self._boundary_probe_payload())
            return json.dumps(self._valid_probe_payload())
        if (
            self.scenario == "multiple_boundary_probe"
            and state.get("pending_action_batch") is None
        ):
            if self._probe_attempt(state, trace_context) == 1:
                return json.dumps(self._multiple_boundary_probe_payload())
            return json.dumps(self._valid_probe_payload(count=2))
        if self.scenario == "wrong_batch" and call_number == 2:
            pending = state.get("pending_action_batch") or {}
            payload = self._normal_payload(state)
            if payload["tool"] == "submit_action_batch":
                payload["arguments"]["batch_id"] = f"{pending.get('batch_id', 'batch_01')}_wrong"
            return json.dumps(payload)
        if self.scenario == "partial_actions" and call_number == 2:
            payload = self._normal_payload(state)
            if payload["tool"] == "submit_action_batch":
                actions = payload["arguments"]["actions"]
                payload["arguments"]["actions"] = dict(list(actions.items())[:1])
            return json.dumps(payload)
        if self.scenario == "invalid_action" and call_number == 2:
            return json.dumps(self._invalid_action_payload(state))
        if self.scenario == "repeated_environment_rejection":
            pending = state.get("pending_action_batch")
            if pending and self._action_attempt(state, trace_context) == 1:
                payload = self._normal_payload(state)
                payload["arguments"]["batch_id"] = f"{pending['batch_id']}_wrong"
                return json.dumps(payload)
        if self.scenario == "action_repair_exhausted":
            pending = state.get("pending_action_batch") or {}
            if pending:
                attempt = self._action_attempt(state, trace_context)
                batch_number = int(pending["batch_number"])
                if (batch_number == 1 and attempt <= 2) or (
                    batch_number == 2 and attempt == 1
                ):
                    return json.dumps(self._invalid_action_payload(state))
        return json.dumps(self._normal_payload(state))

    async def generate(self, *, request: Any, schema: dict[str, Any], trace_context: dict[str, Any]) -> GeneratorResponse:
        del schema
        context = dict(trace_context)
        self.calls.append(context)
        self.wire_requests.append(_visible_request_text(request))
        call_number = self._count(context)
        if (
            self.scenario == "provider_error"
            and context.get("condition") == "dynamic"
            and int(context.get("interaction") or 0)
            == int(self.episode["evaluation"]["transition"]["after_action_batch"]) * 2 + 1
        ):
            raise RuntimeError("simulated provider outage")
        state = _state_from_request(request)
        content = self._payload(state, call_number, context)
        usage = {
            "input_tokens": 0,
            "output_tokens": len(content),
            "total_tokens": len(content),
        }
        return GeneratorResponse(
            content=content,
            model=self.config.model,
            usage=usage,
            raw={"simulation": True, "scenario": self.scenario, "call": call_number},
            finish_reason="stop",
        )


def load_episode(dataset: Path, episode_id: str | None = None) -> dict[str, Any]:
    episodes = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if not episodes:
        raise ValueError(f"dataset is empty: {dataset}")
    if episode_id is None:
        return episodes[0]
    for episode in episodes:
        if str(episode.get("episode_id")) == episode_id:
            return episode
    raise ValueError(f"episode ID not found: {episode_id}")


def runtime_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("run config must be a mapping")
    policy = config.get("interaction")
    if not isinstance(policy, dict):
        raise ValueError("runtime config requires an interaction mapping")
    branches = resolve_runtime_branches(config.get("branches"))
    return dict(policy), branches


async def simulate_episode(
    episode: dict[str, Any],
    scenario: str,
    *,
    policy: dict[str, Any] | None = None,
    include_no_change_control: bool = True,
    include_eval_case_given: bool = True,
    branches: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], SimulatedModel]:
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if policy is None:
        raise ValueError("simulate_episode requires an explicit runtime policy")
    model = SimulatedModel(episode, scenario)
    default_branches = {
        name: {
            "sampling": name,
            "rule_change": name != "static_iid",
            "history_policy": "eval_case_hidden",
        }
        for name in ENABLED_BRANCHES
    }
    if include_no_change_control:
        default_branches["flip_50_no_change_control"] = {
            "sampling": "flip_50",
            "rule_change": False,
            "history_policy": "eval_case_hidden",
        }
    if include_eval_case_given:
        default_branches[EVAL_CASE_GIVEN_SETTING] = {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_given",
        }
    if scenario in PROBE_POLICY_SCENARIOS:
        default_branches[PROBE_POLICY_SETTING] = {
            "sampling": "flip_50",
            "rule_change": True,
            "history_policy": "eval_case_hidden",
            "enable_probe_skip": False,
            "least_probe_count": 4,
        }
    selected_branches = resolve_runtime_branches(branches or default_branches)
    artifact = await run_branched_agent_episode(
        episode,
        model,
        interaction_policy=policy,
        branches=selected_branches,
    )
    return artifact, model


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_run(
    output_dir: Path,
    dataset: Path,
    episode: dict[str, Any],
    scenario: str,
    artifact: dict[str, Any],
    policy: dict[str, Any],
    run_config: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_configs = {}
    for name, branch in (artifact.get("branches") or {}).items():
        config = {
            "sampling": branch["sampling_branch"],
            "rule_change": branch["condition"] == "dynamic",
            "history_policy": branch["history_policy"],
        }
        if not branch.get("enable_probe_skip", True):
            config["enable_probe_skip"] = False
        if branch.get("least_probe_count") is not None:
            config["least_probe_count"] = branch["least_probe_count"]
        branch_configs[name] = config
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    episode_sha = hashlib.sha256(
        json.dumps(episode, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(output_dir / f"{episode['episode_id']}.json", {**artifact, "episode_sha256": episode_sha})
    _write_json(output_dir / "metrics.json", aggregate_branched_metrics([artifact]))
    manifest = {
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": artifact.get("run_status"),
        "simulation_scenario": scenario,
        "simulation_description": SCENARIO_DESCRIPTIONS[scenario],
        "run_identity": {
            "dataset_sha256": dataset_sha,
            "episode_schema_versions": [episode["schema_version"]],
            "policy": "agent",
            "scenario": scenario,
            "episode_ids": [episode["episode_id"]],
            "episode_sha256": {episode["episode_id"]: episode_sha},
            "branches": branch_configs,
        },
        "dataset": {"path": str(dataset), "sha256": dataset_sha},
        "run_config": {
            "path": str(run_config),
            "sha256": hashlib.sha256(run_config.read_bytes()).hexdigest(),
        },
        "code": code_identity(),
        "policy": "agent",
        "evaluation_mode": "branched",
        "branches": branch_configs,
        "interaction_policy": policy,
        "history": {
            "policies": artifact.get("history_policies") or ["eval_case_hidden"],
            "version": HISTORY_POLICY_VERSION,
        },
        "episode_count": 1,
        "episode_ids": [episode["episode_id"]],
        "failed_episode_ids": [] if artifact.get("run_status") == "completed" else [episode["episode_id"]],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPOSITORY_ROOT / "datasets/full-384/dynamic_episodes.jsonl")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs/simulated_trajectory")
    parser.add_argument(
        "--scenario", action="append", choices=ALL_SCENARIOS, dest="scenarios"
    )
    parser.add_argument("--viewer-output", type=Path, help="also publish the simulated runs for the viewer")
    parser.add_argument("--gzip", action="store_true", help="write compressed viewer mirrors")
    args = parser.parse_args()
    scenarios = args.scenarios or list(ALL_SCENARIOS)
    dataset = args.dataset.resolve()
    run_config = args.run_config.resolve()
    episode = load_episode(dataset, args.episode_id)
    policy, branches = runtime_config(run_config)
    run_dirs: list[Path] = []
    for scenario in scenarios:
        artifact, _ = asyncio.run(
            simulate_episode(
                episode,
                scenario,
                policy=policy,
                branches=branches,
            )
        )
        run_dir = write_run(
            args.output_dir.resolve() / f"simulated_{scenario}",
            dataset,
            episode,
            scenario,
            artifact,
            policy,
            run_config,
        )
        run_dirs.append(run_dir)
        print(json.dumps({"scenario": scenario, "run_status": artifact.get("run_status"), "run_dir": str(run_dir)}))
    if args.viewer_output:
        manifest = generate_viewer_data(dataset, args.viewer_output.resolve(), run_dirs, replace=True)
        if args.gzip:
            write_gzip_mirror(args.viewer_output.resolve())
        print(json.dumps({"viewer_output": str(args.viewer_output.resolve()), "episode_count": manifest["episode_count"], "run_count": len(manifest["runs"])}))


if __name__ == "__main__":
    main()
