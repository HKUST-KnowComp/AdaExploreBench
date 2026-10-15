from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from .calculator import calculate_many
from .decisions import format_error_messages
from protocol.precision import (
    has_at_most_decimal_places,
    round_field_value,
    round_field_values,
)
from protocol.mechanism import (
    ACTION_COUNT,
    hydrate_rule,
    intervention_score,
    utilities_from_score,
)
from .conversation import StateCard, TaskSpec
from .history import validate_history_policy
from synthesis.datasets.contracts import (
    DYNAMIC_EPISODE_SCHEMA_VERSION,
    STATIC_EPISODE_SCHEMA_VERSION,
    validate_episode,
)
from protocol.constraints import constraint_violations


class BenchmarkEnvironment:
    """Evaluator-private sequential benchmark environment.

    A static condition is a runtime projection of the same episode and case
    order that keeps the initial rule active for all twelve batches.
    """

    def __init__(
        self,
        episode: dict[str, Any],
        *,
        interaction_policy: dict[str, Any],
        condition: str = "dynamic",
        sampling_branch: str | None = None,
        history_policy: str = "eval_case_given",
        enable_probe_skip: bool = True,
        least_probe_count: int | None = None,
        syntax_repair_limit: int = 1,
    ) -> None:
        self.dataset_mode = str(episode.get("dataset_mode") or "")
        expected_schema = {
            "dynamic": DYNAMIC_EPISODE_SCHEMA_VERSION,
            "static": STATIC_EPISODE_SCHEMA_VERSION,
        }.get(self.dataset_mode)
        if expected_schema is None or episode.get("schema_version") != expected_schema:
            raise ValueError("environment requires a recognized static or dynamic episode")
        if condition not in {"dynamic", "static"}:
            raise ValueError("condition must be dynamic or static")
        if self.dataset_mode == "static" and condition != "static":
            raise ValueError("a static dataset episode only supports static evaluation")
        self.condition = condition
        self.history_policy = validate_history_policy(history_policy)
        (
            self.enable_probe_skip,
            self.least_probe_count,
        ) = self._validate_probe_policy(
            enable_probe_skip, least_probe_count
        )
        self._integrated_episode = deepcopy(episode)
        validate_episode(self._integrated_episode, self.dataset_mode)
        self.episode = deepcopy(self._integrated_episode)
        self.sampling_branch = sampling_branch
        if self.dataset_mode == "dynamic":
            self._select_dynamic_timeline(sampling_branch)
        # Dataset releases use phase/branch-specific identifiers (for example
        # ``pre_case_001`` or ``flip_50_case_001``).  Those identifiers are
        # evaluator metadata and must never be shown to the model.  Materialize
        # an opaque, sequential id namespace for each runtime branch instead.
        self._anonymize_runtime_context_ids()
        self._materialize_private_runtime()
        self._field_order = self._shuffled_field_order()
        self._validate_episode()
        self._validate_interaction_policy(interaction_policy)
        self.done = False
        self.termination: dict[str, Any] | None = None
        self.interactions_used = 0
        self.invalid_calls: list[dict[str, Any]] = []
        self.invalid_probe_calls = 0
        self.calculator_calls = 0
        self.calculator_expressions = 0
        self.calculator_expression_errors = 0
        # The calculator batches all computation into one call: after a
        # successful call it stays blocked until a probe or batch action runs.
        self.calculator_blocked = False
        # Batch 1 is revealed only after the agent chooses to probe or skip.
        # Cumulative Probes is constrained by one call per batch window and the
        # per-call case limit.
        self.pending_batch: dict[str, Any] | None = None
        self.next_batch_index = 0
        self._probe_count = 0
        self.probe_used_this_window = False
        # Repair allowance per batch window: a rejected call may be repaired
        # ``syntax_repair_limit`` times before the environment advances.
        self.syntax_repair_limit = max(0, int(syntax_repair_limit))
        self.repair_attempts_this_window = 0
        self.probe_batches_by_batch: dict[int, int] = {
            i: 0 for i in range(self.action_batch_count)
        }
        self.probe_cases_by_batch: dict[int, int] = {
            i: 0 for i in range(self.action_batch_count)
        }
        self.completed_contexts: dict[str, dict[str, Any]] = {}
        self.probe_contexts: dict[str, dict[str, Any]] = {}
        self.public_history: list[dict[str, Any]] = []
        self.private_history: list[dict[str, Any]] = []
        self.interaction_log: list[dict[str, Any]] = []
        self.action_results: list[dict[str, Any]] = []
        self._probe_index = 0
        self._pool = {
            item["context_id"]: item for item in self.episode["cases"]
        }
        # Do not preload batch 1: the first interaction must choose
        # probe_batch or request_next_batch.

    def _select_dynamic_timeline(self, sampling_branch: str | None) -> None:
        if not sampling_branch:
            raise ValueError("dynamic runtime requires an explicit sampling branch")
        source = self._integrated_episode
        branches = source["post_branches"]
        if sampling_branch not in branches:
            raise ValueError(f"dataset lacks requested sampling branch: {sampling_branch}")
        self.episode = deepcopy(source)
        shared = source["shared_prefix"]
        post = branches[sampling_branch]
        # ``_anonymize_runtime_context_ids`` mutates the selected timeline in
        # place.  Do not attach the release objects directly here: the shared
        # prefix is reused by every branch and shallow references would rewrite
        # ``_integrated_episode`` on the first branch selection.  That loses
        # release ``context_id`` values for later branches and corrupts joins
        # in artifacts/viewer output.
        self.episode["cases"] = deepcopy(shared["cases"]) + deepcopy(post["cases"])
        self.episode["batches"] = deepcopy(shared["batches"]) + deepcopy(post["batches"])

    def select_runtime_branch(self, sampling_branch: str, condition: str) -> None:
        """Switch only the not-yet-run post timeline after a shared prefix fork."""

        if self.dataset_mode != "dynamic":
            raise ValueError("runtime branch selection requires a dynamic episode")
        if condition not in {"dynamic", "static"}:
            raise ValueError("runtime branch condition is invalid")
        if self.next_batch_index > self.after_action_batch:
            raise ValueError("runtime branch cannot change after post batches begin")
        self.condition = condition
        self.sampling_branch = sampling_branch
        self._select_dynamic_timeline(sampling_branch)
        self._anonymize_runtime_context_ids()
        self._materialize_private_runtime()
        self._field_order = self._shuffled_field_order()
        self._pool = {
            item["context_id"]: item for item in self.episode["cases"]
        }

    def _anonymize_runtime_context_ids(self) -> None:
        """Replace release ids with opaque sequential ids for model-facing data.

        A fresh mapping is built whenever a runtime timeline is selected.  The
        shared prefix therefore receives the same ``case_001`` … ids in every
        branch, while each branch's post-fork cases continue the sequence.  The
        original ids are retained privately so metrics/artifacts can still be
        joined to the source release, and as input aliases for old checkpoints.
        """

        aliases: dict[str, str] = {}
        for index, item in enumerate(self.episode["cases"], start=1):
            original = str(item["context_id"])
            runtime_id = f"case_{index:03d}"
            aliases[original] = runtime_id
            item["_source_context_id"] = original
            item["context_id"] = runtime_id

        for batch in self.episode["batches"]:
            batch["context_ids"] = [
                aliases.get(str(context_id), str(context_id))
                for context_id in batch["context_ids"]
            ]
        self._context_id_aliases = aliases

    def _materialize_private_runtime(self) -> None:
        fields = self.episode["agent_task"]["fields"]
        for case in self.episode["cases"]:
            case["raw_values"] = round_field_values(case["raw_values"], fields)
        evaluation = self.episode["evaluation"]
        if self.dataset_mode == "static":
            self._rules = {"static": hydrate_rule(evaluation["rule"], fields)}
        else:
            transition = evaluation["transition"]
            self._rules = {
                "pre_dynamic": hydrate_rule(transition["pre_rule"], fields),
                "post_dynamic": hydrate_rule(transition["post_rule"], fields),
            }
        phase_by_id = {
            context_id: batch["phase"]
            for batch in self.episode["batches"]
            for context_id in batch["context_ids"]
        }
        for case in self.episode["cases"]:
            case["phase"] = phase_by_id[case["context_id"]]
            phases = (
                ("static",)
                if self.dataset_mode == "static"
                else ("pre_dynamic", "post_dynamic")
            )
            case["private_oracle"] = {
                phase: self._oracle(self._rules[phase], case["raw_values"])
                for phase in phases
            }

    def _shuffled_field_order(self) -> list[str]:
        """Seeded field display order so position never leaks rule relevance.

        Dataset releases list the rule-relevant fields first, which would let
        the model infer the hidden rule from field position alone.  Shuffle
        once per episode with the episode seed so every model-facing view of
        the fields (schema descriptions and per-case raw values) uses the
        same order and replays stay reproducible.
        """

        order = list(self.episode["agent_task"]["fields"])
        random.Random(int(self.episode["seed"])).shuffle(order)
        return order

    @staticmethod
    def _oracle(rule: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
        score = intervention_score(rule, values)
        utilities = utilities_from_score(score)
        best = max(range(ACTION_COUNT), key=utilities.__getitem__)
        return {
            "score": round(score, 6),
            "utilities": {
                f"action_{index}": round(value, 6)
                for index, value in enumerate(utilities)
            },
            "optimal_action_id": f"action_{best}",
        }

    def _validate_episode(self) -> None:
        interaction = self.episode["agent_task"]["interaction_protocol"]
        required_positive = (
            "action_batch_count",
            "action_batch_size",
        )
        for key in required_positive:
            value = interaction.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid dynamic interaction setting: {key}")
        strata = self.episode["evaluation"]["batch_contract"]["strata_per_batch"]
        if (
            not isinstance(strata, dict)
            or set(strata) != {"iid"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in strata.values()
            )
            or sum(strata.values()) != interaction["action_batch_size"]
        ):
            raise ValueError("invalid dynamic interaction setting: strata_per_batch")
        self.action_batch_count = int(interaction["action_batch_count"])
        self.action_batch_size = int(interaction["action_batch_size"])
        self.strata_per_batch = dict(strata)
        if self.dataset_mode == "static":
            self._validate_static_episode()
            return
        event = self.episode["evaluation"]["transition"]
        after_action_batch = int(event.get("after_action_batch") or 0)
        if not 0 < after_action_batch < self.action_batch_count:
            raise ValueError("dynamic event must occur between the first and last batch")
        self.after_action_batch = after_action_batch
        dynamic_type = str(
            event.get("type") or "field_replacement"
        )
        operations = event.get("operations")
        if not isinstance(operations, list):
            raise ValueError("dynamic event requires an operations list")
        pre_rule = self._rules["pre_dynamic"]
        post_rule = self._rules["post_dynamic"]
        if dynamic_type == "compound":
            if len(operations) != 2 or {
                operation.get("type") for operation in operations
            } != {"field_replacement", "direction_reversal"}:
                raise ValueError(
                    "compound dynamic requires one replacement and one reversal"
                )
            replacement = next(
                operation
                for operation in operations
                if operation.get("type") == "field_replacement"
            )
            reversal = next(
                operation
                for operation in operations
                if operation.get("type") == "direction_reversal"
            )
            outgoing = replacement.get("outgoing_field")
            incoming = replacement.get("incoming_field")
            target = reversal.get("target_field")
            pre_relevant = list(pre_rule.get("relevant_fields") or [])
            post_relevant = list(post_rule.get("relevant_fields") or [])
            shared = [field for field in pre_relevant if field in post_relevant]
            if (
                outgoing not in pre_relevant
                or outgoing in post_relevant
                or incoming in pre_relevant
                or incoming not in post_relevant
            ):
                raise ValueError("compound replacement rule diff is invalid")
            if target not in shared:
                raise ValueError("compound reversal target must remain relevant")
            changed = [
                field
                for field in shared
                if pre_rule["field_directions"].get(field)
                != post_rule["field_directions"].get(field)
            ]
            if changed != [target] or int(
                pre_rule["field_directions"][target]
            ) != -int(post_rule["field_directions"][target]):
                raise ValueError("compound episode must reverse exactly one shared field")
        elif len(operations) != 1:
            raise ValueError("single dynamic release requires exactly one operation")
        elif operations[0].get("type") != dynamic_type:
            raise ValueError("dynamic operation type does not match taxonomy")
        elif dynamic_type == "direction_reversal":
            target = operations[0].get("target_field")
            if target not in pre_rule.get("relevant_fields", []):
                raise ValueError("direction target must be relevant")
            changed = [
                field
                for field in pre_rule["relevant_fields"]
                if pre_rule["field_directions"].get(field)
                != post_rule["field_directions"].get(field)
            ]
            if changed != [target]:
                raise ValueError("direction episode must reverse exactly one field")
        elif dynamic_type != "field_replacement":
            raise ValueError("unsupported dynamic operation type")
        batches = self.episode["batches"]
        if len(batches) != self.action_batch_count:
            raise ValueError("dynamic timeline has the wrong action batch count")
        observed: list[str] = []
        for index, batch in enumerate(batches, start=1):
            if batch.get("batch_id") != f"batch_{index:02d}":
                raise ValueError("dynamic batch IDs must be sequential")
            if len(batch.get("context_ids") or []) != self.action_batch_size:
                raise ValueError("dynamic action batch has the wrong case count")
            batch_cases = {
                item["context_id"]: item for item in self.episode["cases"]
            }
            observed_strata = {
                stratum: sum(
                    batch_cases[context_id]["stratum"] == stratum
                    for context_id in batch["context_ids"]
                )
                for stratum in self.strata_per_batch
            }
            if observed_strata != self.strata_per_batch:
                raise ValueError("dynamic action batch does not match declared strata")
            expected_phase = (
                "pre_dynamic" if index <= after_action_batch else "post_dynamic"
            )
            if batch.get("phase") != expected_phase:
                raise ValueError("action batch phase does not match dynamic schedule")
            observed.extend(batch["context_ids"])
        pool_ids = [item["context_id"] for item in self.episode["cases"]]
        expected_cases = self.action_batch_count * self.action_batch_size
        if (
            len(observed) != expected_cases
            or len(set(observed)) != expected_cases
            or observed != pool_ids
        ):
            raise ValueError("timeline must cover the ordered action-case pool exactly once")

    def _validate_interaction_policy(self, policy: dict[str, Any]) -> None:
        expected = {
            "max_probe_cases_per_batch",
            "allow_abstention",
        }
        if not isinstance(policy, dict) or set(policy) != expected:
            raise ValueError(
                f"runtime interaction policy must define exactly {sorted(expected)}"
            )
        maximum = policy["max_probe_cases_per_batch"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("max_probe_cases_per_batch must be a positive integer")
        if not isinstance(policy["allow_abstention"], bool):
            raise ValueError("allow_abstention must be boolean")
        self.max_probe_cases_per_batch = maximum
        self.allow_abstention = policy["allow_abstention"]
        if (
            self.least_probe_count is not None
            and self.least_probe_count > self.max_probe_cases_per_batch
        ):
            raise ValueError(
                "invalid config: least_probe_count cannot exceed "
                "max_probe_cases_per_batch"
            )

    @staticmethod
    def _validate_probe_policy(
        enable_probe_skip: Any, least_probe_count: Any
    ) -> tuple[bool, int | None]:
        if not isinstance(enable_probe_skip, bool):
            raise ValueError("enable_probe_skip must be boolean")
        if least_probe_count is None:
            return enable_probe_skip, None
        if isinstance(least_probe_count, str) and least_probe_count.strip().lower() in {
            "none",
            "null",
        }:
            return enable_probe_skip, None
        if (
            isinstance(least_probe_count, bool)
            or not isinstance(least_probe_count, int)
            or least_probe_count <= 0
        ):
            raise ValueError("least_probe_count must be a positive integer or none")
        return enable_probe_skip, int(least_probe_count)

    def _validate_static_episode(self) -> None:
        if not isinstance(self.episode["evaluation"].get("rule"), dict):
            raise ValueError("static episode requires one stable rule")
        batches = self.episode["batches"]
        if len(batches) != self.action_batch_count:
            raise ValueError("static timeline has the wrong action batch count")
        observed: list[str] = []
        for index, batch in enumerate(batches, start=1):
            if batch.get("batch_id") != f"batch_{index:02d}":
                raise ValueError("static batch IDs must be sequential")
            if batch.get("phase") != "static":
                raise ValueError("static timeline batch has a dynamic phase")
            if len(batch.get("context_ids") or []) != self.action_batch_size:
                raise ValueError("static action batch has the wrong case count")
            batch_cases = {
                item["context_id"]: item for item in self.episode["cases"]
            }
            observed_strata = {
                stratum: sum(
                    batch_cases[context_id]["stratum"] == stratum
                    for context_id in batch["context_ids"]
                )
                for stratum in self.strata_per_batch
            }
            if observed_strata != self.strata_per_batch:
                raise ValueError("static action batch does not match declared strata")
            observed.extend(batch["context_ids"])
        pool_ids = [item["context_id"] for item in self.episode["cases"]]
        if observed != pool_ids or len(set(observed)) != len(observed):
            raise ValueError("static timeline must cover its action cases exactly once")
        if any(
            item.get("phase") != "static"
            or set(item.get("private_oracle") or {}) != {"static"}
            for item in self.episode["cases"]
        ):
            raise ValueError("static action cases require only a static oracle")
        self.after_action_batch = self.action_batch_count

    @property
    def actions(self) -> list[dict[str, Any]]:
        return list(self.episode["agent_task"]["actions"])

    @property
    def raw_fields(self) -> dict[str, dict[str, Any]]:
        return dict(self.episode["agent_task"]["fields"])

    @property
    def rule(self) -> dict[str, Any]:
        return self._rule_for_phase(self.active_rule_phase)

    @property
    def active_rule_phase(self) -> str:
        if self.dataset_mode == "static":
            return "static"
        if self.condition == "static":
            return "pre_dynamic"
        return (
            "pre_dynamic"
            if self.next_batch_index < self.after_action_batch
            else "post_dynamic"
        )

    def _rule_for_phase(self, phase: str) -> dict[str, Any]:
        if self.dataset_mode == "static":
            if phase != "static":
                raise ValueError("static episode has no dynamic rule phase")
            return dict(self._rules["static"])
        return dict(self._rules[phase])

    @property
    def completed_batch_count(self) -> int:
        return self.next_batch_index

    @property
    def action_utility(self) -> float:
        return round(
            sum(float(item["utility"]) for item in self.action_results), 2
        )

    @property
    def probe_count(self) -> int:
        return self._probe_count

    def _load_next_batch(self) -> None:
        if self.next_batch_index >= self.action_batch_count:
            raise ValueError("all action batches have already been completed")
        batch = deepcopy(
            self.episode["batches"][self.next_batch_index]
        )
        self.pending_batch = {
            "batch_id": batch["batch_id"],
            "batch_number": batch["batch_number"],
            "context_ids": batch["context_ids"],
        }

    def _public_case(self, item: dict[str, Any]) -> dict[str, Any]:
        values = item["raw_values"]
        return {
            "context_id": item["context_id"],
            "raw_values": {
                field_id: round_field_value(
                    values[field_id], self.raw_fields[field_id]["type"]
                )
                for field_id in self._field_order
            },
        }

    def _public_action_identity(self, context_id: str) -> dict[str, str]:
        return {"context_id": context_id}

    @staticmethod
    def _private_context_id(item: dict[str, Any]) -> str:
        """Return the source-release id used for evaluator-private records."""

        return str(item.get("_source_context_id") or item["context_id"])

    def task_spec(self) -> TaskSpec:
        """Return the one-time, non-dynamic task description."""
        task = self.episode["agent_task"]
        interaction = task["interaction_protocol"]
        return TaskSpec(
            {
                "scenario_description": task["scenario_description"],
                "agent_role": task["agent_role"],
                "actions": deepcopy(self.actions),
                "raw_field_schema": {
                    field_id: deepcopy(task["fields"][field_id])
                    for field_id in self._field_order
                },
                "feasibility_constraints": deepcopy(
                    task.get("feasibility_constraints") or []
                ),
                "interaction_protocol": {
                    "action_batch_count": interaction["action_batch_count"],
                    "action_batch_size": interaction["action_batch_size"],
                    "max_probe_cases_per_batch": self.max_probe_cases_per_batch,
                    "allow_abstention": self.allow_abstention,
                    "enable_probe_skip": self.enable_probe_skip,
                    "least_probe_count": self.least_probe_count,
                },
            }
        )

    def _tool_usability(self) -> list[dict[str, Any]]:
        """Per-turn availability for tools exposed by the active setting."""
        if self.done:
            tools = [
                {"tool": tool, "available": False, "reason": "the episode is complete"}
                for tool in (
                    "probe_batch",
                    "submit_action_batch",
                    "calculator",
                )
            ]
            if self.enable_probe_skip:
                tools.insert(
                    1,
                    {
                        "tool": "request_next_batch",
                        "available": False,
                        "reason": "the episode is complete",
                    },
                )
            return tools
        pending = self.pending_batch
        probed = self.probe_used_this_window
        if pending:
            probe_reason = (
                f"action batch {pending['batch_id']} is pending; submit it with "
                "submit_action_batch before probing again, so probe evidence "
                "cannot be used on already-revealed cases"
            )
            request_reason = (
                f"action batch {pending['batch_id']} is already pending; "
                "submit it with submit_action_batch"
            )
            submit_reason = None
        elif probed:
            probe_reason = (
                "probe_batch was already used for the upcoming batch window; "
                "call request_next_batch to reveal it"
            )
            request_reason = None
            submit_reason = (
                "no action batch is pending; reveal one with request_next_batch first"
            )
        else:
            probe_reason = None
            request_reason = None
            submit_reason = (
                "no action batch is pending; reveal one with probe_batch "
                "or request_next_batch first"
            )
        calculator_reason = (
            None
            if not self.calculator_blocked
            else (
                "calculator was already used since the last probe or batch action; "
                "submit every pending expression in one calculator call"
            )
        )
        tools = [
            {"tool": "probe_batch", "available": probe_reason is None, "reason": probe_reason},
            {"tool": "submit_action_batch", "available": submit_reason is None, "reason": submit_reason},
            {"tool": "calculator", "available": calculator_reason is None, "reason": calculator_reason},
        ]
        if self.enable_probe_skip:
            tools.insert(
                1,
                {
                    "tool": "request_next_batch",
                    "available": request_reason is None,
                    "reason": request_reason,
                },
            )
        return tools

    def state_card(self) -> StateCard:
        """Return only current control state; history belongs to the session."""
        pending = None
        if self.pending_batch:
            pending = {
                "batch_id": self.pending_batch["batch_id"],
                "batch_number": self.pending_batch["batch_number"],
                "cases": [
                    self._public_case(self._pool[context_id])
                    for context_id in self.pending_batch["context_ids"]
                ],
            }
        return StateCard(
            {
                "cumulative_utility": self.action_utility,
                "cumulative_probes": self.probe_count,
                "remaining_action_batches": self.action_batch_count
                - self.completed_batch_count,
                "pending_action_batch": pending,
                "tool_usability": self._tool_usability(),
            }
        )

    def agent_state(self) -> dict[str, Any]:
        """Legacy inspection view without cumulative evidence snapshots.

        The benchmark runner never uses this aggregate view; it exists for
        callers that still inspect an environment directly while migrating to
        ``task_spec()`` and ``state_card()``.
        """
        return {**deepcopy(self.task_spec()), **deepcopy(self.state_card())}

    @staticmethod
    def _require_exact_keys(scope: str, value: Any, expected: set[str]) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{scope} must be an object")
        received = set(value)
        if received == expected:
            return
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        detail = []
        if missing:
            detail.append(f"missing key(s): {', '.join(missing)}")
        if extra:
            detail.append(f"unexpected key(s): {', '.join(extra)}")
        raise ValueError(
            format_error_messages(detail, prefix=f"{scope} has invalid keys")
        )

    def _validate_value(
        self, field_id: str, value: Any, *, partial_probe_hint: bool = False
    ) -> int | float:
        if field_id not in self.raw_fields:
            raise ValueError(f"field is not editable: {field_id}")
        meta = self.raw_fields[field_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            message = f"field {field_id} must be numeric; received {value!r}"
            if partial_probe_hint:
                message += (
                    "; to vary only a few fields, use an existing-context probe "
                    "with base_context_id and changes instead of a full "
                    "new-context values object"
                )
            raise ValueError(message)
        if meta["type"] == "integer" and not isinstance(value, int):
            raise ValueError(
                f"field {field_id} must be a JSON integer; received {value!r}"
            )
        if meta["type"] == "decimal" and not has_at_most_decimal_places(value):
            raise ValueError(
                f"field {field_id} must use at most two decimal places; "
                f"received {value!r}"
            )
        numeric = float(value)
        # Reveal only the violated bound so a rejected probe can be repaired
        # without exposing the full sampling range.
        if numeric > float(meta["maximum"]):
            raise ValueError(
                f"field {field_id} exceeds its allowed maximum; max: {meta['maximum']}"
            )
        if numeric < float(meta["minimum"]):
            raise ValueError(
                f"field {field_id} is below its allowed minimum; min: {meta['minimum']}"
            )
        if meta["type"] == "integer":
            return value
        return round(numeric, 2)

    def _validate_feasibility(self, values: dict[str, Any]) -> None:
        violations = constraint_violations(
            list(self.episode["agent_task"].get("feasibility_constraints") or []),
            values,
        )
        if violations:
            raise ValueError(
                format_error_messages(
                    [
                    f"{item['constraint_id']}: {item['message']}"
                    for item in violations
                    ],
                    prefix="infeasible_context",
                )
            )

    def _raw_utility_vector(
        self,
        values: dict[str, Any],
        rule: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        utilities = utilities_from_score(
            intervention_score(rule or self.rule, values)
        )
        return {
            action["action_id"]: utilities[index]
            for index, action in enumerate(self.actions)
        }

    def _utility_vector(self, values: dict[str, Any]) -> dict[str, float]:
        return {
            action_id: round(value, 2)
            for action_id, value in self._raw_utility_vector(values).items()
        }

    def _action_index(self, action_id: str) -> int:
        for index, action in enumerate(self.actions):
            if action["action_id"] == action_id:
                return index
        raise ValueError(f"unknown action_id: {action_id}")

    def execute(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            raise ValueError("episode is complete")
        self.interactions_used += 1
        log = {
            "interaction": self.interactions_used,
            "tool": tool,
            "arguments": deepcopy(arguments),
            "batch_number": (
                self.pending_batch["batch_number"] if self.pending_batch else None
            ),
            "valid": False,
        }
        try:
            expected = {
                "calculator": {"expressions"},
                "probe_batch": {"probes"},
                "request_next_batch": set(),
                "submit_action_batch": {"batch_id", "actions"},
            }
            if tool not in expected:
                raise ValueError(f"unknown tool: {tool}")
            if tool == "request_next_batch" and not self.enable_probe_skip:
                raise ValueError("tool does not exist: request_next_batch")
            self._require_exact_keys(f"{tool} arguments", arguments, expected[tool])
            if tool == "calculator":
                if self.calculator_blocked:
                    raise ValueError(
                        "calculator was already used since the last probe or batch "
                        "action; submit every pending expression in one calculator "
                        "call (assign results to variables like X = 1 + 2 for "
                        "multi-step calculations)"
                    )
                result = calculate_many(arguments["expressions"])
                self.calculator_calls += 1
                self.calculator_expressions += len(result["results"])
                self.calculator_expression_errors += result[
                    "invalid_expression_count"
                ]
                self.calculator_blocked = True
            elif tool == "probe_batch":
                result = self._probe_batch(arguments["probes"])
                self.calculator_blocked = False
            elif tool == "request_next_batch":
                result = self._request_next_batch()
                self.calculator_blocked = False
            else:
                result = self._submit_action_batch(
                    arguments["batch_id"], arguments["actions"]
                )
                self.calculator_blocked = False
            self.repair_attempts_this_window = 0
            # Return the post-interaction tool state with every environment
            # result.  The same snapshot also remains in ``state_card``; the
            # duplicate makes the result self-contained for adapters/viewers.
            result["tool_usability"] = deepcopy(self._tool_usability())
            log.update(valid=True, result=deepcopy(result))
        except Exception as exc:
            if tool == "probe_batch":
                self.invalid_probe_calls += 1
            failure = {
                "interaction": self.interactions_used,
                "tool": tool,
                "batch_number": (
                    self.pending_batch["batch_number"] if self.pending_batch else None
                ),
                "error_type": "environment_rejection",
                "failure_reason": str(exc),
            }
            self.invalid_calls.append(failure)
            self.private_history.append({"type": "failure", **deepcopy(failure)})
            log["failure_reason"] = str(exc)
            raise
        finally:
            self.interaction_log.append(log)
        return result

    def record_invalid_response(
        self,
        error: str,
        error_type: str = "invalid_tool_schema",
        *,
        tool: str | None = None,
        interaction_already_counted: bool = False,
    ) -> dict[str, Any]:
        if not interaction_already_counted:
            self.interactions_used += 1
            failure = {
                "interaction": self.interactions_used,
                "tool": tool,
                "batch_number": (
                    self.pending_batch["batch_number"] if self.pending_batch else None
                ),
                "error_type": error_type,
                "failure_reason": str(error),
            }
            self.invalid_calls.append(failure)
            if tool == "probe_batch":
                self.invalid_probe_calls += 1
            self.private_history.append({"type": "failure", **deepcopy(failure)})
            self.interaction_log.append(
                {**deepcopy(failure), "arguments": None, "valid": False}
            )
        auto_skipped = False
        skipped_window: str | None = None
        skipped: dict[str, Any] = {}
        if not self.done:
            self.repair_attempts_this_window += 1
            repair_exhausted = (
                self.repair_attempts_this_window > self.syntax_repair_limit
            )
            if self.pending_batch is None and repair_exhausted:
                # Probe evidence is optional for completing an action window,
                # even when the setting normally requires a probe before each
                # batch.  Once the repair budget is exhausted, waive the
                # probe requirement for this window and reveal its action
                # batch so the model can still submit ordinary answers.
                skipped = self._request_next_batch()
                auto_skipped = True
                skipped_window = "probe"
            elif self.pending_batch is not None and repair_exhausted:
                skipped = self._fail_pending_batch(error)
                auto_skipped = True
                skipped_window = "action_batch"
        feedback = {
            "error": {
                "type": error_type,
                "message": str(error),
                # This flag describes whether the rejected interaction itself
                # may be repaired, not whether later episode windows remain.
                # After the second consecutive rejection the environment has
                # already advanced past this window, so retrying it is invalid.
                "retryable": not auto_skipped and not self.done,
            },
        }
        if auto_skipped:
            if skipped_window == "probe":
                if self.enable_probe_skip:
                    feedback["message"] = (
                        "You have used up your repair chance. The episode skipped "
                        "the probe step for this batch and advanced to the action "
                        "batch. Submit the loaded action batch to continue."
                    )
                else:
                    feedback["message"] = (
                        "You have used up your repair chance without submitting a "
                        "valid required probe. The probe requirement is waived for "
                        "this batch; the action batch is now loaded. Submit it to "
                        "continue."
                    )
            else:
                continuation = (
                    "The episode is now complete."
                    if self.done
                    else "Continue with the next batch."
                )
                feedback["message"] = (
                    "You have used up your repair chance. The episode skipped "
                    "the current action batch, and every action in that batch "
                    f"was recorded as unanswered. {continuation}"
                )
            feedback.update(skipped)
        else:
            feedback["repair_attempts_remaining"] = max(
                0,
                1 + self.syntax_repair_limit - self.repair_attempts_this_window,
            )
        if self.pending_batch and not self.done:
            feedback.update(
                {
                    "required_next_tool": "submit_action_batch",
                    "required_batch_id": self.pending_batch["batch_id"],
                }
            )
        # Keep invalid responses self-contained too.  This is evaluated after
        # any automatic probe/action-batch advance, so it describes the next
        # legal interaction rather than the rejected one.
        feedback["tool_usability"] = deepcopy(self._tool_usability())
        return feedback

    def record_truncated_response(self, error: str) -> dict[str, Any]:
        """Skip the current window after repeated over-length responses.

        A response cut off at max_tokens never entered the conversation, so
        unlike a repairable rejection it consumes no repair attempt: the
        window is skipped outright — the probe step when no action batch is
        pending, otherwise the pending batch is recorded as unanswered.
        """
        self.interactions_used += 1
        failure = {
            "interaction": self.interactions_used,
            "tool": None,
            "batch_number": (
                self.pending_batch["batch_number"] if self.pending_batch else None
            ),
            "error_type": "response_truncated",
            "failure_reason": str(error),
        }
        self.invalid_calls.append(failure)
        self.private_history.append({"type": "failure", **deepcopy(failure)})
        self.interaction_log.append(
            {**deepcopy(failure), "arguments": None, "valid": False}
        )
        skipped: dict[str, Any] = {}
        skipped_window: str | None = None
        if not self.done:
            if self.pending_batch is None:
                # A repeatedly truncated probe response has the same outcome
                # as a repeatedly rejected probe: reveal the action batch and
                # let the model answer it without probe evidence.
                skipped = self._request_next_batch()
                skipped_window = "probe"
            else:
                skipped = self._fail_pending_batch(error)
                skipped_window = "action_batch"
        feedback = {
            "error": {
                "type": "response_truncated",
                "message": str(error),
                # The window has already advanced, so retrying it is invalid.
                "retryable": False,
            }
        }
        if skipped_window == "probe":
            if self.enable_probe_skip:
                feedback["message"] = (
                    "Your response exceeded the maximum response length on two "
                    "consecutive attempts. The episode skipped the probe step "
                    "for this batch and advanced to the action batch. Submit "
                    "the loaded action batch to continue."
                )
            else:
                feedback["message"] = (
                    "Your response exceeded the maximum response length on two "
                    "consecutive attempts without submitting a valid required "
                    "probe. The probe requirement is waived for this batch; the "
                    "action batch is now loaded. Submit it to continue."
                )
        else:
            continuation = (
                "The episode is now complete."
                if self.done
                else "Continue with the next batch."
            )
            feedback["message"] = (
                "Your response exceeded the maximum response length on two "
                "consecutive attempts. The episode skipped the current "
                "action batch, and every action in that batch was recorded "
                f"as unanswered. {continuation}"
            )
        feedback.update(skipped)
        if self.pending_batch and not self.done:
            feedback.update(
                {
                    "required_next_tool": "submit_action_batch",
                    "required_batch_id": self.pending_batch["batch_id"],
                }
            )
        feedback["tool_usability"] = deepcopy(self._tool_usability())
        return feedback

    def _terminal_action_counts(self) -> tuple[int, int, int]:
        """Return processed, failed-unanswered, and voluntary-abstention counts."""

        failed_unanswered = sum(
            bool(item.get("unanswered")) and not bool(item.get("abstained"))
            for item in self.action_results
        )
        abstained = sum(bool(item.get("abstained")) for item in self.action_results)
        return len(self.action_results) - failed_unanswered, failed_unanswered, abstained

    def _fail_pending_batch(self, error: str) -> dict[str, Any]:
        """Mark every case in a repeatedly invalid action batch unanswered."""
        if self.pending_batch is None:
            return {}
        batch = self.pending_batch
        results = []
        phase = self.active_rule_phase
        for context_id in batch["context_ids"]:
            item = self._pool[context_id]
            raw_vector = self._raw_utility_vector(item["raw_values"], self._rule_for_phase(phase))
            optimal = max(raw_vector, key=raw_vector.__getitem__)
            row = {
                "context_id": self._private_context_id(item),
                "runtime_context_id": context_id,
                "batch_id": batch["batch_id"],
                "batch_number": batch["batch_number"],
                "phase": "static" if self.condition == "static" else phase,
                "stratum": item["stratum"],
                "action_id": None,
                "utility": 0.0,
                "oracle": {"optimal_action_id": optimal, "utilities": {
                    key: round(value, 2) for key, value in raw_vector.items()
                }},
                "correct": False,
                "stale_rule_optimal_action_id": (
                    item["private_oracle"]["static"]["optimal_action_id"]
                    if self.dataset_mode == "static"
                    else item["private_oracle"]["pre_dynamic"]["optimal_action_id"]
                ),
                "rule_flip": (
                    False
                    if self.dataset_mode == "static"
                    else item["private_oracle"]["pre_dynamic"]["optimal_action_id"]
                    != item["private_oracle"]["post_dynamic"]["optimal_action_id"]
                ),
                "regret": round(raw_vector[optimal], 2),
                "unanswered": True,
                "abstained": False,
            }
            self.action_results.append(row)
            public_row = {
                **self._public_action_identity(context_id),
                "action_id": None,
                "utility": 0.0,
            }
            if self.history_policy == "eval_case_given":
                public_row["utilities"] = deepcopy(row["oracle"]["utilities"])
            results.append(public_row)
        self.private_history.append({
            "type": "submit_action_batch",
            "batch_id": batch["batch_id"],
            "batch_number": batch["batch_number"],
            "results": deepcopy(results),
            "failure_reason": error,
            "unanswered": True,
        })
        self.pending_batch = None
        self.next_batch_index += 1
        self.repair_attempts_this_window = 0
        if self.next_batch_index == self.action_batch_count:
            self.done = True
            completed_count, unanswered_count, abstained_count = (
                self._terminal_action_counts()
            )
            self.termination = {
                # Exhausting repair attempts converts this batch into explicit
                # unanswered cases with utility 0.  The episode still reaches
                # its terminal state normally; keep the reason as a diagnostic
                # rather than classifying the run as a provider/runtime
                # failure.
                "status": "completed",
                "reason": "invalid_action_batch_after_repair",
                "completed_batch_count": self.action_batch_count,
                "completed_count": completed_count,
                "unanswered_count": unanswered_count,
                "abstained_count": abstained_count,
            }
        else:
            self.probe_used_this_window = False
        return {
            "failed_action_batch": batch["batch_id"],
            "results": results,
        }

    def _probe_batch(self, probes: Any) -> dict[str, Any]:
        if self.probe_used_this_window:
            raise ValueError(
                "a probe_batch has already been submitted for the current action batch"
            )
        if self.pending_batch is not None:
            raise ValueError(
                "action batch is already loaded; cannot probe after request_next_batch"
            )
        if not isinstance(probes, list):
            raise ValueError("probe_batch requires a non-empty probes list")
        if (
            self.least_probe_count is not None
            and len(probes) < self.least_probe_count
        ):
            raise ValueError(
                f"You must submit at least {self.least_probe_count} probe cases; "
                f"received {len(probes)}"
            )
        if not probes:
            raise ValueError("probe_batch requires a non-empty probes list")
        if len(probes) > self.max_probe_cases_per_batch:
            raise ValueError(
                f"probe_batch exceeds {self.max_probe_cases_per_batch} cases in one batch"
            )
        validated = []
        rejections: list[tuple[int, str]] = []

        def collect(probe_index: int, error: Exception) -> None:
            rejections.append((probe_index, str(error)))

        for probe_index, raw in enumerate(probes):
            if not isinstance(raw, dict):
                collect(probe_index, ValueError("each probe must be an object"))
                continue
            base_id = str(raw.get("base_context_id") or "").strip()
            # Accept source ids from pre-anonymization checkpoints while never
            # emitting them in model-facing results.
            if (
                base_id not in self.completed_contexts
                and base_id not in self.probe_contexts
            ):
                base_id = self._context_id_aliases.get(base_id, base_id)
            changes = raw.get("changes")
            values = raw.get("values")
            existing_mode = bool(base_id) or changes is not None
            new_mode = values is not None
            if existing_mode == new_mode:
                collect(
                    probe_index,
                    ValueError("each probe must use exactly one probe mode"),
                )
                continue
            if existing_mode:
                try:
                    self._require_exact_keys(
                        "existing-context probe", raw, {"base_context_id", "changes"}
                    )
                except ValueError as exc:
                    collect(probe_index, exc)
                if (
                    self.history_policy == "eval_case_hidden"
                    and base_id in self.completed_contexts
                ):
                    collect(
                        probe_index,
                        ValueError(
                            "eval-case-hidden history forbids using a completed action "
                            "context as a probe base"
                        ),
                    )
                    continue
                base = self.probe_contexts.get(base_id) or self.completed_contexts.get(base_id)
                if base is None:
                    collect(
                        probe_index,
                        ValueError(
                            "existing-context probe requires a completed action or "
                            "prior probe context"
                        ),
                    )
                    continue
                if not isinstance(changes, dict):
                    collect(
                        probe_index,
                        ValueError("existing-context probe requires a changes object"),
                    )
                    continue
                resolved = dict(base["raw_values"])
                for field_id, value in changes.items():
                    try:
                        resolved[field_id] = self._validate_value(field_id, value)
                    except Exception as exc:
                        collect(probe_index, exc)
                mode = "existing_context"
            else:
                try:
                    self._require_exact_keys("new-context probe", raw, {"values"})
                except ValueError as exc:
                    collect(probe_index, exc)
                if not isinstance(values, dict):
                    collect(
                        probe_index,
                        ValueError("new-context values must be an object"),
                    )
                    continue
                try:
                    self._require_exact_keys(
                        "new-context values", values, set(self.raw_fields)
                    )
                except ValueError as exc:
                    collect(probe_index, exc)
                resolved = {}
                for field_id in self.raw_fields:
                    if field_id not in values:
                        continue
                    try:
                        resolved[field_id] = self._validate_value(
                            field_id, values[field_id], partial_probe_hint=True
                        )
                    except Exception as exc:
                        collect(probe_index, exc)
                base_id = ""
                mode = "new_context"
            # Feasibility checks require a complete, individually valid value
            # set.  If field/schema errors were collected, report those rather
            # than producing a secondary KeyError from an incomplete payload.
            if set(resolved) == set(self.raw_fields):
                try:
                    self._validate_feasibility(resolved)
                except Exception as exc:
                    collect(probe_index, exc)
            if any(index == probe_index for index, _ in rejections):
                continue
            validated.append((base_id or None, mode, resolved))

        if rejections:
            if len(rejections) == 1:
                raise ValueError(rejections[0][1])
            details = [
                f"probe {index + 1}: {reason}" for index, reason in rejections
            ]
            raise ValueError(
                format_error_messages(
                    details,
                    prefix="probe_batch rejected for multiple reasons",
                )
            )

        results = []
        for base_id, mode, values in validated:
            self._probe_index += 1
            context_id = f"probe_{self._probe_index:03d}"
            result = {
                "context_id": context_id,
                "probe_mode": mode,
                "utilities": self._utility_vector(values),
            }
            if base_id is not None:
                result["base_context_id"] = base_id
            results.append(result)
            self.probe_contexts[context_id] = {"raw_values": deepcopy(values)}
        self._probe_count += len(results)
        self.probe_used_this_window = True
        self.repair_attempts_this_window = 0
        batch_index = self.next_batch_index
        self.probe_batches_by_batch[batch_index] += 1
        self.probe_cases_by_batch[batch_index] += len(results)
        event = {
            "type": "probe_batch",
            "batch_number": self.next_batch_index,
            "probe_cases_used": len(results),
            "cumulative_utility": self.action_utility,
            "cumulative_probes": self.probe_count,
            "results": results,
        }
        self.public_history.append(deepcopy(event))
        self.private_history.append(
            {
                **deepcopy(event),
                "interaction": self.interactions_used,
                "private_oracle": deepcopy(results),
                "regret": None,
                "failure_reason": None,
            }
        )
        self._load_next_batch()
        pending_cases = [
            self._public_case(self._pool[context_id])
            for context_id in self.pending_batch["context_ids"]
        ]
        return {
            "results": results,
            "cumulative_utility": self.action_utility,
            "cumulative_probes": self.probe_count,
            "pending_action_batch": {
                "batch_id": self.pending_batch["batch_id"],
                "batch_number": self.pending_batch["batch_number"],
                "cases": pending_cases,
            },
        }

    def _request_next_batch(self) -> dict[str, Any]:
        if self.pending_batch is not None:
            raise ValueError("action batch is already loaded")
        self._load_next_batch()
        self.repair_attempts_this_window = 0
        pending_cases = [
            self._public_case(self._pool[context_id])
            for context_id in self.pending_batch["context_ids"]
        ]
        return {
            "pending_action_batch": {
                "batch_id": self.pending_batch["batch_id"],
                "batch_number": self.pending_batch["batch_number"],
                "cases": pending_cases,
            }
        }

    def _submit_action_batch(
        self, batch_id: Any, actions: Any
    ) -> dict[str, Any]:
        if self.pending_batch is None:
            raise ValueError("no action batch is pending")
        if batch_id != self.pending_batch["batch_id"]:
            raise ValueError(
                f"batch_id must equal current pending batch {self.pending_batch['batch_id']}"
            )
        if not isinstance(actions, dict):
            raise ValueError("actions must be an object")
        expected_ids = set(self.pending_batch["context_ids"])
        public_id_to_runtime = {
            str(context_id): str(context_id)
            for context_id in expected_ids
        }
        canonical_keys = {
            public_id_to_runtime.get(
                str(context_id),
                self._context_id_aliases.get(str(context_id), str(context_id)),
            )
            for context_id in actions
        }
        unknown = sorted(canonical_keys - expected_ids)
        abstained_ids = sorted(expected_ids - canonical_keys)
        key_rejections = []
        if unknown:
            key_rejections.append(f"actions has unexpected key(s): {', '.join(unknown)}")
        if abstained_ids and not self.allow_abstention:
            key_rejections.append(
                f"actions has missing key(s): {', '.join(abstained_ids)}"
            )
        selected: dict[str, str] = {}
        duplicate_rejections = []
        for context_id, action_id in actions.items():
            raw_id = str(context_id)
            runtime_id = public_id_to_runtime.get(
                raw_id,
                self._context_id_aliases.get(raw_id, raw_id),
            )
            if runtime_id not in expected_ids:
                continue
            if runtime_id in selected:
                duplicate_rejections.append(f"actions has duplicate key for {runtime_id}")
                continue
            selected[runtime_id] = str(action_id)
        invalid_action_rejections = []
        for action_id in selected.values():
            try:
                self._action_index(action_id)
            except ValueError as exc:
                invalid_action_rejections.append(str(exc))
        all_rejections = key_rejections + duplicate_rejections + invalid_action_rejections
        if all_rejections:
            raise ValueError(format_error_messages(all_rejections))

        batch_number = self.pending_batch["batch_number"]
        public_results = []
        private_results = []
        for context_id in self.pending_batch["context_ids"]:
            item = self._pool[context_id]
            rule_phase = self.active_rule_phase
            if self.condition == "dynamic" and str(item["phase"]) != rule_phase:
                raise ValueError("action case phase does not match active rule")
            raw_vector = self._raw_utility_vector(
                item["raw_values"], self._rule_for_phase(rule_phase)
            )
            vector = {
                action_id: round(value, 2)
                for action_id, value in raw_vector.items()
            }
            optimal_action_id = max(raw_vector, key=raw_vector.__getitem__)
            abstained = context_id not in selected
            if abstained:
                action_id = None
                utility = 0.0
                regret = round(raw_vector[optimal_action_id], 2)
            else:
                action_id = selected[context_id]
                utility = vector[action_id]
                regret = round(
                    raw_vector[optimal_action_id] - raw_vector[action_id], 2
                )
            public = {
                **self._public_action_identity(context_id),
                "action_id": action_id,
                "utility": utility,
            }
            if self.history_policy == "eval_case_given":
                # The full-memory condition aligns action feedback with probe
                # feedback: the agent learns every action's utility on each
                # submitted case, not only the utility it earned.
                public["utilities"] = vector
            private = {
                **public,
                "context_id": self._private_context_id(item),
                "runtime_context_id": context_id,
                "batch_id": batch_id,
                "batch_number": batch_number,
                "phase": (
                    "static" if self.condition == "static" else rule_phase
                ),
                "stratum": item["stratum"],
                "oracle": {
                    "optimal_action_id": optimal_action_id,
                    "utilities": vector,
                },
                "correct": not abstained and action_id == optimal_action_id,
                "stale_rule_optimal_action_id": (
                    item["private_oracle"]["static"]["optimal_action_id"]
                    if self.dataset_mode == "static"
                    else item["private_oracle"]["pre_dynamic"]["optimal_action_id"]
                ),
                "rule_flip": (
                    False
                    if self.dataset_mode == "static"
                    else item["private_oracle"]["pre_dynamic"]["optimal_action_id"]
                    != item["private_oracle"]["post_dynamic"]["optimal_action_id"]
                ),
                "regret": regret,
                "unanswered": abstained,
                "abstained": abstained,
            }
            public_results.append(public)
            private_results.append(private)
            self.action_results.append(private)
            self.completed_contexts[context_id] = {
                "raw_values": deepcopy(item["raw_values"])
            }

        event = {
            "type": "submit_action_batch",
            "batch_id": batch_id,
            "batch_number": batch_number,
            "results": public_results,
            "batch_utility": round(
                sum(item["utility"] for item in public_results), 2
            ),
            "cumulative_utility": self.action_utility,
            "cumulative_probes": self.probe_count,
            "abstained_count": len(abstained_ids),
        }
        self.public_history.append(deepcopy(event))
        self.private_history.append(
            {
                **deepcopy(event),
                "interaction": self.interactions_used,
                "results": private_results,
                "private_oracle": [
                    item["oracle"] for item in private_results
                ],
                "regret": round(
                    sum(item["regret"] for item in private_results), 2
                ),
                "failure_reason": None,
            }
        )
        self.pending_batch = None
        self.next_batch_index += 1
        self.repair_attempts_this_window = 0
        if self.next_batch_index == self.action_batch_count:
            self.done = True
            completed_count, unanswered_count, abstained_count = (
                self._terminal_action_counts()
            )
            self.termination = {
                # Unanswered cases are a supported outcome (including model
                # abstentions and exhausted repairs), not a failed episode.
                "status": "completed",
                "reason": (
                    "invalid_action_batch_after_repair"
                    if unanswered_count
                    else "all_action_batches_completed"
                ),
                "completed_batch_count": self.action_batch_count,
                "completed_count": completed_count,
                "unanswered_count": unanswered_count,
                "abstained_count": abstained_count,
            }
        else:
            self.probe_used_this_window = False
        return {
            "batch_id": batch_id,
            "results": public_results,
            "batch_utility": event["batch_utility"],
            "cumulative_utility": self.action_utility,
            "cumulative_probes": self.probe_count,
        }

    def _finalize_failure(
        self, reason: str, error_message: str | None
    ) -> None:
        answered = {item["context_id"] for item in self.action_results}
        for item in self.episode["cases"]:
            if self._private_context_id(item) in answered:
                continue
            rule_phase = (
                "static"
                if self.dataset_mode == "static"
                else (
                    "pre_dynamic"
                    if self.condition == "static" or item["phase"] == "pre_dynamic"
                    else "post_dynamic"
                )
            )
            raw_vector = self._raw_utility_vector(
                item["raw_values"], self._rule_for_phase(rule_phase)
            )
            vector = {
                action_id: round(value, 2)
                for action_id, value in raw_vector.items()
            }
            optimal_action_id = max(raw_vector, key=raw_vector.__getitem__)
            self.action_results.append(
                {
                    "context_id": self._private_context_id(item),
                    "runtime_context_id": item["context_id"],
                    "batch_id": None,
                    "batch_number": None,
                    "phase": (
                        "static" if self.condition == "static" else rule_phase
                    ),
                    "stratum": item["stratum"],
                    "action_id": None,
                    "utility": 0.0,
                    "oracle": {
                        "optimal_action_id": optimal_action_id,
                        "utilities": vector,
                    },
                    "correct": False,
                    "stale_rule_optimal_action_id": (
                        item["private_oracle"]["static"]["optimal_action_id"]
                        if self.dataset_mode == "static"
                        else item["private_oracle"]["pre_dynamic"]["optimal_action_id"]
                    ),
                    "rule_flip": (
                        False
                        if self.dataset_mode == "static"
                        else item["private_oracle"]["pre_dynamic"]["optimal_action_id"]
                        != item["private_oracle"]["post_dynamic"]["optimal_action_id"]
                    ),
                    "regret": round(raw_vector[optimal_action_id], 2),
                    "unanswered": True,
                    "abstained": False,
                }
            )
        self.done = True
        self.termination = {
            "status": "failed",
            "reason": reason,
            "error_message": error_message,
            "completed_batch_count": self.next_batch_index,
            "completed_count": len(answered),
            "unanswered_count": len(self._pool) - len(answered),
        }

    def terminate_unanswered(
        self,
        *,
        reason: str,
        stage: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        del stage, error_type
        if not self.done:
            self._finalize_failure(reason, error_message)

    def artifact_state(self) -> dict[str, Any]:
        return {
            "episode_seed": self.episode["seed"],
            "condition": self.condition,
            "history_policy": self.history_policy,
            "batches": deepcopy(self.episode["batches"]),
            "interactions_used": self.interactions_used,
            "completed_batch_count": self.next_batch_index,
            "active_rule_phase": self.active_rule_phase,
            "dynamic_activated": (
                self.condition == "dynamic"
                and self.next_batch_index >= self.after_action_batch
            ),
            "probe_count": self.probe_count,
            "action_utility": self.action_utility,
            "allow_abstention": self.allow_abstention,
            "enable_probe_skip": self.enable_probe_skip,
            "least_probe_count": self.least_probe_count,
            "probe_batches_by_batch": deepcopy(self.probe_batches_by_batch),
            "probe_cases_by_batch": deepcopy(self.probe_cases_by_batch),
            "invalid_probe_calls": self.invalid_probe_calls,
            "calculator_calls": self.calculator_calls,
            "calculator_expressions": self.calculator_expressions,
            "calculator_expression_errors": self.calculator_expression_errors,
            "invalid_calls": deepcopy(self.invalid_calls),
            "interaction_log": deepcopy(self.interaction_log),
        }


def replay_episode(
    episode: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    interaction_policy: dict[str, Any],
) -> dict[str, Any]:
    env = BenchmarkEnvironment(episode, interaction_policy=interaction_policy)
    for call in calls:
        try:
            env.execute(call["tool"], deepcopy(call.get("arguments") or {}))
        except ValueError:
            pass
        if env.done:
            break
    return {
        "termination": deepcopy(env.termination),
        "public_history": deepcopy(env.public_history),
        "private_history": deepcopy(env.private_history),
        "action_results": deepcopy(env.action_results),
        "private_state": env.artifact_state(),
    }
