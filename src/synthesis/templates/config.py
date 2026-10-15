from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_PROMPTS = (
    "scenarios/system.txt", "scenarios/user.j2", "scenarios/audit.j2", "scenarios/repair.j2",
    "fields/discover_system.txt", "fields/discover_user.j2", "fields/discover_audit.j2",
    "fields/discover_set_audit.j2", "fields/discover_assignment_repair.j2",
    "fields/semantics_system.txt", "fields/semantics_user.j2", "fields/semantics_audit.j2",
    "fields/parameters_system.txt", "fields/parameters_user.j2", "fields/parameters_audit.j2",
    "fields/repair.j2", "fields/semantics_repair.j2",
    "fields/parameters_repair.j2",
    "relations/field_constraints_system.txt", "relations/field_constraints_user.j2",
    "relations/field_constraints_audit.j2", "relations/field_constraints_repair.j2",
    "relations/lexicographic_orders_system.txt", "relations/lexicographic_orders_user.j2",
    "relations/lexicographic_orders_audit.j2", "relations/lexicographic_orders_repair.j2",
    "dynamics/replacements_system.txt", "dynamics/replacements_user.j2", "dynamics/replacements_audit.j2",
    "dynamics/repair.j2", "audit/system.txt", "audit/domain_diversity.j2",
)


@dataclass(frozen=True)
class Paths:
    skeleton: Path
    canonical_schema: Path
    stage_schemas: Path
    prompts: Path
    canonical: Path
    audit: Path
    manifest: Path
    runs: Path


@dataclass(frozen=True)
class AgentSettings:
    max_completion_tokens: int
    effort: str
    output_format: str
    concurrency: int
    request_timeout_seconds: float
    request_retry_limit: int
    api_mode: str
    background_poll_interval_seconds: float
    limit: int = 0


@dataclass(frozen=True)
class BatchingSettings:
    context_window_tokens: int
    safety_margin_tokens: int
    max_field_audit_per_batch: int
    max_field_parameters_per_batch: int
    estimated_field_parameter_tokens: int


@dataclass(frozen=True)
class FieldParameterSettings:
    beta_concentration_range: tuple[float, float]


@dataclass(frozen=True)
class TaskAction:
    display_name: str
    description: str

    def prompt_value(self) -> dict[str, str]:
        return {
            "display_name": self.display_name,
            "description": self.description,
        }


@dataclass(frozen=True)
class TaskBlueprint:
    task_id: str
    description: str
    agent_role: str
    actions: tuple[TaskAction, ...]

    def prompt_value(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent_role": self.agent_role,
            "actions": [action.prompt_value() for action in self.actions],
        }


@dataclass(frozen=True)
class SynthesisConfig:
    source: Path
    model_config: Path
    audit_model_config: Path
    paths: Paths
    domains: tuple[str, ...]
    domain_descriptions: dict[str, str]
    blueprint_path: Path | None
    task_blueprints: dict[str, tuple[TaskBlueprint, ...]]
    scenario_batch_size: int
    minimum_reversal_fields: int
    generation: AgentSettings
    audit: AgentSettings
    repair: AgentSettings
    batching: BatchingSettings
    field_parameters: FieldParameterSettings
    parse_retry_limit: int
    catalog_set_drop_limit: int
    semantics_drop_limit: int
    template_retry_limit: int
    scenario_refill_limit: int


def _path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _agent(raw: dict[str, Any], *, repair: bool = False) -> AgentSettings:
    required = {
        "max_completion_tokens", "effort", "output_format", "concurrency",
        "request_timeout_seconds", "request_retry_limit",
        "api_mode", "background_poll_interval_seconds",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"agent settings missing {sorted(missing)}")
    output_format = str(raw["output_format"])
    if output_format != "json_schema":
        raise ValueError("template synthesis requires json_schema output")
    concurrency = int(raw["concurrency"])
    if concurrency < 1:
        raise ValueError("agent concurrency must be positive")
    limit = int(raw.get("limit", 0))
    if repair and limit < 0:
        raise ValueError("repair limit cannot be negative")
    max_completion_tokens = int(raw["max_completion_tokens"])
    if max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be positive")
    request_timeout_seconds = float(raw["request_timeout_seconds"])
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    request_retry_limit = int(raw["request_retry_limit"])
    if request_retry_limit < 0:
        raise ValueError("request_retry_limit cannot be negative")
    api_mode = str(raw["api_mode"])
    if api_mode != "responses_background":
        raise ValueError("template synthesis requires responses_background API mode")
    background_poll_interval_seconds = float(raw["background_poll_interval_seconds"])
    if background_poll_interval_seconds <= 0:
        raise ValueError("background poll interval must be positive")
    return AgentSettings(
        max_completion_tokens=max_completion_tokens,
        effort=str(raw["effort"]),
        output_format=output_format,
        concurrency=concurrency,
        request_timeout_seconds=request_timeout_seconds,
        request_retry_limit=request_retry_limit,
        api_mode=api_mode,
        background_poll_interval_seconds=background_poll_interval_seconds,
        limit=limit,
    )


def load_task_blueprints(
    path: str | Path,
    domains: tuple[str, ...],
) -> dict[str, tuple[TaskBlueprint, ...]]:
    source = _path(path)
    if not source.is_file():
        raise FileNotFoundError(f"task blueprint file does not exist: {source}")
    raw = yaml.safe_load(source.read_text()) or {}
    if (
        not isinstance(raw, dict)
        or set(raw) != {"domains"}
        or not isinstance(raw["domains"], dict)
    ):
        raise ValueError("task blueprints must define exactly one domains mapping")
    configured = raw["domains"]
    if set(configured) != set(domains):
        raise ValueError(
            "task blueprint domains do not match template synthesis domains: "
            f"expected={list(domains)}, observed={list(configured)}"
        )
    output: dict[str, tuple[TaskBlueprint, ...]] = {}
    observed_task_ids: set[str] = set()
    observed_roles: set[str] = set()
    expected_keys = {"task_id", "description", "agent_role", "actions"}
    for domain in domains:
        rows = configured[domain]
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"task blueprints for {domain} must be a non-empty list"
            )
        parsed = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != expected_keys:
                raise ValueError(
                    f"task blueprint {domain}[{index}] must define exactly "
                    f"{sorted(expected_keys)}"
                )
            values = {
                key: str(row[key]).strip()
                for key in ("task_id", "description", "agent_role")
            }
            empty = [key for key, value in values.items() if not value]
            if empty:
                raise ValueError(
                    f"task blueprint {domain}[{index}] has empty {sorted(empty)}"
                )
            action_rows = row["actions"]
            if not isinstance(action_rows, list) or len(action_rows) != 4:
                raise ValueError(
                    f"task blueprint {domain}[{index}] must define exactly 4 actions"
                )
            actions = []
            for action_index, action in enumerate(action_rows):
                if (
                    not isinstance(action, dict)
                    or set(action) != {"display_name", "description"}
                ):
                    raise ValueError(
                        f"task blueprint {domain}[{index}] action {action_index} "
                        "must define exactly display_name and description"
                    )
                action_values = {
                    key: str(action[key]).strip()
                    for key in ("display_name", "description")
                }
                if not all(action_values.values()):
                    raise ValueError(
                        f"task blueprint {domain}[{index}] action {action_index} "
                        "has an empty value"
                    )
                actions.append(TaskAction(**action_values))
            action_names = [action.display_name for action in actions]
            if len(action_names) != len(set(action_names)):
                raise ValueError(
                    f"task blueprint {domain}[{index}] action display names "
                    "must be unique"
                )
            task_id = values["task_id"]
            role = values["agent_role"]
            if task_id in observed_task_ids:
                raise ValueError(f"duplicate task blueprint task_id: {task_id}")
            if role in observed_roles:
                raise ValueError(f"duplicate task blueprint agent_role: {role}")
            observed_task_ids.add(task_id)
            observed_roles.add(role)
            parsed.append(TaskBlueprint(**values, actions=tuple(actions)))
        output[domain] = tuple(parsed)
    return output


def load_config(path: str | Path = "configs/synthesis/templates.yaml") -> SynthesisConfig:
    source = _path(path).resolve()
    raw = yaml.safe_load(source.read_text()) or {}
    path_raw = raw.get("paths") or {}
    required_paths = {
        "skeleton", "canonical_schema", "stage_schemas", "prompts",
        "canonical", "audit", "manifest", "runs",
    }
    missing = required_paths - set(path_raw)
    if missing:
        raise ValueError(f"synthesis paths missing {sorted(missing)}")
    paths = Paths(**{key: _path(path_raw[key]) for key in required_paths})
    domain_raw = raw.get("domains") or {}
    if not isinstance(domain_raw, dict) or not domain_raw:
        raise ValueError("domains must be a non-empty mapping of IDs to descriptions")
    malformed_domains = [
        str(domain_id) for domain_id, value in domain_raw.items()
        if not isinstance(value, dict) or set(value) != {"description"}
    ]
    if malformed_domains:
        raise ValueError(
            "each domain must define exactly one description: "
            f"{sorted(malformed_domains)}"
        )
    domain_descriptions = {
        str(domain_id): str(value.get("description") or "").strip()
        for domain_id, value in domain_raw.items()
    }
    if not all(domain_descriptions.values()):
        raise ValueError("every configured domain must define a non-empty description")
    domains = tuple(domain_descriptions)
    scenario_batch_size = int(raw.get("scenario_batch_size", 0))
    if scenario_batch_size < 1:
        raise ValueError("scenario_batch_size must be positive")
    blueprint_value = raw.get("blueprint_path")
    if blueprint_value is not None and not isinstance(blueprint_value, (str, Path)):
        raise ValueError("blueprint_path must be null or a file path")
    if blueprint_value is not None and not str(blueprint_value).strip():
        raise ValueError("blueprint_path must be null or a non-empty file path")
    blueprint_path = _path(blueprint_value).resolve() if blueprint_value is not None else None
    task_blueprints = (
        load_task_blueprints(blueprint_path, domains)
        if blueprint_path is not None else {}
    )
    minimum_reversal_fields = int(raw.get("minimum_reversal_fields", 0))
    if minimum_reversal_fields < 1:
        raise ValueError("minimum_reversal_fields must be positive")
    batching_raw = raw.get("batching") or {}
    batching = BatchingSettings(**{
        key: int(batching_raw[key])
        for key in BatchingSettings.__dataclass_fields__
    })
    if batching.context_window_tokens <= batching.safety_margin_tokens:
        raise ValueError("batching context window must exceed its safety margin")
    if batching.max_field_audit_per_batch < 1:
        raise ValueError("field audit batch limit must be positive")
    if batching.max_field_parameters_per_batch < 1:
        raise ValueError("field parameter batch limit must be positive")
    if batching.estimated_field_parameter_tokens < 1:
        raise ValueError("batching per-item token estimates must be positive")
    field_parameters_raw = raw.get("field_parameters") or {}
    concentration_range = field_parameters_raw.get("beta_concentration_range")
    if not isinstance(concentration_range, list) or len(concentration_range) != 2:
        raise ValueError("field parameter beta_concentration_range must have two values")
    concentration_minimum, concentration_maximum = map(float, concentration_range)
    if not 0 < concentration_minimum < concentration_maximum:
        raise ValueError("field parameter beta concentration range must be positive and ordered")
    field_parameters = FieldParameterSettings(
        beta_concentration_range=(concentration_minimum, concentration_maximum),
    )
    parse_retry_limit = int(raw.get("parse_retry_limit", 2))
    if parse_retry_limit < 0:
        raise ValueError("parse_retry_limit cannot be negative")
    catalog_set_drop_limit = int(raw.get("catalog_set_drop_limit", 1))
    if catalog_set_drop_limit < 0:
        raise ValueError("catalog set drop limit cannot be negative")
    semantics_drop_limit = int(raw.get("semantics_drop_limit", 1))
    if semantics_drop_limit < 0:
        raise ValueError("semantics_drop_limit cannot be negative")
    template_retry_limit = int(raw.get("template_retry_limit", 2))
    if template_retry_limit < 0:
        raise ValueError("template_retry_limit cannot be negative")
    scenario_refill_limit = int(raw.get("scenario_refill_limit", 8))
    if scenario_refill_limit < 0:
        raise ValueError("scenario_refill_limit cannot be negative")
    return SynthesisConfig(
        source=source,
        model_config=_path(raw["model_config"]),
        audit_model_config=_path(raw.get("audit_model_config", raw["model_config"])),
        paths=paths,
        domains=domains,
        domain_descriptions=domain_descriptions,
        blueprint_path=blueprint_path,
        task_blueprints=task_blueprints,
        scenario_batch_size=scenario_batch_size,
        minimum_reversal_fields=minimum_reversal_fields,
        generation=_agent(raw.get("generation") or {}),
        audit=_agent(raw.get("audit") or {}),
        repair=_agent(raw.get("repair") or {}, repair=True),
        batching=batching,
        field_parameters=field_parameters,
        parse_retry_limit=parse_retry_limit,
        catalog_set_drop_limit=catalog_set_drop_limit,
        semantics_drop_limit=semantics_drop_limit,
        template_retry_limit=template_retry_limit,
        scenario_refill_limit=scenario_refill_limit,
    )


def validate_resources(config: SynthesisConfig) -> None:
    required = [
        config.model_config, config.audit_model_config,
        config.paths.skeleton,
        config.paths.canonical_schema, config.paths.stage_schemas, config.paths.prompts,
    ]
    if config.blueprint_path is not None:
        required.append(config.blueprint_path)
    required.extend(config.paths.prompts / path for path in REQUIRED_PROMPTS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing synthesis resources: {missing}")
    skeleton = yaml.safe_load(config.paths.skeleton.read_text()) or {}
    skeleton_concentration_range = tuple(
        map(float, skeleton["defaults"]["sampling"]["beta_concentration_range"])
    )
    if skeleton_concentration_range != config.field_parameters.beta_concentration_range:
        raise ValueError(
            "skeleton beta_concentration_range must match template synthesis config"
        )
