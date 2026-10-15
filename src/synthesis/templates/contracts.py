from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:  # editable source trees can still run --check before install
    Draft202012Validator = None  # type: ignore[assignment]


SCHEMAS = {
    "scenario_expansion": "scenario_expansion.schema.json",
    "scenario_generation": "scenario_generation.schema.json",
    "scenario_cores": "scenario_cores.schema.json",
    "field_catalog": "field_catalog.schema.json",
    "field_semantics": "field_semantics.schema.json",
    "field_parameters": "field_parameters.schema.json",
    "field_constraints": "field_constraints.schema.json",
    "lexicographic_orders": "lexicographic_orders.schema.json",
    "replacements": "replacements.schema.json",
    "field_catalog_local_audit": "field_catalog_local_audit.schema.json",
    "field_catalog_set_audit": "field_catalog_set_audit.schema.json",
    "semantic_audit": "semantic_audit.schema.json",
    "field_semantics_audit": "field_semantics_audit.schema.json",
    "overall_audit": "overall_audit.schema.json",
    "component_patch": "component_patch.schema.json",
    "field_constraints_repair": "field_constraints_repair.schema.json",
}


class ContractError(ValueError):
    pass


class Contracts:
    def __init__(self, canonical: Path, stages: Path) -> None:
        self._paths = {name: stages / filename for name, filename in SCHEMAS.items()}
        self._paths["canonical"] = canonical
        self._schemas: dict[str, dict[str, Any]] = {}
        self._validators: dict[str, Draft202012Validator] = {}

    def schema(self, name: str) -> dict[str, Any]:
        if name not in self._paths:
            raise KeyError(f"unknown schema {name}")
        if name not in self._schemas:
            value = json.loads(self._paths[name].read_text())
            if Draft202012Validator is not None:
                Draft202012Validator.check_schema(value)
            self._schemas[name] = value
        return self._schemas[name]

    def validate(self, name: str, value: Any) -> None:
        schema = self.schema(name)
        if Draft202012Validator is not None:
            if name not in self._validators:
                self._validators[name] = Draft202012Validator(schema)
            errors = sorted(self._validators[name].iter_errors(value), key=lambda e: list(e.path))
            rendered = [f"/{'/'.join(map(str, error.absolute_path))}: {error.message}" for error in errors[:20]]
        else:
            rendered = _fallback_errors(schema, value, schema)
        if rendered:
            raise ContractError(f"{name} contract failed: " + "; ".join(rendered[:20]))

    def validate_template(self, value: Any) -> None:
        """Validate one assembled template against the canonical template definition."""
        canonical = self.schema("canonical")
        schema = {
            "$schema": canonical.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
            "$defs": deepcopy(canonical["$defs"]),
            **deepcopy(canonical["$defs"]["template"]),
        }
        if Draft202012Validator is not None:
            validator = Draft202012Validator(schema)
            errors = sorted(
                validator.iter_errors(value), key=lambda error: list(error.path)
            )
            rendered = [
                f"/{'/'.join(map(str, error.absolute_path))}: {error.message}"
                for error in errors[:20]
            ]
        else:
            rendered = _fallback_errors(schema, value, schema)
        if rendered:
            raise ContractError(
                "canonical template contract failed: " + "; ".join(rendered[:20])
            )

    def bound_schema(
        self, name: str, enum_bindings: dict[str, tuple[Any, ...]]
    ) -> dict[str, Any]:
        """Return a per-call schema whose references are bound to known identifiers."""
        schema = deepcopy(self.schema(name))
        for pointer, values in enum_bindings.items():
            if not values:
                raise ContractError(f"empty enum binding for {name}{pointer}")
            target: Any = schema
            try:
                for token in pointer.removeprefix("/").split("/"):
                    token = token.replace("~1", "/").replace("~0", "~")
                    target = target[int(token)] if isinstance(target, list) else target[token]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ContractError(f"invalid schema binding {name}{pointer}") from exc
            if not isinstance(target, dict):
                raise ContractError(f"schema binding target is not an object: {name}{pointer}")
            target["enum"] = list(values)
        if Draft202012Validator is not None:
            Draft202012Validator.check_schema(schema)
        return schema

    def check_all(self) -> None:
        for name in self._paths:
            self.schema(name)


def provider_schema(value: Any) -> Any:
    """Project the local contract onto OpenAI strict-schema capabilities.

    Local validation retains keywords such as ``uniqueItems`` that the
    provider rejects in response_format. Program validators still enforce the
    omitted invariants after parsing.
    """
    if isinstance(value, dict):
        return {
            key: provider_schema(child)
            for key, child in value.items()
            if key not in {"uniqueItems"}
        }
    if isinstance(value, list):
        return [provider_schema(child) for child in value]
    return value


def _fallback_errors(schema: dict[str, Any], value: Any, root: dict[str, Any], path: str = "") -> list[str]:
    """Validate the JSON-Schema subset used by synthesis when jsonschema is absent."""
    if "$ref" in schema:
        target: Any = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return _fallback_errors(target, value, root, path)
    if "anyOf" in schema:
        alternatives = [
            _fallback_errors(option, value, root, path)
            for option in schema["anyOf"]
        ]
        if any(not errors for errors in alternatives):
            return []
        return [f"{path or '/'}: value does not match any allowed schema"]
    errors: list[str] = []
    label = path or "/"
    if "const" in schema and value != schema["const"]:
        errors.append(f"{label}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label}: value is not in enum")
    expected = schema.get("type")
    valid_type = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }.get(expected, lambda item: True)
    if expected and not valid_type(value):
        return [f"{label}: expected {expected}"]
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            errors.append(f"{label}: string is too short")
        if schema.get("pattern") and re.search(schema["pattern"], value) is None:
            errors.append(f"{label}: string does not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{label}: number is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{label}: number is above maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{label}: number is not above exclusive minimum")
    if isinstance(value, dict):
        required = set(schema.get("required") or ())
        missing = required - set(value)
        if missing:
            errors.append(f"{label}: missing {sorted(missing)}")
        if len(value) < int(schema.get("minProperties", 0)):
            errors.append(f"{label}: too few properties")
        if "maxProperties" in schema and len(value) > int(schema["maxProperties"]):
            errors.append(f"{label}: too many properties")
        properties = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            child = properties.get(key)
            if child is None:
                if additional is False:
                    errors.append(f"{label}: unexpected property {key}")
                    continue
                if isinstance(additional, dict):
                    child = additional
            if isinstance(child, dict):
                errors.extend(_fallback_errors(child, item, root, f"{path}/{key}"))
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            errors.append(f"{label}: too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{label}: too many items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{label}: items are not unique")
        prefix = schema.get("prefixItems") or []
        for index, child in enumerate(prefix[:len(value)]):
            errors.extend(_fallback_errors(child, value[index], root, f"{path}/{index}"))
        items = schema.get("items", True)
        start = len(prefix)
        if items is False and len(value) > start:
            errors.append(f"{label}: additional array items are forbidden")
        elif isinstance(items, dict):
            for index, item in enumerate(value[start:] if prefix else value, start if prefix else 0):
                errors.extend(_fallback_errors(items, item, root, f"{path}/{index}"))
    return errors
