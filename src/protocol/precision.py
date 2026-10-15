"""Shared numeric presentation rules for model-facing benchmark values."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


# Decimal inputs are part of the agent-facing protocol.  Dataset synthesis may
# retain more digits internally for calibration, but values sent to the agent
# and values accepted as probe inputs use this presentation precision.
DECIMAL_PLACES = 2


def has_at_most_decimal_places(value: Any, places: int = DECIMAL_PLACES) -> bool:
    """Return whether a JSON numeric value has no more than ``places`` decimals."""

    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    if not parsed.is_finite():
        return False
    exponent = parsed.as_tuple().exponent
    return isinstance(exponent, int) and exponent >= -places


def round_field_value(value: Any, field_type: str) -> int | float | Any:
    """Return a field value in the canonical agent-facing representation."""

    if field_type == "integer":
        return int(value)
    if field_type == "decimal":
        return round(float(value), DECIMAL_PLACES)
    return value


def round_field_values(
    values: dict[str, Any], field_schema: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Round known numeric fields without changing field order or unknown keys."""

    return {
        field_id: round_field_value(
            value, str(field_schema.get(field_id, {}).get("type") or "")
        )
        for field_id, value in values.items()
    }
