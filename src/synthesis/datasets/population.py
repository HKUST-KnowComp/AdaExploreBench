"""Bounded natural-population sampling for dataset synthesis."""

from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import math
import os
import random
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

from protocol.constraints import constraint_violations
from .template_pool import TemplateSelection

_NORMAL = NormalDist()
_EPSILON = 1e-12


class PopulationBudgetExhausted(RuntimeError):
    """The fallback draw budget could not fill a legal pool."""


def _beta_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
    result = d
    for index in range(1, 201):
        doubled = 2 * index
        coefficient = index * (b - index) * x / ((qam + doubled) * (a + doubled))
        d, c = 1.0 + coefficient * d, 1.0 + coefficient / c
        d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
        c = c if abs(c) >= 1e-300 else 1e-300
        result *= d * c
        coefficient = -(a + index) * (qab + index) * x / ((a + doubled) * (qap + doubled))
        d, c = 1.0 + coefficient * d, 1.0 + coefficient / c
        d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
        c = c if abs(c) >= 1e-300 else 1e-300
        change = d * c
        result *= change
        if abs(change - 1.0) < 2e-13:
            return result
    raise ArithmeticError("beta continued fraction did not converge")


def _beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def _beta_inverse(probability: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(36):
        middle = (low + high) / 2.0
        if _beta_cdf(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


@dataclass(frozen=True)
class _BetaLookup:
    values: tuple[float, ...]

    def inverse(self, probability: float) -> float:
        position = min(1.0, max(0.0, probability)) * (len(self.values) - 1)
        left = min(int(position), len(self.values) - 2)
        weight = position - left
        return self.values[left] * (1.0 - weight) + self.values[left + 1] * weight


_LOOKUPS: dict[tuple[float, float, int], _BetaLookup] = {}
_CONDITIONAL_TABLES: dict[tuple[Any, ...], tuple[tuple[float, ...], float]] = {}
_LOOKUP_CACHE_DIRECTORY: Path | None = None


def configure_population_cache(directory: Path | None) -> None:
    global _LOOKUP_CACHE_DIRECTORY
    _LOOKUP_CACHE_DIRECTORY = (
        None if directory is None else Path(directory) / "population_beta_lookups"
    )


def _build_lookup(shape: tuple[float, float], knots: int) -> _BetaLookup:
    values = [
        0.0,
        *(
            _beta_inverse(index / knots, *shape)
            for index in range(1, knots)
        ),
        1.0,
    ]
    return _BetaLookup(tuple(values))


def _cached_lookup(shape: tuple[float, float], knots: int) -> _BetaLookup:
    if _LOOKUP_CACHE_DIRECTORY is None:
        return _build_lookup(shape, knots)
    identity = json.dumps([*shape, knots], separators=(",", ":"))
    digest = hashlib.sha256(identity.encode()).hexdigest()
    directory = _LOOKUP_CACHE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.bin"
    lock_path = directory / f"{digest}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            values = array("d")
            try:
                with path.open("rb") as stream:
                    values.fromfile(stream, knots + 1)
            except EOFError:
                values = array("d")
            if len(values) == knots + 1:
                return _BetaLookup(tuple(values))
        lookup = _build_lookup(shape, knots)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            array("d", lookup.values).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return lookup


def _lookup(shape: tuple[float, float], knots: int) -> _BetaLookup:
    key = (*shape, knots)
    if key not in _LOOKUPS:
        _LOOKUPS[key] = _cached_lookup(shape, knots)
    return _LOOKUPS[key]


@dataclass
class BaseProposals:
    field_ids: tuple[str, ...]
    seed: int
    uniforms: list[tuple[float, ...]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def ensure(self, count: int) -> None:
        width = len(self.field_ids) + 3
        while len(self.uniforms) < count:
            self.uniforms.append(tuple(self._rng.random() for _ in range(width)))


@dataclass
class PopulationPool:
    records: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    raw_draw_count: int = 0

    @property
    def states(self) -> list[dict[str, Any]]:
        return [values for values, _ in self.records]


def prepare_base_proposals(selection: TemplateSelection, field_ids: list[str], seed: int, count: int = 0) -> BaseProposals:
    del selection
    base = BaseProposals(tuple(map(str, field_ids)), seed)
    base.ensure(count)
    return base


def _profiles(selection: TemplateSelection) -> list[tuple[str, float]]:
    return [(str(key), float(value["weight"])) for key, value in selection.template["operational_profiles"].items()]


def _weighted_choice(items: list[tuple[str, float]], probability: float) -> str:
    target = probability * sum(weight for _, weight in items)
    running = 0.0
    for key, weight in items:
        running += weight
        if target <= running:
            return key
    return items[-1][0]


def _normal_inverse(probability: float) -> float:
    return _NORMAL.inv_cdf(min(1.0 - _EPSILON, max(_EPSILON, probability)))


def _shape(selection: TemplateSelection, field_id: str, profile: str) -> tuple[float, float]:
    return tuple(map(float, selection.fields[field_id]["profile_beta"][profile]))


def _value(selection: TemplateSelection, field_id: str, profile: str, fraction: float, config: Any) -> int | float:
    meta = selection.fields[field_id]
    beta_value = _lookup(_shape(selection, field_id, profile), config.beta_lookup_knots).inverse(fraction)
    raw = float(meta["minimum"]) + beta_value * (float(meta["maximum"]) - float(meta["minimum"]))
    return int(round(raw)) if meta["type"] == "integer" else round(raw, config.saved_decimal_places)


def _marginal_cdf(selection: TemplateSelection, field_id: str, profile: str, value: float) -> float:
    meta = selection.fields[field_id]
    fraction = (value - float(meta["minimum"])) / (float(meta["maximum"]) - float(meta["minimum"]))
    return _beta_cdf(min(1.0, max(0.0, fraction)), *_shape(selection, field_id, profile))


def _coupled_pair(selection: TemplateSelection, fields: tuple[str, ...], group_id: str | None) -> tuple[str, str] | None:
    if group_id is None:
        return None
    group = selection.template["dynamics"]["replacements"].get(group_id)
    if group is None:
        raise ValueError(f"unknown active replacement group: {group_id}")
    members = tuple(str(item) for item in group["members"] if item in fields)
    return members if len(members) == 2 else None


def _unconditional_state(selection: TemplateSelection, fields: tuple[str, ...], profile: str, uniforms: tuple[float, ...], pair: tuple[str, str] | None, rho: float, config: Any) -> dict[str, Any]:
    fractions = {field_id: uniforms[index] for index, field_id in enumerate(fields)}
    if pair is not None:
        left, right = pair
        noise = _normal_inverse(uniforms[len(fields)])
        fractions[right] = _NORMAL.cdf(rho * _normal_inverse(fractions[left]) + math.sqrt(max(0.0, 1.0 - rho * rho)) * noise)
    return {field_id: _value(selection, field_id, profile, fractions[field_id], config) for field_id in fields}


def _step(meta: dict[str, Any], places: int) -> float:
    return 1.0 if meta["type"] == "integer" else 10.0 ** -places


def _allowed_intervals(sampling: dict[str, Any], selection: TemplateSelection, variable: str, fixed: dict[str, float], places: int) -> list[tuple[float, float]]:
    meta = selection.fields[variable]
    step = _step(meta, places)
    intervals: list[tuple[float, float]] = []
    for branch in sampling["branches"]:
        low, high = float(meta["minimum"]), float(meta["maximum"])
        valid = True
        for atom in branch:
            coefficients = atom["coefficients"]
            coefficient = float(coefficients.get(variable, 0.0))
            offset = float(atom["constant"]) + sum(float(value) * float(coefficients.get(key, 0.0)) for key, value in fixed.items())
            op = atom["op"]
            if abs(coefficient) <= 1e-15:
                checks = {"le": offset <= 0, "lt": offset < 0, "ge": offset >= 0, "gt": offset > 0, "eq": abs(offset) <= 1e-12}
                if not checks.get(op, False):
                    valid = False
                    break
                continue
            boundary = -offset / coefficient
            effective = op if coefficient > 0 else {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt", "eq": "eq"}[op]
            if effective == "eq":
                snapped = round(boundary / step) * step
                if abs(snapped - boundary) > 1e-9:
                    valid = False
                    break
                low, high = max(low, snapped), min(high, snapped)
            elif effective in {"ge", "gt"}:
                snapped = (math.floor(boundary / step) + 1 if effective == "gt" else math.ceil(boundary / step - 1e-12)) * step
                low = max(low, snapped)
            else:
                snapped = (math.ceil(boundary / step) - 1 if effective == "lt" else math.floor(boundary / step + 1e-12)) * step
                high = min(high, snapped)
        if valid and low <= high + 1e-12:
            intervals.append((low, high))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for low, high in intervals:
        if merged and low <= merged[-1][1] + step + 1e-12:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _latent_bounds(selection: TemplateSelection, field_id: str, profile: str, interval: tuple[float, float], places: int) -> tuple[float, float]:
    meta = selection.fields[field_id]
    half = _step(meta, places) / 2.0
    low = _marginal_cdf(selection, field_id, profile, max(float(meta["minimum"]), interval[0] - half))
    high = _marginal_cdf(selection, field_id, profile, min(float(meta["maximum"]), interval[1] + half))
    return _normal_inverse(low), _normal_inverse(high)


def _interval_masses(selection: TemplateSelection, field_id: str, profile: str, intervals: list[tuple[float, float]], mean: float, deviation: float, places: int) -> list[float]:
    return [max(0.0, _NORMAL.cdf((high - mean) / deviation) - _NORMAL.cdf((low - mean) / deviation)) for low, high in (_latent_bounds(selection, field_id, profile, interval, places) for interval in intervals)]


def _sample_interval(selection: TemplateSelection, field_id: str, profile: str, intervals: list[tuple[float, float]], probability: float, mean: float, deviation: float, config: Any) -> int | float:
    masses = _interval_masses(selection, field_id, profile, intervals, mean, deviation, config.saved_decimal_places)
    total = sum(masses)
    if total <= 0:
        raise ArithmeticError("conditional whitelist interval has zero probability mass")
    target, running = probability * total, 0.0
    for interval, mass in zip(intervals, masses):
        if target <= running + mass:
            low, _ = _latent_bounds(selection, field_id, profile, interval, config.saved_decimal_places)
            probability_z = _NORMAL.cdf((low - mean) / deviation) + target - running
            value = _value(selection, field_id, profile, _NORMAL.cdf(mean + deviation * _normal_inverse(probability_z)), config)
            lower, upper = interval
            bounded = min(upper, max(lower, float(value)))
            return int(round(bounded)) if selection.fields[field_id]["type"] == "integer" else round(bounded, config.saved_decimal_places)
        running += mass
    raise AssertionError("conditional interval selection fell outside cumulative mass")


def _conditional_state(selection: TemplateSelection, fields: tuple[str, ...], profile: str, uniforms: tuple[float, ...], pair: tuple[str, str] | None, rho: float, sampling: dict[str, Any], config: Any) -> dict[str, Any]:
    involved = list(map(str, sampling["fields"]))
    component = list(dict.fromkeys((*involved, *(pair or ())))) if pair is not None and set(pair) & set(involved) else involved
    if len(component) > 2:
        raise ValueError("conditional whitelist is incompatible with the coupling component")
    output: dict[str, Any] = {}
    if len(component) == 1:
        variable = component[0]
        output[variable] = _sample_interval(selection, variable, profile, _allowed_intervals(sampling, selection, variable, {}, config.saved_decimal_places), uniforms[fields.index(variable)], 0.0, 1.0, config)
    elif len(component) == 2:
        first, second = component
        effective_rho = rho if pair is not None and set(component) == set(pair) else 0.0
        deviation = math.sqrt(max(_EPSILON, 1.0 - effective_rho * effective_rho))
        points = config.conditional_integration_points
        cumulative, total = _conditional_table(selection, profile, pair, rho, sampling, config)
        if total <= 0:
            raise ArithmeticError("conditional whitelist has zero numerical probability mass")
        index = min(bisect.bisect_right(cumulative, uniforms[fields.index(first)] * total), points - 1)
        first_u = (index + uniforms[-2]) / points
        output[first] = _value(selection, first, profile, first_u, config)
        intervals = _allowed_intervals(sampling, selection, second, {first: float(output[first])}, config.saved_decimal_places)
        output[second] = _sample_interval(selection, second, profile, intervals, uniforms[fields.index(second)], effective_rho * _normal_inverse(first_u), deviation, config)
    remaining = tuple(field_id for field_id in fields if field_id not in output)
    remaining_pair = pair if pair is not None and set(pair).issubset(remaining) else None
    ordinary_uniforms = tuple(uniforms[fields.index(field_id)] for field_id in remaining) + uniforms[-3:]
    output.update(_unconditional_state(selection, remaining, profile, ordinary_uniforms, remaining_pair, rho, config))
    return output


def _sampling_kind(sampling: dict[str, Any], pair: tuple[str, str] | None) -> str:
    if sampling.get("kind") != "conditional":
        return "fallback"
    involved = set(map(str, sampling.get("fields") or []))
    return "fallback" if pair is not None and involved & set(pair) and len(involved | set(pair)) > 2 else "conditional"


def _conditional_table(selection: TemplateSelection, profile: str, pair: tuple[str, str] | None, rho: float, sampling: dict[str, Any], config: Any) -> tuple[tuple[float, ...], float]:
    key = (
        selection.template_id,
        json.dumps({key: value for key, value in sampling.items() if key != "constraints"}, sort_keys=True, separators=(",", ":")),
        profile,
        pair,
        rho,
        config.conditional_integration_points,
        config.beta_lookup_knots,
        config.saved_decimal_places,
    )
    if key in _CONDITIONAL_TABLES:
        return _CONDITIONAL_TABLES[key]
    involved = list(map(str, sampling.get("fields") or []))
    component = list(dict.fromkeys((*involved, *(pair or ())))) if pair is not None and set(pair) & set(involved) else involved
    if len(component) != 2:
        raise ValueError("conditional table requires a two-field component")
    first, second = component
    effective_rho = rho if pair is not None and set(component) == set(pair) else 0.0
    deviation = math.sqrt(max(_EPSILON, 1.0 - effective_rho * effective_rho))
    cumulative: list[float] = []
    total = 0.0
    for index in range(config.conditional_integration_points):
        first_u = (index + 0.5) / config.conditional_integration_points
        first_value = _value(selection, first, profile, first_u, config)
        intervals = _allowed_intervals(sampling, selection, second, {first: float(first_value)}, config.saved_decimal_places)
        total += sum(_interval_masses(selection, second, profile, intervals, effective_rho * _normal_inverse(first_u), deviation, config.saved_decimal_places))
        cumulative.append(total)
    result = tuple(cumulative), total
    _CONDITIONAL_TABLES[key] = result
    return result


def _conditional_profile_mass(selection: TemplateSelection, profile: str, pair: tuple[str, str] | None, rho: float, sampling: dict[str, Any], config: Any) -> float:
    involved = list(map(str, sampling.get("fields") or []))
    component = list(dict.fromkeys((*involved, *(pair or ())))) if pair is not None and set(pair) & set(involved) else involved
    if not component:
        return 1.0
    if len(component) == 1:
        intervals = _allowed_intervals(sampling, selection, component[0], {}, config.saved_decimal_places)
        return sum(_interval_masses(selection, component[0], profile, intervals, 0.0, 1.0, config.saved_decimal_places))
    _, total = _conditional_table(selection, profile, pair, rho, sampling, config)
    return total / config.conditional_integration_points


def sample_states(selection: TemplateSelection, field_ids: list[str], base: BaseProposals, compiled_constraints: dict[str, Any], *, replacement_group_id: str | None, effective_coupling: float | None, config: Any, count: int | None = None, pool: PopulationPool | None = None, _include_profiles: bool = False) -> list[Any]:
    fields = tuple(map(str, field_ids))
    if base.field_ids != fields:
        raise ValueError("base proposal fields do not match the requested population")
    target = count if count is not None else len(base.uniforms)
    result = pool if pool is not None else PopulationPool()
    if target < len(result.records):
        raise ValueError("population pool cannot shrink")
    pair = _coupled_pair(selection, fields, replacement_group_id)
    rho = float(effective_coupling or 0.0)
    constraints = list(compiled_constraints.get("constraints") or [])
    sampling = dict(compiled_constraints.get("sampling") or {"kind": "conditional", "fields": [], "branches": [[]]})
    kind = _sampling_kind(sampling, pair)
    needed = target - len(result.records)
    raw_limit = result.raw_draw_count + (
        needed * int(config.fallback_raw_draw_multiplier)
        if kind == "fallback"
        else needed
    )
    profiles = _profiles(selection)
    if kind == "conditional" and constraints:
        profiles = [
            (profile, weight * _conditional_profile_mass(selection, profile, pair, rho, sampling, config))
            for profile, weight in profiles
        ]
        if sum(weight for _, weight in profiles) <= 0:
            raise ArithmeticError("conditional whitelist has zero profile-weighted mass")
    while len(result.records) < target:
        if result.raw_draw_count >= raw_limit:
            raise PopulationBudgetExhausted(f"fallback exhausted {raw_limit} raw draws for {target} legal records")
        base.ensure(result.raw_draw_count + 1)
        row = base.uniforms[result.raw_draw_count]
        result.raw_draw_count += 1
        profile = _weighted_choice(profiles, row[-1])
        values = _conditional_state(selection, fields, profile, row, pair, rho, sampling, config) if kind == "conditional" else _unconditional_state(selection, fields, profile, row, pair, rho, config)
        invalid = constraint_violations(constraints, values)
        if kind == "fallback" and invalid:
            continue
        if invalid:
            raise AssertionError("conditional whitelist sampler produced a constraint-invalid state")
        result.records.append((values, profile))
    records = result.records[:target]
    return records if _include_profiles else [values for values, _ in records]


def sample_population_records(selection: TemplateSelection, field_ids: list[str], base: BaseProposals, compiled_constraints: dict[str, Any], **kwargs: Any) -> list[tuple[dict[str, Any], str]]:
    return sample_states(selection, field_ids, base, compiled_constraints, _include_profiles=True, **kwargs)


def sample_unconstrained_states(selection: TemplateSelection, field_ids: list[str], base: BaseProposals, *, replacement_group_id: str | None, effective_coupling: float | None, config: Any, count: int | None = None) -> list[dict[str, Any]]:
    return sample_states(selection, field_ids, base, {"constraints": [], "sampling": {"kind": "conditional", "fields": [], "branches": [[]]}}, replacement_group_id=replacement_group_id, effective_coupling=effective_coupling, config=config, count=count)


def sample_raw_values(selection: TemplateSelection, field_ids: list[str], seed: int, compiled_constraints: dict[str, Any] | None = None, **kwargs: Any) -> tuple[dict[str, Any], str]:
    base = prepare_base_proposals(selection, field_ids, seed)
    return sample_population_records(selection, field_ids, base, compiled_constraints or {"constraints": []}, count=1, **kwargs)[0]


def sample_profile_values(
    selection: TemplateSelection,
    field_ids: list[str],
    profile: str,
    seed: int,
    config: Any,
) -> dict[str, Any]:
    rng = random.Random(seed)
    return {
        field_id: _value(selection, field_id, profile, rng.random(), config)
        for field_id in field_ids
    }
