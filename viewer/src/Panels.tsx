import { AlertTriangle, ArrowRight, BarChart3, Brain, CheckCircle2, ChevronRight, CircleDot, Database, Eye, FlaskConical, Gauge, Info, Layers3, XCircle } from "lucide-react";
import { useState, type ReactNode } from "react";
import { ACTION_COLORS, FAMILY_COLORS, ParetoFrontierChart, SettingMetricBars, UtilityLandscape } from "./Charts";
import { SHOW_REASONING } from "./config";
import type { DynamicScope, Episode, EpisodeSummary, Manifest, ProbeAllocationRow, ProbePoint, RunBranch, RunEpisode, RunSummary, RunSummaryView, TraceEvent } from "./types";

export type Selection = { type: "case" | "probe"; id: string } | null;

const pretty = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const DYNAMIC_SCOPES: DynamicScope[] = ["all", "field_replacement", "direction_reversal", "compound"];
export const availableDynamicScopes = (manifest: Manifest): DynamicScope[] =>
  DYNAMIC_SCOPES.filter((scope) => scope === "all" || (manifest.dynamic_type_counts?.[scope] || 0) > 0);
const dynamicLabel = (scope: DynamicScope, long = false) => scope === "all"
  ? "All dynamics"
  : scope === "field_replacement"
    ? (long ? "Field replacement" : "Replacement")
    : scope === "direction_reversal"
      ? (long ? "Direction reversal" : "Reversal")
      : "Compound";
const fmt = (value: unknown, digits = 2) => typeof value === "number" ? value.toFixed(digits) : "—";
const compact = (value: unknown) => typeof value === "number" ? String(Number(value.toFixed(2))) : String(value);
const traceJson = (value: unknown) => JSON.stringify(value, null, 2).replace(
  /(-?\d+\.\d+)(?=\s*[,}\]])/g,
  (number) => Number(number).toFixed(2),
);
const jsonValueEqual = (left: unknown, right: unknown): boolean => {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length
      && left.every((value, index) => jsonValueEqual(value, right[index]));
  }
  if (!left || !right || typeof left !== "object" || typeof right !== "object") return false;
  const leftObject = left as Record<string, unknown>;
  const rightObject = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftObject);
  const rightKeys = Object.keys(rightObject);
  return leftKeys.length === rightKeys.length
    && leftKeys.every((key) => Object.prototype.hasOwnProperty.call(rightObject, key)
      && jsonValueEqual(leftObject[key], rightObject[key]));
};
export const environmentResultForWire = (result: unknown, stateCard: unknown, stage: unknown = "reveal"): unknown => {
  if (!result || typeof result !== "object" || Array.isArray(result)) return result;
  const compactResult = { ...(result as Record<string, unknown>) };
  delete compactResult.type;
  if (stage === "reveal" && stateCard && typeof stateCard === "object" && !Array.isArray(stateCard)) {
    Object.entries(stateCard as Record<string, unknown>).forEach(([key, value]) => {
      if (Object.prototype.hasOwnProperty.call(compactResult, key) && jsonValueEqual(compactResult[key], value)) {
        delete compactResult[key];
      }
    });
  }
  return Object.keys(compactResult).length ? compactResult : null;
};
function scoreFormula(rule: Record<string, any>): { title: string; lines: string[]; parameters: string[] } {
  const clip = "clip(y) = max(−1, min(1, y))";
  const ranges = Object.entries(rule.field_ranges || {}).map(([field, range]: [string, any]) =>
    `n_${field}(x) = 2·(x − ${compact(range.minimum)})/(${compact(range.maximum)} − ${compact(range.minimum)}) − 1`,
  );
  if (rule.family === "additive_aggregation") {
    const terms = (rule.relevant_fields || []).map(
      (field: string) => `${compact(rule.weights[field])}·n_${field}(x_${field})`,
    );
    return {
      title: "Additive aggregation",
      lines: [
        `raw evidence E = ${compact(rule.bias)} + ${terms.join(" + ")}`,
        "score = E / |negative anchor| when E < 0",
        "score = E / positive anchor when E > 0",
        "score is clipped to [−1, 1]",
      ],
      parameters: [clip, ...ranges, `calibration = ${JSON.stringify(rule.score_calibration)}`],
    };
  }
  if (["conjunctive_gating", "disjunctive_sufficiency", "lexicographic_priority"].includes(rule.family)) {
    const margins = (rule.relevant_fields || []).map(
      (field: string) => `m_${field} = ${compact(rule.field_directions[field])}·[n_${field}(x_${field}) − n_${field}(${compact(rule.thresholds[field])})]`,
    );
    const evidence = rule.family === "conjunctive_gating"
      ? "E = min(all field margins)"
      : rule.family === "disjunctive_sufficiency"
        ? "E = max(all field margins)"
        : `E = first margin in priority order [${(rule.priority_order || []).join(", ")}] with |m| > ${compact(rule.indifference_band)}; otherwise use the final margin`;
    return {
      title: pretty(rule.family),
      lines: [
        ...margins,
        evidence,
        "score = piecewise zero-preserving calibrated E, clipped to [−1, 1]",
      ],
      parameters: [clip, ...ranges, `calibration = ${JSON.stringify(rule.score_calibration)}`],
    };
  }
  return { title: "Hidden rule", lines: [JSON.stringify(rule)], parameters: [] };
}

export function Sidebar({
  manifest, selectedId, search, family, dynamicScope, onSearch, onFamily, onDynamicScope, onSelect, onOverview,
}: {
  manifest: Manifest; selectedId: string; search: string; family: string;
  dynamicScope: DynamicScope; onSearch: (value: string) => void; onFamily: (value: string) => void;
  onDynamicScope: (value: DynamicScope) => void; onSelect: (id: string) => void; onOverview: () => void;
}) {
  const families = Object.keys(manifest.family_counts);
  const scopes = availableDynamicScopes(manifest);
  const episodes = manifest.episodes.filter((episode) =>
    (!family || episode.rule_family === family) &&
    (dynamicScope === "all" || episode.dynamic_type === dynamicScope) &&
    (!search || `${episode.episode_id} ${episode.domain}`.toLowerCase().includes(search.toLowerCase())),
  );
  return <aside className="sidebar">
    <div className="brand"><div className="brand-mark">R</div><div><strong>AdaExploreBench</strong><span>Research Viewer</span></div></div>
    <div className="dataset-chip"><Database size={14} /> {manifest.episode_count} episodes{manifest.episodes.length < manifest.episode_count ? ` · ${manifest.episodes.length} examples` : ""} · {manifest.release.name} · {manifest.benchmark_config.probe_budget} probe contexts max</div>
    <button className={`overall-link ${!selectedId ? "selected" : ""}`} onClick={onOverview}><BarChart3 size={16} /><span><strong>Overview</strong><small>Compare all model runs</small></span><ChevronRight size={15} /></button>
    <input className="search" aria-label="Search episodes" placeholder={manifest.episodes.length < manifest.episode_count ? "Search examples…" : "Search episodes…"} value={search} onChange={(event) => onSearch(event.target.value)} />
    {scopes.length > 1 && <div className="dynamic-filter" role="group" aria-label="Filter by dynamic type">
      {scopes.map((scope) => <button key={scope} className={dynamicScope === scope ? "active" : ""} onClick={() => onDynamicScope(scope)}>{dynamicLabel(scope)}</button>)}
    </div>}
    <div className="family-filter">
      <button className={!family ? "active" : ""} onClick={() => onFamily("")}>All</button>
      {families.map((item) => <button key={item} className={family === item ? "active" : ""} title={pretty(item)} onClick={() => onFamily(item)}><i style={{ background: FAMILY_COLORS[item] }} /></button>)}
    </div>
    <div className="episode-list">
      {episodes.map((episode: EpisodeSummary) => <button key={episode.episode_id} className={`episode-row ${selectedId === episode.episode_id ? "selected" : ""}`} onClick={() => onSelect(episode.episode_id)}>
        <i className="family-bar" style={{ background: FAMILY_COLORS[episode.rule_family] }} />
        <span><strong>{episode.episode_id}</strong><small>{dynamicLabel(episode.dynamic_type)} · {episode.sampling_branches.join(" / ")} · change after B{episode.after_action_batch} · {episode.domain} · {episode.raw_field_count ?? episode.relevant_field_count + episode.distractor_field_count} raw · {episode.relevant_field_count} relevant · {episode.distractor_field_count} distractor fields</small></span>
        <ChevronRight size={15} />
      </button>)}
    </div>
  </aside>;
}

function Stat({ label, value, note }: { label: string; value: string | number | null | undefined; note?: string }) {
  return <div className="stat"><span>{label}</span><strong>{value ?? "—"}</strong>{note && <small>{note}</small>}</div>;
}

const HYPERPARAMETER_ORDER = ["model", "provider", "conversation_protocol", "temperature", "max_tokens", "effort", "output_format", "seed", "policy", "max_probe_cases_per_batch", "allow_abstention", "action_batch_count", "action_batch_size", "branches"];

function RunHyperparameters({ run }: { run: RunSummary }) {
  const entries = Object.entries(run.hyperparameters || {});
  if (!entries.length) return null;
  const ordered = [
    ...HYPERPARAMETER_ORDER.flatMap((key) => entries.filter(([name]) => name === key)),
    ...entries.filter(([name]) => !HYPERPARAMETER_ORDER.includes(name)),
  ];
  return <details className="trace-detail run-hyperparameters"><summary>Run hyperparameters</summary>
    <dl>{ordered.map(([key, value]) => <div key={key}><dt>{pretty(key)}</dt><dd>{value === null || value === undefined ? "—" : String(value)}</dd></div>)}</dl>
  </details>;
}

const RUN_COLORS = [
  "#356f95",
  "#c06c2d",
  "#7b5aa6",
  "#2f7d6d",
  "#a34355",
  "#756b25",
  "#317e92",
  "#a1548b",
  "#6c7041",
  "#4964a8",
  "#9b593f",
  "#4e7c50",
];
export const modelColor = (_model: string, index: number) => {
  const colorIndex = Math.max(0, Math.trunc(index));
  if (colorIndex < RUN_COLORS.length) return RUN_COLORS[colorIndex];
  const generatedIndex = colorIndex - RUN_COLORS.length;
  const hue = (23 + generatedIndex * 137.50776405003785) % 360;
  const saturation = 55 + generatedIndex % 3 * 5;
  const lightness = 37 + generatedIndex % 2 * 8;
  return `hsl(${hue.toFixed(6)} ${saturation}% ${lightness}%)`;
};
const percent = (value: number | null | undefined) => `${((value || 0) * 100).toFixed(1)}%`;
const SAMPLING_BRANCH_ORDER = ["flip_50", "flip_50_no_change_control", "static_iid", "flip_25", "flip_75", "flip_50_eval_case_given", "flip_50_full_history"];
const SAMPLING_BRANCH_STYLE: Record<string, { label: string; color: string; dash?: string }> = {
  flip_50: { label: "Default", color: "#d17816" },
  static_iid: { label: "Static IID", color: "#2f7d6d" },
  flip_25: { label: "25% flip", color: "#3979b7" },
  flip_50_no_change_control: { label: "50% flip · no-change control", color: "#89704b", dash: "6 4" },
  flip_75: { label: "75% flip", color: "#9a4f91" },
  flip_50_eval_case_given: { label: "50% flip · eval cases given", color: "#9b6548", dash: "3 3" },
  flip_50_full_history: { label: "50% flip · eval cases given (legacy)", color: "#9b6548", dash: "3 3" },
};
// ``flip_50_full_history`` was renamed to ``flip_50_eval_case_given``.  The
// latter is the canonical viewer setting name; old artifacts are normalized
// during publication and the lookup fallback below keeps older data readable.
const CANONICAL_BRANCH_NAMES: Record<string, string> = {
  flip_50_full_history: "flip_50_eval_case_given",
};

const isNoChangeControl = (name: string) => name.endsWith("_no_change_control");

export function orderedSamplingBranches(names: Iterable<string>): string[] {
  const present = new Set(names);
  return [
    ...SAMPLING_BRANCH_ORDER.filter((name) => present.has(name)),
    ...[...present].filter((name) => !SAMPLING_BRANCH_ORDER.includes(name)).sort(),
  ];
}

export function overviewSamplingBranches(names: Iterable<string>): string[] {
  return orderedSamplingBranches(
    [...names].map((name) => CANONICAL_BRANCH_NAMES[name] || name),
  );
}

type DynamicBatchMetric = {
  batch_number?: number;
  case_count?: number;
  answered_cases?: number;
  probe_cases_in_preceding_window?: number;
  active_rule_accuracy?: number | null;
  partial_credit_accuracy?: number | null;
  partial_credit_random_baseline?: number | null;
  static_accuracy?: number | null;
  old_rule_alignment?: number | null;
  stale_policy_accuracy?: number | null;
  flip_case_count?: number;
  nonflip_case_count?: number;
  flip_adoption_accuracy?: number | null;
  nonflip_stability_accuracy?: number | null;
  flipped_case_breakdown?: {
    denominator: number;
    new_rule_adoption: number;
    stale_rule_adherence: number;
    neither_rule: number;
  };
  stratum_metrics?: Record<string, {
    case_count: number;
    active_rule_accuracy: number | null;
    static_accuracy?: number | null;
    old_rule_alignment: number | null;
    stale_policy_accuracy: number | null;
    flipped_case_breakdown: DynamicBatchMetric["flipped_case_breakdown"];
  }>;
};

type DynamicSeries = "active_rule_accuracy" | "partial_credit_accuracy" | "static_accuracy" | "old_rule_alignment" | "stale_policy_accuracy";

function dynamicSeriesPoints(
  rows: DynamicBatchMetric[],
  key: DynamicSeries,
  x: (index: number) => number,
  y: (value: number) => number,
) {
  const points: string[] = [];
  rows.forEach((row, index) => {
    if (row[key] != null) points.push(`${x(index)},${y(Number(row[key]))}`);
  });
  return points.join(" ");
}

function ComparisonBars({ rows, runs, value, label = pretty }: { rows: string[]; runs: RunSummary[]; value: (run: RunSummary, row: string) => number | null | undefined; label?: (row: string) => string }) {
  return <div className="comparison-bars">{rows.map((row) => <div className="comparison-row" key={row}>
    <strong>{label(row)}</strong>
    <div>{runs.map((run, index) => { const score = value(run, row) || 0; return <div className="comparison-bar" key={run.run_id}>
      <span>{run.label}</span><i><b style={{ width: `${score * 100}%`, background: modelColor(run.model_key, index) }} /></i><code>{percent(score)}</code>
    </div>; })}</div>
  </div>)}</div>;
}

const summaryView = (run: RunSummary, scope: DynamicScope): RunSummaryView => run.views?.[scope] || {
  episode_count: run.episode_ids.length,
  metrics: run.metrics,
  batch_accuracy: run.batch_accuracy || [],
  batch_dynamic: run.batch_dynamic || {},
  breakdowns: run.breakdowns || { by_family: {}, by_domain: {} },
};

const numOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
export const paretoRunLabel = (label: string) => label.replace(/\s+\d{4}$/, "");
export const branchLabel = (name: string) => SAMPLING_BRANCH_STYLE[name]?.label || pretty(name);
const branchColor = (name: string) => SAMPLING_BRANCH_STYLE[name]?.color || "#59645f";
const branchDash = (name: string) => SAMPLING_BRANCH_STYLE[name]?.dash;
const branchLegendBackground = (name: string) => branchDash(name)
  ? `repeating-linear-gradient(90deg, ${branchColor(name)} 0 6px, transparent 6px 10px)`
  : branchColor(name);
const sourceSamplingBranch = (name: string) => name.replace(/_no_change_control$/, "");

function samplingBranchesForViews(runs: RunSummary[], scope: DynamicScope): string[] {
  return overviewSamplingBranches(
    runs.flatMap((run) => Object.keys(summaryView(run, scope).branches || {})),
  );
}

type UtilitySummary = {
  actionUtility: number | null;
  optimal: number | null;
};

function utilitySummary(metrics: Record<string, unknown> | undefined): UtilitySummary {
  const source = (metrics || {}) as Record<string, unknown>;
  const actionUtility = numOrNull(source.action_utility) ?? numOrNull(source.cumulative_utility);
  const regret = numOrNull(source.cumulative_regret);
  return {
    actionUtility,
    optimal: actionUtility != null && regret != null ? actionUtility + regret : null,
  };
}

function branchMetric(view: RunSummaryView, branch: string, key: string): number | null {
  const branchView = view.branches?.[branch]
    || (branch === "flip_50_eval_case_given" ? view.branches?.flip_50_full_history : undefined);
  return numOrNull((branchView?.metrics as Record<string, unknown> | undefined)?.[key]);
}

function branchProbeCount(view: RunSummaryView, branch: string): number | null {
  return branchMetric(view, branch, "probe_count")
    ?? branchMetric(view, branch, "probe_cases");
}

function branchAccuracyMetric(metrics: Record<string, unknown> | undefined, branch: string): number | null {
  const source = metrics || {};
  const key = branch === "static_iid" || branch.endsWith("_no_change_control")
    ? "static_overall_accuracy"
    : "dynamic_post_rule_accuracy";
  return numOrNull(source[key]) ?? numOrNull(source.action_accuracy);
}

function probeRowsFromBranch(run: RunBranch): ProbeAllocationRow[] {
  const metrics = (run.metrics || {}) as Record<string, any>;
  const probeRows = (metrics.probe_metrics?.by_batch || []) as Array<Record<string, any>>;
  const batchRows = new Map<number, Record<string, any>>(
    ((metrics.batch_metrics || []) as Array<Record<string, any>>).map((row) => [Number(row.batch_number), row]),
  );
  return probeRows
    .map((row) => {
      const batch = Number(row.batch_number);
      const previous = batchRows.get(batch - 1);
      return {
        batch_number: batch,
        probe_cases: Number(row.probe_cases || 0),
        action_utility_on_hand: previous
          ? numOrNull(previous.action_utility) ?? numOrNull(previous.cumulative_utility)
          : 0,
      };
    })
    .sort((a, b) => a.batch_number - b.batch_number);
}

function RunSwitcher({ runs, selectedRunId, onSelect, ariaLabel }: { runs: RunSummary[]; selectedRunId: string; onSelect: (runId: string) => void; ariaLabel: string }) {
  if (runs.length < 2) return null;
  return <div className="trajectory-run-switch" role="group" aria-label={ariaLabel}>{runs.map((run) => <button className={run.run_id === selectedRunId ? "active" : ""} key={run.run_id} onClick={() => onSelect(run.run_id)}>{run.label}</button>)}</div>;
}

const ACTION_UTILITY_COLOR = "#2f7d6d";
const UTILITY_STACK_RATIONALE = "Full bar = optimal action utility (oracle upper bound). The filled portion is realized action utility; the unfilled remainder is regret. Mean Unique Probes reports exploration breadth.";

function utilitySegments(summary: UtilitySummary) {
  if (summary.optimal == null) return null;
  const optimal = Number(summary.optimal);
  const track = optimal;
  if (track <= 0) return null;
  const actionUtility = summary.actionUtility ?? 0;
  const segments = [
    { key: "action", label: "Realized action utility", value: Math.max(0, actionUtility), width: Math.max(0, actionUtility) / track, color: ACTION_UTILITY_COLOR, negative: false },
  ].filter((segment) => segment.width > 0.0005);
  return { track, actionUtility, segments };
}

function UtilityStack({ summary, label }: { summary: UtilitySummary; label: string }) {
  const data = utilitySegments(summary);
  if (!data) return null;
  return <div className="utility-stack" role="img" aria-label={`${label} action utility breakdown`} title={`${label} · oracle cap ${fmt(data.track)} · action utility ${fmt(data.actionUtility)}`}>
    {data.segments.map((segment) => <span
      key={segment.key}
      className={`utility-stack-seg${segment.negative ? " negative" : ""}`}
      style={{ width: `${Math.min(segment.width, 1) * 100}%`, backgroundColor: segment.color }}
      title={`${label} · ${segment.label}: ${fmt(segment.value)} · ${percent(segment.width)} of oracle cap`}
    >{segment.width >= 0.08 ? percent(segment.width) : ""}</span>)}
  </div>;
}

function UtilityStackLegend() {
  return <div className="batch-chart-legend utility-legend">
    <span><i style={{ background: "#fff", border: "1px solid #b6bdb8" }} />Full bar = optimal action utility</span>
    <span><i style={{ background: ACTION_UTILITY_COLOR }} />Realized action utility</span>
    <span><i style={{ background: "#e5e8e3" }} />Regret gap (unfilled)</span>
  </div>;
}

type UtilityComparisonBar = { key: string; label: string; title: string; summary: UtilitySummary; share: number };

function utilityComparisonBar(key: string, label: string, title: string, summary: UtilitySummary): UtilityComparisonBar | null {
  const data = utilitySegments(summary);
  return data ? { key, label, title, summary, share: data.actionUtility / data.track } : null;
}

function UtilityComparisonRows({ rows }: { rows: Array<{ key: string; label: ReactNode; bars: UtilityComparisonBar[] }> }) {
  return <div className="comparison-bars utility-stacked">{rows.map((row) => <div className="comparison-row" key={row.key}>
    <strong>{row.label}</strong>
    <div>{row.bars.map((bar) => <div className="comparison-bar utility-bar" key={bar.key}>
      <span>{bar.label}</span>
      <UtilityStack summary={bar.summary} label={bar.title} />
      <div className="utility-bar-values">
        <code title={`${bar.title} · realized share of optimal action utility`}>{percent(bar.share)}</code>
        <strong>Action utility {fmt(bar.summary.actionUtility)}</strong>
      </div>
    </div>)}</div>
  </div>)}</div>;
}

type ProbeAllocationEntry = {
  name: string;
  rows: ProbeAllocationRow[];
  pre?: number | null;
  post?: number | null;
};

function ProbeAllocationChart({ entry, forkAfterBatch }: { entry: ProbeAllocationEntry; forkAfterBatch?: number | null }) {
  const { name, rows } = entry;
  if (!rows.length) return null;
  const width = 470;
  const height = 205;
  const margin = { left: 30, right: 44, top: 16, bottom: 32 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const maxCases = Math.max(1, ...rows.map((row) => Number(row.probe_cases || 0)));
  const utilities = rows.map((row) => row.action_utility_on_hand).filter((value): value is number => value != null);
  const minUtility = Math.min(0, ...utilities);
  const maxUtility = Math.max(0, ...utilities);
  const x = (index: number) => margin.left + (index + 0.5) * innerW / rows.length;
  const barWidth = innerW / rows.length * 0.6;
  const yCases = (value: number) => height - margin.bottom - (value / maxCases) * innerH;
  const yUtility = (value: number) => height - margin.bottom - ((value - minUtility) / (maxUtility - minUtility || 1)) * innerH;
  const forkX = forkAfterBatch && forkAfterBatch < rows.length ? margin.left + forkAfterBatch * innerW / rows.length : null;
  const linePoints = rows.map((row, index) => {
    const utility = row.action_utility_on_hand;
    return utility == null ? null : `${x(index)},${yUtility(utility)}`;
  }).filter(Boolean).join(" ");
  const caseTicks = [0, 0.5, 1].map((fraction) => ({ value: Math.round(maxCases * fraction), y: yCases(maxCases * fraction) }));
  const utilityTicks = [0, 0.5, 1].map((fraction) => {
    const value = minUtility + (maxUtility - minUtility) * fraction;
    return { value, y: yUtility(value) };
  });
  const pre = entry.pre ?? (forkAfterBatch ? rows.filter((row) => row.batch_number <= forkAfterBatch).reduce((sum, row) => sum + Number(row.probe_cases || 0), 0) : null);
  const post = entry.post ?? (forkAfterBatch ? rows.filter((row) => row.batch_number > forkAfterBatch).reduce((sum, row) => sum + Number(row.probe_cases || 0), 0) : null);
  return <article className="probe-mini-chart">
    <header><i style={{ background: branchColor(name) }} /><strong>{branchLabel(name)}</strong>{pre != null && post != null && <span className="probe-chips"><b>pre-dynamic {fmt(pre, 1)}</b><b>post-dynamic {fmt(post, 1)}</b></span>}</header>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${branchLabel(name)} probe allocation by action batch`}>
      <line className="probe-axis-line" x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} />
      <line className="probe-axis-line" x1={width - margin.right} x2={width - margin.right} y1={margin.top} y2={height - margin.bottom} />
      {caseTicks.map((tick, index) => <g key={`case-tick-${index}`}><line className="probe-axis-tick" x1={margin.left - 3} x2={margin.left} y1={tick.y} y2={tick.y} /><text className="probe-axis-label" x={margin.left - 6} y={tick.y + 3} textAnchor="end">{tick.value}</text></g>)}
      {utilityTicks.map((tick, index) => <g key={`utility-tick-${index}`}><line className="probe-axis-tick" x1={width - margin.right} x2={width - margin.right + 3} y1={tick.y} y2={tick.y} /><text className="probe-axis-label" x={width - margin.right + 6} y={tick.y + 3}>{fmt(tick.value)}</text></g>)}
      {minUtility < 0 && <line className="probe-zero-line" x1={margin.left} x2={width - margin.right} y1={yUtility(0)} y2={yUtility(0)} />}
      {rows.map((row, index) => {
        const cases = Number(row.probe_cases || 0);
        const actionUtility = row.action_utility_on_hand;
        const phase = forkAfterBatch ? (row.batch_number <= forkAfterBatch ? "pre" : "post") : "all";
        return <rect key={row.batch_number} className={`probe-bar probe-bar-${phase}`} x={x(index) - barWidth / 2} y={yCases(cases)} width={barWidth} height={height - margin.bottom - yCases(cases)} style={{ fill: branchColor(name), opacity: phase === "pre" ? 0.38 : 0.9 }}>
          <title>{`Batch ${row.batch_number}: ${cases} probe cases before batch · action utility so far ${fmt(actionUtility)}`}</title>
        </rect>;
      })}
      {forkX != null && <g>
        <line className="probe-fork-line" x1={forkX} x2={forkX} y1={margin.top} y2={height - margin.bottom} />
        <text className="probe-fork-label" x={forkX + 4} y={margin.top + 10}>rule change</text>
      </g>}
      {linePoints && <polyline className="probe-utility-line" points={linePoints} />}
      {rows.map((row, index) => {
        const utility = row.action_utility_on_hand;
        return utility == null ? null : <circle key={`u-${row.batch_number}`} className="probe-utility-point" cx={x(index)} cy={yUtility(Number(utility))} r="2.6" />;
      })}
      {rows.map((row, index) => <text key={`x-${row.batch_number}`} className="probe-axis-label" x={x(index)} y={height - 16} textAnchor="middle">B{row.batch_number}</text>)}
      <text className="probe-axis-label" x={margin.left} y={margin.top - 5}>probe cases</text>
      <text className="probe-axis-label" x={width - margin.right} y={margin.top - 5}>action utility</text>
    </svg>
  </article>;
}

function ProbeAllocationPanel({ entries, forkAfterBatch, note, title = "Probe Allocation", modelSwitch }: { entries: ProbeAllocationEntry[]; forkAfterBatch?: number | null; note: string; title?: string; modelSwitch?: ReactNode }) {
  const visible = entries.filter((entry) => entry.rows.length);
  if (!visible.length) return null;
  return <section className="panel probe-allocation-panel">
    <div className="section-heading"><div><span className="eyebrow">Exploration over time</span><h2>{title}</h2></div><span className="privacy-badge">bars: probe cases before each batch · line: action utility so far</span></div>
    {modelSwitch}
    <p className="rationale">{note}</p>
    <div className="probe-allocation-grid">{visible.map((entry) => <ProbeAllocationChart key={entry.name} entry={entry} forkAfterBatch={forkAfterBatch} />)}</div>
  </section>;
}

function OverallBatchAccuracyCurve({ runs, dynamicScope }: { runs: RunSummary[]; dynamicScope: DynamicScope }) {
  const stratum = "all";
  const visibleRuns = runs.filter((run) => Object.values(
    summaryView(run, dynamicScope).branches || {},
  ).some((view) => (view.batch_dynamic?.[stratum] || []).length > 0));
  const [selectedRunId, setSelectedRunId] = useState("");
  if (!visibleRuns.length) return null;
  const selectedRun = visibleRuns.find((run) => run.run_id === selectedRunId) || visibleRuns[0];
  const selectedView = summaryView(selectedRun, dynamicScope);
  const branchRows = orderedSamplingBranches(Object.keys(selectedView.branches || {})).flatMap((name) => {
    const rows = selectedView.branches?.[name]?.batch_dynamic?.[stratum] || [];
    return rows.length ? [{ name, rows }] : [];
  });
  if (!branchRows.length) return null;
  const width = 1040;
  const height = 330;
  const margin = { left: 58, right: 26, top: 24, bottom: 52 };
  const batchCount = Math.max(...branchRows.map(({ rows }) => rows.length));
  const x = (index: number) => margin.left + index * (width - margin.left - margin.right) / Math.max(batchCount - 1, 1);
  const y = (value: number) => height - margin.bottom - value * (height - margin.top - margin.bottom);
  const caseCount = branchRows[0]?.rows[0]?.case_count || 0;
  const baselineSeries = Array.from({ length: batchCount }, (_, index) => {
    const values = branchRows
      .map(({ rows }) => rows[index]?.partial_credit_random_baseline)
      .filter((value): value is number => value != null);
    return values.length ? { index, value: values.reduce((sum, value) => sum + value, 0) / values.length } : null;
  }).filter((point): point is { index: number; value: number } => point != null);
  const baselinePoints = baselineSeries.map((point) => `${x(point.index)},${y(point.value)}`).join(" ");

  return <section className="panel batch-accuracy-panel overall-batch-accuracy">
    <div className="section-heading"><div><span className="eyebrow">Sequential learning · {dynamicScope === "all" ? "all dynamic types" : pretty(dynamicScope)}</span><h2>Partial-Credit Accuracy Curve</h2></div><span className="privacy-badge">{caseCount} cases × batch × branch</span></div>
    {visibleRuns.length > 1 && <RunSwitcher runs={visibleRuns} selectedRunId={selectedRun.run_id} onSelect={setSelectedRunId} ariaLabel="Accuracy curve model" />}
    <p className="rationale">All sampling branches reuse the exact pre-change conversation for each episode, then separate after its configured Batch 5, 6, or 7 fork. Static IID and the matched 50% flip no-change control keep the pre-rule; the 25%, 50%, and 75% flip branches activate the post-rule. Each point is the batch's mean per-case partial credit — (chosen utility − worst) / (best − worst) over the case's oracle utilities — so partially learned rules land between 0% and 100%. The dashed gray line is the expected score of a uniformly random policy.</p>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${dynamicLabel(dynamicScope, true)} sampling-branch partial-credit accuracy comparison`}>
      {[0, .25, .5, .75, 1].map((tick) => <g key={tick}>
        <line className="batch-grid-line" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text className="batch-axis-label" x={margin.left - 9} y={y(tick) + 4} textAnchor="end">{Math.round(tick * 100)}%</text>
      </g>)}
      {Array.from({ length: batchCount }, (_, index) => <g key={index}>
        <line className="batch-tick" x1={x(index)} x2={x(index)} y1={height - margin.bottom} y2={height - margin.bottom + 5} />
        <text className="batch-axis-label" x={x(index)} y={height - 27} textAnchor="middle">B{index + 1}</text>
      </g>)}
      {baselinePoints && <polyline data-series="partial-credit-random-baseline" className="batch-line overall-batch-line" style={{ stroke: "#8a918c", strokeWidth: 2, strokeDasharray: "5 4" }} points={baselinePoints} />}
      {branchRows.map(({ name, rows }) => <g key={name}>
        <polyline data-series={`sampling-${name}`} className="batch-line overall-batch-line" style={{ stroke: branchColor(name), strokeWidth: 3, strokeDasharray: branchDash(name) }} points={dynamicSeriesPoints(rows, "partial_credit_accuracy", x, y)} />
        {rows.map((row, index) => row.partial_credit_accuracy == null ? null : <circle key={row.batch_number} className="batch-point overall-batch-point" style={{ fill: branchColor(name), strokeDasharray: branchDash(name) }} cx={x(index)} cy={y(Number(row.partial_credit_accuracy))} r="4">
          <title>{`${selectedRun.label} · ${branchLabel(name)} · batch ${row.batch_number} · ${pretty(stratum)}: ${percent(row.partial_credit_accuracy)} partial credit · ${row.answered_cases}/${row.case_count} answered`}</title>
        </circle>)}
      </g>)}
    </svg>
    <div className="batch-chart-legend">{branchRows.map(({ name }) => <span key={name}><i className={`legend-line${branchDash(name) ? " dashed" : ""}`} style={{ background: branchLegendBackground(name) }} />{branchLabel(name)} · partial-credit accuracy</span>)}{baselinePoints && <span><i className="legend-line dashed" style={{ background: "#8a918c" }} />uniform-random baseline</span>}</div>
    {branchRows.filter(({ name }) => name !== "static_iid" && !isNoChangeControl(name)).map(({ name }) => {
      const sourceRows = selectedView.branches?.[name]?.batch_dynamic?.all || [];
      const adaptationRows = selectedView.branches?.[name]?.adaptation_curve
        || sourceRows.filter((row) => row.flip_adoption_accuracy != null || row.nonflip_stability_accuracy != null).map((row) => ({
          batch_number: row.batch_number || 0,
          flip_case_count: row.flip_case_count,
          nonflip_case_count: row.nonflip_case_count,
          flip_adoption_accuracy: row.flip_adoption_accuracy,
          nonflip_stability_accuracy: row.nonflip_stability_accuracy,
          flipped_case_breakdown: row.flipped_case_breakdown,
        }));
      if (!adaptationRows.length) return null;
      const stabilityValues = adaptationRows
        .map((row) => row.nonflip_stability_accuracy)
        .filter((value): value is number => value != null);
      const meanStability = stabilityValues.length ? stabilityValues.reduce((sum, value) => sum + value, 0) / stabilityValues.length : null;
      return <div className="overall-flip-row" key={name}>
        <strong>{branchLabel(name)} post-rule behavior</strong>
        <span className="flip-stability-label">non-flip stability {meanStability == null ? "—" : percent(meanStability)}</span>
        <div className="flip-strip" style={{ gridTemplateColumns: `repeat(${adaptationRows.length}, minmax(0, 1fr))` }}>{adaptationRows.map((row) => {
          const breakdown = row.flipped_case_breakdown || { denominator: 0, new_rule_adoption: 0, stale_rule_adherence: 0, neither_rule: 0 };
          const denominator = breakdown?.denominator || 0;
          return <div className="flip-strip-cell" key={row.batch_number} title={`${selectedRun.label} · ${branchLabel(name)} · post batch P${row.batch_number}: ${denominator} flipped cases · adoption ${percent(row.flip_adoption_accuracy)} · non-flip stability ${percent(row.nonflip_stability_accuracy)}`}>
            {denominator > 0 && <><i className="flip-new" style={{ width: `${100 * breakdown.new_rule_adoption / denominator}%` }} /><i className="flip-stale" style={{ width: `${100 * breakdown.stale_rule_adherence / denominator}%` }} /><i className="flip-neither" style={{ width: `${100 * breakdown.neither_rule / denominator}%` }} /></>}
          </div>;
        })}</div>
      </div>;
    })}
    <div className="flip-strip-legend"><span><i className="flip-new" />Matches post-change optimum</span><span><i className="flip-stale" />Matches pre-change optimum</span><span><i className="flip-neither" />Matches neither optimum</span><span>Each strip is aligned to relative post batch P1, P2, …</span></div>
  </section>;
}

function OverallProbeCurve({ runs, dynamicScope }: { runs: RunSummary[]; dynamicScope: DynamicScope }) {
  const visibleRuns = runs.filter((run) => Object.keys(summaryView(run, dynamicScope).branches || {}).length > 0);
  const [selectedRunId, setSelectedRunId] = useState("");
  if (!visibleRuns.length) return null;
  const selectedRun = visibleRuns.find((run) => run.run_id === selectedRunId) || visibleRuns[0];
  const view = summaryView(selectedRun, dynamicScope);
  const entries = orderedSamplingBranches(Object.keys(view.branches || {})).flatMap((name) => {
    const branchView = view.branches?.[name];
    return branchView ? [{ name, view: branchView }] : [];
  });
  if (!entries.length) return null;
  return <ProbeAllocationPanel
    entries={entries.map(({ name, view: branchView }) => ({ name, rows: branchView.probe_by_batch || [], pre: branchView.probe_pre_dynamic_cases, post: branchView.probe_post_dynamic_cases }))}
    note={`${selectedRun.label} · macro-averaged per batch across episode groups. Change points vary between episodes, so bars are not phase-colored here; use the pre/post chips.`}
    modelSwitch={visibleRuns.length > 1 ? <RunSwitcher runs={visibleRuns} selectedRunId={selectedRun.run_id} onSelect={setSelectedRunId} ariaLabel="Probe curve model" /> : undefined}
  />;
}

function OverallRelevantFieldsPanel({ runs, dynamicScope }: { runs: RunSummary[]; dynamicScope: DynamicScope }) {
  const entries = runs.map((run, index) => ({
    run,
    index,
    summary: summaryView(run, dynamicScope).branches?.flip_50?.relevant_fields,
  })).filter((entry) => entry.summary);
  if (!entries.length) return null;
  return <section className="panel chart-card relevant-fields-panel">
    <div className="chart-heading"><div><span className="eyebrow">Rule identification · Default setting</span><h3>Relevant Fields · flip_50</h3></div></div>
    <p className="chart-note">Per-episode field sets are scored without regard to order, then TP/FP/FN counts are micro-pooled per phase before F1 is computed. Pre-change uses the three pre-rule relevant fields as ground truth; Post-Change uses the three post-rule relevant fields. Change outcomes include only episodes with both completed assessments and compare answer sets exactly — field identity only; direction correctness is scored in the Change Recognition panel. Episodes whose answer set stayed identical are classified in the unchanged block.</p>
    <div className="relevant-field-runs">{entries.map(({ run, index, summary }) => {
      if (!summary) return null;
      const phases = [
        { key: "pre", label: "Pre-Change", value: summary.pre_change },
        { key: "post", label: "Post-Change", value: summary.post_change },
      ];
      const outcomes = [
        ["Both correct", summary.changed_outcomes.both_correct],
        ["Pre correct · final incorrect", summary.changed_outcomes.pre_correct_final_incorrect],
        ["Pre incorrect · final correct", summary.changed_outcomes.pre_incorrect_final_correct],
        ["Both incorrect", summary.changed_outcomes.both_incorrect],
      ];
      const unchangedOutcomes = [
        ["Both correct", summary.unchanged_outcomes?.both_correct ?? 0],
        ["Pre correct · final incorrect", summary.unchanged_outcomes?.pre_correct_final_incorrect ?? 0],
        ["Pre incorrect · final correct", summary.unchanged_outcomes?.pre_incorrect_final_correct ?? 0],
        ["Both incorrect", summary.unchanged_outcomes?.both_incorrect ?? 0],
      ];
      return <article className="relevant-field-run" key={run.run_id} style={{ borderTopColor: modelColor(run.model_key, index) }}>
        <h4>{run.label}</h4>
        <div className="relevant-field-phases">{phases.map((phase) => <section key={phase.key}>
          <header><strong>{phase.label}</strong><small>n={phase.value.episode_count}</small></header>
          <dl>
            <div><dt>F1</dt><dd>{phase.value.f1 == null ? "—" : percent(phase.value.f1)}</dd></div>
            <div><dt>Precision</dt><dd>{phase.value.precision == null ? "—" : percent(phase.value.precision)}</dd></div>
            <div><dt>Recall</dt><dd>{phase.value.recall == null ? "—" : percent(phase.value.recall)}</dd></div>
          </dl>
        </section>)}</div>
        <div className="relevant-field-change-summary">
          <header><strong>Answer-set changes</strong><span>{summary.changed_episode_count} / {summary.paired_episode_count} paired episodes changed the answer set</span></header>
          <dl>{outcomes.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
          <header><strong>Unchanged answer sets</strong><span>{summary.unchanged_episode_count} / {summary.paired_episode_count} paired episodes</span></header>
          <dl>{unchangedOutcomes.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
        </div>
      </article>;
    })}</div>
  </section>;
}

function OverallAdaptationPanel({ runs, dynamicScope }: { runs: RunSummary[]; dynamicScope: DynamicScope }) {
  const entries = runs.map((run, index) => ({
    run,
    index,
    summary: summaryView(run, dynamicScope).branches?.flip_50?.relevant_fields,
  })).filter((entry) => entry.summary?.change_detection_outcomes);
  if (!entries.length) return null;
  return <section className="panel chart-card adaptation-panel">
    <div className="chart-heading"><div><span className="eyebrow">Change recognition · Default setting</span><h3>Change Recognition · flip_50</h3></div></div>
    <p className="chart-note">Each episode's predicted and actual rule changes are compared as exact-match atoms — a field leaving or entering the relevant set with its direction, or a direction flip on a retained field — so this is the one headline metric that scores direction. Change-set precision, recall, and F1 are macro-averaged per episode over eligible episodes (both assessments completed). A change type counts as detected only when every atom of that type matches; the atom rows report the per-atom detection rate, so partial replacement detection is visible there. Quadrants cover paired episodes whose rule change contains both a field replacement and a direction reversal.</p>
    <div className="relevant-field-runs">{entries.map(({ run, index, summary }) => {
      if (!summary) return null;
      const atoms = summary.change_atom_detection;
      const quadrants = summary.change_detection_outcomes;
      const atomRows: Array<[string, number, number]> = atoms ? [
        ["Field replacement atoms", atoms.field_replacement_atoms_detected, atoms.field_replacement_atoms_total],
        ["Direction reversal atoms", atoms.direction_reversal_atoms_detected, atoms.direction_reversal_atoms_total],
      ] : [];
      const quadrantRows: Array<[string, number]> = quadrants ? [
        ["Both changes detected", quadrants.both_detected],
        ["Field replacement only", quadrants.field_replacement_only],
        ["Direction reversal only", quadrants.direction_reversal_only],
        ["Neither detected", quadrants.neither_detected],
      ] : [];
      return <article className="relevant-field-run" key={run.run_id} style={{ borderTopColor: modelColor(run.model_key, index) }}>
        <h4>{run.label}</h4>
        <div className="relevant-field-phases">
          <section>
            <header><strong>Change-set scores</strong><small>n={summary.adaptation_episode_count ?? 0} eligible</small></header>
            <dl>
              <div><dt>F1</dt><dd>{summary.change_recognition == null ? "—" : percent(summary.change_recognition)}</dd></div>
              <div><dt>Precision</dt><dd>{summary.adaptation_precision == null ? "—" : percent(summary.adaptation_precision)}</dd></div>
              <div><dt>Recall</dt><dd>{summary.adaptation_recall == null ? "—" : percent(summary.adaptation_recall)}</dd></div>
            </dl>
          </section>
          <section>
            <header><strong>Change atoms detected</strong><small>all atoms of a type must match for detection</small></header>
            <dl>{atomRows.map(([label, detected, total]) => <div key={label}><dt>{label}</dt><dd>{detected}/{total} ({total ? percent(detected / total) : "—"})</dd></div>)}</dl>
          </section>
        </div>
        {quadrants && <div className="relevant-field-change-summary">
          <header><strong>Change detection</strong><span>{quadrants.episode_count} paired episodes with both change types</span></header>
          <dl>{quadrantRows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
        </div>}
      </article>;
    })}</div>
  </section>;
}


export function OverallPanel({ manifest, dynamicScope }: { manifest: Manifest; dynamicScope: DynamicScope }) {
  const runs = manifest.runs.filter((run) => !["oracle", "random"].includes(run.model_key));
  const scopeLabel = dynamicLabel(dynamicScope, true);
  const scopedEpisodeCount = dynamicScope === "all" ? manifest.episode_count : manifest.dynamic_type_counts[dynamicScope] || 0;
  const settings = samplingBranchesForViews(runs, dynamicScope).map((key) => ({ key, label: branchLabel(key) }));
  const flipMetricFor = (run: RunSummary, key: string) => branchMetric(summaryView(run, dynamicScope), "flip_50", key);
  const paretoPoints = runs.map((run, index) => ({
    runId: run.run_id,
    label: paretoRunLabel(run.label),
    color: modelColor(run.model_key, index),
    probeCount: branchProbeCount(summaryView(run, dynamicScope), "flip_50"),
    calibratedAccuracy: flipMetricFor(run, "calibrated_accuracy"),
  }));
  const rankedRuns = runs
    .map((run, index) => ({ run, index, calibrated: flipMetricFor(run, "calibrated_accuracy") }))
    .sort((left, right) => {
      if (left.calibrated == null && right.calibrated == null) return left.index - right.index;
      if (left.calibrated == null) return 1;
      if (right.calibrated == null) return -1;
      return right.calibrated - left.calibrated || left.index - right.index;
    });
  const hasMainMetrics = paretoPoints.some((point) => point.probeCount != null && point.calibratedAccuracy != null);
  return <div className="page-stack overall-page">
    <section className="overall-hero panel">
      <div><span className="eyebrow">Dataset-level evaluation · {scopeLabel}</span><h2>Model performance across {scopedEpisodeCount} episode{scopedEpisodeCount === 1 ? "" : "s"} × {settings.length} branches<span> · default = flip_50</span></h2><p>{manifest.dataset}</p></div>
      <div className="overall-legend">{runs.map((run, index) => <span key={run.run_id}><i style={{ background: modelColor(run.model_key, index) }} />{run.label}</span>)}</div>
    </section>
    <section className="panel main-metrics-panel">
      <div className="chart-heading"><div><span className="eyebrow">Default setting · flip_50</span><h3>Five main metrics</h3></div></div>
      <p className="chart-note">Compare task performance, exploration breadth, factor recognition, change recognition, and post-change verification. Scores below use all episodes in the selected dynamic scope and are ordered by calibrated accuracy.</p>
    </section>
    <section className="model-scorecards">{rankedRuns.map(({ run, index }) => {
      const runView = summaryView(run, dynamicScope);
      const flipCalibrated = branchMetric(runView, "flip_50", "calibrated_accuracy");
      const flipPartialCredit = branchMetric(runView, "flip_50", "partial_credit_accuracy");
      const flipRandomBaseline = branchMetric(runView, "flip_50", "partial_credit_random_baseline");
      const flipUniqueProbes = branchMetric(runView, "flip_50", "mean_unique_probes");
      const flipFactorRecognition = branchMetric(runView, "flip_50", "factor_recognition");
      const flipChangeRecognition = branchMetric(runView, "flip_50", "change_recognition");
      const flipReprobes = branchMetric(runView, "flip_50", "mean_reprobes_after_change");
      const flipBranchView = runView.branches?.["flip_50"] || runView.branches?.["flip_50_full_history"];
      const flipFieldSummary = flipBranchView?.relevant_fields;
      const flipEpisodes = flipBranchView?.episode_count || 0;
      const eligibility = flipFieldSummary?.change_recognition_eligibility ?? null;
      const assessedCount = flipFieldSummary
        ? flipFieldSummary.pre_change.episode_count + flipFieldSummary.post_change.episode_count
        : 0;
      const identificationCoverage = flipFieldSummary && flipEpisodes
        ? ` · valid ${assessedCount}/${flipEpisodes * 2} assessments (${percent(assessedCount / (flipEpisodes * 2))})`
        : "";
      const changeCoverage = eligibility && eligibility.rate != null
        ? ` · eligible ${eligibility.eligible}/${eligibility.total} episodes (${percent(eligibility.rate)})`
        : "";
      return <article className="panel model-scorecard" key={run.run_id} style={{ borderTopColor: modelColor(run.model_key, index) }}>
      <div className="run-title"><span className="status-orb success" /><div><span className="eyebrow">{run.run_id}</span><h2>{run.label}</h2>{SHOW_REASONING && <small>{run.reasoning?.parameter || "reasoning"}: {run.reasoning?.value || "unspecified"}{run.reasoning?.thought_signature_calls ? ` · thought signatures ${run.reasoning.thought_signature_calls}/${run.reasoning.total_calls}` : ""}</small>}</div></div>
      <div className="scorecard-main-metrics">
        <div><strong>{flipCalibrated == null ? "—" : percent(flipCalibrated)}</strong><span>Calibrated Accuracy ↑</span><small>Measures Inductive Performance · main leaderboard metric · flip_50 · chance-corrected: random 0%, oracle 100% · raw partial-credit {flipPartialCredit == null ? "—" : percent(flipPartialCredit)} · random baseline {flipRandomBaseline == null ? "—" : percent(flipRandomBaseline)}</small></div>
        <div><strong>{flipUniqueProbes == null ? "—" : fmt(flipUniqueProbes)}</strong><span>Mean Unique Probes</span><small>Unique probe points per episode; duplicates are deduplicated within pre-/post-change phases.</small></div>
        <div><strong>{flipFactorRecognition == null ? "—" : percent(flipFactorRecognition)}</strong><span>Factor Recognition ↑</span><small>Micro F1 of reported relevant field sets against ground truth, averaged over pre-/post-change assessments.{identificationCoverage}</small></div>
        <div><strong>{flipChangeRecognition == null ? "—" : percent(flipChangeRecognition)}</strong><span>Change Recognition ↑</span><small>F1 of predicted vs actual field-and-direction change atoms, over episodes with both assessments completed.{changeCoverage}</small></div>
        <div><strong>{flipReprobes == null ? "—" : fmt(flipReprobes)}</strong><span>Mean Re-probes after Change</span><small>Distinct pre-change probe points probed again after the rule change, per episode.</small></div>
      </div>
      <RunHyperparameters run={run} />
    </article>; })}</section>
    {hasMainMetrics && <section className="panel chart-card main-metrics-panel">
      <div className="chart-heading"><div><span className="eyebrow">Diagnostic · Pareto frontier</span><h3>Performance and exploration · flip_50</h3></div></div>
      <p className="chart-note">The five headline capabilities are final task performance, exploration breadth, factor recognition, change recognition, and post-change verification. The chart retains the calibrated-accuracy versus probe-demand frontier as a compact diagnostic.</p>
      <ParetoFrontierChart points={paretoPoints} randomBaseline={0} />
    </section>}
    <OverallRelevantFieldsPanel runs={runs} dynamicScope={dynamicScope} />
    <OverallAdaptationPanel runs={runs} dynamicScope={dynamicScope} />
    {settings.length > 0 && <>
      <section className="panel chart-card by-setting-panel">
        <div className="chart-heading"><div><span className="eyebrow">Inductive Performance</span><h3>Calibrated Accuracy by setting ↑</h3></div></div>
        <p className="chart-note">Default (flip_50) is shown first; the remaining {settings.length - 1} settings are diagnostic settings.</p>
        <SettingMetricBars
          settings={settings}
          series={runs.map((run, index) => ({
            runId: run.run_id,
            label: run.label,
            color: modelColor(run.model_key, index),
            values: settings.map(({ key }) => branchMetric(summaryView(run, dynamicScope), key, "calibrated_accuracy")),
          }))}
          format={(value) => percent(value)}
          zeroReference
          ariaLabel="calibrated accuracy by setting"
        />
      </section>
      <section className="panel chart-card by-setting-panel">
        <div className="chart-heading"><div><span className="eyebrow">Exploration Breadth</span><h3>Mean Unique Probes by setting</h3></div></div>
        <p className="chart-note">Default (flip_50) is shown first; the remaining {settings.length - 1} settings are diagnostic settings.</p>
        <SettingMetricBars
          settings={settings}
          series={runs.map((run, index) => ({
            runId: run.run_id,
            label: run.label,
            color: modelColor(run.model_key, index),
            values: settings.map(({ key }) => branchMetric(summaryView(run, dynamicScope), key, "mean_unique_probes")),
          }))}
          format={(value) => fmt(value)}
          ariaLabel="mean unique probes by setting"
        />
      </section>
    </>}
    <OverallBatchAccuracyCurve runs={runs} dynamicScope={dynamicScope} />
    <OverallProbeCurve runs={runs} dynamicScope={dynamicScope} />
    <section className="panel metric-guide"><div className="section-heading"><div><span className="eyebrow">How to read this page</span><h2>Metric definitions</h2></div><span className="privacy-badge">higher/lower direction included</span></div>
      <p className="aggregation-note">Main Metrics are macro-averaged over episode trajectories, except Factor Recognition, which micro-pools TP/FP/FN counts across episodes before computing F1. The five headline metrics separate final task performance, exploration breadth, factor recognition, change recognition, and post-change verification. Relevant-field assessments stay out-of-band and are scored only after the trajectory completes.</p>
      <div className="metric-guide-grid">
        <article><strong>Inductive Performance</strong><code>Calibrated Accuracy = (partial-credit accuracy − random baseline)/(1 − random baseline) · random 0% · oracle 100%</code><p>Main leaderboard dimension on flip_50. Calibrated Accuracy applies a κ-style chance correction to per-case partial credit, so a random agent lands at 0% and the oracle at 100%, while below-random runs go negative.</p></article>
        <article><strong>Mean Unique Probes</strong><code>mean over episodes of |S_pre| + |S_post| · duplicates deduplicated within each phase</code><p>Main leaderboard dimension on flip_50. Every successful probe materializes to a point on the agent-facing value grid; unique probes count each phase's distinct points, and a post-change probe that repeats a pre-change point still counts — that overlap is measured separately by Mean Re-probes after Change. Branches without a rule change report the single-phase count.</p></article>
        <article><strong>Factor Recognition</strong><code>micro F1 of reported relevant-field sets · higher is better</code><p>Uses the private out-of-band field assessments: pre-change reports are pooled against the pre-rule relevant fields and post-change reports against the post-rule fields, and the headline is the mean of the defined phase F1s. Field identity only; direction is scored by Change Recognition.</p></article>
        <article><strong>Change Recognition</strong><code>F1 over the predicted versus actual field-and-direction change atoms · higher is better</code><p>Scores whether the model recognized the identity and direction changes in the latent rule, independently of eventual action recovery. Only episodes whose pre-change and post-change assessments both complete are eligible; the paired-assessment success rate is reported alongside. It is the only main metric that scores direction, and it is undefined on branches without a rule change.</p></article>
        <article><strong>Mean Re-probes after Change</strong><code>mean over episodes of |S_post ∩ S_pre| · re-probing one pre-change point N times counts once</code><p>Counts the distinct pre-change probe points the agent probed again after the rule change — verification effort, not a normalized rate. The value is bounded by the pre-change unique probe count, so it co-varies with exploration breadth. Undefined on branches without a rule change.</p></article>
        <article><strong>Partial-Credit Accuracy ↑</strong><code>mean per-case (u − min)/(max − min) over the case's oracle utilities · range [0,1] · higher is better</code><p>Diagnostic companion to Calibrated Accuracy, which derives from it. Each submitted action earns credit from the case's own oracle utilities — the worst action scores 0, the oracle scores 1, and the credit equals 1 − per-case normalized regret — then credits are averaged within the flip_50 trajectory. It remains available in each scorecard's main-metric block, shown raw next to its random baseline.</p></article>
      </div>
    </section>
  </div>;
}

export function MechanismPanel({ episode }: { episode: Episode }) {
  const directionReversal = episode.dynamic?.event.dynamic_type === "direction_reversal";
  const compound = episode.dynamic?.event.dynamic_type === "compound";
  const compoundOperations = (episode.dynamic?.event.operations || []) as Array<Record<string, any>>;
  const compoundReplacement = compoundOperations.find((operation) => operation.operation_type === "field_replacement");
  const compoundReversal = compoundOperations.find((operation) => operation.operation_type === "direction_reversal");
  return <div className="page-stack">
    {episode.dynamic && <section className="panel">
      <div className="section-heading"><div><span className="eyebrow">Hidden change point</span><h2>{compound ? "Compound dynamic change" : directionReversal ? "Dynamic direction reversal" : "Dynamic field replacement"}</h2></div><span className="privacy-badge">after batch {episode.dynamic.event.after_action_batch}</span></div>
      <div className="parameter-list">
        {compound ? <>
          <div><span>Field replacement</span><code>{compoundReplacement?.outgoing_field} → {compoundReplacement?.incoming_field}</code></div>
          <div><span>Direction reversal</span><code>{compoundReversal?.target_field}: {compoundReversal?.pre_direction} → {compoundReversal?.post_direction}</code></div>
          <div><span>Stable relevant</span><code>{episode.dynamic.event.stable_relevant_field}</code></div>
          <div><span>Permanent distractor</span><code>{episode.dynamic.event.permanent_distractor}</code></div>
        </> : directionReversal
          ? <div><span>Direction change</span><code>{episode.dynamic.event.target_field}: {episode.dynamic.event.pre_direction} → {episode.dynamic.event.post_direction}</code></div>
          : <div><span>Transition</span><code>{episode.dynamic.event.outgoing_field} → {episode.dynamic.event.incoming_field}</code></div>}
        <div><span>Population flip</span><code>{percent(episode.dynamic.population_change.flip_rate)}</code></div>
        <div><span>Pre-change relevant</span><code>{episode.dynamic.pre_dynamic.relevant_fields.join(", ")}</code></div>
        <div><span>Post-change relevant</span><code>{episode.dynamic.post_dynamic.relevant_fields.join(", ")}</code></div>
      </div>
      <p className="rationale">{compound ? "Both hidden operations become active atomically at the same change point." : directionReversal ? episode.dynamic.event.why_direction_reverses : episode.dynamic.event.change_event}</p>
      {compound && <div className="parameter-list event-rationale-list">
        <div><span>Replacement trigger</span><code>{compoundReplacement?.operational_trigger}</code></div>
        <div><span>Replacement rationale</span><code>{compoundReplacement?.change_event || compoundReplacement?.semantic_rationale}</code></div>
        <div><span>Reversal trigger</span><code>{compoundReversal?.operational_trigger}</code></div>
        <div><span>Reversal rationale</span><code>{compoundReversal?.why_direction_reverses}</code></div>
      </div>}
      {episode.dynamic.event.operational_trigger && <>
        <div className="parameter-list event-rationale-list">
          <div><span>Operational trigger</span><code>{episode.dynamic.event.operational_trigger}</code></div>
          {directionReversal ? <>
            <div><span>Before semantics</span><code>{episode.dynamic.event.decision_semantics_before}</code></div>
            <div><span>After semantics</span><code>{episode.dynamic.event.decision_semantics_after}</code></div>
          </> : <>
            <div><span>Authority before</span><code>{episode.dynamic.event.authority_before}</code></div>
            <div><span>Authority after</span><code>{episode.dynamic.event.authority_after}</code></div>
          </>}
          {episode.dynamic.event.effective_scope && <div><span>Effective scope</span><code>{episode.dynamic.event.effective_scope}</code></div>}
          {episode.dynamic.event.expected_duration && <div><span>Expected duration</span><code>{episode.dynamic.event.expected_duration}</code></div>}
          {episode.dynamic.event.ending_condition && <div><span>Ending condition</span><code>{episode.dynamic.event.ending_condition}</code></div>}
        </div>
        <p className="rationale"><strong>Reality check:</strong> {episode.dynamic.event.realism_rationale || episode.dynamic.event.why_incoming_authoritative}</p>
      </>}
    </section>}
    <section className="panel mechanism-flow-panel">
        <div className="section-heading"><div><span className="eyebrow">Evaluator-private ground truth</span><h2>Mechanism flow</h2></div><Eye size={20} /></div>
        <div className="mechanism-flow">
          <div className="flow-fields">{episode.relevant_fields.map((field) => <span key={field}>{field}</span>)}</div>
          <ArrowRight />
          <div className="flow-node family-node" style={{ borderColor: FAMILY_COLORS[episode.mechanism.family] }}><Layers3 size={18} /><strong>{pretty(episode.mechanism.family)}</strong><small>hidden computation</small></div>
          <ArrowRight />
          <div className="flow-node"><Gauge size={18} /><strong>Intervention score</strong><small>continuous · [−1, 1]</small></div>
          <ArrowRight />
          <div className="flow-actions">{episode.scenario.actions.map((action, index) => <span key={action.action_id} style={{ borderColor: ACTION_COLORS[index] }}>{action.action_id}<small>level {index + 1}</small></span>)}</div>
        </div>
        <p className="rationale">{episode.mechanism.domain_rationale}</p>
    </section>
    <section className="two-column rule-comparison">
      {(["pre_change", "post_change"] as const).map((phase) => { const view = episode.mechanism_views[phase]; return <article className="panel" key={phase}>
        <div className="section-heading"><div><span className="eyebrow">Evaluator-private rule parameters</span><h2>{phase === "pre_change" ? "Pre-change rule" : "Post-change rule"}</h2></div><span className="privacy-badge">{view.relevant_fields.join(" · ")}</span></div>
        <div className="parameter-list">{Object.entries(view.mechanism.detail).map(([key, value]) => <div key={key}><span>{pretty(key)}</span><code>{typeof value === "object" ? JSON.stringify(value) : String(value)}</code></div>)}</div>
      </article>; })}
    </section>
    <div className="mechanism-formulas">
      {(["pre_change", "post_change"] as const).map((phase) => <FormulaPanel episode={episode} rulePhase={phase} key={phase} />)}
    </div>
    <section className="panel">
      <div className="section-heading"><div><span className="eyebrow">Input schema</span><h2>Fields used by this episode</h2></div><span className="privacy-badge">truth view</span></div>
      <div className="field-grid">{Object.entries(episode.fields).map(([field, meta]) => {
        const pre = episode.dynamic?.pre_dynamic.relevant_fields.includes(field) ?? episode.relevant_fields.includes(field);
        const post = episode.dynamic?.post_dynamic.relevant_fields.includes(field) ?? pre;
        const relevant = pre || post;
        const reversalTarget = directionReversal && episode.dynamic?.event.target_field === field;
        const role = reversalTarget ? "Direction-reversal target" : pre && post ? "Shared relevant" : pre ? "Outgoing relevant" : post ? "Incoming relevant" : "Permanent distractor";
        return <article key={field} className={`field-card ${relevant ? "relevant" : "distractor"}`}><div><CircleDot size={15} /><strong>{field}</strong></div><span className="field-role">{role}</span><p>{meta.description || "Raw operational field"}</p><small>{meta.minimum} → {meta.maximum}{meta.unit ? ` ${meta.unit}` : ""} · {meta.type}</small></article>;
      })}</div>
    </section>
  </div>;
}

function FormulaPanel({ episode, rulePhase }: { episode: Episode; rulePhase: "pre_change" | "post_change" }) {
  const view = episode.mechanism_views[rulePhase];
  const formula = scoreFormula(view.mechanism_raw);
  const actions = episode.scenario.actions;
  const boundaries = actions.slice(0, -1).map((_, index) => {
    const anchors = actions.length === 4 ? [-.75, -.25, .25, .75] : actions.length === 3 ? [-2 / 3, 0, 2 / 3] : actions.map((_, item) => actions.length > 1 ? -1 + (2 * item) / (actions.length - 1) : 0);
    return (anchors[index] + anchors[index + 1]) / 2;
  });
  return <section className="panel formula-panel">
    <div className="section-heading"><div><span className="eyebrow">Evaluator-private exact computation</span><h2>{rulePhase === "pre_change" ? "Pre-change" : "Post-change"} intervention-score formula</h2></div><span className="privacy-badge">{pretty(view.mechanism.family)}</span></div>
    <p className="rationale">This is the exact {rulePhase === "pre_change" ? "pre-change" : "post-change"} rule, not a fitted approximation.</p>
    <div className="formula-name">{formula.title}</div>
    <pre className="formula-block">{formula.lines.join("\n")}</pre>
    <div className="formula-definitions"><strong>Normalization and helper definitions</strong>{formula.parameters.map((line) => <code key={line}>{line}</code>)}</div>
    <div className="formula-actions"><strong>Score → oracle action</strong>{actions.map((action, index) => {
      const lower = index === 0 ? "−∞" : boundaries[index - 1].toFixed(2);
      const upper = index === actions.length - 1 ? "+∞" : boundaries[index].toFixed(2);
      return <span key={action.action_id}><i style={{ background: ACTION_COLORS[index] }} />{lower} &lt; score ≤ {upper} → {action.action_id}</span>;
    })}</div>
  </section>;
}

type ModelRunData = { summary: RunSummary; data: RunEpisode };

function EpisodeAccuracyCurve({ modelRuns, forkAfterBatch }: { modelRuns: ModelRunData[]; forkAfterBatch: number }) {
  const [selectedRunId, setSelectedRunId] = useState("");
  const selected = modelRuns.find((modelRun) => modelRun.summary.run_id === selectedRunId) || modelRuns[0];
  if (!selected) return null;
  const series = orderedSamplingBranches(Object.keys(selected.data.branches || {})).flatMap((name) => {
    const rows = ((selected.data.branches[name]?.metrics?.batch_metrics || []) as DynamicBatchMetric[]);
    return rows.length ? [{ name, rows }] : [];
  });
  if (!series.length) return null;
  const width = 1040;
  const height = 330;
  const margin = { left: 58, right: 26, top: 24, bottom: 52 };
  const batchCount = Math.max(...series.map(({ rows }) => rows.length));
  const x = (index: number) => margin.left + index * (width - margin.left - margin.right) / Math.max(batchCount - 1, 1);
  const y = (value: number) => height - margin.bottom - value * (height - margin.top - margin.bottom);
  const hasFork = forkAfterBatch > 0 && batchCount > forkAfterBatch;
  const forkX = hasFork ? (x(forkAfterBatch - 1) + x(forkAfterBatch)) / 2 : null;
  return <section className="panel batch-accuracy-panel overall-batch-accuracy">
    <div className="section-heading"><div><span className="eyebrow">Sequential learning</span><h2>Accuracy Curve</h2></div><span className="privacy-badge">{selected.summary.label} · {series.length} branches × {batchCount} batches</span></div>
    <RunSwitcher runs={modelRuns.map((modelRun) => modelRun.summary)} selectedRunId={selected.summary.run_id} onSelect={setSelectedRunId} ariaLabel="Accuracy curve model" />
    <p className="rationale">All branches reuse the exact pre-change conversation, then separate after the Batch {forkAfterBatch} fork: Static IID and the matched no-change control keep the pre-rule, while the flip branches activate the post-change rule.</p>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Episode accuracy curve">
      {[0, .25, .5, .75, 1].map((tick) => <g key={tick}>
        <line className="batch-grid-line" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text className="batch-axis-label" x={margin.left - 9} y={y(tick) + 4} textAnchor="end">{Math.round(tick * 100)}%</text>
      </g>)}
      {Array.from({ length: batchCount }, (_, index) => <g key={index}>
        <line className="batch-tick" x1={x(index)} x2={x(index)} y1={height - margin.bottom} y2={height - margin.bottom + 5} />
        <text className="batch-axis-label" x={x(index)} y={height - 27} textAnchor="middle">B{index + 1}</text>
      </g>)}
      {forkX != null && <g>
        <line x1={forkX} x2={forkX} y1={margin.top} y2={height - margin.bottom} stroke="#c75b4b" strokeDasharray="6 5" strokeWidth="2" />
        <text x={forkX + 6} y={margin.top + 12} fill="#a13e31" fontSize="11">rule change</text>
      </g>}
      {series.map((entry) => <g key={entry.name}>
        <polyline data-series={entry.name} className="batch-line overall-batch-line" style={{ stroke: branchColor(entry.name), strokeWidth: 3, strokeDasharray: branchDash(entry.name) }} points={dynamicSeriesPoints(entry.rows, "active_rule_accuracy", x, y)} />
        {entry.rows.map((row, index) => row.active_rule_accuracy == null ? null : <circle key={row.batch_number || index} cx={x(index)} cy={y(Number(row.active_rule_accuracy))} r="3.5" style={{ fill: branchColor(entry.name) }}>
          <title>{`${selected.summary.label} · ${branchLabel(entry.name)} · batch ${row.batch_number || index + 1}: ${percent(row.active_rule_accuracy)} active-rule accuracy`}</title>
        </circle>)}
      </g>)}
    </svg>
    <div className="batch-chart-legend">
      {series.map((entry) => <span key={entry.name}><i className={`legend-line${branchDash(entry.name) ? " dashed" : ""}`} style={{ background: branchLegendBackground(entry.name) }} />{branchLabel(entry.name)}</span>)}
    </div>
  </section>;
}

export function PerformancePanel({
  episode, modelRuns, selected, onSelect,
}: { episode: Episode; modelRuns: ModelRunData[]; selected: Selection; onSelect: (selection: Selection) => void }) {
  const branchNames = orderedSamplingBranches(modelRuns.flatMap((modelRun) => Object.keys(modelRun.data.branches || {})));
  const batchShape = `${episode.interaction_spec.action_batch_count} batches × ${episode.interaction_spec.action_batch_size} cases per branch`;
  const forkAfterBatch = modelRuns[0]?.data.fork_after_batch ?? episode.dynamic?.event.after_action_batch ?? 0;
  const runDataById = new Map(modelRuns.map((modelRun) => [modelRun.summary.run_id, modelRun.data]));
  const [probeModelId, setProbeModelId] = useState("");
  const probeModel = modelRuns.find((modelRun) => modelRun.summary.run_id === probeModelId) || modelRuns[0];
  const [landscapeModelId, setLandscapeModelId] = useState("");
  const landscapeModel = modelRuns.find((modelRun) => modelRun.summary.run_id === landscapeModelId) || modelRuns[0];
  const landscapeBranchNames = orderedSamplingBranches(Object.keys(landscapeModel?.data.branches || {}));
  const [landscapeBranch, setLandscapeBranch] = useState("");
  const activeLandscapeBranch = landscapeBranchNames.includes(landscapeBranch) ? landscapeBranch : landscapeBranchNames[0] || "";
  const landscapeRun = activeLandscapeBranch ? landscapeModel?.data.branches[activeLandscapeBranch] || null : null;
  const landscapeEpisode = landscapeRun
    ? { ...episode, cases: episode.cases.filter((item) => item.sampling_branch === "shared_prefix" || item.sampling_branch === activeLandscapeBranch || item.sampling_branch === landscapeRun.source_sampling_branch || (isNoChangeControl(activeLandscapeBranch) && item.sampling_branch === sourceSamplingBranch(activeLandscapeBranch))) }
    : episode;
  const landscapeProbes = landscapeRun
    ? landscapeRun.probes.map((probe) => ({
      ...probe,
      context_id: `${landscapeRun.model_key || landscapeRun.run_label || "model"}:${probe.context_id}`,
      source_model: landscapeRun.run_label || landscapeRun.model_key || "Model",
    }))
    : [];
  const utilityRows = branchNames
    .map((name) => ({
      key: name,
      label: <><i className="utility-branch-dot" style={{ background: branchColor(name) }} />{branchLabel(name)}</>,
      bars: modelRuns.flatMap((modelRun) => {
        const bar = utilityComparisonBar(modelRun.summary.run_id, modelRun.summary.label, `${modelRun.summary.label} · ${branchLabel(name)}`, utilitySummary(modelRun.data.branches[name]?.metrics));
        return bar ? [bar] : [];
      }),
    }))
    .filter((row) => row.bars.length > 0);
  const selectedDetail = selected?.type === "case"
    ? landscapeEpisode.cases.find((item) => item.context_id === selected.id)
    : landscapeProbes.find((item) => item.context_id === selected?.id);
  return <div className="page-stack">
    {branchNames.length > 0 && <>
      <section className="panel overall-chart accuracy-breakdown">
        <div className="section-heading"><div><span className="eyebrow">Accuracy slices</span><h2>Accuracy Comparison</h2></div><span className="privacy-badge">{batchShape}</span></div>
        <ComparisonBars rows={branchNames} runs={modelRuns.map((modelRun) => modelRun.summary)} label={branchLabel} value={(run, row) => branchAccuracyMetric(runDataById.get(run.run_id)?.branches[row]?.metrics, row)} />
      </section>
      {utilityRows.length > 0 && <section className="panel overall-chart utility-breakdown">
        <div className="section-heading"><div><span className="eyebrow">Diagnostic · action outcomes</span><h2>Action Utility Comparison</h2></div><span className="privacy-badge">{batchShape}</span></div>
        <UtilityComparisonRows rows={utilityRows} />
        <p className="rationale">{UTILITY_STACK_RATIONALE} Right figure = realized action utility as a share of optimal action utility.</p>
        <UtilityStackLegend />
      </section>}
      <EpisodeAccuracyCurve modelRuns={modelRuns} forkAfterBatch={forkAfterBatch} />
      {probeModel && <ProbeAllocationPanel
        entries={branchNames.flatMap((name) => {
          const branchRun = probeModel.data.branches[name];
          return branchRun ? [{ name, rows: probeRowsFromBranch(branchRun) }] : [];
        })}
        forkAfterBatch={probeModel.data.fork_after_batch || forkAfterBatch}
        note={`${probeModel.summary.label} · probes before each action batch, paired with cumulative action utility when that probe round was chosen. Pre-dynamic bars end at the rule change after Batch ${probeModel.data.fork_after_batch || forkAfterBatch}.`}
        modelSwitch={<RunSwitcher runs={modelRuns.map((modelRun) => modelRun.summary)} selectedRunId={probeModel.summary.run_id} onSelect={setProbeModelId} ariaLabel="Probe curve model" />}
      />}
    </>}
    {landscapeRun && landscapeModel && <section className="panel landscape-panel">
      <div className="landscape-selectors">
        <RunSwitcher runs={modelRuns.map((modelRun) => modelRun.summary)} selectedRunId={landscapeModel.summary.run_id} onSelect={(runId) => { setLandscapeModelId(runId); onSelect(null); }} ariaLabel="Landscape model" />
        <div className="trajectory-run-switch" role="group" aria-label="Landscape branch">{landscapeBranchNames.map((name) => <button key={name} className={name === activeLandscapeBranch ? "active" : ""} onClick={() => { setLandscapeBranch(name); onSelect(null); }}>{name.replaceAll("_", " ")}</button>)}</div>
      </div>
      <UtilityLandscape episode={landscapeEpisode} runs={[landscapeRun]} probes={landscapeProbes} selected={selected} onSelect={onSelect} />
    </section>}
    {selectedDetail && <section className="panel point-inspector">
      <div className="section-heading"><div><span className="eyebrow">Linked selection</span><h2>{selected?.type === "case" ? (selectedDetail as typeof episode.cases[number]).context_id : (selectedDetail as ProbePoint).context_id}</h2></div><button className="text-button" onClick={() => onSelect(null)}>Clear</button></div>
      {selected?.type === "probe" && <p className="rationale">{(selectedDetail as ProbePoint).base_context_id ? <>Counterfactual probe from available context <strong>{(selectedDetail as ProbePoint).base_context_id}</strong>. Changed fields: <code>{Object.keys((selectedDetail as ProbePoint).changes || {}).join(", ") || "none"}</code>.</> : "Synthetic probe context; it is not a private action case."}</p>}
      <div className="value-grid">{Object.entries(selectedDetail.raw_values).map(([field, value]) => <div key={field}><span>{field}</span><strong>{fmt(value)}</strong></div>)}</div>
    </section>}
  </div>;
}

function EventIcon({ event }: { event: TraceEvent }) {
  if (event.valid === false) return event.error_type === "provider_error" ? <AlertTriangle /> : <XCircle />;
  if (event.response?.tool === "probe_batch") return <FlaskConical />;
  if (event.response?.tool === "submit_action_batch") return <CheckCircle2 />;
  return <CircleDot />;
}

function eventTitle(event: TraceEvent) {
  if (event.status === "awaiting_response") return "Awaiting model response";
  if (event.error_type === "provider_error") return "Provider failure";
  if (event.valid === false) return pretty(event.error_type || "Invalid response");
  return pretty(event.response?.tool || "Model interaction");
}

export function TrajectoryPanel({ episode, runSummaries, runs, branch, forkAfterBatch, sharedPrefixInteractions }: { episode: Episode; runSummaries: RunSummary[]; runs: RunBranch[]; branch: string; forkAfterBatch: number; sharedPrefixInteractions: number }) {
  const lanes = runs.filter((run) => run.tool_trace.length || run.termination);
  return <div className="page-stack">
    <section className={`trajectory-summary-grid ${lanes.length === 1 ? "single" : ""}`}>{lanes.map((run, index) => {
      const summary = runSummaries[index];
      return <article className="panel run-summary" key={`${index}-${run.trace_schema_version}`}><div className="run-title"><span className={`status-orb ${run.run_success ? "success" : "failed"}`} /> <div><span className="eyebrow">{summary?.label || `Run ${index + 1}`} · {branch.replaceAll("_", " ")}</span><h2>{run.run_success ? "Completed" : "Failed"}</h2>{SHOW_REASONING && <small>{summary?.reasoning?.parameter || "reasoning"}: {summary?.reasoning?.value || "unspecified"}{summary?.reasoning?.thought_signature_calls ? ` · signatures ${summary.reasoning.thought_signature_calls}/${summary.reasoning.total_calls}` : ""}</small>}</div></div><div className="run-metrics"><Stat label="Completion" value={`${fmt(Number(run.metrics.completion_rate) * 100, 0)}%`} /><Stat label="Accuracy" value={`${fmt(Number(run.metrics.action_accuracy) * 100, 1)}%`} /><Stat label="Mean regret" value={fmt(run.metrics.mean_regret)} /><Stat label="Probe contexts" value={Number(run.metrics.probes || 0)} /></div>{summary && <RunHyperparameters run={summary} />}{run.termination && <p className="termination-line">{String(run.termination.reason || "unknown")} · {String(run.termination.last_error_type || "no error")}</p>}</article>;
    })}</section>
    <section className={`trajectory-lanes ${lanes.length > 1 ? "dual" : ""}`}>{lanes.map((run, index) => <TrajectoryLane key={`${branch}-${index}-${run.trace_schema_version}`} run={run} label={runSummaries[index]?.label || `Run ${index + 1}`} episode={episode} branch={branch} forkAfterBatch={forkAfterBatch} sharedPrefixInteractions={sharedPrefixInteractions} />)}</section>
  </div>;
}

function TrajectoryLane({ run, label, episode, branch, forkAfterBatch, sharedPrefixInteractions }: { run: RunBranch; label: string; episode: Episode; branch: string; forkAfterBatch: number; sharedPrefixInteractions: number }) {
  const sharedCount = Math.min(sharedPrefixInteractions, run.tool_trace.length);
  const selectedBranchLabel = branch.replaceAll("_", " ");
  const conversationEvents = (run.conversation?.events || []) as Array<Record<string, unknown>>;
  const visibleEvents = (run.agent_visible_history?.events || []) as Array<Record<string, unknown>>;
  const persistentMapping = run.agent_visible_history?.event_mapping || [];
  const conversationById = new Map(conversationEvents.map((item) => [String(item.event_id || ""), item] as const).filter(([id]) => Boolean(id)));
  const visibleById = new Map(visibleEvents.map((item) => [String(item.event_id || ""), item] as const).filter(([id]) => Boolean(id)));
  const persistentMappingById = new Map(persistentMapping.map((item) => [String(item.source_event_id || ""), item] as const).filter(([id]) => Boolean(id)));
  const sourceAssistantFor = (event: TraceEvent) => conversationEvents.find((item) => item.type === "assistant_turn" && Number(item.interaction) === Number(event.interaction));
  const sourceEnvironmentFor = (event: TraceEvent) => conversationEvents.find((item) => item.type === "environment_turn" && Number(item.interaction) === Number(event.interaction));
  const projectedEvent = (source: Record<string, unknown> | undefined, disposition: string | undefined, visible: Record<string, unknown> | undefined) => {
    if (!source) return visible;
    if (visible) return visible;
    if (disposition === "rewritten_reveal") {
      const original = source.result && typeof source.result === "object" && !Array.isArray(source.result) ? source.result as Record<string, unknown> : {};
      const batchNumber = source.batch_number ?? "unknown";
      const sensitive = new Set(["pending_action_batch", "required_batch_id", "required_next_tool", "tool_usability", "type", "message"]);
      const retained = Object.fromEntries(Object.entries(original).filter(([key]) => !sensitive.has(key)));
      if (!Object.keys(retained).length) return undefined;
      const message = typeof original.message === "string" && original.message.trim()
        ? `${original.message.trim()} The revealed action cases and transient action state are hidden from later agent context.`
        : original.error && typeof original.error === "object" && !Array.isArray(original.error)
          ? `The environment ended the probe opportunity for action batch ${batchNumber} after the repair limit was exhausted. The rejected tool response and transient action state are hidden from later agent context.`
          : "The revealed action cases and transient state are hidden from later agent context.";
      return { ...source, error: null, state_card: {}, result: { message, ...retained } };
    }
    if (disposition === "rewritten_completion") {
      const original = source.result && typeof source.result === "object" && !Array.isArray(source.result) ? source.result as Record<string, unknown> : {};
      const rows = Array.isArray(original.results) ? original.results.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row) && (row.context_id != null || row.action_case_id != null)).map((row) => ({ context_id: row.context_id ?? row.action_case_id, utility: row.utility })) : [];
      const batchUtility = original.batch_utility ?? rows.reduce((sum, row) => sum + Number(row.utility || 0), 0);
      const state = source.state_card && typeof source.state_card === "object" && !Array.isArray(source.state_card) ? source.state_card as Record<string, unknown> : {};
      const batchNumber = source.batch_number ?? "";
      const failed = Object.prototype.hasOwnProperty.call(original, "failed_action_batch");
      const message = failed
        ? `The environment ended action batch ${batchNumber} after the repair limit was exhausted. No actions were accepted; every case was recorded as unanswered. Invalid action attempts and action-solving messages are hidden from later agent context.`
        : `The environment accepted action batch ${batchNumber}. Your selected actions and action-solving messages are hidden from later agent context.`;
      const result: Record<string, unknown> = {
        message,
        batch_number: batchNumber,
        results: rows,
        batch_utility: batchUtility,
        cumulative_utility: state.cumulative_utility ?? state.action_utility ?? original.cumulative_utility ?? original.action_utility,
        cumulative_probes: state.cumulative_probes ?? state.probe_count ?? original.cumulative_probes ?? original.probe_count,
      };
      return { ...source, error: null, result };
    }
    return visible;
  };
  const projectionAfter = (current: TraceEvent, next: TraceEvent | undefined) => {
    if (!["eval_case_hidden", "outcome_only"].includes(run.history_policy ?? "")) return [];
    const currentMapping = new Map(
      (current.request_context?.event_mapping || []).map((item) => [String(item.source_event_id || ""), item.disposition] as const),
    );
    const targetMapping = next?.request_context?.event_mapping || persistentMapping;
    return targetMapping
      .filter((item) => item.disposition && item.disposition !== "retained")
      .map((item) => ({ item, source: conversationById.get(String(item.source_event_id || "")), visible: visibleById.get(String(item.visible_event_id || "")) }))
      .filter(({ item }) => currentMapping.get(String(item.source_event_id || "")) !== item.disposition);
  };
  const batchFor = (source: Record<string, unknown> | undefined) => String(source?.batch_number ?? "");
  const sourceLabel = (source: Record<string, unknown> | undefined) => source?.type === "assistant_turn" ? "Model response" : "Environment feedback";
  const assistantWireContent = (event: TraceEvent, source: Record<string, unknown> | undefined) =>
    event.raw_response || (typeof source?.visible_content === "string" ? source.visible_content : traceJson(event.response || {}));
  const environmentWireContent = (event: TraceEvent, source: Record<string, unknown> | undefined) => {
    const rawResult = source ? source.result : event.result;
    const stateCard = source?.state_card;
    const result = environmentResultForWire(rawResult, stateCard, source?.stage);
    const sections: string[] = [];
    const hasStateCard = Boolean(stateCard && typeof stateCard === "object" && !Array.isArray(stateCard) && Object.keys(stateCard as Record<string, unknown>).length);
    const hasProbeEvidence = Boolean(
      source?.stage === "reveal"
      && result
      && typeof result === "object"
      && !Array.isArray(result)
      && Array.isArray((result as Record<string, unknown>).results),
    );
    const label = hasProbeEvidence && !hasStateCard ? "Probe result" : "Environment result";
    if (result != null) sections.push(`${label}:\n${traceJson(result)}`);
    if (hasStateCard) sections.push(`Current state:\n${traceJson(stateCard)}`);
    return sections.join("\n\n") || traceJson(event.result || {});
  };
  const renderProjectionDiff = (current: TraceEvent, next: TraceEvent | undefined) => {
    const changes = projectionAfter(current, next);
    if (!changes.length) return null;
    return <div className="history-diff" role="note">
      <div className="history-diff-heading"><Info size={15} /><strong>After this step · {next ? "next request" : "stored agent-visible history"} projection</strong></div>
      {changes.map(({ item, source, visible }) => {
        const disposition = item.disposition || "";
        const removedAction = disposition === "removed_action_solving";
        const removedReveal = disposition === "removed_empty_reveal";
        const removed = removedAction || removedReveal;
        const reveal = disposition === "rewritten_reveal";
        const completion = disposition === "rewritten_completion";
        return <div className="history-diff-row" key={`${item.source_event_id}-${disposition}`}>
          <div className="history-diff-label"><span className="history-projected-label">{removed ? "Deleted" : "Replaced"}</span><strong>{sourceLabel(source)} · {String(source?.event_id || item.source_event_id || "unknown")}</strong></div>
          <p className="history-diff-explanation">{removedAction
            ? `Action-solving ${sourceLabel(source).toLowerCase()} removed; selected actions, calculator work, schema-error repairs, and reasoning from this span do not enter later agent context.`
            : removedReveal
              ? "Batch reveal removed: it contained no probe evidence after pending cases and transient state were hidden."
            : reveal
              ? "Batch reveal rewritten to keep safe probe evidence while removing action cases and transient state."
              : `Completion feedback rewritten as an environment-authored outcome for action batch ${batchFor(source)}; selected actions and solving messages are explicitly hidden.`}</p>
          {completion && Boolean(visible) && <span className="history-black-label">Replacement is shown directly with the environment event above.</span>}
        </div>;
      })}
    </div>;
  };
  const renderEvent = (event: TraceEvent, index: number) => {
    const probes = run.probes.filter((probe) => probe.interaction === event.interaction);
    const actionRows = event.response?.tool === "submit_action_batch" ? run.action_results.filter((item) => {
      const actions = (event.response?.arguments.actions || {}) as Record<string, string>;
      return item.context_id in actions || `action_${String(item.context_id).replace(/^case_/, "")}` in actions;
    }) : [];
    const nextEvent = run.tool_trace[index + 1];
    const sourceAssistant = sourceAssistantFor(event);
    const sourceEnvironment = sourceEnvironmentFor(event);
    const assistantMapping = sourceAssistant ? persistentMappingById.get(String(sourceAssistant.event_id || "")) : undefined;
    const environmentMapping = sourceEnvironment ? persistentMappingById.get(String(sourceEnvironment.event_id || "")) : undefined;
    const assistantRemoved = assistantMapping?.disposition === "removed_action_solving";
    const environmentRemoved = environmentMapping?.disposition === "removed_action_solving" || environmentMapping?.disposition === "removed_empty_reveal";
    const environmentRewritten = environmentMapping?.disposition === "rewritten_reveal" || environmentMapping?.disposition === "rewritten_completion";
    const environmentChanged = environmentRemoved || environmentRewritten;
    const environmentVisible = environmentMapping ? visibleById.get(String(environmentMapping.visible_event_id || "")) : undefined;
    const environmentReplacement = environmentRewritten ? projectedEvent(sourceEnvironment, environmentMapping?.disposition, environmentVisible) : undefined;
    const diff = renderProjectionDiff(event, nextEvent);
    return <details className={`timeline-event ${event.valid === false ? "invalid" : ""}`} key={event.request_id || index} open={event.valid === false}>
      <summary className="timeline-event-header"><span className="event-index">{event.interaction || index + 1}</span><span className="event-icon"><EventIcon event={event} /></span><span className="event-title"><strong>{eventTitle(event)}{(assistantRemoved || environmentChanged) && <span className="event-history-marker">history projected</span>}</strong><small>{fmt(event.latency_seconds, 1)}s · {event.usage?.total_tokens || 0} tokens</small></span><ChevronRight className="event-chevron" size={16} /></summary>
      <div className="event-body">
        {probes.length > 0 && <div className="probe-batch-summary"><div className="batch-stat"><strong>{probes.length}</strong><span>probe contexts</span></div><div className="batch-stat"><strong>{episode.scenario.actions.length}</strong><span>utilities / context</span></div></div>}
        {probes.length > 0 && <div className="mini-probe-list">{probes.map((probe) => <div key={probe.context_id}><span>{probe.context_id}</span><code>s {fmt(probe.score)} · u [{probe.returned_utilities.map((utility) => fmt(utility)).join(", ")}]</code></div>)}</div>}
        {actionRows.length > 0 && <div className="action-batch-list">{actionRows.map((item) => <div key={item.context_id} className={item.correct ? "correct" : "wrong"}><strong>{item.context_id}</strong><span>{item.action_id ?? "unanswered"}</span><span>utility {fmt(item.utility)}</span><span>regret {fmt(item.regret)}</span></div>)}</div>}
        {event.error && <div className="error-callout"><AlertTriangle size={16} />{event.error}</div>}
        {SHOW_REASONING && <details className="trace-detail"><summary><Brain size={15} /> Reasoning</summary><pre>{event.reasoning_content || (event.reasoning_signature_present ? "Minimal reasoning was enabled. The provider returned an encrypted thought signature, but Vertex generateContent did not return a readable thought summary." : "Reasoning was not provided by the model.")}</pre></details>}
        <section className={`trace-detail trace-detail-open ${assistantRemoved ? "history-removed-panel" : ""}`}><h4>Model response · wire content {assistantRemoved && <span className="history-projected-label">deleted from later agent context</span>}</h4><pre className={assistantRemoved ? "history-removed" : ""}>{assistantWireContent(event, sourceAssistant)}</pre></section>
        <section className={`trace-detail trace-detail-open ${environmentChanged ? "history-rewritten-panel" : ""}`}><h4>Environment feedback · wire content {environmentRemoved ? <span className="history-projected-label">deleted from later agent context</span> : environmentRewritten ? <span className="history-projected-label">rewritten for later agent context</span> : null}</h4>{environmentReplacement ? <div className="history-replacement-grid"><div className="history-replacement-column"><span className="history-projected-label">Raw history · removed</span><pre className="history-removed">{environmentWireContent(event, sourceEnvironment)}</pre></div><div className="history-replacement-column"><span className="history-black-label">Agent-visible replacement</span><pre className="history-visible">{environmentWireContent(event, environmentReplacement)}</pre></div></div> : <pre className={environmentChanged ? "history-removed" : ""}>{environmentWireContent(event, sourceEnvironment)}</pre>}</section>
        {diff}
      </div>
    </details>;
  };
  const initialPrompt = run.initial_user_message ? <details className="timeline-event initial-event">
    <summary className="timeline-event-header"><span className="event-index">0</span><span className="event-icon"><CircleDot size={16} /></span><span className="event-title"><strong>Initial prompt</strong><small>system instruction + first user message</small></span><ChevronRight className="event-chevron" size={16} /></summary>
    <div className="event-body"><section className="trace-detail trace-detail-open"><h4>System prompt</h4><pre>{run.conversation?.system_instruction || episode.system_prompt}</pre></section><section className="trace-detail trace-detail-open"><h4>User message</h4><pre>{run.initial_user_message}</pre></section></div>
  </details> : null;
  const answeredCases = run.termination?.completed_count != null
    ? Number(run.termination.completed_count)
    : run.action_results.filter((item) => !item.unanswered).length;
  const totalCases = episode.cases.length;
  const unansweredCases = Math.max(0, totalCases - answeredCases);
  return <div className="trajectory-lane">
    <div className="lane-heading"><strong>{label} · {selectedBranchLabel}</strong><span>{run.tool_trace.length} complete interactions · {run.probes.length} probe contexts</span></div>
    {["eval_case_hidden", "outcome_only"].includes(run.history_policy ?? "") && <div className="history-legend"><span><i className="history-legend-black" />Black content is agent-visible in the next request.</span><span><i className="history-legend-projected" />Blue text marks raw history deleted or replaced by runtime projection.</span></div>}
    {initialPrompt}
    <section className="trajectory-segment shared-segment" aria-label={`Shared prefix through Batch ${forkAfterBatch}`}>
      <div className="segment-heading"><strong>Shared prefix through Batch {forkAfterBatch}</strong><span>{sharedCount} interactions · identical model history</span></div>
      {run.tool_trace.slice(0, sharedCount).map(renderEvent)}
    </section>
    <div className={`trajectory-fork ${branch}`}><span>Fork after Batch {forkAfterBatch}</span><strong>{selectedBranchLabel}</strong></div>
    <section className="trajectory-segment branch-segment" aria-label={`${selectedBranchLabel} after Batch ${forkAfterBatch}`}>
      <div className="segment-heading"><strong>{selectedBranchLabel} after Batch {forkAfterBatch}</strong><span>{run.tool_trace.length - sharedCount} branch-only interactions</span></div>
      {run.tool_trace.slice(sharedCount).map((event, index) => renderEvent(event, sharedCount + index))}
    </section>
    {run.termination && <div className={`terminal-card ${run.run_success ? "success" : "failed"}`}><EventIcon event={{ interaction: 0, valid: run.run_success, error_type: String(run.termination.last_error_type || "") }} /><div><strong>{run.run_success ? "Episode completed" : "Episode terminated"}</strong><p>{String(run.termination.reason || "unknown")} · {answeredCases}/{totalCases} cases answered{unansweredCases > 0 ? ` · ${unansweredCases} unanswered` : ""}</p></div></div>}
  </div>;
}
