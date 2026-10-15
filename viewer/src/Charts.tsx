import { useMemo } from "react";
import { line, scaleBand, scaleLinear } from "d3";
import type { Episode, FieldCurve, ProbePoint, RunBranch } from "./types";

export const ACTION_COLORS = ["#2878b5", "#c88719", "#b44552", "#6f4aa8"];
export const FAMILY_COLORS: Record<string, string> = {
  additive_aggregation: "#278477",
  conjunctive_gating: "#5568a9",
  disjunctive_sufficiency: "#a16a2d",
  lexicographic_priority: "#a74e6b",
};

const fmt = (value: number | null | undefined, digits = 3) =>
  value == null ? "—" : Number(value).toFixed(digits);

function actionAnchors(actionCount: number): number[] {
  if (actionCount === 4) return [-.75, -.25, .25, .75];
  if (actionCount === 3) return [-2 / 3, 0, 2 / 3];
  return Array.from({ length: actionCount }, (_, index) =>
    actionCount > 1 ? -1 + (2 * index) / (actionCount - 1) : 0,
  );
}

function actionBoundaries(actionCount: number): number[] {
  const anchors = actionAnchors(actionCount);
  return anchors.slice(0, -1).map((anchor, index) => (anchor + anchors[index + 1]) / 2);
}

export function utilitiesFromScore(score: number, actionCount = 3): number[] {
  const anchors = actionAnchors(actionCount);
  const beta = actionCount === 4 ? 8 : 3;
  const logits = anchors.map((anchor) => -beta * (score - anchor) ** 2);
  const peak = Math.max(...logits);
  const weights = logits.map((value) => Math.exp(value - peak));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((weight) => (actionCount * weight / total - 1) / (actionCount - 1));
}

function ActionLegend({ episode, line = false }: { episode: Episode; line?: boolean }) {
  return <div className="legend-group" aria-label="Action color legend">
    <strong>Color = action</strong>
    <div className="legend-items">{episode.scenario.actions.map((action, index) =>
      <span key={action.action_id}><i className={line ? "legend-line" : "legend-color-dot"} style={{ background: ACTION_COLORS[index] }} />{action.action_id}</span>,
    )}</div>
  </div>;
}

type Selection = { type: "case" | "probe"; id: string } | null;

export function FieldResponseChart({
  curve,
  episode,
  probes,
  selected,
  onSelect,
}: {
  curve: FieldCurve;
  episode: Episode;
  probes: ProbePoint[];
  selected: Selection;
  onSelect: (selection: Selection) => void;
}) {
  const width = 760;
  const height = 245;
  const margin = { left: 44, right: 18, top: 20, bottom: 36 };
  const x = scaleLinear().domain([0, 1]).range([margin.left, width - margin.right]);
  const y = scaleLinear().domain([-1.05, 1.05]).range([height - margin.bottom, margin.top]);
  const boundaries = actionBoundaries(episode.scenario.actions.length);
  const scorePath = useMemo(
    () => line<(typeof curve.samples)[number]>()
      .defined((item) => item.feasible && item.score != null)
      .x((item) => x(item.normalized_x))
      .y((item) => y(item.score as number))(curve.samples) || "",
    [curve],
  );
  const meta = episode.fields[curve.field_id];
  const probeModels = [...new Set(probes.map((item) => item.source_model || "Model"))];
  const fixedRelevant = episode.relevant_fields
    .filter((field) => field !== curve.field_id)
    .map((field) => [field, episode.reference_values[field]] as const);

  return (
    <article className="chart-card">
      <div className="chart-heading">
        <div>
          <span className="eyebrow">One-variable slice of the hidden function</span>
          <h3>{curve.field_id}</h3>
        </div>
        <span className="reference-pill">reference {curve.reference_value}</span>
      </div>
      <div className="slice-definition">
        <strong>What this graph shows</strong>
        <span><code>{curve.field_id}</code> varies along X; Y is the resulting hidden intervention score.</span>
        <span><b>Held fixed at reference:</b> {fixedRelevant.length ? fixedRelevant.map(([field, value]) => <code key={field}>{field} = {value}</code>) : "none"}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${curve.field_id} intervention score response`}>
        {episode.scenario.actions.map((_, index) => {
          const lower = index === 0 ? -1.05 : boundaries[index - 1];
          const upper = index === episode.scenario.actions.length - 1 ? 1.05 : boundaries[index];
          return <rect key={`band-${index}`} x={margin.left} y={y(upper)} width={width - margin.left - margin.right} height={y(lower) - y(upper)} fill={ACTION_COLORS[index]} opacity=".045" />;
        })}
        {[ -1, -0.5, 0, 0.5, 1 ].map((tick) => (
          <g key={tick}>
            <line className="grid-line" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
            <text className="axis-label" x={margin.left - 8} y={y(tick) + 4} textAnchor="end">{tick}</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line className="tick-line" x1={x(tick)} x2={x(tick)} y1={height - margin.bottom} y2={height - margin.bottom + 5} />
            <text className="axis-label" x={x(tick)} y={height - 12} textAnchor="middle">
              {fmt(meta.minimum + tick * (meta.maximum - meta.minimum), meta.type === "integer" ? 0 : 1)}
            </text>
          </g>
        ))}
        {boundaries.map((boundary) => <g key={boundary}><line className="score-boundary" x1={margin.left} x2={width-margin.right} y1={y(boundary)} y2={y(boundary)} /><text className="boundary-label" x={width-margin.right-3} y={y(boundary)-4} textAnchor="end">action boundary {fmt(boundary)}</text></g>)}
        <line className="reference-line" x1={x((curve.reference_value-meta.minimum)/(meta.maximum-meta.minimum))} x2={x((curve.reference_value-meta.minimum)/(meta.maximum-meta.minimum))} y1={margin.top} y2={height-margin.bottom} />
        <path d={scorePath} fill="none" stroke="#263b34" strokeWidth="2.6" />
        {episode.cases.map((item) => {
          const action = item.oracle_action_index;
          const isSelected = selected?.type === "case" && selected.id === item.context_id;
          return (
            <circle
              key={item.context_id}
              className="data-point"
              cx={x(item.normalized_values[curve.field_id])}
              cy={y(item.score)}
              r={isSelected ? 6 : 3.2}
              fill={ACTION_COLORS[action]}
              opacity={isSelected ? 1 : 0.48}
              onClick={() => onSelect({ type: "case", id: item.context_id })}
            >
              <title>{`${item.context_id} · full-context score ${fmt(item.score)} · oracle action_${action}`}</title>
            </circle>
          );
        })}
        {probes.map((item) => {
          const isSelected = selected?.type === "probe" && selected.id === item.context_id;
          const cx = x(item.normalized_values[curve.field_id]);
          const cy = y(item.score);
          return (
            <rect key={`${item.context_id}-${item.interaction}`} className="data-point probe-point" x={cx - (isSelected ? 5 : 4)} y={cy - (isSelected ? 5 : 4)} width={isSelected ? 10 : 8} height={isSelected ? 10 : 8} transform={`rotate(45 ${cx} ${cy})`} fill="#fff" stroke={ACTION_COLORS[item.oracle_action_index]} strokeWidth={isSelected ? 3 : 2} onClick={() => onSelect({ type: "probe", id: item.context_id })}><title>{`${item.source_model || "Model"} · ${item.context_id} · I${item.interaction} · full-context score ${fmt(item.score)} · implied oracle action_${item.oracle_action_index} · all action utilities returned · ${item.base_context_id ? `counterfactual from ${item.base_context_id} · changed ${Object.keys(item.changes || {}).join(", ")}` : "synthetic new context"}`}</title></rect>
          );
        })}
        <text className="axis-title" x={width / 2} y={height - 1} textAnchor="middle">{curve.field_id}{meta.unit ? ` (${meta.unit})` : ""}</text>
        <text className="axis-title" transform={`translate(12 ${height / 2}) rotate(-90)`} textAnchor="middle">Intervention score</text>
      </svg>
      <div className="chart-legend">
        <div className="legend-group"><strong>Background band = implied oracle action</strong><div className="legend-items">{episode.scenario.actions.map((action,index)=><span key={action.action_id}><i className="legend-line" style={{background:ACTION_COLORS[index]}} />{action.action_id}</span>)}</div></div>
        <div className="legend-group"><strong>Mark = source</strong><div className="legend-items">
          <span><i className="legend-case" />Evaluation case · full-context score</span>
          {probeModels.map((model) => <span key={model}><i className="legend-probe" />{model} probe · full-context score (◇)</span>)}
        </div></div>
      </div>
      <p className="chart-note">This is a <strong>one-dimensional conditional slice</strong> through the full {episode.relevant_fields.length}-field mechanism: <strong>score = f({curve.field_id} varies; {fixedRelevant.length} other relevant field{fixedRelevant.length === 1 ? "" : "s"} fixed)</strong>. The dark line belongs to that slice. Circles and diamonds instead use each case’s complete context, so vertical distance from the line reflects different values in the other fields—not model error.</p>
    </article>
  );
}

export type ParetoFrontierPoint = {
  runId: string;
  label: string;
  color: string;
  probeCount: number | null;
  calibratedAccuracy: number | null;
};

// A point dominates another when it is at least as good on both axes (accuracy
// >=, probes <=) and strictly better on at least one; exact ties dominate nothing.
export const paretoFrontierIndices = (
  points: Array<Pick<ParetoFrontierPoint, "probeCount" | "calibratedAccuracy">>,
): number[] =>
  points.flatMap((point, index) => {
    const { probeCount, calibratedAccuracy } = point;
    if (probeCount == null || calibratedAccuracy == null) return [];
    const dominated = points.some((other, otherIndex) => {
      if (otherIndex === index || other.probeCount == null || other.calibratedAccuracy == null) return false;
      return other.calibratedAccuracy >= calibratedAccuracy
        && other.probeCount <= probeCount
        && (other.calibratedAccuracy > calibratedAccuracy || other.probeCount < probeCount);
    });
    return dominated ? [] : [index];
  });

const paddedDomain = (values: number[]): [number, number] => {
  if (!values.length) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [min - 1, max + 1];
  const pad = (max - min) * .08;
  return [min - pad, max + pad];
};

export function ParetoFrontierChart({ points, randomBaseline = null }: { points: ParetoFrontierPoint[]; randomBaseline?: number | null }) {
  const width = 1080;
  const height = 380;
  const margin = { left: 56, right: 26, top: 20, bottom: 44 };
  const x = scaleLinear()
    .domain(paddedDomain(points.map((point) => point.probeCount).filter((value): value is number => value != null)))
    .range([margin.left, width - margin.right]);
  const y = scaleLinear()
    .domain(paddedDomain([0, ...points.map((point) => point.calibratedAccuracy).filter((value): value is number => value != null)]))
    .range([height - margin.bottom, margin.top]);
  const frontier = new Set(paretoFrontierIndices(points));
  const frontierPath = line<ParetoFrontierPoint>()
    .defined((point) => point.probeCount != null && point.calibratedAccuracy != null)
    .x((point) => x(point.probeCount as number))
    .y((point) => y(point.calibratedAccuracy as number))(
      points
        .map((point, index) => ({ point, index }))
        .filter(({ index }) => frontier.has(index))
        .sort((left, right) => (left.point.probeCount as number) - (right.point.probeCount as number))
        .map(({ point }) => point),
    ) || "";
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="inductive performance versus exploration demand in the default setting">
      {y.ticks(5).map((tick) => (
        <g key={`y-${tick}`}>
          <line className="grid-line" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
          <text className="axis-label" x={margin.left - 8} y={y(tick) + 4} textAnchor="end">{tick}</text>
        </g>
      ))}
      {x.ticks(6).map((tick) => (
        <g key={`x-${tick}`}>
          <line className="tick-line" x1={x(tick)} x2={x(tick)} y1={height - margin.bottom} y2={height - margin.bottom + 5} />
          <text className="axis-label" x={x(tick)} y={height - 16} textAnchor="middle">{tick}</text>
        </g>
      ))}
      {randomBaseline != null && <g>
        <line className="pareto-random-line" x1={margin.left} x2={width - margin.right} y1={y(randomBaseline)} y2={y(randomBaseline)} />
        <text className="pareto-random-label" x={width - margin.right - 3} y={y(randomBaseline) - 5} textAnchor="end">random</text>
      </g>}
      {frontierPath && <path className="pareto-frontier-line" d={frontierPath} />}
      {points.map((point, index) => {
        if (point.probeCount == null || point.calibratedAccuracy == null) return null;
        const cx = x(point.probeCount);
        const cy = y(point.calibratedAccuracy);
        const labelOnLeft = cx > width * .72;
        return <g key={point.runId}>
          <circle
            className="data-point pareto-point"
            data-testid="pareto-point"
            data-run-id={point.runId}
            data-frontier={frontier.has(index) ? "true" : "false"}
            cx={cx}
            cy={cy}
            r={6}
            fill={point.color}
            stroke={frontier.has(index) ? "#26332e" : "#fbfcf8"}
            strokeWidth={frontier.has(index) ? 3 : 1.5}
          >
            <title>{`${point.label} · calibrated accuracy ${(point.calibratedAccuracy * 100).toFixed(1)}% · probe cases ${point.probeCount.toFixed(2)}`}</title>
          </circle>
          <text
            className="pareto-model-label"
            data-testid="pareto-model-label"
            x={cx + (labelOnLeft ? -10 : 10)}
            y={cy < margin.top + 18 ? cy + 18 : cy - 10}
            textAnchor={labelOnLeft ? "end" : "start"}
          >{point.label}</text>
        </g>;
      })}
      <text className="axis-title" x={width / 2} y={height - 1} textAnchor="middle">Exploration Demand · mean probe cases per episode ↓</text>
      <text className="axis-title" transform={`translate(14 ${height / 2}) rotate(-90)`} textAnchor="middle">Inductive Performance · Calibrated Accuracy ↑</text>
    </svg>
  );
}

export type SettingMetricSetting = { key: string; label: string };
export type SettingMetricSeries = { runId: string; label: string; color: string; values: Array<number | null> };

export function SettingMetricBars({
  settings,
  series,
  format,
  zeroReference = false,
  ariaLabel,
}: {
  settings: SettingMetricSetting[];
  series: SettingMetricSeries[];
  format: (value: number) => string;
  zeroReference?: boolean;
  ariaLabel: string;
}) {
  if (!settings.length) return null;
  const width = 1080;
  const height = 340;
  const margin = { left: 52, right: 20, top: 24, bottom: 40 };
  const plotHeight = height - margin.top - margin.bottom;
  const x = scaleBand<string>()
    .domain(settings.map(({ key }) => key))
    .range([margin.left, width - margin.right])
    .paddingInner(.25)
    .paddingOuter(.12);
  const subBand = scaleBand<string>()
    .domain(series.map(({ runId }) => runId))
    .range([0, x.bandwidth()])
    .paddingInner(.18);
  const plotted = series.flatMap(({ values }) => values.filter((value): value is number => value != null));
  const [minValue, maxValue] = paddedDomain(plotted);
  const y = scaleLinear().domain([minValue, maxValue]).range([height - margin.bottom, margin.top]);
  const baseline = y(Math.min(Math.max(minValue, 0), maxValue));
  const showZeroLine = zeroReference && minValue < 0 && maxValue > 0;
  const separatorX = settings.length > 1
    ? (x(settings[0].key) ?? 0) + x.bandwidth() + (x.step() - x.bandwidth()) / 2
    : null;
  return (
    <div className="setting-bars">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
      {separatorX != null && <rect className="setting-band-default" x={margin.left} y={margin.top} width={separatorX - margin.left} height={plotHeight} />}
      {y.ticks(5).map((tick) => (
        <g key={`y-${tick}`}>
          <line className="grid-line" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
          <text className="axis-label" x={margin.left - 8} y={y(tick) + 4} textAnchor="end">{format(tick)}</text>
        </g>
      ))}
      {showZeroLine && <line className="setting-zero-line" x1={margin.left} x2={width - margin.right} y1={y(0)} y2={y(0)} />}
      {separatorX != null && <g>
        <line className="setting-separator" x1={separatorX} x2={separatorX} y1={margin.top} y2={height - margin.bottom} />
        <text className="setting-separator-label" x={separatorX + 6} y={margin.top + 10}>diagnostics</text>
      </g>}
      {series.map(({ runId, label, color, values }) => settings.map(({ key, label: settingLabel }, settingIndex) => {
        const value = values[settingIndex];
        if (value == null) return null;
        const barX = (x(key) ?? 0) + (subBand(runId) ?? 0);
        const barWidth = subBand.bandwidth();
        const barTop = Math.min(y(value), baseline);
        const barHeight = Math.max(Math.abs(baseline - y(value)), 1);
        return <g key={`${runId}-${key}`} className="setting-bar">
          <rect x={barX} y={barTop} width={barWidth} height={barHeight} fill={color}>
            <title>{`${label} · ${settingLabel} · ${format(value)}`}</title>
          </rect>
          <text className="setting-value-label" x={barX + barWidth / 2} y={value >= 0 ? barTop - 5 : barTop + barHeight + 12} textAnchor="middle">{format(value)}</text>
        </g>;
      }))}
      {settings.map(({ key, label }, index) => (
        <text key={key} className={`axis-label${index === 0 ? " setting-label-default" : ""}`} x={(x(key) ?? 0) + x.bandwidth() / 2} y={height - 14} textAnchor="middle">{label}</text>
      ))}
      </svg>
      <div className="batch-chart-legend setting-bars-legend" aria-label="Model run color legend">
        {series.map(({ runId, label, color }) => <span key={runId}><i style={{ background: color }} />{label}</span>)}
      </div>
    </div>
  );
}

export function UtilityLandscape({
  episode,
  runs,
  probes,
  selected,
  onSelect,
}: {
  episode: Episode;
  runs: RunBranch[];
  probes: ProbePoint[];
  selected: Selection;
  onSelect: (selection: Selection) => void;
}) {
  const width = 920, height = 390;
  const margin = { left: 50, right: 24, top: 22, bottom: 82 };
  const x = scaleLinear().domain([-1, 1]).range([margin.left, width - margin.right]);
  const y = scaleLinear().domain([-.55, 1.05]).range([height - margin.bottom, margin.top]);
  const curve = Array.from({ length: 101 }, (_, index) => -1 + index / 50);
  const actionPaths = episode.scenario.actions.map((_, action) => {
    return line<number>().x((score) => x(score)).y((score) => y(utilitiesFromScore(score, episode.scenario.actions.length)[action]))(curve) || "";
  });
  const scoredResults = runs.flatMap((run, runIndex) => run.action_results.map((result) => ({ result, runIndex, label: run.run_label || `Model ${runIndex + 1}` })));
  const resultGroups = useMemo(() => {
    const grouped = new Map<string, typeof scoredResults>();
    for (const row of scoredResults) {
      const utility = row.result.unanswered ? 0 : row.result.utility;
      const key = `${row.runIndex}|${row.result.score}|${utility}|${row.result.unanswered ? "unanswered" : "answered"}`;
      grouped.set(key, [...(grouped.get(key) || []), row]);
    }
    return [...grouped.values()];
  }, [scoredResults]);
  const modelLabels = [...new Set(runs.map((run, index) => run.run_label || run.model_key || `Model ${index + 1}`))];
  const probeCount = probes.length;
  const probeGroups = useMemo(() => {
    const grouped = new Map<string, Array<{ probe: ProbePoint; actionIndex: number; utility: number }>>();
    for (const probe of probes) {
      probe.returned_utilities.forEach((returned, actionIndex) => {
        const utility = returned ?? probe.utilities[actionIndex];
        const key = `${probe.source_model || "Model"}|${probe.score}|${actionIndex}|${utility}`;
        grouped.set(key, [...(grouped.get(key) || []), { probe, actionIndex, utility }]);
      });
    }
    return [...grouped.values()];
  }, [probes]);
  const answeredCount = scoredResults.filter(({ result }) => !result.unanswered).length;
  const answeredCoordinates = resultGroups.filter((group) => !group[0].result.unanswered).length;
  const actionCaseCount = episode.cases.length;
  const probeBudget = Number(episode.interaction_spec.probe_budget ?? 0);
  const unansweredRailY = height - 15;
  return (
    <article className="chart-card landscape-card">
      <div className="chart-heading">
        <div><span className="eyebrow">Decision landscape</span><h3>Utility V.S. Intervention Score</h3></div>
        <div className="chart-counts" aria-label="Model interaction totals">
          <span><strong>{probeCount}</strong> / {probeBudget} probe contexts · {probeGroups.length} visible coordinates</span>
          <span><strong>{answeredCount}</strong> / {actionCaseCount} actions · {answeredCoordinates} visible coordinates</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Score and utility landscape">
        {[-0.5, 0, 0.5, 1].map((tick) => <g key={`y${tick}`}><line className="grid-line" x1={margin.left} x2={width-margin.right} y1={y(tick)} y2={y(tick)} /><text className="axis-label" x={margin.left-8} y={y(tick)+4} textAnchor="end">{tick}</text></g>)}
        {[-1, -0.5, 0, 0.5, 1].map((tick) => <g key={`x${tick}`}><text className="axis-label" x={x(tick)} y={height-61} textAnchor="middle">{tick}</text></g>)}
        {actionPaths.map((path, index) => <path key={index} data-testid={`utility-curve-${index}`} data-mapping="centered-softmax" d={path} fill="none" stroke={ACTION_COLORS[index]} strokeWidth="2.5" />)}
        <line className="unanswered-rail" x1={margin.left} x2={width - margin.right} y1={unansweredRailY} y2={unansweredRailY} />
        <text className="unanswered-rail-label" x={margin.left} y={unansweredRailY - 6}>dedicated unanswered tray · scored utility = 0</text>
        {resultGroups.map((group) => {
          const { result: item, runIndex, label } = group[0];
          const isSelected = selected?.type === "case" && group.some((row) => row.result.context_id === selected.id);
          const episodeCase = episode.cases.find((row) => row.context_id === item.context_id);
          const oracleActionId = episode.scenario.actions[item.oracle_action_index]?.action_id;
          const oracleUtility = (oracleActionId ? item.oracle?.utilities?.[oracleActionId] : undefined) ?? episodeCase?.utilities[item.oracle_action_index];
          const pointX = x(item.score) + (runIndex - (runs.length - 1) / 2) * 5;
          const plottedUtility = item.unanswered ? 0 : item.utility;
          const selectedActions = [...new Set(group.map((row) => row.result.selected_action_index).filter((action) => action != null))] as number[];
          const actionColor = selectedActions.length === 1 ? ACTION_COLORS[selectedActions[0]] : "#4e5a55";
          const contexts = group.map((row) => row.result.context_id).join(", ");
          return <g key={`${runIndex}-${item.score}-${plottedUtility}-${item.unanswered}`} onClick={() => onSelect({ type: "case", id: item.context_id })} className="data-point">
            {!item.unanswered && oracleUtility != null && <g className="regret-gap"><line x1={pointX} x2={pointX} y1={y(item.utility)} y2={y(oracleUtility)} /><line x1={pointX - 3} x2={pointX + 3} y1={y(oracleUtility)} y2={y(oracleUtility)} /></g>}
            {item.unanswered ? <g data-testid="unanswered-point" data-utility={plottedUtility} data-display-lane="unanswered" stroke="#9b2f3f" strokeWidth={isSelected ? 3.5 : 2.5} opacity={isSelected ? 1 : .82}>
              <line x1={pointX - 5} x2={pointX + 5} y1={unansweredRailY - 5} y2={unansweredRailY + 5} />
              <line x1={pointX + 5} x2={pointX - 5} y1={unansweredRailY - 5} y2={unansweredRailY + 5} />
              <title>{`${label} · ${group.length} unanswered cases · utility 0\n${contexts}`}</title>
            </g> : <rect data-display-lane="scored" x={pointX - (isSelected ? 6 : 4)} y={y(plottedUtility) - (isSelected ? 6 : 4)} width={isSelected ? 12 : 8} height={isSelected ? 12 : 8} rx="1" fill={actionColor} opacity={isSelected ? 1 : .68}><title>{`${label} · ${group.length} answered case${group.length === 1 ? "" : "s"} · ${item.action_id} · utility ${fmt(plottedUtility)} · regret ${fmt(item.regret)}\n${contexts}`}</title></rect>}
            {group.length > 1 && <text className="result-multiplicity" x={pointX + 8} y={(item.unanswered ? unansweredRailY : y(plottedUtility)) + 12}>×{group.length}</text>}
          </g>;
        })}
        {probeGroups.map((group) => {
          const { probe: item, actionIndex, utility } = group[0];
          const px = x(item.score), py = y(utility);
          const isSelected = selected?.type === "probe" && group.some(({ probe }) => probe.context_id === selected.id);
          const size = isSelected ? 13 : 9;
          const detail = group.map(({ probe }) => `${probe.context_id}: all action utilities${probe.base_context_id ? ` from ${probe.base_context_id} + {${Object.keys(probe.changes || {}).join(", ")}}` : " synthetic"}`).join("\n");
          return <g key={`${item.source_model}-${item.score}-${actionIndex}-${utility}`} className="data-point" onClick={() => onSelect({ type: "probe", id: item.context_id })}>
            <rect x={px-size/2} y={py-size/2} width={size} height={size} transform={`rotate(45 ${px} ${py})`} fill="#fff" stroke={ACTION_COLORS[actionIndex]} strokeWidth="2.2"><title>{`${group.length} probe utility point${group.length === 1 ? "" : "s"} for action_${actionIndex}\n${detail}`}</title></rect>
            {group.length > 1 && <text className="probe-multiplicity" x={px + 8} y={py - 8}>×{group.length}</text>}
          </g>;
        })}
        <text className="axis-title" x={width/2} y={height-43} textAnchor="middle">Intervention score</text>
        <text className="axis-title" transform={`translate(14 ${height/2}) rotate(-90)`} textAnchor="middle">Utility</text>
      </svg>
      <div className="chart-legend landscape-legend">
        <ActionLegend episode={episode} line />
        <div className="legend-group"><strong>Mark = model result</strong><div className="legend-items">
          {modelLabels.map((model) => <span key={model}><i className="legend-scored square" />{model} scored answer</span>)}
          {modelLabels.map((model) => <span key={`${model}-probe`}><i className="legend-probe" />{model} counterfactual probe (◇)</span>)}
          <span><i className="legend-multiplicity">×N</i>Overlapping probe queries at one coordinate</span>
          <span><i className="legend-regret" />Regret gap · top cap = oracle utility</span>
          <span><i className="legend-unanswered" />Unanswered (×) · utility fixed at 0</span>
        </div></div>
      </div>
      <p className="chart-note"><strong>X:</strong> evaluator-private hidden score for the context. <strong>Y:</strong> utility for each probed action or the selected scored action. Every context probe returns {episode.scenario.actions.length} utilities and therefore places one diamond on each action curve. A probe based on an action case is a counterfactual edit made <strong>after that case’s submitted answer was locked</strong>. Unanswered cases are scored at utility 0 and placed in the separate bottom tray.</p>
    </article>
  );
}

export function MechanismHeatmap({ episode }: { episode: Episode }) {
  if (!episode.heatmap) return null;
  const { heatmap } = episode;
  const width = 520, height = 455;
  const margin = { left: 66, right: 22, top: 20, bottom: 50 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const x = scaleLinear().domain([0, heatmap.size]).range([margin.left, margin.left + innerWidth]);
  const color = scaleLinear<string>().domain([-1, 0, 1]).range(["#355f91", "#f4f1e8", "#a13f50"]);
  const xMeta = episode.fields[heatmap.x_field], yMeta = episode.fields[heatmap.y_field];
  const fixedRelevant = episode.relevant_fields
    .filter((field) => field !== heatmap.x_field && field !== heatmap.y_field)
    .map((field) => [field, episode.reference_values[field]] as const);
  const xValue = scaleLinear().domain([xMeta.minimum, xMeta.maximum]).range([margin.left, margin.left + innerWidth]);
  const yValue = scaleLinear().domain([yMeta.minimum, yMeta.maximum]).range([margin.top + innerHeight, margin.top]);
  const ticks = [0, .5, 1];
  return <article className="chart-card heatmap-card">
    <div className="chart-heading"><div><span className="eyebrow">Two-field mechanism surface</span><h3>{heatmap.x_field} × {heatmap.y_field}</h3></div></div>
    <div className="slice-definition">
      <strong>What this graph shows</strong>
      <span><code>{heatmap.x_field}</code> varies along X, <code>{heatmap.y_field}</code> varies along Y, and cell color is the hidden intervention score.</span>
      <span><b>Held fixed at reference:</b> {fixedRelevant.length ? fixedRelevant.map(([field, value]) => <code key={field}>{field} = {value}</code>) : "none—the full relevant mechanism is shown"}</span>
    </div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Mechanism interaction heatmap">
      {heatmap.cells.map((cell) => <rect key={`${cell.xi}-${cell.yi}`} x={x(cell.xi)} y={margin.top + innerHeight - ((cell.yi + 1) * innerHeight / heatmap.size)} width={innerWidth/heatmap.size + .5} height={innerHeight/heatmap.size + .5} fill={cell.feasible && cell.score != null ? color(cell.score) : "#d6d7d2"}><title>{`${heatmap.x_field}=${cell.x}, ${heatmap.y_field}=${cell.y}, score=${fmt(cell.score)}`}</title></rect>)}
      {heatmap.region && <rect className="exception-region" x={xValue(heatmap.region.x[0])} y={yValue(heatmap.region.y[1])} width={xValue(heatmap.region.x[1]) - xValue(heatmap.region.x[0])} height={yValue(heatmap.region.y[0]) - yValue(heatmap.region.y[1])} />}
      {ticks.map((tick) => <g key={`x-${tick}`}><line className="tick-line" x1={margin.left + tick * innerWidth} x2={margin.left + tick * innerWidth} y1={margin.top + innerHeight} y2={margin.top + innerHeight + 5} /><text className="axis-label" x={margin.left + tick * innerWidth} y={height - 27} textAnchor="middle">{fmt(xMeta.minimum + tick * (xMeta.maximum - xMeta.minimum), xMeta.type === "integer" ? 0 : 1)}</text></g>)}
      {ticks.map((tick) => <g key={`y-${tick}`}><line className="tick-line" x1={margin.left - 5} x2={margin.left} y1={margin.top + (1 - tick) * innerHeight} y2={margin.top + (1 - tick) * innerHeight} /><text className="axis-label" x={margin.left - 9} y={margin.top + (1 - tick) * innerHeight + 3} textAnchor="end">{fmt(yMeta.minimum + tick * (yMeta.maximum - yMeta.minimum), yMeta.type === "integer" ? 0 : 1)}</text></g>)}
      <text className="axis-title" x={margin.left+innerWidth/2} y={height-2} textAnchor="middle">{heatmap.x_field}{xMeta.unit ? ` (${xMeta.unit})` : ""}</text>
      <text className="axis-title" transform={`translate(13 ${margin.top+innerHeight/2}) rotate(-90)`} textAnchor="middle">{heatmap.y_field}{yMeta.unit ? ` (${yMeta.unit})` : ""}</text>
    </svg>
    <div className="heatmap-legend" aria-label="Heatmap legend"><strong>Cell color = hidden intervention score</strong><div className="color-scale"><i /><div><span>−1 · low</span><span>0 · neutral</span><span>+1 · high</span></div></div><div className="legend-items"><span><i className="legend-infeasible" />Infeasible field combination</span>{heatmap.region && <span><i className="legend-exception" />Exception region</span>}</div></div>
    <p className="chart-note">This is a <strong>two-dimensional conditional slice</strong> through the full {episode.relevant_fields.length}-field mechanism: two fields vary and {fixedRelevant.length} other relevant field{fixedRelevant.length === 1 ? " is" : "s are"} fixed. It shows <strong>hidden intervention score, not utility</strong>; the upper chart maps this score to each action’s utility.</p>
  </article>;
}
