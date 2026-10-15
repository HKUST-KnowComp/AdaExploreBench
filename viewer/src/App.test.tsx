import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App, { isolateEpisodeBranch } from "./App";
import { environmentResultForWire, modelColor, orderedSamplingBranches, overviewSamplingBranches, paretoRunLabel, TrajectoryPanel } from "./Panels";
import type { Episode, RunBranch } from "./types";

test("environment wire result omits fields duplicated by current state", () => {
  const pending = { batch_id: "batch_001", cases: [{ context_id: "case_001" }] };
  const usability = { probe_batch: false, submit_action_batch: true };
  expect(environmentResultForWire(
    { type: "probe_batch_result", results: [{ context_id: "probe_001" }], pending_action_batch: pending, tool_usability: usability, cumulative_utility: 3, cumulative_probes: 1 },
    { tool_usability: usability, cumulative_utility: 3, cumulative_probes: 1, pending_action_batch: pending },
  )).toEqual({ results: [{ context_id: "probe_001" }] });
});

const dynamicBatches = Array.from({ length: 12 }, (_, index) => ({
  batch_number: index + 1,
  case_count: 8,
  answered_cases: 8,
  active_rule_accuracy: .5 + index * .01,
  partial_credit_accuracy: .45 + index * .01,
  partial_credit_random_baseline: .33,
  old_rule_alignment: index >= 5 ? .7 - index * .01 : null,
  stale_policy_accuracy: index >= 5 ? .45 : null,
  flip_adoption_accuracy: index >= 5 ? .5 + (index - 5) * .02 : null,
  nonflip_stability_accuracy: index >= 5 ? .8 : null,
  cumulative_utility: index * 3,
  action_utility: index * 3,
  probe_count: index,
  cumulative_probe_cases_before_batch: index,
  flipped_case_breakdown: {
    denominator: index >= 5 ? 4 : 0,
    new_rule_adoption: index >= 5 ? 2 : 0,
    stale_rule_adherence: index >= 5 ? 1 : 0,
    neither_rule: index >= 5 ? 1 : 0,
  },
}));

const probeByBatch = Array.from({ length: 12 }, (_, index) => ({
  batch_number: index + 1,
  probe_cases: (index + 1) % 3,
  action_utility_on_hand: index * 3,
}));

const branchAccuracies: Record<string, number> = { static_iid: .8, flip_25: .6, flip_50: .45, flip_50_no_change_control: .78, flip_75: .3 };
const branchProbeCases: Record<string, number> = { static_iid: 6, flip_25: 11, flip_50: 9.5, flip_50_no_change_control: 5.5, flip_75: 13 };
const branchCalibratedAccuracy: Record<string, number | null> = { static_iid: .55, flip_25: .38, flip_50_no_change_control: .52, flip_75: .21 };
const branchUtilities: Record<string, { cumulative: number; regret: number }> = {
  static_iid: { cumulative: 40, regret: 5 },
  flip_25: { cumulative: 36, regret: 9 },
  flip_50: { cumulative: 30, regret: 15 },
  flip_50_no_change_control: { cumulative: 40, regret: 5 },
  flip_75: { cumulative: 24, regret: 21 },
};
const relevantFieldSummary = {
  pre_change: { episode_count: 5, precision: .8, recall: .7, f1: .74 },
  post_change: { episode_count: 5, precision: .6, recall: .5, f1: .54 },
  final: { episode_count: 0, precision: null, recall: null, f1: null },
  paired_episode_count: 5,
  changed_episode_count: 4,
  unchanged_episode_count: 1,
  factor_recognition: .64,
  factor_recognition_pre: .74,
  factor_recognition_post: .54,
  factor_recognition_final: null,
  change_recognition: .575,
  change_recognition_eligibility: { eligible: 5, total: 5, rate: 1 },
  adaptation_episode_count: 5,
  adaptation_precision: .75,
  adaptation_recall: .5,
  change_atom_detection: {
    field_replacement_atoms_detected: 6,
    field_replacement_atoms_total: 10,
    direction_reversal_atoms_detected: 2,
    direction_reversal_atoms_total: 5,
  },
  change_detection_outcomes: {
    both_detected: 2,
    field_replacement_only: 1,
    direction_reversal_only: 1,
    neither_detected: 1,
    episode_count: 5,
  },
  unchanged_outcomes: {
    both_correct: 0,
    pre_correct_final_incorrect: 0,
    pre_incorrect_final_correct: 0,
    both_incorrect: 1,
  },
  changed_outcomes: {
    both_correct: 1,
    pre_correct_final_incorrect: 1,
    pre_incorrect_final_correct: 1,
    both_incorrect: 1,
  },
};
const samplingBranchViews = Object.fromEntries(
  ["static_iid", "flip_25", "flip_50", "flip_50_no_change_control", "flip_75"].map((name) => {
    const utility = branchUtilities[name];
    const noChangeControl = name.endsWith("_no_change_control");
    return [name, {
    episode_count: name === "flip_50" ? 5 : 1,
    metrics: { action_accuracy: branchAccuracies[name], static_overall_accuracy: name === "static_iid" || noChangeControl ? branchAccuracies[name] : null, dynamic_post_rule_accuracy: name === "static_iid" || noChangeControl ? null : branchAccuracies[name], probe_cases: branchProbeCases[name] ?? 8, cumulative_utility: utility.cumulative, action_utility: utility.cumulative, cumulative_regret: utility.regret, mean_unique_probes: name === "static_iid" || noChangeControl ? 7.25 : 9.5, mean_reprobes_after_change: name === "static_iid" || noChangeControl ? null : 3, factor_recognition: name === "static_iid" || noChangeControl ? .7 : .64, change_recognition: name === "static_iid" || noChangeControl ? null : .3, ...(name === "flip_50" ? { calibrated_accuracy: .43, partial_credit_accuracy: .62, partial_credit_random_baseline: .33 } : { calibrated_accuracy: branchCalibratedAccuracy[name] ?? null }) },
    batch_accuracy: [],
    batch_dynamic: { all: dynamicBatches },
    probe_by_batch: probeByBatch,
    probe_pre_dynamic_cases: 10,
    probe_post_dynamic_cases: 5,
    relevant_fields: name === "flip_50" ? relevantFieldSummary : null,
    breakdowns: { by_family: {}, by_domain: {} },
  }]; }),
);
const allView = {
  episode_count: 4,
  metrics: { action_accuracy: .5, cumulative_utility: 40, action_utility: 40, cumulative_regret: 5, probe_cases: 8, mean_unique_probes: 8.75, mean_reprobes_after_change: 2.5, factor_recognition: .66, change_recognition: .31 },
  batch_accuracy: [],
  batch_dynamic: { all: dynamicBatches },
  probe_by_batch: probeByBatch,
  probe_pre_dynamic_cases: 10,
  probe_post_dynamic_cases: 5,
  breakdowns: {
    by_family: { additive_aggregation: { action_accuracy: .5, cumulative_utility: 40, action_utility: 40, cumulative_regret: 5 } },
    by_domain: { finance: { action_accuracy: .5, cumulative_utility: 40, action_utility: 40, cumulative_regret: 5 } },
  },
  branches: samplingBranchViews,
};

const manifest = {
  title: "AdaExploreBench v3.2 Research Viewer", release: { name: "AdaExploreBench v3.2", slug: "v3_2_three_dynamic", version: "3.2.0" }, dataset: "test", episode_count: 1,
  benchmark_config: { schema_version: "adaexplorebench_v3_dynamic", probe_budget: 176, max_probe_rounds: 11 },
  family_counts: { additive_aggregation: 1 },
  dynamic_type_counts: { field_replacement: 0, direction_reversal: 1, compound: 0 },
  runs: [
    { run_id: "run_deepseek", label: "DeepSeek", model_key: "deepseek", episode_ids: ["test_episode"], metrics: { action_accuracy: .4, completion_rate: 1, action_utility: 5, probe_count: 8, mean_regret: .3, family_diagnostic_accuracy: .35 }, batch_accuracy: [{ batch_number: 1, accuracy: .5, correct_cases: 4, answered_cases: 8, expected_cases: 8 }], batch_dynamic: { all: dynamicBatches }, breakdowns: { by_family: { additive_aggregation: { action_accuracy: .4 } }, by_domain: { finance: { action_accuracy: .4 } } }, hyperparameters: { model: "deepseek-v4-flash", conversation_protocol: "deepseek_chat", temperature: 0, max_tokens: 32768, effort: "low", output_format: "json_object", seed: 7, policy: "agent", max_probe_cases_per_batch: 16, allow_abstention: true, branches: "static_iid, flip_25, flip_50, flip_75" }, views: { all: allView } },
    { run_id: "run_gemini", label: "Gemini", model_key: "gemini", episode_ids: ["test_episode"], metrics: { action_accuracy: .5, completion_rate: 1, action_utility: 6, probe_count: 7, mean_regret: .2, family_diagnostic_accuracy: .45 }, batch_accuracy: [{ batch_number: 1, accuracy: .625, correct_cases: 5, answered_cases: 8, expected_cases: 8 }], batch_dynamic: { all: dynamicBatches }, breakdowns: { by_family: { additive_aggregation: { action_accuracy: .5 } }, by_domain: { finance: { action_accuracy: .5 } } }, views: { all: allView } },
  ],
  episodes: [{ episode_id: "test_episode", domain: "finance", rule_family: "additive_aggregation", dynamic_type: "direction_reversal", sampling_branches: ["static_iid", "flip_50"], after_action_batch: 5, raw_field_count: 3, relevant_field_count: 2, distractor_field_count: 1, runs: ["run_deepseek", "run_gemini"] }],
};

const branch = {
  run_id: "run_deepseek", run_label: "DeepSeek", model_key: "deepseek",
  run_status: "completed", run_success: true, termination: { reason: "all_cases_completed" },
  metrics: {
    action_accuracy: .75, cumulative_utility: 30, action_utility: 30, cumulative_regret: 15, probe_count: 9.5,
    batch_metrics: dynamicBatches,
    probe_metrics: { by_batch: dynamicBatches.map((row) => ({ batch_number: row.batch_number, probe_batches: 1, probe_cases: row.batch_number % 3 })) },
  }, probes: [], action_results: [], tool_trace: [
    { request_id: "request_001", interaction: 1, valid: true, request_messages: [{ role: "system", content: "instructions" }, { role: "user", content: "initial state" }], raw_response: "first raw", response: { tool: "probe_batch", arguments: {} }, result: { ok: true }, usage: {} },
    { request_id: "request_002", interaction: 2, valid: true, request_messages: [{ role: "system", content: "instructions" }, { role: "user", content: "initial state" }, { role: "assistant", content: "first raw" }, { role: "user", content: "next state" }], raw_response: "second raw", response: { tool: "request_action_batch", arguments: {} }, result: { ok: true }, usage: {} },
  ], trace_schema_version: "test",
  conversation: { schema_version: "adaexplorebench_benchmark_conversation_v1", system_instruction: "instructions", system_prompt_hash: "abc", task_spec: {}, events: [] },
  initial_user_message: "Input fields:\n\nCurrent state:\n{}",
};

const run = {
  episode_id: "test_episode", run_id: "run_deepseek", run_label: "DeepSeek", model_key: "deepseek",
  fork_after_batch: 5, shared_prefix_interactions: 1,
  run_status: "completed", run_success: true, trace_schema_version: "test",
  branches: { flip_50: branch, flip_50_no_change_control: { ...branch, run_label: "DeepSeek · no-change control" } },
};

const episode = {
  episode_id: "test_episode", schema_version: "adaexplorebench_v3_dynamic", setting: "dynamic",
  taxonomy: { rule_family: "additive_aggregation" },
  scenario: { domain: "finance", template_id: "test_template", agent_role: "operator", actions: [{ action_id: "action_0" }, { action_id: "action_1" }, { action_id: "action_2" }] },
  interaction_spec: { probe_budget: 32, action_case_count: 96, action_batch_count: 12, action_batch_size: 8, max_probe_rounds: 11, max_interactions: 30 },
  fields: { a: { type: "number", minimum: 0, maximum: 1 }, b: { type: "number", minimum: 0, maximum: 1 }, c: { type: "number", minimum: 0, maximum: 1 } },
  relevant_fields: ["a", "b"], distractor_fields: ["c"], constraints: [],
  mechanism: { family: "additive_aggregation", detail: { weights: { a: 1, b: 1 }, bias: 0 }, domain_rationale: "test" },
  mechanism_raw: { family: "additive_aggregation", relevant_fields: ["a", "b"], field_ranges: { a: { minimum: 0, maximum: 1 }, b: { minimum: 0, maximum: 1 } }, weights: { a: 1, b: 1 }, bias: 0, score_calibration: { method: "zero_preserving_balanced_action_quantiles", negative_anchor: -1, positive_anchor: 1 } }, reference_values: { a: .5, b: .5, c: .5 }, system_prompt: "prompt", cases: [], field_curves: [],
  mechanism_views: {
    pre_change: { relevant_fields: ["a", "b"], mechanism: { family: "additive_aggregation", detail: { weights: { a: 1, b: 1 }, bias: 0 }, domain_rationale: "pre" }, mechanism_raw: { family: "additive_aggregation", relevant_fields: ["a", "b"], field_ranges: { a: { minimum: 0, maximum: 1 }, b: { minimum: 0, maximum: 1 } }, weights: { a: 1, b: 1 }, bias: 0, score_calibration: { method: "zero_preserving_balanced_action_quantiles", negative_anchor: -1, positive_anchor: 1 } }, field_curves: [], heatmap: null },
    post_change: { relevant_fields: ["a", "b"], mechanism: { family: "additive_aggregation", detail: { weights: { a: -1, b: 1 }, bias: 0 }, domain_rationale: "post" }, mechanism_raw: { family: "additive_aggregation", relevant_fields: ["a", "b"], field_ranges: { a: { minimum: 0, maximum: 1 }, b: { minimum: 0, maximum: 1 } }, weights: { a: -1, b: 1 }, bias: 0, score_calibration: { method: "zero_preserving_balanced_action_quantiles", negative_anchor: -1, positive_anchor: 1 } }, field_curves: [], heatmap: null },
  },
  dynamic: { event: { dynamic_type: "direction_reversal", after_action_batch: 5, target_field: "a", pre_direction: 1, post_direction: -1, sampling_design: { candidate_slot_id: "candidate_001", available_post_branches: ["static_iid", "flip_50"] } }, pre_dynamic: { relevant_fields: ["a", "b"], mechanism: { family: "additive_aggregation", detail: {}, domain_rationale: "pre" }, mechanism_raw: {} }, post_dynamic: { relevant_fields: ["a", "b"], mechanism: { family: "additive_aggregation", detail: {}, domain_rationale: "post" }, mechanism_raw: {} }, population_change: { flip_rate: .5, action_transition_matrix: [], mean_score_shift: 0, mean_absolute_score_shift: .4 } },
  heatmap: { x_field: "a", y_field: "b", size: 1, region: null, cells: [{ x: .5, y: .5, xi: 0, yi: 0, feasible: true, score: 0 }] }, generation_diagnostics: {},
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn((path: string) => Promise.resolve({
    ok: true,
    json: () => Promise.resolve(path.includes("manifest") ? manifest : path.includes("/runs/") ? run : episode),
  })));
});

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

test("keeps discovered control branches in the canonical order", () => {
  expect(orderedSamplingBranches(["flip_75", "unseen_branch", "flip_50_no_change_control", "static_iid", "flip_50"])).toEqual([
    "flip_50",
    "flip_50_no_change_control",
    "static_iid",
    "flip_75",
    "unseen_branch",
  ]);
});

test("merges legacy full-history aliases out of overview setting comparisons", () => {
  expect(overviewSamplingBranches([
    "flip_50_eval_case_given",
    "flip_50_full_history",
    "flip_75",
    "flip_50",
  ])).toEqual(["flip_50", "flip_75", "flip_50_eval_case_given"]);
});

test("assigns distinct colors as more runs from any model family are added", () => {
  const colors = Array.from({ length: 100 }, (_, index) => modelColor("gpt", index));
  expect(new Set(colors).size).toBe(colors.length);
});

test("omits the MMDD suffix only from Pareto point labels", () => {
  expect(paretoRunLabel("GPT-5.6 Luna High 0910")).toBe("GPT-5.6 Luna High");
  expect(paretoRunLabel("Model without date")).toBe("Model without date");
});

test("isolates shared-prefix and selected-branch cases", () => {
  const source = {
    cases: [
      { context_id: "pre", sampling_branch: "shared_prefix" },
      { context_id: "static", sampling_branch: "static_iid" },
      { context_id: "flip25", sampling_branch: "flip_25" },
      { context_id: "flip50", sampling_branch: "flip_50" },
    ],
  } as unknown as Episode;
  expect(isolateEpisodeBranch(source, "flip_25").cases.map((item) => item.context_id)).toEqual([
    "pre",
    "flip25",
  ]);
  expect(isolateEpisodeBranch(source, "flip_50_no_change_control").cases.map((item) => item.context_id)).toEqual([
    "pre",
    "flip50",
  ]);
  expect(isolateEpisodeBranch(source, "flip_50_probe_required_least4", "flip_50").cases.map((item) => item.context_id)).toEqual([
    "pre",
    "flip50",
  ]);
});

test("loads an overall model comparison by default", async () => {
  const rendered = render(<App />);
  expect((await screen.findAllByText("Overview")).length).toBeGreaterThan(0);
  expect(screen.getByText("Model performance across 1 episode × 5 branches")).toBeInTheDocument();
  expect(screen.getByText("· default = flip_50")).toBeInTheDocument();
  expect(screen.queryByText("Accuracy Comparison")).not.toBeInTheDocument();
  expect(screen.queryByText("Action Utility Comparison")).not.toBeInTheDocument();
  expect(screen.getByText("Partial-Credit Accuracy Curve")).toBeInTheDocument();
  expect(screen.getByText("Probe Allocation")).toBeInTheDocument();
  const partialCreditCurve = screen.getByRole("img", { name: /sampling-branch partial-credit accuracy comparison/ });
  expect(partialCreditCurve.querySelectorAll('[data-series="partial-credit-random-baseline"]')).toHaveLength(1);
  expect(partialCreditCurve.querySelectorAll('[data-series="sampling-flip_50"]')).toHaveLength(1);
  expect(screen.queryByText(/Static overall accuracy/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Dynamic overall accuracy/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Static first-N accuracy/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Dynamic post-rule accuracy/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Probe efficiency/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Dynamic adaptation score/)).not.toBeInTheDocument();
  expect(screen.getAllByText("Inductive Performance").length).toBeGreaterThanOrEqual(2);
  expect(screen.getAllByText("Partial-Credit Accuracy ↑").length).toBeGreaterThanOrEqual(1);
  rendered.container.querySelectorAll(".model-scorecard").forEach((card) => {
    expect(within(card as HTMLElement).getByText("Calibrated Accuracy ↑")).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("Mean Unique Probes")).toBeInTheDocument();
    for (const label of ["Factor Recognition ↑", "Change Recognition ↑", "Mean Re-probes after Change", "64.0%", "30.0%", "3.00"]) {
      expect(within(card as HTMLElement).getByText(label)).toBeInTheDocument();
    }
    expect(card.querySelectorAll(".scorecard-main-metrics > div")).toHaveLength(5);
    expect(within(card as HTMLElement).getByText(/Measures Inductive Performance/)).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("Unique probe points per episode; duplicates are deduplicated within pre-/post-change phases.")).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText(/Micro F1 of reported relevant field sets against ground truth, averaged over pre-\/post-change assessments/)).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText(/F1 of predicted vs actual field-and-direction change atoms, over episodes with both assessments completed/)).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("Distinct pre-change probe points probed again after the rule change, per episode.")).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("43.0%")).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText(/raw partial-credit 62\.0% · random baseline 33\.0%/)).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText(/valid 10\/10 assessments \(100\.0%\)/)).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText(/eligible 5\/5 episodes \(100\.0%\)/)).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("9.50")).toBeInTheDocument();
  });
  expect(screen.getByText("Five main metrics")).toBeInTheDocument();
  expect(screen.getByText("Relevant Fields · flip_50")).toBeInTheDocument();
  expect(screen.getByText("Change Recognition · flip_50")).toBeInTheDocument();
  const relevantFieldRuns = rendered.container.querySelectorAll(".relevant-fields-panel .relevant-field-run");
  expect(relevantFieldRuns).toHaveLength(2);
  relevantFieldRuns.forEach((card) => {
    expect(within(card as HTMLElement).getByText("Pre-Change")).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("Post-Change")).toBeInTheDocument();
    expect(within(card as HTMLElement).getAllByText("n=5")).toHaveLength(2);
    expect(within(card as HTMLElement).getByText(/4 \/ 5 paired episodes changed the answer set/)).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("Unchanged answer sets")).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("1 / 5 paired episodes")).toBeInTheDocument();
    expect(within(card as HTMLElement).getAllByText("1")).toHaveLength(5);
  });
  expect(screen.getByText(/direction correctness is scored in the Change Recognition panel/)).toBeInTheDocument();
  const adaptationRuns = rendered.container.querySelectorAll(".adaptation-panel .relevant-field-run");
  expect(adaptationRuns).toHaveLength(2);
  adaptationRuns.forEach((card) => {
    const view = within(card as HTMLElement);
    expect(view.getByText("Change-set scores")).toBeInTheDocument();
    expect(view.getByText("n=5 eligible")).toBeInTheDocument();
    expect(view.getByText("57.5%")).toBeInTheDocument();
    expect(view.getByText("75.0%")).toBeInTheDocument();
    expect(view.getByText("50.0%")).toBeInTheDocument();
    expect(view.getByText("Field replacement atoms")).toBeInTheDocument();
    expect(view.getByText("6/10 (60.0%)")).toBeInTheDocument();
    expect(view.getByText("Direction reversal atoms")).toBeInTheDocument();
    expect(view.getByText("2/5 (40.0%)")).toBeInTheDocument();
    expect(view.getByText("5 paired episodes with both change types")).toBeInTheDocument();
    expect(view.getByText("Both changes detected")).toBeInTheDocument();
    expect(view.getByText("Field replacement only")).toBeInTheDocument();
    expect(view.getByText("Direction reversal only")).toBeInTheDocument();
    expect(view.getByText("Neither detected")).toBeInTheDocument();
    expect(view.getAllByText("1")).toHaveLength(3);
    expect(view.getAllByText("2")).toHaveLength(1);
  });
  const paretoChart = screen.getByRole("img", { name: "inductive performance versus exploration demand in the default setting" });
  expect(paretoChart.querySelectorAll('[data-testid="pareto-point"]')).toHaveLength(2);
  expect(paretoChart.querySelectorAll('[data-frontier="true"]')).toHaveLength(2);
  expect(paretoChart.querySelector(".pareto-random-line")).not.toBeNull();
  expect(paretoChart.querySelector(".pareto-frontier-line")).not.toBeNull();
  const accuracyBySetting = screen.getByRole("img", { name: "calibrated accuracy by setting" });
  const probesBySetting = screen.getByRole("img", { name: "mean unique probes by setting" });
  expect(accuracyBySetting.querySelectorAll(".setting-bar")).toHaveLength(10);
  expect(probesBySetting.querySelectorAll(".setting-bar")).toHaveLength(10);
  const defaultSettingLabels = rendered.container.querySelectorAll(".setting-label-default");
  expect(defaultSettingLabels).toHaveLength(2);
  defaultSettingLabels.forEach((label) => expect(label.textContent).toBe("Default"));
  expect(rendered.container.querySelectorAll(".setting-separator")).toHaveLength(2);
  expect(rendered.container.querySelectorAll(".setting-separator-label")).toHaveLength(2);
  expect(rendered.container.querySelectorAll(".setting-zero-line")).toHaveLength(0);
  expect(rendered.container.querySelectorAll(".model-scorecard .scorecard-metric-grid")).toHaveLength(0);
  expect(rendered.container.querySelectorAll(".model-scorecard .scorecard-grid")).toHaveLength(0);
  expect(screen.getByText("Metric definitions")).toBeInTheDocument();
  expect(screen.getByText(/Main Metrics are macro-averaged/)).toBeInTheDocument();
  expect(screen.getAllByText("Exploration Breadth").length).toBeGreaterThanOrEqual(1);
  expect(screen.queryByText("Accuracy by stratum")).not.toBeInTheDocument();
  expect(screen.queryByText("Diagnostic accuracy")).not.toBeInTheDocument();
  expect(screen.queryByText("Boundary accuracy ↑")).not.toBeInTheDocument();
  expect(screen.queryByText("Efficiency and usage")).not.toBeInTheDocument();
});

test("renders run hyperparameters in the overall scorecards", async () => {
  render(<App />);
  expect((await screen.findAllByText("Overview")).length).toBeGreaterThan(0);
  expect(screen.getAllByText("Run hyperparameters")).toHaveLength(1);
  expect(screen.getByText("deepseek-v4-flash")).toBeInTheDocument();
  expect(screen.getByText("static_iid, flip_25, flip_50, flip_75")).toBeInTheDocument();
});

test("ranks scorecards by flip_50 partial-credit accuracy with ties in manifest order", async () => {
  const geminiView = {
    ...allView,
    branches: {
      ...samplingBranchViews,
      flip_50: { ...samplingBranchViews.flip_50, metrics: { ...samplingBranchViews.flip_50.metrics, calibrated_accuracy: .53, partial_credit_accuracy: .7 } },
    },
  };
  const rankedManifest = { ...manifest, runs: [manifest.runs[0], { ...manifest.runs[1], views: { all: geminiView } }] };
  vi.stubGlobal("fetch", vi.fn((path: string) => Promise.resolve({
    ok: true,
    json: () => Promise.resolve(path.includes("manifest") ? rankedManifest : path.includes("/runs/") ? run : episode),
  })));
  const rendered = render(<App />);
  expect((await screen.findAllByText("Overview")).length).toBeGreaterThan(0);
  const cards = rendered.container.querySelectorAll(".model-scorecard");
  expect(cards).toHaveLength(2);
  expect(cards[0].textContent).toContain("Gemini");
  expect(cards[1].textContent).toContain("DeepSeek");
  const frontierPoints = rendered.container.querySelectorAll('[data-testid="pareto-point"][data-frontier="true"]');
  expect(frontierPoints).toHaveLength(1);
});

test("only offers dynamic-type tabs with episodes", async () => {
  render(<App />);
  expect(await screen.findByText("Model performance across 1 episode × 5 branches")).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "All dynamics" })).toHaveLength(1);
  expect(screen.queryByRole("button", { name: "Direction reversal" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Reversal" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Field replacement" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Replacement" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Compound" })).not.toBeInTheDocument();
});

test("splits overall metrics and the sidebar by dynamic type", async () => {
  render(<App />);
  expect(await screen.findByText("Model performance across 1 episode × 5 branches")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Reversal" }));
  expect(screen.getByText("Model performance across 1 episode × 0 branches")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /test_episode/i })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "All dynamics" }));
  expect(screen.getByText("Model performance across 1 episode × 5 branches")).toBeInTheDocument();
});

test("opens the episode explorer and current schema", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByRole("button", { name: /test_episode/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /test_episode/i }));
  expect(await screen.findByText("adaexplorebench_v3_dynamic")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^Mechanism$/i })).toBeInTheDocument();
});

test("performance shows branch metrics and one decision landscape", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByRole("button", { name: /test_episode/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /test_episode/i }));
  fireEvent.click(await screen.findByRole("button", { name: /^Performance$/i }));
  expect(await screen.findByText("Accuracy Comparison")).toBeInTheDocument();
  expect(screen.getByText("Action Utility Comparison")).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "DeepSeek · Default action utility breakdown" })).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Gemini · Default action utility breakdown" })).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "DeepSeek · 50% flip · no-change control action utility breakdown" })).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Gemini · 50% flip · no-change control action utility breakdown" })).toBeInTheDocument();
  expect(screen.getAllByTitle(/Default · realized share of optimal action utility/)).toHaveLength(2);
  expect(screen.queryByText(/ · N /)).not.toBeInTheDocument();
  expect(screen.getByText("Accuracy Curve")).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Episode accuracy curve" })).toBeInTheDocument();
  expect(screen.getByText("Probe Allocation")).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Default probe allocation by action batch" })).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "50% flip · no-change control probe allocation by action batch" })).toBeInTheDocument();
  expect(screen.getAllByRole("img", { name: "Score and utility landscape" })).toHaveLength(1);
  expect(screen.getByText("Utility V.S. Intervention Score")).toBeInTheDocument();
  expect(screen.queryByRole("group", { name: "Select model" })).not.toBeInTheDocument();
  expect(screen.queryByRole("group", { name: "Select evaluation branch" })).not.toBeInTheDocument();
  expect(screen.queryByText("System prompt")).not.toBeInTheDocument();
  expect(screen.queryByText("Rule family")).not.toBeInTheDocument();
  expect(screen.queryByText("Interaction cap")).not.toBeInTheDocument();
});

test("separates mechanism truth from model performance", async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByRole("button", { name: /test_episode/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /test_episode/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Mechanism$/i })).toBeInTheDocument());
  expect(screen.queryByRole("group", { name: "Select model" })).not.toBeInTheDocument();
  expect(screen.getByText("Mechanism flow")).toBeInTheDocument();
  expect(screen.getByText("Dynamic direction reversal")).toBeInTheDocument();
  expect(screen.getByText("Pre-change intervention-score formula")).toBeInTheDocument();
  expect(screen.getByText("Post-change intervention-score formula")).toBeInTheDocument();
  expect(screen.getByText("Fields used by this episode")).toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "Score and utility landscape" })).not.toBeInTheDocument();
  expect(screen.queryByText(/case matrix/i)).not.toBeInTheDocument();

  const tabs = screen.getByRole("navigation", { name: "Episode views" }).querySelectorAll("button");
  expect([...tabs].map((button) => button.textContent?.replace(/\d+$/, "").trim())).toEqual(["Mechanism", "Performance", "Trajectory"]);
  fireEvent.click(screen.getByRole("button", { name: /^Performance$/i }));
  expect(await screen.findByRole("img", { name: "Score and utility landscape" })).toBeInTheDocument();
  expect(screen.queryByRole("group", { name: "Select model" })).not.toBeInTheDocument();
  expect(screen.queryByRole("group", { name: "Select evaluation branch" })).not.toBeInTheDocument();
  expect(screen.getAllByLabelText("Action color legend")).toHaveLength(1);
  expect(screen.getByRole("group", { name: "Probe curve model" })).toBeInTheDocument();
  const accuracyModel = screen.getByRole("group", { name: "Accuracy curve model" });
  fireEvent.click(within(accuracyModel).getByRole("button", { name: /Gemini/ }));
  expect(within(accuracyModel).getByRole("button", { name: /Gemini/ })).toHaveClass("active");
  expect(screen.getByText(/Gemini · 2 branches × 12 batches/)).toBeInTheDocument();
  fireEvent.click(within(accuracyModel).getByRole("button", { name: /DeepSeek/ }));
  const landscapeModel = screen.getByRole("group", { name: "Landscape model" });
  const landscapeBranch = screen.getByRole("group", { name: "Landscape branch" });
  expect(within(landscapeBranch).getByRole("button", { name: "flip 50" })).toHaveClass("active");
  fireEvent.click(within(landscapeBranch).getByRole("button", { name: "flip 50 no change control" }));
  expect(within(landscapeBranch).getByRole("button", { name: "flip 50 no change control" })).toHaveClass("active");
  expect(within(landscapeBranch).getByRole("button", { name: "flip 50" })).not.toHaveClass("active");
  fireEvent.click(within(landscapeModel).getByRole("button", { name: /Gemini/ }));
  expect(within(landscapeModel).getByRole("button", { name: /Gemini/ })).toHaveClass("active");
  expect(within(landscapeModel).getByRole("button", { name: /DeepSeek/ })).not.toHaveClass("active");
});

test("renders one initial prompt followed by continuous interaction responses", async () => {
  const rendered = render(<App />);
  const view = within(rendered.container);
  await waitFor(() => expect(view.getByRole("button", { name: /test_episode/i })).toBeInTheDocument());
  fireEvent.click(view.getByRole("button", { name: /test_episode/i }));
  await waitFor(() => expect(view.getByRole("button", { name: /^Trajectory/i })).toBeInTheDocument());
  fireEvent.click(view.getByRole("button", { name: /^Trajectory/i }));
  expect(await view.findByText("Initial prompt")).toBeInTheDocument();
  expect(rendered.container.querySelector(".trajectory-summary-grid")).toHaveClass("single");
  expect(view.getAllByText("Initial prompt")).toHaveLength(1);
  expect(view.getAllByText("Model response", { exact: false, selector: "h4" })).toHaveLength(2);
  expect(view.queryByText("Exact prompt messages")).not.toBeInTheDocument();
  expect(view.queryByText("Raw and parsed response")).not.toBeInTheDocument();
});

test("renders one full trajectory with outcome-only history changes annotated", () => {
  const pendingBatch = { batch_id: "batch_001", batch_number: 1, cases: [{ context_id: "case_001", raw_values: { a: .4 } }] };
  const probeResults = [{ context_id: "probe_001", utilities: { action_0: 1 } }];
  const revealEvent = {
    type: "environment_turn", event_id: "event_0003", interaction: 1, stage: "reveal", batch_id: "batch_001", batch_number: 1,
    result: { results: probeResults, cumulative_utility: 0, cumulative_probes: 1, pending_action_batch: pendingBatch, tool_usability: { calculator: true } },
    state_card: { pending_action_batch: pendingBatch, tool_usability: { calculator: true }, cumulative_utility: 0, cumulative_probes: 1 }, error: null,
  };
  const completionEvent = {
    type: "environment_turn", event_id: "event_0007", interaction: 3, stage: "completion", batch_id: "batch_001", batch_number: 1,
    result: { results: [{ context_id: "case_001", action_id: "action_1", utility: 2, regret: 0 }], batch_utility: 2, detailed_diagnostics: { selected_action: "action_1" } },
    state_card: { pending_action_batch: null, tool_usability: { calculator: true }, cumulative_utility: 2, cumulative_probes: 1 }, error: null,
  };
  const outcomeRun = {
    ...branch,
    history_policy: "eval_case_hidden",
    history_view_status: "available",
    conversation: {
      schema_version: "adaexplorebench_benchmark_conversation_v4",
      system_instruction: "instructions",
      system_prompt_hash: "abc",
      task_spec: {},
      events: [
        { type: "environment_turn", event_id: "event_0001", state_card: {}, interaction: null },
        { type: "assistant_turn", event_id: "event_0002", interaction: 1, stage: "control", visible_content: "request batch", parsed_call: { tool: "request_action_batch", arguments: {} }, valid: true },
        revealEvent,
        { type: "assistant_turn", event_id: "event_0004", interaction: 2, stage: "action_solving", batch_id: "batch_001", batch_number: 1, visible_content: "calculate", parsed_call: { tool: "calculator", arguments: { expression: "1 + 1" } }, valid: true },
        { type: "environment_turn", event_id: "event_0005", interaction: 2, stage: "action_solving", batch_id: "batch_001", batch_number: 1, result: { value: 2 }, state_card: { pending_action_batch: { batch_id: "batch_001" }, cumulative_utility: 0, cumulative_probes: 1 }, error: null },
        { type: "assistant_turn", event_id: "event_0006", interaction: 3, stage: "action_solving", batch_id: "batch_001", batch_number: 1, visible_content: "submit", parsed_call: { tool: "submit_action_batch", arguments: { actions: { case_001: "action_1" } } }, valid: true },
        completionEvent,
      ],
    },
    agent_visible_history: {
      schema_version: "eval_case_visibility_v1",
      history_policy: "eval_case_hidden",
      system_instruction: "instructions",
      task_spec: {},
      events: [
        { type: "environment_turn", event_id: "event_0001", state_card: {}, interaction: null },
        { type: "assistant_turn", event_id: "event_0002", interaction: 1, stage: "control", visible_content: "request batch", parsed_call: { tool: "request_action_batch", arguments: {} }, valid: true },
        { ...revealEvent, result: { message: "The revealed action cases and transient state are hidden from later agent context.", results: [{ context_id: "probe_001", utilities: { action_0: 1 } }], cumulative_utility: 0, cumulative_probes: 1 }, state_card: {} },
        { ...completionEvent, result: { message: "The environment accepted action batch 1. Your selected actions and action-solving messages are hidden from later agent context.", batch_number: 1, results: [{ context_id: "case_001", utility: 2 }], batch_utility: 2, cumulative_utility: 2, cumulative_probes: 1 }, error: null },
      ],
      event_mapping: [
        { source_event_id: "event_0001", visible_event_id: "event_0001", disposition: "retained" },
        { source_event_id: "event_0002", visible_event_id: "event_0002", disposition: "retained" },
        { source_event_id: "event_0003", visible_event_id: "event_0003", disposition: "rewritten_reveal" },
        { source_event_id: "event_0004", visible_event_id: null, disposition: "removed_action_solving" },
        { source_event_id: "event_0005", visible_event_id: null, disposition: "removed_action_solving" },
        { source_event_id: "event_0006", visible_event_id: null, disposition: "removed_action_solving" },
        { source_event_id: "event_0007", visible_event_id: "event_0007", disposition: "rewritten_completion" },
      ],
    },
    tool_trace: [
      { ...branch.tool_trace[0], request_id: "request_001", interaction: 1, raw_response: "request batch", response: { tool: "request_action_batch", arguments: {} }, result: revealEvent.result, request_context: { event_mapping: [{ source_event_id: "event_0001", visible_event_id: "event_0001", disposition: "retained" }] } },
      { ...branch.tool_trace[1], request_id: "request_002", interaction: 2, raw_response: "calculate", response: { tool: "calculator", arguments: { expression: "1 + 1" } }, result: { value: 2 }, request_context: { event_mapping: [
        { source_event_id: "event_0001", visible_event_id: "event_0001", disposition: "retained" },
        { source_event_id: "event_0002", visible_event_id: "event_0002", disposition: "retained" },
        { source_event_id: "event_0003", visible_event_id: "event_0003", disposition: "retained" },
      ] } },
      { ...branch.tool_trace[1], request_id: "request_003", interaction: 3, raw_response: "submit", response: { tool: "submit_action_batch", arguments: { actions: { case_001: "action_1" } } }, result: completionEvent.result, request_context: { event_mapping: [
        { source_event_id: "event_0001", visible_event_id: "event_0001", disposition: "retained" },
        { source_event_id: "event_0002", visible_event_id: "event_0002", disposition: "retained" },
        { source_event_id: "event_0003", visible_event_id: "event_0003", disposition: "retained" },
        { source_event_id: "event_0004", visible_event_id: "event_0004", disposition: "retained" },
        { source_event_id: "event_0005", visible_event_id: "event_0005", disposition: "retained" },
      ] } },
    ],
  } as unknown as RunBranch;
  const rendered = render(<TrajectoryPanel episode={episode as unknown as Episode} runSummaries={[]} runs={[outcomeRun]} branch="flip_50" forkAfterBatch={5} sharedPrefixInteractions={1} />);
  const view = within(rendered.container);
  expect(view.queryByRole("group", { name: "History view" })).not.toBeInTheDocument();
  expect(view.getByRole("region", { name: "Shared prefix through Batch 5" })).toBeInTheDocument();
  expect(rendered.container.querySelectorAll("details.timeline-event")).toHaveLength(4);
  expect(rendered.container.querySelectorAll("details.timeline-event > summary.timeline-event-header")).toHaveLength(4);
  expect(rendered.container.querySelectorAll("details.timeline-event[open]")).toHaveLength(0);
  expect(rendered.container.querySelectorAll("details.timeline-event details.trace-detail")).toHaveLength(3);
  expect(rendered.container.querySelectorAll("details.timeline-event details.trace-detail[open]")).toHaveLength(0);
  expect(view.getAllByText("Model response", { exact: false, selector: "h4" })).toHaveLength(3);
  expect(view.getAllByText("Environment feedback", { exact: false, selector: "h4" })).toHaveLength(3);
  expect(view.getAllByRole("note")).toHaveLength(1);
  expect(view.getByText(/Batch reveal rewritten to keep safe probe evidence/)).toBeInTheDocument();
  expect(view.getAllByText(/Action-solving model response removed/)).toHaveLength(2);
  expect(view.getByText(/Action-solving environment feedback removed/)).toBeInTheDocument();
  expect(view.getByText(/Completion feedback rewritten/)).toBeInTheDocument();
  expect(view.getAllByText("Agent-visible replacement")).toHaveLength(2);
  expect(view.getAllByText(/The environment accepted action batch 1\./).length).toBeGreaterThan(0);
  expect(rendered.container.textContent).toContain("Probe result:");
  const retainedProbeBlock = [...rendered.container.querySelectorAll("pre.history-visible")]
    .find((node) => node.textContent?.includes("Probe result:"));
  expect(retainedProbeBlock?.textContent).not.toContain("Current state:");
  expect(rendered.container.textContent).not.toContain('"type": "retained_probe_evidence"');
  expect(rendered.container.textContent).not.toContain('"type": "action_batch_outcome"');
  expect(rendered.container.querySelectorAll("pre.history-removed")).toHaveLength(5);
  expect(rendered.container.querySelectorAll(".history-replacement-grid")).toHaveLength(2);
  expect(rendered.container.textContent).not.toContain('"type": "assistant_turn"');
  expect(rendered.container.textContent).not.toContain('"visible_content"');
  expect(rendered.container.textContent).not.toContain('"parsed_call"');
  expect(rendered.container.textContent).toContain('Environment result:');
  expect(rendered.container.textContent).toContain('Current state:');
  const firstInteraction = rendered.container.querySelector("details.timeline-event:not(.initial-event)") as HTMLDetailsElement;
  const firstSummary = firstInteraction.querySelector(":scope > summary") as HTMLElement;
  fireEvent.click(firstSummary);
  expect(firstInteraction.open).toBe(true);
});

test("eval-case-given diagnostic has no redundant history switch", () => {
  const givenRun = { ...branch, history_policy: "eval_case_given", setting_id: "flip_50_eval_case_given" } as unknown as RunBranch;
  const rendered = render(<TrajectoryPanel episode={episode as unknown as Episode} runSummaries={[]} runs={[givenRun]} branch="flip_50_eval_case_given" forkAfterBatch={5} sharedPrefixInteractions={1} />);
  expect(within(rendered.container).queryByRole("group", { name: "History view" })).not.toBeInTheDocument();
  expect(within(rendered.container).getByRole("region", { name: "Shared prefix through Batch 5" })).toBeInTheDocument();
});

test("keeps the shared prefix separate and switches the post-change path", async () => {
  const rendered = render(<App />);
  const view = within(rendered.container);
  await waitFor(() => expect(view.getByRole("button", { name: /test_episode/i })).toBeInTheDocument());
  fireEvent.click(view.getByRole("button", { name: /test_episode/i }));
  await waitFor(() => expect(view.getByRole("button", { name: /^Trajectory/i })).toBeInTheDocument());
  fireEvent.click(view.getByRole("button", { name: /^Trajectory/i }));
  expect(await view.findByRole("region", { name: "Shared prefix through Batch 5" })).toBeInTheDocument();
  expect(view.getByRole("region", { name: "flip 50 after Batch 5" })).toBeInTheDocument();
  fireEvent.click(view.getByRole("button", { name: /no-change control/i }));
  expect(await view.findByRole("region", { name: "flip 50 no change control after Batch 5" })).toBeInTheDocument();
  expect(view.queryByRole("region", { name: "flip 50 after Batch 5" })).not.toBeInTheDocument();
});
