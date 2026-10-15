import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { ParetoFrontierChart, paretoFrontierIndices, SettingMetricBars, utilitiesFromScore, UtilityLandscape } from "./Charts";
import type { Episode, RunBranch } from "./types";

const episode = {
  interaction_spec: { probe_budget: 32 },
  scenario: { actions: [{ action_id: "action_0" }, { action_id: "action_1" }, { action_id: "action_2" }] },
  cases: [{
    context_id: "case_1", score: .4, utilities: [.1, .8, .6], oracle_action_index: 1,
  }],
} as unknown as Episode;

const run = {
  run_label: "DeepSeek",
  probes: [],
  action_results: [{
    context_id: "case_1", action_id: null, utility: .8, score: .4, regret: 1.8,
    selected_action_index: null, oracle_action_index: 1, unanswered: true, correct: false,
  }],
} as unknown as RunBranch;

test("pins unanswered results to zero and excludes them from action count", () => {
  render(<UtilityLandscape episode={episode} runs={[run]} probes={[]} selected={null} onSelect={() => undefined} />);

  expect(screen.getByTestId("unanswered-point")).toHaveAttribute("data-utility", "0");
  expect(screen.getByTestId("unanswered-point")).toHaveAttribute("data-display-lane", "unanswered");
  expect(screen.getByText("Unanswered (×) · utility fixed at 0")).toBeInTheDocument();
  expect(screen.getByLabelText("Model interaction totals")).toHaveTextContent("0 / 32 probe contexts");
  expect(screen.getByLabelText("Model interaction totals")).toHaveTextContent("0 visible coordinates");
  expect(screen.getByLabelText("Model interaction totals")).toHaveTextContent("0 / 1 actions");
  expect(screen.getByLabelText("Model interaction totals")).toHaveTextContent("0 visible coordinates");
  expect(screen.getByText("Unanswered (×) · utility fixed at 0")).toBeInTheDocument();
  expect(screen.getAllByTestId(/utility-curve-/)).toHaveLength(3);
  expect(screen.getByTestId("utility-curve-1")).toHaveAttribute("data-mapping", "centered-softmax");
});

test("uses the v3.2 centered-softmax score-to-utility mapping", () => {
  for (const score of [-1, -.4, 0, .4, 1]) {
    const utilities = utilitiesFromScore(score);
    expect(utilities.reduce((sum, utility) => sum + utility, 0)).toBeCloseTo(0, 12);
  }
  expect(utilitiesFromScore(-2 / 3)[0]).toBeGreaterThan(0);
  expect(utilitiesFromScore(0)[1]).toBeGreaterThan(0);
  expect(utilitiesFromScore(2 / 3)[2]).toBeGreaterThan(0);
});

test("supports the v6 four-action utility mapping", () => {
  const utilities = utilitiesFromScore(0, 4);
  expect(utilities).toHaveLength(4);
  expect(utilities.reduce((sum, utility) => sum + utility, 0)).toBeCloseTo(0, 12);
  expect(utilitiesFromScore(-.75, 4)[0]).toBeGreaterThan(utilitiesFromScore(-.75, 4)[3]);
  expect(utilitiesFromScore(.75, 4)[3]).toBeGreaterThan(utilitiesFromScore(.75, 4)[0]);
});

test("paretoFrontierIndices keeps non-dominated points, ties, and drops dominated and null points", () => {
  const points = [
    { probeCount: 10, calibratedAccuracy: .5 },
    { probeCount: 8, calibratedAccuracy: .6 },
    { probeCount: 12, calibratedAccuracy: .8 },
    { probeCount: 8, calibratedAccuracy: .6 },
    { probeCount: null, calibratedAccuracy: .9 },
    { probeCount: 9, calibratedAccuracy: .5 },
  ];
  expect(paretoFrontierIndices(points)).toEqual([1, 2, 3]);
  expect(paretoFrontierIndices([])).toEqual([]);
});

test("pareto chart plots one circle per scored model with frontier and random reference", () => {
  render(<ParetoFrontierChart
    points={[
      { runId: "run_alpha", label: "Alpha", color: "#111111", probeCount: 9.5, calibratedAccuracy: .62 },
      { runId: "run_beta", label: "Beta", color: "#222222", probeCount: 12, calibratedAccuracy: .55 },
      { runId: "run_gamma", label: "Gamma", color: "#333333", probeCount: null, calibratedAccuracy: null },
    ]}
    randomBaseline={.31}
  />);
  const chart = screen.getByRole("img", { name: "inductive performance versus exploration demand in the default setting" });
  expect(chart.querySelectorAll('[data-testid="pareto-point"]')).toHaveLength(2);
  expect(chart.querySelectorAll('[data-frontier="true"]')).toHaveLength(1);
  expect([...chart.querySelectorAll('[data-testid="pareto-point"]')].map((node) => node.getAttribute("r"))).toEqual(["6", "6"]);
  expect([...chart.querySelectorAll('[data-testid="pareto-model-label"]')].map((node) => node.textContent)).toEqual(["Alpha", "Beta"]);
  expect(chart.querySelector('[data-run-id="run_gamma"]')).toBeNull();
  expect(chart.querySelector(".pareto-random-line")).not.toBeNull();
  expect(chart.querySelector(".pareto-frontier-line")).not.toBeNull();
  expect(screen.getByText("random")).toBeInTheDocument();
  const titles = [...chart.querySelectorAll("title")].map((node) => node.textContent);
  expect(titles).toContain("Alpha · calibrated accuracy 62.0% · probe cases 9.50");
  expect(titles).toContain("Beta · calibrated accuracy 55.0% · probe cases 12.00");
  expect(screen.getByText("Exploration Demand · mean probe cases per episode ↓")).toBeInTheDocument();
  expect(screen.getByText("Inductive Performance · Calibrated Accuracy ↑")).toBeInTheDocument();
});

test("setting metric bars highlight the default setting and separate diagnostics", () => {
  render(<SettingMetricBars
    settings={[
      { key: "flip_50", label: "Default" },
      { key: "flip_75", label: "75% flip" },
    ]}
    series={[
      { runId: "run_alpha", label: "Alpha", color: "#111111", values: [.43, null] },
      { runId: "run_beta", label: "Beta", color: "#222222", values: [.21, .11] },
    ]}
    format={(value) => `${(value * 100).toFixed(1)}%`}
    zeroReference
    ariaLabel="calibrated accuracy by setting"
  />);
  const chart = screen.getByRole("img", { name: "calibrated accuracy by setting" });
  expect(chart.querySelectorAll(".setting-bar")).toHaveLength(3);
  expect(chart.querySelectorAll(".setting-label-default")).toHaveLength(1);
  expect(chart.querySelector(".setting-label-default")?.textContent).toBe("Default");
  expect(chart.querySelector(".setting-separator")).not.toBeNull();
  expect(chart.querySelector(".setting-separator-label")?.textContent).toBe("diagnostics");
  // All values above the random baseline: the zoomed y-domain excludes 0, so no zero line.
  expect(chart.querySelector(".setting-zero-line")).toBeNull();
  const titles = [...chart.querySelectorAll("title")].map((node) => node.textContent);
  expect(titles).toContain("Alpha · Default · 43.0%");
  expect(titles).toContain("Beta · 75% flip · 11.0%");
  const legend = screen.getByLabelText("Model run color legend");
  expect(legend.textContent).toContain("Alpha");
  expect(legend.textContent).toContain("Beta");
});

test("setting metric bars draw the zero line when the domain crosses it", () => {
  render(<SettingMetricBars
    settings={[
      { key: "flip_50", label: "Default" },
      { key: "flip_75", label: "75% flip" },
    ]}
    series={[
      { runId: "run_alpha", label: "Alpha", color: "#111111", values: [.43, -.12] },
    ]}
    format={(value) => `${(value * 100).toFixed(1)}%`}
    zeroReference
    ariaLabel="calibrated accuracy by setting with negative"
  />);
  const chart = screen.getByRole("img", { name: "calibrated accuracy by setting with negative" });
  expect(chart.querySelector(".setting-zero-line")).not.toBeNull();
});
