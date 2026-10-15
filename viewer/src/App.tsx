import { Fragment, useEffect, useMemo, useState } from "react";
import { Activity, CheckCircle2, FlaskConical, LayoutDashboard, Menu, Network, RefreshCw, XCircle } from "lucide-react";
import { loadEpisode, loadManifest, loadRunEpisode } from "./api";
import { SHOW_REASONING } from "./config";
import { MechanismPanel, OverallPanel, PerformancePanel, Sidebar, TrajectoryPanel, availableDynamicScopes, branchLabel, orderedSamplingBranches, type Selection } from "./Panels";
import type { DynamicScope, Episode, Manifest, RunEpisode, RunSummary } from "./types";

type Tab = "mechanism" | "performance" | "trajectory";

export const isolateEpisodeBranch = (
  episode: Episode,
  branch: string,
  sourceBranch = branch,
): Episode => ({
  ...episode,
  cases: episode.cases.filter((item) => (
    item.sampling_branch === "shared_prefix" ||
    item.sampling_branch === branch ||
    item.sampling_branch === sourceBranch ||
    (branch.endsWith("_no_change_control") && item.sampling_branch === branch.replace(/_no_change_control$/, "")) ||
    (branch.endsWith("_eval_case_given") && item.sampling_branch === branch.replace(/_eval_case_given$/, "")) ||
    (branch.endsWith("_full_history") && item.sampling_branch === branch.replace(/_full_history$/, ""))
  )),
});

function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [runs, setRuns] = useState<Array<{ summary: RunSummary; data: RunEpisode }>>([]);
  const [episodeId, setEpisodeId] = useState("");
  const requestedTab = new URLSearchParams(window.location.search).get("tab");
  const initialTab = requestedTab === "overview" ? "performance" : requestedTab as Tab | null;
  const [tab, setTab] = useState<Tab>(["mechanism", "performance", "trajectory"].includes(initialTab || "") ? initialTab! : "mechanism");
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("");
  const [dynamicScope, setDynamicScope] = useState<DynamicScope>("all");
  const [branch, setBranch] = useState("");
  const [selection, setSelection] = useState<Selection>(null);
  const [selectedRunId, setSelectedRunId] = useState(() => new URLSearchParams(window.location.search).get("model") || "");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [error, setError] = useState("");
  const [overall, setOverall] = useState(() => !new URLSearchParams(window.location.search).has("episode"));

  useEffect(() => {
    loadManifest()
      .then((data) => {
        setManifest(data);
        const requested = new URLSearchParams(window.location.search).get("episode");
        setEpisodeId(data.episodes.find((item) => item.episode_id === requested)?.episode_id || data.episodes[0]?.episode_id || "");
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (manifest && !availableDynamicScopes(manifest).includes(dynamicScope)) {
      setDynamicScope("all");
    }
  }, [manifest, dynamicScope]);

  useEffect(() => {
    if (!manifest || !episodeId || overall) return;
    setEpisode(null);
    setRuns([]);
    setSelection(null);
    setBranch("");
    const available = manifest.runs.filter((run) => run.episode_ids.includes(episodeId));
    Promise.all([
      loadEpisode(episodeId),
      ...available.map((run) => loadRunEpisode(run.run_id, episodeId).then((data) => ({ summary: run, data }))),
    ]).then(([episodeData, ...runData]) => {
      setEpisode(episodeData as Episode);
      setRuns(runData as Array<{ summary: RunSummary; data: RunEpisode }>);
    }).catch((reason) => setError(String(reason)));
  }, [manifest, episodeId, overall]);

  const modelRuns = useMemo(
    () => {
      const selected = new Map<string, { summary: RunSummary; data: RunEpisode }>();
      for (const item of runs.filter((candidate) => !["oracle", "random"].includes(candidate.summary.model_key))) {
        if (!selected.has(item.summary.model_key)) {
          selected.set(item.summary.model_key, item);
        }
      }
      return [...selected.values()];
    },
    [runs],
  );
  const selectedModel = modelRuns.find((item) => item.summary.run_id === selectedRunId) || modelRuns[0];
  const selectedBranch = selectedModel
    ? selectedModel.data.branches[branch] || selectedModel.data.branches[orderedSamplingBranches(Object.keys(selectedModel.data.branches))[0]]
    : null;
  const branchEpisode = useMemo(
    () => episode && branch
      ? isolateEpisodeBranch(episode, branch, selectedBranch?.source_sampling_branch)
      : episode,
    [episode, branch, selectedBranch?.source_sampling_branch],
  );

  useEffect(() => {
    if (selectedModel && !selectedModel.data.branches[branch]) {
      setBranch(orderedSamplingBranches(Object.keys(selectedModel.data.branches))[0] || "");
    }
  }, [selectedModel, branch]);

  useEffect(() => {
    if (modelRuns.length && !modelRuns.some((item) => item.summary.run_id === selectedRunId)) {
      setSelectedRunId(modelRuns[0].summary.run_id);
    }
  }, [modelRuns, selectedRunId]);

  if (error) return <main className="fatal-state"><Activity /><h1>Viewer data could not be loaded</h1><p>{error}</p><code>npm run data</code></main>;
  if (!manifest || (!overall && !episode)) return <main className="loading-state"><RefreshCw className="spin" /><span>Loading research artifacts…</span></main>;

  return <div className={`app-shell ${sidebarOpen ? "sidebar-open" : ""}`}>
    <Sidebar
      manifest={manifest}
      selectedId={overall ? "" : episodeId}
      search={search}
      family={family}
      dynamicScope={dynamicScope}
      onSearch={setSearch}
      onFamily={setFamily}
      onDynamicScope={setDynamicScope}
      onSelect={(id) => { setEpisodeId(id); setOverall(false); setSidebarOpen(false); }}
      onOverview={() => { setOverall(true); setSidebarOpen(false); }}
    />
    <main className="main-content">
      <header className="topbar">
        <button className="mobile-menu" aria-label="Toggle episode menu" onClick={() => setSidebarOpen((value) => !value)}><Menu /></button>
        {overall ? <div className="title-block"><span className="eyebrow">Benchmark overview</span><h1>Overview</h1><p>{manifest.dataset}</p></div> : episode && <div className="title-block"><span className="eyebrow">{episode.scenario.domain} · {episode.setting} · {episode.dynamic!.event.sampling_design.available_post_branches.join(" / ")}</span><h1>{episode.episode_id}</h1><p>{episode.scenario.template_id}</p></div>}
        <div className="header-badges"><span>{overall ? manifest.benchmark_config.schema_version : episode?.schema_version}</span><span className={`availability ${(overall ? manifest.runs.length : modelRuns.length) ? "available" : ""}`}>{overall ? `${manifest.runs.length} model runs` : modelRuns.length ? `${modelRuns.length} model trajectories` : "dataset only"}</span></div>
      </header>
      {!overall && <nav className="tabs" aria-label="Episode views">
        <button className={tab === "mechanism" ? "active" : ""} onClick={() => setTab("mechanism")}><Network />Mechanism</button>
        <button className={tab === "performance" ? "active" : ""} onClick={() => setTab("performance")}><LayoutDashboard />Performance</button>
        <button className={tab === "trajectory" ? "active" : ""} onClick={() => setTab("trajectory")}><FlaskConical />Trajectory <span>{modelRuns.length}</span></button>
      </nav>}
      {!overall && tab === "trajectory" && modelRuns.length > 0 && <div className="model-selector-bar">
        <div><span className="eyebrow">Model view</span><strong>Select one trajectory</strong></div>
        <div className="model-selector" role="group" aria-label="Select model">
          {modelRuns.map((item) => <button key={item.summary.run_id} className={selectedModel?.summary.run_id === item.summary.run_id ? "active" : ""} onClick={() => { setSelectedRunId(item.summary.run_id); setSelection(null); }}>
            {item.data.run_success ? <CheckCircle2 /> : <XCircle />}
            <span title={item.summary.simulation_description || undefined}><strong>{item.summary.label}</strong><small>{item.data.run_success ? "Completed" : "Failed"}{item.summary.simulation_scenario ? ` · ${item.summary.simulation_scenario}` : ""}{SHOW_REASONING ? ` · ${item.summary.reasoning?.parameter || "reasoning"}: ${item.summary.reasoning?.value || "unspecified"}` : ""}</small></span>
          </button>)}
        </div>
        <div className="model-selector branch-selector" role="group" aria-label="Select evaluation branch">
          {orderedSamplingBranches(Object.keys(selectedModel?.data.branches || {})).map((name, index, names) => <Fragment key={name}>
            <button className={branch === name ? "active" : ""} onClick={() => { setBranch(name); setSelection(null); }}><span><strong>{branchLabel(name)}</strong><small>{name.endsWith("_eval_case_given") || name.endsWith("_full_history") ? "evaluation cases remain available" : name.endsWith("_no_change_control") || name === "static_iid" ? "pre-change rule remains active" : `evaluation cases hidden · rule changes after Batch ${selectedModel?.data.fork_after_batch}`}</small></span></button>
            {name === "flip_50" && index < names.length - 1 && <span className="branch-selector-divider">diagnostics</span>}
          </Fragment>)}
        </div>
      </div>}
      <div className="content-area">
        {overall && <OverallPanel manifest={manifest} dynamicScope={dynamicScope} />}
        {!overall && episode && tab === "mechanism" && <MechanismPanel episode={episode} />}
        {!overall && episode && tab === "performance" && <PerformancePanel episode={episode} modelRuns={modelRuns} selected={selection} onSelect={setSelection} />}
        {!overall && branchEpisode && tab === "trajectory" && (selectedModel && selectedBranch
          ? <TrajectoryPanel key={`${selectedModel.summary.run_id}-${branch}`} episode={branchEpisode} runs={[selectedBranch]} runSummaries={[selectedModel.summary]} branch={branch} forkAfterBatch={selectedModel.data.fork_after_batch} sharedPrefixInteractions={selectedModel.data.shared_prefix_interactions} />
          : <section className="empty-state"><FlaskConical /><h2>No model trajectory for this episode</h2><p>The episode design and all private cases remain available in the other views.</p></section>)}
      </div>
    </main>
  </div>;
}

export default App;
