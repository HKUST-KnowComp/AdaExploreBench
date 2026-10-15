import { useEffect, useState } from "react";
import { Activity, ArrowUpRight, RefreshCw } from "lucide-react";
import { loadManifest } from "./api";
import { OverallPanel } from "./Panels";
import type { Manifest } from "./types";

export default function OverviewApp() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    loadManifest().then(setManifest).catch((reason) => setError(String(reason)));
  }, []);

  if (error) return <main className="fatal-state"><Activity /><h1>Results could not be loaded</h1><p>{error}</p></main>;
  if (!manifest) return <main className="loading-state"><RefreshCw className="spin" /><span>Loading benchmark results…</span></main>;

  return <div className="overview-shell">
    <header className="topbar">
      <div className="title-block"><span className="eyebrow">Proactive and adaptive exploration</span><h1>AdaExploreBench</h1><p>Explore how LLM agents learn and adapt in changing environments.</p></div>
      <div className="header-badges"><span>{manifest.episode_count} episodes</span><span className="available">{manifest.runs.length} models</span></div>
      <a className="repository-link" href="https://github.com/HKUST-KnowComp/AdaExploreBench" target="_blank" rel="noreferrer">Code &amp; data <ArrowUpRight size={16} /></a>
    </header>
    <main className="content-area"><OverallPanel manifest={manifest} dynamicScope="all" /></main>
    <footer className="overview-footer">AdaExploreBench · Evaluating proactive and adaptive exploration in LLM agents</footer>
  </div>;
}
