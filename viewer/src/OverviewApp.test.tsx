import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import OverviewApp from "./OverviewApp";
import { loadEpisode, loadManifest, loadRunEpisode } from "./api";

vi.mock("./api", () => ({ loadManifest: vi.fn(), loadEpisode: vi.fn(), loadRunEpisode: vi.fn() }));
vi.mock("./Panels", () => ({ OverallPanel: ({ manifest }: { manifest: { runs: { label: string }[] } }) => <section aria-label="Model overview">{manifest.runs.map((run) => <p key={run.label}>{run.label}</p>)}</section> }));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

test("shows eight models using only aggregate data, including episode deep links", async () => {
  window.history.replaceState({}, "", "/?episode=old-link&tab=trajectory");
  vi.mocked(loadManifest).mockResolvedValue({ episode_count: 256, runs: Array.from({ length: 8 }, (_, i) => ({ label: `Model ${i + 1}` })) } as Awaited<ReturnType<typeof loadManifest>>);
  render(<OverviewApp />);
  await waitFor(() => expect(screen.getByText("8 models")).toBeInTheDocument());
  expect(screen.getByRole("heading", { name: "AdaExploreBench" })).toBeInTheDocument();
  expect(screen.getByText("256 episodes")).toBeInTheDocument();
  expect(screen.getByText("Model 8")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /trajectory|episode/i })).not.toBeInTheDocument();
  expect(loadEpisode).not.toHaveBeenCalled();
  expect(loadRunEpisode).not.toHaveBeenCalled();
  window.history.replaceState({}, "", "/");
});
