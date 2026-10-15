import type { Episode, Manifest, RunEpisode } from "./types";

async function getJson<T>(path: string): Promise<T> {
  if (import.meta.env.VITE_COMPRESSED_DATA === "true") {
    const response = await fetch(`${path}.gz`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Failed to load ${path}.gz: ${response.status}`);
    const stream = response.body?.pipeThrough(new DecompressionStream("gzip"));
    if (!stream) throw new Error(`Failed to decompress ${path}.gz`);
    return new Response(stream).json() as Promise<T>;
  }

  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
  return response.json() as Promise<T>;
}

const publicationDir = import.meta.env.VITE_DATA_DIR || "data";
const dataPath = (path: string) => `${import.meta.env.BASE_URL}${publicationDir}/${path}`;

export const loadManifest = () => getJson<Manifest>(dataPath("manifest.json"));
export const loadEpisode = (episodeId: string) => getJson<Episode>(dataPath(`episodes/${episodeId}.json`));
export const loadRunEpisode = (runId: string, episodeId: string) =>
  getJson<RunEpisode>(dataPath(`runs/${runId}/${episodeId}.json`));
