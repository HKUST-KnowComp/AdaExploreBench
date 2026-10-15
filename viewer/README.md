# AdaExploreBench Overview

The official [GitHub Pages overview](https://hkust-knowcomp.github.io/AdaExploreBench/) compares eight models across the 256-episode benchmark. It displays full-dataset metrics, setting comparisons, and exploration analyses from one compressed aggregate file.

## Local preview

Use Node.js 22.12 or newer. From the repository root:

```bash
npm --prefix viewer ci
VITE_COMPRESSED_DATA=true npm --prefix viewer run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:8000`.

## Publication

`public/data/manifest.json.gz` contains the complete overview. Its `publication` contract records `mode: overview` and `model_count: 8`. Model summaries retain full-dataset branch aggregates, while episode and trajectory payloads are excluded.

The workflow in `.github/workflows/deploy-pages.yml` validates the publication, tests the overview, builds the site, and deploys it to GitHub Pages after pushes to `main`.

```bash
python viewer/validate_public_data.py --compressed
npm --prefix viewer run test -- src/OverviewApp.test.tsx
VITE_COMPRESSED_DATA=true npm --prefix viewer run build
```

## Local result generation

The Python publication tools can also process completed benchmark runs for local research:

```bash
python viewer/generate_data.py \
  --dataset datasets/adaexplorebench/data/test.jsonl \
  --run-dir outputs/your-completed-run \
  --output viewer/public/data-local \
  --replace \
  --gzip
```

Use `python viewer/generate_data.py --help` for publication options. Repeat `--run-dir` to compare runs. Summary caches can be prepared independently with `scripts/precompute_metrics.py` and are keyed by dataset identity and run artifacts. Keep locally generated artifacts outside the official `public/data` publication.
