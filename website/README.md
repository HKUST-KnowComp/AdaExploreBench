# AdaExploreBench project website

Static project page for the AdaExploreBench paper. There is no build step: `index.html`, `style.css`, `app.js`, and the generated JSON under `assets/data/` are served directly.

## Regenerate the data

From the repository root:

```bash
python3 website/scripts/build_data.py
```

The script reads the ReRule checkout path when present, otherwise `datasets/selected-256/dynamic_episodes.jsonl`. It selects one episode per domain on the `flip_50` branch, computes reference utilities from each hidden pre-change rule with the runtime's deterministic scoring mechanism, and writes:

- `website/assets/data/episodes.json`
- `website/assets/data/site-data.json`

The dataset split is distributed through Hugging Face and is not committed to the repository. If it is not on disk, download it from <https://huggingface.co/datasets/Roryaccout/AdaExploreBench> or pass `--source /path/to/dynamic_episodes.jsonl`.

Preview locally:

```bash
cd website
python3 -m http.server 8000
```

Then open <http://localhost:8000/>.

## Deployment

The page is a plain static bundle for the repository's GitHub Pages workflow at
`https://hkust-knowcomp.github.io/AdaExploreBench/`. Every local asset path is relative, so the
subpath works without configuration.

The current workflow publishes `viewer/dist`. To serve this project page at the repository root,
point the `upload-pages-artifact` step in `.github/workflows/deploy-pages.yml` at `website/`.
