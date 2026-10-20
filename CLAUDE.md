# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A dual-mode visualization tool for KM (kernel-density / retrospective) and EB (event-boundary / realtime) response distributions across video clips. Two independent apps share the same data format:

- **JS app** (`js-app/`) — static React + Plotly site, deployed to GitHub Pages. This is the primary, publicly-accessible version.
- **Python Dash app** (`src/app_2x2.py`) — local interactive prototype; served on port 8051.

## Running the apps

### JavaScript app (primary)

```bash
cd js-app
npm install          # first time only
npm run dev          # dev server → http://localhost:5173
npm run build        # production build → dist/
npm run preview      # preview prod build → http://localhost:4173
```

### Python Dash app (local only)

```bash
conda activate events_dashboard   # or: conda activate storyboard
python src/app_2x2.py
# Open http://127.0.0.1:8051
```

## Data pipeline

Raw CSV/Parquet → JS JSON is a one-step conversion:

```bash
# Default: reads data/2x2_distributions.parquet, writes js-app/public/data/data.json
python3 scripts/generate_js_data.py

# Custom dataset
python3 scripts/generate_js_data.py \
  --input path/to/data.csv \
  --image-base "https://your-host/frames/" \
  --video-base "https://your-host/videos/" \
  --output js-app/public/data/my_dataset.json
```

After generating, register the dataset in `js-app/public/data/datasets.json`.

## Architecture

### JS app (`js-app/src/`)

| File | Role |
|---|---|
| `App.jsx` | Root component; owns all shared state (selected clip, conditions, hover position) |
| `figureBuilders.js` | All Plotly figure constructors (dist, scatter, timeline, tile) |
| `dataUtils.js` | Data normalization (max→1 / z-score), `frameUrl()` helper |
| `constants.js` | Shared style tokens |
| `appConfig.js` | **Password** (`APP_PASSWORD`) and Google Form URL — change before deploying |
| `components/` | `DistPlot`, `ScatterPlot`, `FrameView`, `VideoPlayer`, `PasswordGate`, `HowToUse` |

Data flow: `generate_js_data.py` writes a JSON file with `meta` + `clips` + `data` keys. `meta` carries condition names, colours, and asset base URLs. Components read only from this precomputed structure — no server needed at runtime.

### Python Dash app (`src/app_2x2.py`)

Single-file app. On startup it loads the parquet/CSV, precomputes a `clip_data` dict (per-clip arrays for all four conditions), and pre-filters peak rows into `peaks_df`. All callbacks use this in-memory lookup — the full DataFrame is never queried in hot paths.

Key design points:
- `TIME_OFFSET = 10` — clips have a 10-second overlap pre-removed; display time = `selected_frame_time + 10`.
- The last shape on the distribution plot is always a highlight rect updated via Plotly `Patch()` (avoids full figure re-render on scatter hover).
- `make_scatter_fig` uses a stale reference `cdf` (bug: should be `cd`'s arrays) — if editing, check scatter peak index computation.

### Condition naming convention (both apps)

Column names drive auto-detection:
- `*km*` or `*sb*` in `_dist` column → KM-type (warm amber colours)
- `*eb*` in `_dist` column → EB-type (cool teal colours)
- Peaks column: exact `<cond>_peaks`, or truncated prefix match (e.g. `sb_cont_dist` → `sb_peaks`)

## Deploying to GitHub Pages

```bash
cd js-app/dist
git config http.postBuffer 524288000   # needed for large dist/ pushes
git add -A && git commit -m "Deploy"
git push --force https://<username>:<token>@github.com/<username>/storyboard-visualization.git main:gh-pages
```

Then: **Settings → Pages → Branch: `gh-pages` → `/ (root)`**.

## Asset URLs

The active `IMAGE_BASE` / `VIDEO_BASE` are set in `js-app/src/constants.js`. Two hosts are used depending on the dataset:

| Dataset | IMAGE_BASE | VIDEO_BASE |
|---|---|---|
| Mr. Bean | `https://cdmlab.a2hosted.com/adi/segmented_stimuli_frames/` | `https://cdmlab.a2hosted.com/adi/segmented_stimuli/` |
| 2×2 / Storyboards fMRI | `https://adibuoy23.github.io/event_representations/video_frames/` | `https://adibuoy23.github.io/event_representations/videos/` |

The URL pattern for assets is:
```
<IMAGE_BASE><clip_name>/framesNNNN.jpg   # 4-digit zero-padded frame number
<VIDEO_BASE><clip_name>.mp4
```

When switching datasets, swap the active constants in `constants.js` (the inactive set is kept commented). These URLs are also embedded in each JSON dataset file by `generate_js_data.py` (via `--image-base` / `--video-base`) and read from `meta.image_base` / `meta.video_base` at runtime.
