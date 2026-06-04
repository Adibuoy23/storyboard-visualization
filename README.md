# KM and EB Visualization

An interactive browser-based viewer for comparing kernel-density (KM) and event-boundary (EB) response distributions across video clips.

**Live demo:** https://adibuoy23.github.io/storyboard-visualization/

---

## What it shows

| Panel | Description |
|---|---|
| **Distribution plot** | KM and EB response curves over time for the selected clip. Hover to see the video frame at that moment. |
| **Scatter (Peak Space)** | Every detected peak for all clips plotted as (sum, difference) of the two selected conditions. Selected clip is highlighted. Hover to highlight the corresponding peak in the distribution plot. |
| **Timeline / Tile** | Peak frames from the selected clip arranged chronologically or as a grid. Hover to enlarge and highlight the peak in the distribution plot. |
| **Video player** | The clip video; seeking is driven by hover events on all three plots. |

---

## Running locally (Python Dash app)

Requires Conda. Download from https://docs.anaconda.com/free/miniconda/ if needed.

```bash
# 1. Clone
git clone https://github.com/Adibuoy23/storyboard-visualization.git
cd storyboard-visualization

# 2. Create environment and install dependencies
conda create -n storyboard python=3.9
conda activate storyboard
pip install -r requirements.txt

# 3. Run
python src/app_2x2.py
# Open http://127.0.0.1:8051
```

---

## Forking and using your own data

### 1 — Fork the repository

Click **Fork** on https://github.com/Adibuoy23/storyboard-visualization, then clone:

```bash
git clone https://github.com/<your-username>/storyboard-visualization.git
cd storyboard-visualization
```

---

### 2 — Prepare your data

Your CSV or Parquet file must have these columns:

| Column | Type | Description |
|---|---|---|
| `clip_name` | string | Unique identifier for each clip |
| `time_ms` | int | Time in milliseconds within the clip |
| `pdf_len` | int | Total length of the distribution (ms) |
| `num_frames` | int | Total video frames in the clip |
| `<cond>_dist` | float | One or more distribution columns |
| `<cond>_peaks` | int (0/1) | Matching peak-indicator for each `_dist` column |
| `clip` *(optional)* | int | Sort key; clips are ordered by this if present |

**Minimum viable dataset** — one KM column and one EB column:

```
clip_name, time_ms, pdf_len, num_frames, km_dist, km_peaks, eb_dist, eb_peaks
my_clip_1, 0, 60000, 1800, 0.0012, 0, 0.0008, 0
my_clip_1, 1, 60000, 1800, 0.0013, 0, 0.0009, 1
...
```

**Condition naming** — conditions are auto-detected from column names:
- A `<cond>_dist` column whose name contains **`km`** → KM-type (warm amber colours)
- A `<cond>_dist` column whose name contains **`eb`** → EB-type (cool teal colours)
- At least one of each is required

Examples that work: `km_dist`, `rt_km_dist`, `retro_km_dist`, `eb_dist`, `retro_eb_dist`

---

### 3 — Point the app at your video/image assets

Open `js-app/src/constants.js` and update the two URL constants:

```js
export const IMAGE_BASE   = 'https://<your-host>/video_frames/';
export const VIDEO_BASE   = 'https://<your-host>/videos/';
export const CLIP_TOTAL_S = 190.0;   // total clip duration in seconds
```

The app constructs URLs like:
```
# Frame image
IMAGE_BASE + clip_name + "/frames" + frameNumber.padStart(4,'0') + ".jpg"
# e.g. https://your-host/video_frames/my_clip_1/frames0042.jpg

# Video
VIDEO_BASE + clip_name + ".mp4"
# e.g. https://your-host/videos/my_clip_1.mp4
```

If you don't have frame images hosted, tooltips will show broken icons but everything else works fine.

---

### 4 — Install dependencies

```bash
# Python (data generation)
pip install pandas numpy pyarrow

# JavaScript (app build)
cd js-app && npm install
```

---

### 5 — Generate the data file

```bash
# Default — looks for data/2x2_distributions.parquet, falls back to .csv
python3 scripts/generate_js_data.py

# Custom path
python3 scripts/generate_js_data.py --input path/to/your_data.csv

# Adjust downsampling (default 50; 180k pts → 3.6k pts for display)
python3 scripts/generate_js_data.py --downsample 30
```

This writes `js-app/public/data/data.json` (~5–10 MB depending on clip count).

---

### 6 — Run locally

```bash
cd js-app
npm run dev       # dev server with live reload → http://localhost:5173
npm run build     # production build → dist/
npm run preview   # preview the production build → http://localhost:4173
```

---

### 7 — Deploy to GitHub Pages

```bash
cd js-app
npm run deploy    # builds and pushes dist/ to the gh-pages branch
```

In your GitHub repo go to **Settings → Pages → Branch: `gh-pages` → `/ (root)`**.

Your app will be live at:
```
https://<your-username>.github.io/storyboard-visualization/
```

---

## Project structure

```
storyboard-visualization/
├── data/                          raw data files (CSV / Parquet)
├── scripts/
│   └── generate_js_data.py        converts data → js-app/public/data/data.json
├── src/                           Python Dash app (for local use)
│   └── app_2x2.py
├── js-app/                        JavaScript app (static hosting)
│   ├── public/data/data.json      generated — do not edit manually
│   ├── src/
│   │   ├── constants.js           ← edit IMAGE_BASE / VIDEO_BASE here
│   │   ├── dataUtils.js           normalize, frameUrl helpers
│   │   ├── figureBuilders.js      Plotly figure constructors
│   │   ├── App.jsx                layout + shared state
│   │   └── components/
│   │       ├── DistPlot.jsx
│   │       ├── ScatterPlot.jsx
│   │       ├── FrameView.jsx
│   │       └── VideoPlayer.jsx
│   └── dist/                      production build (deploy this folder)
└── requirements.txt               Python dependencies
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No KM-type conditions detected" | Ensure at least one `_dist` column name contains `km` |
| "No EB-type conditions detected" | Ensure at least one `_dist` column name contains `eb` |
| Missing `clip_name` column | Rename your clip identifier column to `clip_name` |
| Broken frame images | Update `IMAGE_BASE` in `js-app/src/constants.js` |
| GitHub Pages shows blank page | Enable Pages in repo Settings; set branch to `gh-pages`, folder to `/ (root)` |
