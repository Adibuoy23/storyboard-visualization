# KM and EB Visualization

An interactive browser-based viewer for comparing kernel-density (KM) and event-boundary (EB) response distributions across video clips.

**Live demo:** https://adibuoy23.github.io/storyboard-visualization/
*(Access-restricted — request credentials via the form on the page)*

---

## What it shows

| Panel | Description |
|---|---|
| **Distribution plot** | KM and EB response curves over time for the selected clip. Hover to see the video frame at that moment. |
| **Scatter (Peak Space)** | Every detected peak for all clips plotted as (sum, difference) of the two selected conditions. Selected clip is highlighted. Hover to highlight the corresponding peak in the distribution plot. |
| **Timeline / Tile** | Peak frames from the selected clip arranged chronologically or as a grid. Hover to enlarge and highlight the peak in the distribution plot. |
| **Video player** | The clip video; seeking is driven by hover events on all three plots. |

---

## Included datasets

| Dataset | Description | Reference |
|---|---|---|
| **Storyboards fMRI** | KM (retrospective) and EB (realtime) distributions collected within coarse-grained events of BBC Sherlock, applied to open-access fMRI data | Upadhyayula et al. (2025), [doi:10.1101/2025.08.30.673233](https://doi.org/10.1101/2025.08.30.673233) |
| **2×2 KM & EB** | Four conditions (realtime/retrospective × KM/EB) across 3-min clips with 10-sec overlap | Upadhyayula et al. (2026, in prep) |
| **Mr. Bean** | Realtime EB and retrospective KM distributions for Mr. Bean clips (10-sec overlap) | — |

### Downloading the preprocessed data

The **Download data** button in the app header downloads a zip of the preprocessed, per-participant response CSVs for the selected dataset (key-moment selections and event-boundary button presses). These are the inputs to the KDE distributions shown in the plots. Each zip contains a `README.txt` listing the files and their columns.

The bundles are built by `scripts/package_preprocessed.py` from the source files listed in `scripts/preprocessed_manifest.json`, and written to `js-app/public/downloads/`. To add or update a bundle, edit the manifest and re-run:

```bash
python3 scripts/package_preprocessed.py            # all datasets
python3 scripts/package_preprocessed.py mrbean     # one dataset
```

Then add a `"download"` key to the dataset's entry in `js-app/public/data/datasets.json` (see step 6 below).

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
| `time_ms` | int | Time in milliseconds within the clip *(reconstructed from row index if absent)* |
| `pdf_len` | int | Total length of the distribution in ms *(computed from max `time_ms` if absent)* |
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
- A `<cond>_dist` column whose name contains **`km`** or **`sb`** → KM-type (warm amber colours)
- A `<cond>_dist` column whose name contains **`eb`** → EB-type (cool teal colours)
- At least one of each is required
- Peaks columns are fuzzy-matched: `sb_cont_dist` → `sb_peaks` is fine

Examples that work: `km_dist`, `rt_km_dist`, `retro_km_dist`, `eb_dist`, `rt_eb_dist`, `sb_cont_dist`

---

### 3 — Point the app at your video/image assets

Pass `--image-base` and `--video-base` when generating your data file (see step 4). These are stored in the dataset metadata and used automatically — no code changes needed.

The app constructs URLs like:
```
# Frame image
<image_base> + clip_name + "/frames" + frameNumber.padStart(4,'0') + ".jpg"
# e.g. https://your-host/video_frames/my_clip_1/frames0042.jpg

# Video
<video_base> + clip_name + ".mp4"
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

# Custom path with hosted assets
python3 scripts/generate_js_data.py \
  --input path/to/your_data.csv \
  --image-base "https://your-host/frames/" \
  --video-base "https://your-host/videos/" \
  --output js-app/public/data/my_dataset.json

# All options
python3 scripts/generate_js_data.py --help
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--input` | `data/2x2_distributions.parquet` | Path to CSV or Parquet |
| `--output` | `js-app/public/data/data.json` | Output JSON path |
| `--clip-col` | `clip_name` | Column containing clip identifier |
| `--time-offset` | `10` | Seconds added to time axis (use `0` if no overlap) |
| `--image-base` | GitHub Pages default | Base URL for frame images |
| `--video-base` | GitHub Pages default | Base URL for videos |
| `--downsample` | `50` | Keep every Nth point for display (180k → 3.6k pts) |

---

### 6 — Register the dataset

Add an entry to `js-app/public/data/datasets.json`:

```json
[
  { "label": "My Dataset", "file": "my_dataset.json", "download": "my_dataset_preprocessed.zip" }
]
```

The optional `download` key names a zip in `js-app/public/downloads/` (see *Downloading the preprocessed data* above); omit it to hide the download button for that dataset.

---

### 7 — Run locally

```bash
cd js-app
npm run dev       # dev server with live reload → http://localhost:5173
npm run build     # production build → dist/
npm run preview   # preview the production build → http://localhost:4173
```

---

### 8 — Deploy to GitHub Pages

```bash
cd js-app/dist
git config http.postBuffer 524288000
git add -A
git commit -m "Deploy"
git push --force https://<username>:<token>@github.com/<username>/storyboard-visualization.git main:gh-pages
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
├── data/                               raw data files (CSV / Parquet) — gitignored
├── scripts/
│   ├── generate_js_data.py             converts data → JS-ready JSON
│   ├── package_preprocessed.py         bundles per-participant CSVs into downloadable zips
│   └── preprocessed_manifest.json      ← source files for each download bundle
├── src/                                Python Dash app (for local use)
│   └── app_2x2.py
├── js-app/                             JavaScript app (static hosting)
│   ├── public/data/
│   │   ├── datasets.json               ← register datasets here
│   │   ├── data.json                   2×2 KM & EB dataset
│   │   ├── storyboards_fmri.json       Storyboards fMRI dataset
│   │   └── mrbean.json                 Mr. Bean dataset
│   ├── public/downloads/               preprocessed data zips (built by package_preprocessed.py)
│   ├── src/
│   │   ├── appConfig.js                ← set password here
│   │   ├── constants.js                shared style constants
│   │   ├── dataUtils.js                normalize, frameUrl helpers
│   │   ├── figureBuilders.js           Plotly figure constructors
│   │   ├── App.jsx                     layout + shared state
│   │   └── components/
│   │       ├── PasswordGate.jsx        access control
│   │       ├── HowToUse.jsx            tips dropdown
│   │       ├── DistPlot.jsx
│   │       ├── ScatterPlot.jsx
│   │       ├── FrameView.jsx
│   │       └── VideoPlayer.jsx
│   └── dist/                           production build (deploy this folder)
└── requirements.txt                    Python dependencies
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No KM-type conditions detected" | Ensure at least one `_dist` column name contains `km` or `sb` |
| "No EB-type conditions detected" | Ensure at least one `_dist` column name contains `eb` |
| Missing `clip_name` column | Use `--clip-col` to specify the correct column name |
| Blank frame images | Check `--image-base` URL and that frames are hosted at `<base>/<clip_name>/framesNNNN.jpg` |
| GitHub Pages shows blank page | Enable Pages in repo Settings; set branch to `gh-pages`, folder to `/ (root)` |
| Push fails with HTTP 400 | Run `git config http.postBuffer 524288000` before pushing |
