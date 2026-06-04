#!/usr/bin/env python3
"""
generate_js_data.py
===================
Convert any storyboard-visualization CSV/parquet into a data.json file
consumed by the JS app.  Conditions are auto-detected from column names.

Required columns
----------------
  <clip_col>      string  unique identifier for each clip (default: clip_name)
  pdf_len         int     length of the distribution array (ms)
  num_frames      int     total video frames in the clip
  <cond>_dist     float   one or more distribution columns
  <cond>_peaks    int     matching 0/1 peak-indicator columns
                          (fuzzy-matched: sb_cont_dist → sb_peaks is fine)

Optional columns
----------------
  time_ms         int     time in milliseconds; reconstructed from row index
                          if absent
  clip            int     sort key; clips sorted by this if present

Condition naming
----------------
  Columns containing 'km' or 'sb' → KM-type  (warm amber/brown colours)
  Columns containing 'eb'         → EB-type  (cool teal colours)
  At least one of each is required.

Usage
-----
  python scripts/generate_js_data.py [options]

  --input       path/to/data.csv or .parquet
                (default: data/2x2_distributions.parquet → data/2x2_distributions.csv)
  --output      path/to/output.json
                (default: js-app/public/data/data.json)
  --clip-col    column name for clip identifier  (default: clip_name)
  --time-offset seconds added to time axis for display  (default: 10)
  --downsample  keep every Nth point for the dist plot  (default: 50)
"""

import argparse, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Colour palettes (hex, rgba stroke, rgba fill) ─────────────────────────────
_KM_PALETTE = [
    ("#877243", "rgba(135,114,63,0.81)",  "rgba(135,114,63,0.20)"),
    ("#c9a84c", "rgba(225,190,106,0.81)", "rgba(225,190,106,0.20)"),
    ("#5c4a1e", "rgba(92,74,30,0.81)",    "rgba(92,74,30,0.20)"),
    ("#d4a853", "rgba(212,168,83,0.81)",  "rgba(212,168,83,0.20)"),
]
_EB_PALETTE = [
    ("#266a63", "rgba(38,106,99,0.81)",   "rgba(38,106,99,0.20)"),
    ("#40b0a6", "rgba(64,176,166,0.81)",  "rgba(64,176,166,0.20)"),
    ("#1a4d48", "rgba(26,77,72,0.81)",    "rgba(26,77,72,0.20)"),
    ("#6dd4cc", "rgba(109,212,204,0.81)", "rgba(109,212,204,0.20)"),
]
_OTHER_PALETTE = [
    ("#6b4f8a", "rgba(107,79,138,0.81)",  "rgba(107,79,138,0.20)"),
    ("#c45e9b", "rgba(196,94,155,0.81)",  "rgba(196,94,155,0.20)"),
    ("#3d6ea8", "rgba(61,110,168,0.81)",  "rgba(61,110,168,0.20)"),
    ("#e07b39", "rgba(224,123,57,0.81)",  "rgba(224,123,57,0.20)"),
]

_PART_LABELS = {
    "rt": "Realtime", "retro": "Retro",
    "km": "KM", "eb": "EB", "sb": "SB",
    "cont": "Continuous", "hrf": "HRF", "fir": "FIR", "pca": "PCA",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_label(cond: str) -> str:
    return " ".join(_PART_LABELS.get(p, p.capitalize()) for p in cond.split("_"))

def cond_type(cond: str) -> str:
    lower = cond.lower()
    if "km" in lower or "sb" in lower: return "km"
    if "eb" in lower: return "eb"
    return "other"

def find_peaks_col(cond: str, df_cols: list):
    """Find a peaks column for `cond`, allowing truncated prefix matches.
    e.g. sb_cont_dist → tries sb_cont_peaks, then sb_peaks."""
    exact = f"{cond}_peaks"
    if exact in df_cols:
        return exact
    parts = cond.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:i]) + "_peaks"
        if candidate in df_cols:
            return candidate
    return None

def detect_conditions(df: pd.DataFrame):
    conditions = []
    for dc in [c for c in df.columns if c.endswith("_dist")]:
        cond = dc[:-5]
        pk   = find_peaks_col(cond, df.columns.tolist())
        if pk:
            conditions.append((cond, pk))
        else:
            print(f"  Warning: no peaks column found for {dc} — skipped")
    return conditions

def assign_colors(conditions):
    km_i = eb_i = oth_i = 0
    colors = {}
    for cond, _ in conditions:
        t = cond_type(cond)
        if t == "km":
            hex_, rgba, fill = _KM_PALETTE[km_i % len(_KM_PALETTE)]; km_i  += 1
        elif t == "eb":
            hex_, rgba, fill = _EB_PALETTE[eb_i % len(_EB_PALETTE)]; eb_i  += 1
        else:
            hex_, rgba, fill = _OTHER_PALETTE[oth_i % len(_OTHER_PALETTE)]; oth_i += 1
        colors[cond] = {"hex": hex_, "color": rgba, "fill_color": fill}
    return colors

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    default_out = os.path.join(ROOT, "js-app", "public", "data", "data.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default=None)
    parser.add_argument("--output",      default=default_out)
    parser.add_argument("--clip-col",    default="clip_name",
                        help="Column containing clip identifier (default: clip_name)")
    parser.add_argument("--time-offset", default=10.0, type=float,
                        help="Seconds added to time axis for display (default: 10)")
    parser.add_argument("--downsample",  default=50, type=int)
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if args.input:
        path = args.input
        df   = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    elif os.path.exists(os.path.join(ROOT, "data", "2x2_distributions.parquet")):
        path = os.path.join(ROOT, "data", "2x2_distributions.parquet")
        df   = pd.read_parquet(path)
    else:
        path = os.path.join(ROOT, "data", "2x2_distributions.csv")
        df   = pd.read_csv(path)
    print(f"Loaded {path}  →  {len(df):,} rows")

    # ── Clip column ───────────────────────────────────────────────────────────
    clip_col = args.clip_col
    if clip_col not in df.columns:
        raise ValueError(f"Clip column '{clip_col}' not found. "
                         f"Available columns: {df.columns.tolist()}")

    # ── Reconstruct time_ms if missing ────────────────────────────────────────
    if "time_ms" not in df.columns:
        print("  'time_ms' not found — reconstructing from row index within each clip")
        df["time_ms"] = df.groupby(clip_col).cumcount()

    # ── Validate other required columns ───────────────────────────────────────
    for col in ("pdf_len", "num_frames"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found.")

    # ── Clip order ────────────────────────────────────────────────────────────
    if "clip" in df.columns:
        clip_order = (df[[clip_col, "clip"]].drop_duplicates()
                      .sort_values("clip")[clip_col].tolist())
    else:
        clip_order = sorted(df[clip_col].unique())

    print(f"  {len(clip_order)} clips")

    # ── Detect conditions ─────────────────────────────────────────────────────
    cond_pairs = detect_conditions(df)
    if not cond_pairs:
        raise ValueError("No valid condition columns found (need *_dist + *_peaks pairs).")

    conditions = [c for c, _ in cond_pairs]
    peaks_cols = {c: pk for c, pk in cond_pairs}

    km_conds = [c for c in conditions if cond_type(c) == "km"]
    eb_conds = [c for c in conditions if cond_type(c) == "eb"]
    if not km_conds:
        raise ValueError("No KM/SB-type conditions found (column must contain 'km' or 'sb').")
    if not eb_conds:
        raise ValueError("No EB-type conditions found (column must contain 'eb').")

    print(f"  Conditions: {conditions}")
    print(f"  KM/SB: {km_conds}  |  EB: {eb_conds}")
    print(f"  Peaks mapping: { {c: peaks_cols[c] for c in conditions} }")

    colors = assign_colors(cond_pairs)
    TO     = args.time_offset
    meta = {
        "conditions":    conditions,
        "km_conditions": km_conds,
        "eb_conditions": eb_conds,
        "default_cond1": km_conds[0],
        "default_cond2": eb_conds[0],
        "labels":      {c: make_label(c)            for c in conditions},
        "colors":      {c: colors[c]["color"]       for c in conditions},
        "fill_colors": {c: colors[c]["fill_color"]  for c in conditions},
        "hex_colors":  {c: colors[c]["hex"]         for c in conditions},
        "time_offset": TO,
    }

    # ── Per-clip data ─────────────────────────────────────────────────────────
    ds   = args.downsample
    data = {}
    for cn in clip_order:
        print(f"  Processing {cn} ...", end=" ", flush=True)
        cdf = df[df[clip_col] == cn].reset_index(drop=True)

        time_s        = (cdf["time_ms"].values[::ds] / 1000.0 + TO).tolist()
        distributions = {c: cdf[f"{c}_dist"].values[::ds].tolist() for c in conditions}
        norm_params   = {c: {
            "max":  float(cdf[f"{c}_dist"].values.max()),
            "mean": float(cdf[f"{c}_dist"].values.mean()),
            "std":  float(cdf[f"{c}_dist"].values.std()),
        } for c in conditions}

        peaks = {}
        for c in conditions:
            pk   = peaks_cols[c]
            idxs = np.where(cdf[pk].values == 1)[0]
            peaks[c] = {
                "indices":      idxs.tolist(),
                "time_s":       (idxs / 1000.0 + TO).tolist(),
                "dist_at_peak": {c2: cdf[f"{c2}_dist"].values[idxs].tolist()
                                 for c2 in conditions},
            }

        data[cn] = {
            "pdf_len":       int(cdf["pdf_len"].iloc[0]),
            "num_frames":    int(cdf["num_frames"].iloc[0]),
            "time_s":        time_s,
            "distributions": distributions,
            "norm_params":   norm_params,
            "peaks":         peaks,
        }
        total = sum(len(peaks[c]["indices"]) for c in conditions)
        print(f"{total} peaks")

    output = {"meta": meta, "clips": clip_order, "data": data}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Writing {args.output} ...")
    with open(args.output, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"Done — {os.path.getsize(args.output)/1e6:.1f} MB")

if __name__ == "__main__":
    main()
