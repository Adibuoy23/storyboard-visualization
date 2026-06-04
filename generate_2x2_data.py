#!/usr/bin/env python3
"""
generate_2x2_data.py

Generates a combined CSV of KDE distributions for the 2×2 factorial design
(KM vs. EB) × (Realtime vs. Retrospective) across all Sherlock clips.

Output: data/2x2_distributions.csv

Run with:
    /Users/adibuoy23/miniconda3/envs/sem2/bin/python3 generate_2x2_data.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOX_BASE = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/Box-Box/DCL_ARCHIVE/Documents/Events/"
    "exp171_Storyboards&EventBoundaries/Data",
)

DATA_PATHS = {
    "rt_km":    os.path.join(BOX_BASE, "realtime_preprocessed",    "storyboards_preprocessed_data_new.csv"),
    "retro_km": os.path.join(BOX_BASE, "retrospective_preprocessed", "storyboards_preprocessed_data.csv"),
    "rt_eb":    os.path.join(BOX_BASE, "realtime_preprocessed",    "event_boundaries_preprocessed_data.csv"),
    "retro_eb": os.path.join(BOX_BASE, "retrospective_preprocessed", "event_boundaries_preprocessed_data.csv"),
}

OUTPUT_CSV = os.path.join(BASE_DIR, "data", "2x2_distributions.csv")

# KDE bandwidth in seconds
KDE_BANDWIDTH = 0.75  # seconds

# ---------------------------------------------------------------------------
# Load all four datasets
# ---------------------------------------------------------------------------
print("Loading datasets...")
dfs = {}
for key, path in DATA_PATHS.items():
    print(f"  Loading {key}: {path}")
    dfs[key] = pd.read_csv(path)
    print(f"    → {len(dfs[key])} rows, {dfs[key]['video'].nunique()} clips")

# ---------------------------------------------------------------------------
# Determine clip ordering from rt_eb (the authoritative superset)
# ---------------------------------------------------------------------------
clip_order = list(dfs["rt_eb"]["video"].unique())
print(f"\nClip order ({len(clip_order)} clips):")
for i, c in enumerate(clip_order):
    print(f"  {i}: {c}")

# ---------------------------------------------------------------------------
# Helper: fit per-subject KDE and average to group distribution
# ---------------------------------------------------------------------------

def compute_group_kde(df_clip, start_time, end_time, bandwidth_s=KDE_BANDWIDTH):
    """
    For one clip in one condition dataframe, fit a Gaussian KDE per subject
    (bandwidth = bandwidth_s seconds), evaluate on a 1000-pts/s grid,
    average across subjects, and normalise to sum to 1.

    Returns (time_grid, group_dist) where time_grid is in seconds.
    """
    duration = end_time - start_time  # seconds
    n_pts = int(duration * 1000)
    time_grid = np.linspace(start_time, end_time, n_pts)

    subjects = df_clip["subjectID"].unique()
    subject_kdes = []

    for subj in subjects:
        times = df_clip[df_clip["subjectID"] == subj]["selected_frame_time"].dropna().values
        if len(times) < 1:
            continue
        # gaussian_kde bw_method: scotts or a scalar factor
        # We want std dev = bandwidth_s, so we use bw_method = bandwidth_s / std(data)
        # If only one point, use fixed bandwidth directly via a tiny std workaround
        data_std = times.std()
        if data_std < 1e-6 or len(times) == 1:
            # place a Gaussian manually
            kde_vals = np.exp(-0.5 * ((time_grid - times.mean()) / bandwidth_s) ** 2)
            kde_vals /= (bandwidth_s * np.sqrt(2 * np.pi))
        else:
            bw = bandwidth_s / data_std
            kde = gaussian_kde(times, bw_method=bw)
            kde_vals = kde(time_grid)

        subject_kdes.append(kde_vals)

    if len(subject_kdes) == 0:
        return time_grid, np.zeros(n_pts)

    group_dist = np.mean(subject_kdes, axis=0)
    # Normalise to sum to 1
    total = group_dist.sum()
    if total > 0:
        group_dist = group_dist / total

    return time_grid, group_dist


def detect_peaks(dist, df_clip, condition_key):
    """
    Detect top-k peaks in dist where k = int(median responses per subject).
    Returns a 0/1 array the same length as dist.
    """
    # median number of responses per subject (robust to outlier subjects)
    n_per_subj = df_clip.groupby("subjectID").size().median()
    k = max(1, int(round(n_per_subj)))

    peak_indices, properties = find_peaks(dist, distance=50)  # min 50ms separation
    if len(peak_indices) == 0:
        return np.zeros(len(dist), dtype=int)

    # Sort by peak height, take top k
    heights = dist[peak_indices]
    top_k_idx = np.argsort(heights)[::-1][:k]
    top_peaks = peak_indices[top_k_idx]

    indicator = np.zeros(len(dist), dtype=int)
    indicator[top_peaks] = 1
    return indicator


# ---------------------------------------------------------------------------
# Process all clips and build output rows
# ---------------------------------------------------------------------------
condition_keys = ["rt_km", "retro_km", "rt_eb", "retro_eb"]
dist_cols  = ["rt_km_dist",   "retro_km_dist",   "rt_eb_dist",   "retro_eb_dist"]
peaks_cols = ["rt_km_peaks",  "retro_km_peaks",  "rt_eb_peaks",  "retro_eb_peaks"]

all_rows = []

for clip_idx, clip_name in enumerate(clip_order):
    print(f"\nProcessing clip {clip_idx}: {clip_name}")

    # Use rt_eb as the reference for clip metadata
    ref_df = dfs["rt_eb"]
    ref_clip = ref_df[ref_df["video"] == clip_name]
    if len(ref_clip) == 0:
        print(f"  WARNING: clip not found in rt_eb, skipping")
        continue

    start_time = ref_clip["start_time"].iloc[0]
    end_time   = ref_clip["end_time"].iloc[0]
    duration   = end_time - start_time  # seconds
    num_frames = ref_clip["num_frames"].iloc[0]
    n_pts      = int(duration * 1000)

    # time in ms (0-indexed from clip start)
    time_ms = np.arange(n_pts)

    # Compute distributions for each condition
    cond_dists = {}
    cond_peaks = {}

    for key, dist_col, peak_col in zip(condition_keys, dist_cols, peaks_cols):
        cond_df = dfs[key]
        clip_cond = cond_df[cond_df["video"] == clip_name]

        if len(clip_cond) == 0:
            print(f"  {key}: no data for this clip — filling with zeros")
            cond_dists[dist_col]  = np.zeros(n_pts)
            cond_peaks[peak_col]  = np.zeros(n_pts, dtype=int)
        else:
            n_subj = clip_cond["subjectID"].nunique()
            n_resp = len(clip_cond)
            print(f"  {key}: {n_subj} subjects, {n_resp} responses")

            _, dist = compute_group_kde(clip_cond, start_time, end_time)
            peaks   = detect_peaks(dist, clip_cond, key)

            cond_dists[dist_col]  = dist
            cond_peaks[peak_col]  = peaks

    # Build rows for this clip
    clip_rows = pd.DataFrame({
        "clip":            clip_idx,
        "clip_name":       clip_name,
        "pdf_len":         n_pts,
        "num_frames":      num_frames,
        "time_ms":         time_ms,
        "rt_km_dist":      cond_dists["rt_km_dist"],
        "retro_km_dist":   cond_dists["retro_km_dist"],
        "rt_eb_dist":      cond_dists["rt_eb_dist"],
        "retro_eb_dist":   cond_dists["retro_eb_dist"],
        "rt_km_peaks":     cond_peaks["rt_km_peaks"],
        "retro_km_peaks":  cond_peaks["retro_km_peaks"],
        "rt_eb_peaks":     cond_peaks["rt_eb_peaks"],
        "retro_eb_peaks":  cond_peaks["retro_eb_peaks"],
    })

    all_rows.append(clip_rows)
    print(f"  → {n_pts} time points added")

# ---------------------------------------------------------------------------
# Concatenate and save
# ---------------------------------------------------------------------------
print("\nConcatenating all clips...")
output_df = pd.concat(all_rows, ignore_index=True)
print(f"Total rows: {len(output_df)}")
print(f"Clips: {output_df['clip'].nunique()}")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to: {OUTPUT_CSV}")
print("Done.")
