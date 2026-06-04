#!/usr/bin/env python3
"""
app_2x2.py — 2×2 KM & EB Distribution Viewer

Launch:
    /Users/adibuoy23/miniconda3/envs/events_dashboard/bin/python3 src/app_2x2.py
Open http://127.0.0.1:8051
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback, no_update, Patch
from dash.exceptions import PreventUpdate
import dash_player as dp

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_CSV = os.path.join(ROOT_DIR, "data", "2x2_distributions.csv")
DATA_PARQUET = os.path.join(ROOT_DIR, "data", "2x2_distributions.parquet")

IMAGE_BASE  = "https://adibuoy23.github.io/event_representations/video_frames/"
VIDEO_BASE  = "https://adibuoy23.github.io/event_representations/videos/"
TIME_OFFSET = 10        # selected_frame_time=0 corresponds to raw clip t=10 s
# No SKIP_PTS needed — the -10 offset is already applied in preprocessing,
# so selected_frame_time starts at ~0 with real responses from the beginning

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLORS = {
    "rt_km":    "rgba(135, 114, 63, 0.81)",
    "retro_km": "rgba(225, 190, 106, 0.81)",
    "rt_eb":    "rgba(38, 106, 99, 0.81)",
    "retro_eb": "rgba(64, 176, 166, 0.81)",
}
FILL_COLORS = {
    "rt_km":    "rgba(135, 114, 63, 0.20)",
    "retro_km": "rgba(225, 190, 106, 0.20)",
    "rt_eb":    "rgba(38, 106, 99, 0.20)",
    "retro_eb": "rgba(64, 176, 166, 0.20)",
}
HEX_COLORS = {
    "rt_km":    "#877243",
    "retro_km": "#c9a84c",
    "rt_eb":    "#266a63",
    "retro_eb": "#40b0a6",
}
LABELS = {
    "rt_km":    "Realtime KM",
    "retro_km": "Retro KM",
    "rt_eb":    "Realtime EB",
    "retro_eb": "Retro EB",
}
CONDITIONS = ["rt_km", "retro_km", "rt_eb", "retro_eb"]
DIST_COLS  = {c: f"{c}_dist"  for c in CONDITIONS}
PEAKS_COLS = {c: f"{c}_peaks" for c in CONDITIONS}

ALL_COND_OPTIONS = [{"label": LABELS[c], "value": c} for c in CONDITIONS]

# ---------------------------------------------------------------------------
# Load & prep data
# ---------------------------------------------------------------------------
if os.path.exists(DATA_PARQUET):
    print(f"Loading {DATA_PARQUET} ...")
    df = pd.read_parquet(DATA_PARQUET)
else:
    print(f"Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)
df["peak_indices"] = df.groupby("clip").cumcount()
print(f"  → {len(df)} rows, {df['clip'].nunique()} clips")

clip_meta = (df[["clip", "clip_name"]].drop_duplicates()
               .sort_values("clip").reset_index(drop=True))
clip_options = [{"label": r["clip_name"], "value": r["clip_name"]}
                for _, r in clip_meta.iterrows()]
default_clip = clip_options[0]["value"]

# Pre-extract peak rows per condition (all clips)
peaks_df = {c: df[df[PEAKS_COLS[c]] == 1].copy() for c in CONDITIONS}

# Per-clip lookup: avoids filtering the 3M-row dataframe on every hover callback
print("Precomputing per-clip data...")
clip_data = {}
for cn in clip_meta["clip_name"]:
    cdf = df[df["clip_name"] == cn]
    clip_data[cn] = {
        "pdf_len":    int(cdf["pdf_len"].iloc[0]),
        "num_frames": int(cdf["num_frames"].iloc[0]),
        "time_ms":    cdf["time_ms"].values.copy(),
        "dist":       {c: cdf[DIST_COLS[c]].values.copy()  for c in CONDITIONS},
        "peaks":      {c: np.where(cdf[PEAKS_COLS[c]].values == 1)[0] for c in CONDITIONS},
    }
print("  → done")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(arr, mode):
    if mode == "zscore":
        s = arr.std()
        return (arr - arr.mean()) / s if s > 0 else arr
    mx = arr.max()
    return arr / mx if mx > 0 else arr


CLIP_TOTAL_S = 190.0   # each clip is 3 min 10 sec; selected_frame_time already has -10 applied

def frame_url(clip_name, peak_index, pdf_len, num_frames):
    # peak_index is on the selected_frame_time grid (0–180 s * 1000 pts/s).
    # Raw clip time = selected_frame_time + 10 s (the offset removed during preprocessing).
    raw_s = peak_index / 1000.0 + TIME_OFFSET
    f = int(raw_s / CLIP_TOTAL_S * num_frames)
    f = max(0, min(f, int(num_frames) - 1))
    return f"{IMAGE_BASE}{clip_name}/frames{str(f).zfill(4)}.jpg"


def fmt_ts(t_s):
    m = int(t_s // 60)
    s = t_s % 60
    return f"{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def make_dist_fig(clip_name, scale_type, cond1="rt_km", cond2="rt_eb"):
    cd     = clip_data[clip_name]
    time_s = cd["time_ms"] / 1000.0 + TIME_OFFSET
    active = [cond1, cond2]

    dists = {c: normalize(cd["dist"][c], scale_type) for c in active}
    max_y = max(d.max() for d in dists.values()) if dists else 1.0

    fig = go.Figure()
    for c in active:
        fig.add_trace(go.Scatter(
            x=time_s, y=dists[c], name=LABELS[c],
            mode="lines", fill="tozeroy",
            fillcolor=FILL_COLORS[c],
            line=dict(color=COLORS[c], width=2),
            hovertemplate="%{x:.2f}s<extra>" + LABELS[c] + "</extra>",
        ))

    # Peak lines for the two active conditions
    for c in active:
        for ix in cd["peaks"][c]:
            fig.add_shape(
                type="line",
                x0=time_s[ix], x1=time_s[ix],
                y0=0, y1=max_y,
                line=dict(color=COLORS[c], width=1.5, dash="dash"),
            )

    # Hover-highlight placeholder — always the last shape, updated via Patch
    fig.add_shape(
        type="rect", x0=0, x1=0, y0=0, y1=0,
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(0,0,0,0)",
        visible=False,
    )

    fig.update_layout(
        title=dict(text="Response Distributions", font=dict(size=13, color="#555"),
                   x=0.5, xanchor="center"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=390,
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(size=12, color="#555")),
            showgrid=False,
            linecolor="#d0d0d0", linewidth=1.5, showline=True,
            zeroline=False, tickfont=dict(size=10, color="#666"),
            ticks="outside", ticklen=4, tickcolor="#ccc",
        ),
        yaxis=dict(
            title=dict(text="Density", font=dict(size=12, color="#555")),
            showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
            linecolor="#d0d0d0", linewidth=1.5, showline=True,
            zeroline=False, tickfont=dict(size=10, color="#666"),
            ticks="outside", ticklen=4, tickcolor="#ccc",
        ),
        legend=dict(
            orientation="h", y=1.01, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.08)", borderwidth=1,
            font=dict(size=11, color="#333"),
        ),
        margin=dict(l=52, r=12, t=44, b=48),
        font=dict(size=12, color="#333", family="Arial"),
        hovermode="x",
    )
    return fig


def make_scatter_fig(clip_name, cond1, cond2, scale_type):
    """
    Scatter of (KM_norm + EB_norm) vs (KM_norm - EB_norm) at every peak,
    coloured by condition. Selected clip's peaks get a black outline.
    customdata = [peak_index, pdf_len, num_frames, clip_name_str]
    """
    if scale_type == "norm":
        x_title = f"({LABELS[cond1]} + {LABELS[cond2]}) distribution"
        y_title = f"({LABELS[cond1]} − {LABELS[cond2]}) distribution"
    else:
        x_title = f"({LABELS[cond1]} + {LABELS[cond2]}) z-score"
        y_title = f"({LABELS[cond1]} − {LABELS[cond2]}) z-score"

    fig = go.Figure()

    for i, cn in enumerate(clip_meta["clip_name"]):
        cd   = clip_data[cn]
        c1_n = normalize(cd["dist"][cond1], scale_type)
        c2_n = normalize(cd["dist"][cond2], scale_type)
        pdf_len    = cd["pdf_len"]
        num_frames = cd["num_frames"]
        is_selected = (cn == clip_name)

        for cond, color in [(cond1, COLORS[cond1]), (cond2, COLORS[cond2])]:
            idxs = cd["peaks"][cond]
            if len(idxs) == 0:
                continue
            xs = c1_n[idxs] + c2_n[idxs]
            ys = c1_n[idxs] - c2_n[idxs]
            peak_global = cdf["peak_indices"].values[idxs]
            cdata = np.column_stack([
                peak_global,
                np.full(len(idxs), pdf_len),
                np.full(len(idxs), num_frames),
            ])

            opacity   = 0.85 if is_selected else 0.25
            marker_kw = dict(size=10, color=color, opacity=opacity,
                             line=dict(width=2 if is_selected else 0,
                                       color="black"))

            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                name=LABELS[cond] if i == 0 else None,
                showlegend=(i == 0),
                marker=marker_kw,
                customdata=np.hstack([cdata,
                    np.full((len(idxs), 1), cn).reshape(-1, 1)]),
                hovertemplate=(
                    f"{LABELS[cond]}<br>"
                    f"sum=%{{x:.4f}}<br>diff=%{{y:.4f}}<extra></extra>"
                ),
            ))

    # Directional annotations
    fig.add_annotation(x=0.5, y=1.07, xref="paper", yref="paper",
                       text=f"▲ more {LABELS[cond1]}",
                       showarrow=False,
                       font=dict(size=12, color=HEX_COLORS[cond1]),
                       xanchor="center")
    fig.add_annotation(x=0.5, y=-0.13, xref="paper", yref="paper",
                       text=f"▼ more {LABELS[cond2]}",
                       showarrow=False,
                       font=dict(size=12, color=HEX_COLORS[cond2]),
                       xanchor="center")
    fig.add_annotation(x=1.03, y=0.5, xref="paper", yref="paper",
                       text="▶ both",
                       showarrow=False,
                       font=dict(size=12, color="#888"),
                       xanchor="left", yanchor="middle")

    fig.update_layout(
        title=dict(text="Peak Space", font=dict(size=13, color="#555"),
                   x=0.5, xanchor="center"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=310,
        xaxis=dict(
            title=dict(text="Sum", font=dict(size=13, color="#444")),
            showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
            linecolor="#d0d0d0", linewidth=1.5, showline=True,
            zeroline=True, zerolinecolor="rgba(0,0,0,0.18)", zerolinewidth=1,
            tickfont=dict(size=12, color="#555"),
            ticks="outside", ticklen=4, tickcolor="#ccc",
        ),
        yaxis=dict(
            title=dict(text="Difference", font=dict(size=13, color="#444")),
            showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
            linecolor="#d0d0d0", linewidth=1.5, showline=True,
            zeroline=True, zerolinecolor="rgba(0,0,0,0.18)", zerolinewidth=1,
            tickfont=dict(size=12, color="#555"),
            ticks="outside", ticklen=4, tickcolor="#ccc",
        ),
        legend=dict(
            orientation="h", y=1.01, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.08)", borderwidth=1,
            font=dict(size=12, color="#333"),
        ),
        margin=dict(l=55, r=52, t=44, b=55),
        font=dict(size=13, color="#333", family="Arial"),
    )
    return fig


def make_frames_fig(clip_name, cond1, cond2):
    """Tile view: two rows (cond1, cond2), columns = peak frames."""
    active    = [cond1, cond2]
    rows_data = [peaks_df[c][peaks_df[c]["clip_name"] == clip_name] for c in active]
    max_cols  = max((len(r) for r in rows_data), default=1)
    if max_cols == 0:
        return go.Figure()

    frames_fig = make_subplots(rows=2, cols=max_cols, vertical_spacing=0.04)

    for row_i, (cond, cond_peaks) in enumerate(zip(active, rows_data), start=1):
        if cond_peaks.empty:
            continue
        pdf_len    = cond_peaks["pdf_len"].iloc[0]
        num_frames = cond_peaks["num_frames"].iloc[0]

        for col_i in range(max_cols):
            frames_fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="markers",
                           marker_opacity=0, showlegend=False),
                row=row_i, col=col_i + 1)

        for col_i, (_, rec) in enumerate(cond_peaks.iterrows()):
            url = frame_url(clip_name, rec["peak_indices"], pdf_len, num_frames)
            frames_fig.add_layout_image(
                row=row_i, col=col_i + 1,
                source=url,
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                sizex=1, sizey=1)

    frames_fig.update_xaxes(showgrid=False, showticklabels=False,
                             showline=True, linewidth=2,
                             linecolor="rgba(0,0,0,0.15)", mirror=True)
    frames_fig.update_yaxes(showgrid=False, showticklabels=False,
                             showline=True, linewidth=2,
                             linecolor="rgba(0,0,0,0.15)", mirror=True)

    # Colored row labels
    for row_i, cond in enumerate(active):
        y_center = 1.0 - (row_i + 0.5) * 0.5
        frames_fig.add_annotation(
            text=f"<b>{LABELS[cond]}</b>",
            xref="paper", yref="paper",
            x=-0.01, y=y_center,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(color=HEX_COLORS[cond], size=13, family="Arial"),
        )

    frames_fig.update_layout(
        width=max(200 * max_cols, 400), height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=110, r=10, t=10, b=10),
    )
    return frames_fig


def make_timeline_fig(clip_name, cond1, cond2):
    """
    Single horizontal timeline.
    Cond1 peaks float ABOVE with short upward arrows.
    Cond2 peaks float BELOW with short downward arrows.
    Frames are small; hovering shows an enlarged version.
    """
    cd       = clip_data[clip_name]
    duration = cd["time_ms"].max() / 1000.0 + TIME_OFFSET
    x_min, x_max = TIME_OFFSET - 4, duration + 4

    # ── Geometry ─────────────────────────────────────────────────────────────
    # y-range: [-3, 3]
    # Timeline at y = 0
    # Cond1 frames: centre y = +1.2,  bottom = +0.8  → arrow tip = +0.8
    # Cond2 frames: centre y = -1.2,  top    = -0.8  → arrow tip = -0.8
    # Arrow shaft: from y=±0.12 (just off the line) to y=±0.8 → short!
    IMG_W          = 16.0    # seconds wide
    IMG_H          =  0.75   # y-units tall
    Y_TOP_CTR      =  1.22   # centre of cond1 images
    Y_BOT_CTR      = -1.22   # centre of cond2 images
    Y_TOP_ARROW    =  Y_TOP_CTR - IMG_H / 2   # top arrow tip  (+0.845)
    Y_BOT_ARROW    =  Y_BOT_CTR + IMG_H / 2   # bottom arrow tip (-0.845)
    Y_ARROW_BASE   =  0.10   # where arrow tail leaves the timeline (gap)
    Y_TS_TOP       =  0.40   # timestamp y for cond1
    Y_TS_BOT       = -0.40   # timestamp y for cond2

    # Semi-transparent fill under each image band
    FILL_TOP  = FILL_COLORS[cond1]
    FILL_BOT  = FILL_COLORS[cond2]

    fig = go.Figure()

    # Background bands for each condition zone
    fig.add_hrect(y0=Y_TOP_CTR - IMG_H/2 - 0.08,
                  y1=Y_TOP_CTR + IMG_H/2 + 0.08,
                  fillcolor=FILL_TOP, line_width=0, layer="below")
    fig.add_hrect(y0=Y_BOT_CTR - IMG_H/2 - 0.08,
                  y1=Y_BOT_CTR + IMG_H/2 + 0.08,
                  fillcolor=FILL_BOT, line_width=0, layer="below")

    # ── Timeline bar ─────────────────────────────────────────────────────────
    fig.add_shape(type="line",
                  x0=x_min, x1=x_max, y0=0, y1=0,
                  line=dict(color="rgba(30,30,30,0.85)", width=3))

    # Axis anchor (keeps y-range stable)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[-3, 3],
        mode="markers", marker_opacity=0, showlegend=False,
    ))

    # ── Per-condition ─────────────────────────────────────────────────────────
    cfg = [
        (cond1, Y_TOP_CTR, Y_TOP_ARROW,  Y_ARROW_BASE,  Y_TS_TOP),
        (cond2, Y_BOT_CTR, Y_BOT_ARROW, -Y_ARROW_BASE,  Y_TS_BOT),
    ]

    for cond, y_ctr, y_tip, y_base, y_ts in cfg:
        color = HEX_COLORS[cond]
        cond_peaks = peaks_df[cond][peaks_df[cond]["clip_name"] == clip_name]

        # Condition label — coloured pill on the left
        fig.add_annotation(
            text=f"<b> {LABELS[cond]} </b>",
            xref="paper", yref="y",
            x=-0.004, y=y_ctr,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(color="white", size=12, family="Arial"),
            bgcolor=color, bordercolor=color,
            borderwidth=1, borderpad=4, opacity=0.9,
        )

        peak_idxs = cd["peaks"][cond]
        if len(peak_idxs) == 0:
            continue

        pdf_len    = cd["pdf_len"]
        num_frames = cd["num_frames"]

        peak_times, urls, tss = [], [], []

        for pidx in peak_idxs:
            t        = int(pidx) / 1000.0 + TIME_OFFSET
            url      = frame_url(clip_name, int(pidx), pdf_len, num_frames)
            ts_label = fmt_ts(int(pidx) / 1000.0)
            peak_times.append(t)
            urls.append(url)
            tss.append(ts_label)

            # Frame image
            fig.add_layout_image(
                source=url,
                x=t, y=y_ctr + IMG_H / 2,
                xref="x", yref="y",
                xanchor="center", yanchor="top",
                sizex=IMG_W, sizey=IMG_H,
                layer="above",
            )

            # Colored border around each frame
            fig.add_shape(
                type="rect",
                x0=t - IMG_W/2, x1=t + IMG_W/2,
                y0=y_ctr - IMG_H/2, y1=y_ctr + IMG_H/2,
                xref="x", yref="y",
                line=dict(color=color, width=1.5),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

            # Short arrow with solid head: tail just above/below the line
            fig.add_annotation(
                x=t, y=y_tip,      # arrowhead at image edge
                ax=t, ay=y_base,   # tail just off the timeline
                xref="x", yref="y",
                axref="x", ayref="y",
                arrowhead=4,       # solid filled triangle
                arrowsize=1.8,
                arrowwidth=1.5,
                arrowcolor=color,
                showarrow=True, text="",
            )

            # Timestamp below/above the arrow
            fig.add_annotation(
                x=t, y=y_ts,
                text=f"<span style='font-size:9px;color:{color};font-family:Arial'>"
                     f"{ts_label}</span>",
                xref="x", yref="y",
                showarrow=False, xanchor="center", yanchor="middle",
            )

        # Invisible scatter for hover → enlarged tooltip
        fig.add_trace(go.Scatter(
            x=peak_times,
            y=[y_ctr] * len(peak_times),
            mode="markers",
            marker=dict(size=30, color=color, opacity=0.0),
            name=LABELS[cond],
            showlegend=True,
            customdata=list(zip(urls, tss)),
            hovertemplate=f"<b>{LABELS[cond]}</b>  t = %{{x:.1f}} s<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(size=13)),
            range=[x_min, x_max],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            gridwidth=1,
            linecolor="rgba(30,30,30,0.4)",
            linewidth=1.5,
            showline=True,
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            range=[-3, 3],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13, color="#222", family="Arial"),
        legend=dict(
            orientation="h", y=1.04, x=0.5, xanchor="center",
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
        ),
        margin=dict(l=140, r=20, t=20, b=55),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[
        {"href": "https://unpkg.com/purecss@1.0.1/build/pure-min.css",
         "rel": "stylesheet",
         "integrity": "sha384-oAOxQR6DkCoMliIh8yFnu25d7Eq/PHS21PClpwjOTeU2jRSq11vu66rf90/cZr47",
         "crossorigin": "anonymous"},
        "https://unpkg.com/purecss@1.0.1/build/grids-responsive-min.css",
        "https://unpkg.com/purecss@1.0.1/build/base-min.css",
    ],
)
app.title = "2×2 KM & EB Distribution Viewer"

CARD = {
    "backgroundColor": "white",
    "borderRadius": "10px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.04)",
    "padding": "10px",
    "boxSizing": "border-box",
}

app.layout = html.Div([

    # ── Header bar ─────────────────────────────────────────────────────────
    html.Div(
        style={
            "display": "flex", "alignItems": "center", "gap": "18px",
            "padding": "10px 16px 10px 16px",
            "background": "white",
            "borderBottom": "1px solid #e8e8e8",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
            "flexWrap": "wrap",
        },
        children=[
            html.Span("2×2 KM & EB Viewer", style={
                "color": "#2c3e50", "fontSize": 20,
                "fontWeight": "700", "letterSpacing": "-0.3px",
                "whiteSpace": "nowrap",
            }),
            html.Div(style={"width": "1px", "height": "28px",
                            "background": "#e0e0e0", "flexShrink": 0}),
            dcc.Dropdown(clip_options, placeholder="Select a clip",
                         id="clip-dropdown",
                         style={"width": "340px", "fontSize": "13px"}),
            html.Div(style={"width": "1px", "height": "28px",
                            "background": "#e0e0e0", "flexShrink": 0}),
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "8px"},
                     children=[
                         html.Span("Scaling", style={"color": "#777",
                                                      "fontSize": "12px",
                                                      "fontWeight": "600",
                                                      "whiteSpace": "nowrap"}),
                         dcc.RadioItems(
                             [{"label": "Max → 1", "value": "norm"},
                              {"label": "Z-score",  "value": "zscore"}],
                             id="scale-type", value="norm",
                             labelStyle={"display": "inline-block",
                                         "marginRight": "12px",
                                         "fontSize": "12px", "color": "#444"},
                             inputStyle={"marginRight": "4px"}),
                     ]),
            html.Div(style={"width": "1px", "height": "28px",
                            "background": "#e0e0e0", "flexShrink": 0}),
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "8px"},
                     children=[
                         html.Span("Frame view", style={"color": "#777",
                                                         "fontSize": "12px",
                                                         "fontWeight": "600",
                                                         "whiteSpace": "nowrap"}),
                         dcc.RadioItems(
                             [{"label": "Tile",     "value": "Tile"},
                              {"label": "Timeline", "value": "Timeline"}],
                             id="view-type", value="Timeline",
                             labelStyle={"display": "inline-block",
                                         "marginRight": "12px",
                                         "fontSize": "12px", "color": "#444"},
                             inputStyle={"marginRight": "4px"}),
                     ]),
        ],
    ),

    # ── Row 1: video | distribution | scatter ──────────────────────────────
    html.Div(
        style={
            "display": "flex", "alignItems": "stretch",
            "gap": "12px", "padding": "12px 12px 6px 12px",
        },
        children=[

            # Video player card
            html.Div(
                style={**CARD, "flex": "0 0 200px", "width": "200px",
                       "minWidth": 0, "overflow": "hidden",
                       "display": "flex", "flexDirection": "column",
                       "position": "relative", "zIndex": 2},
                children=[
                    html.Div("Video", style={
                        "fontSize": "11px", "fontWeight": "700",
                        "color": "#999", "letterSpacing": "0.8px",
                        "textTransform": "uppercase", "marginBottom": "8px",
                    }),
                    html.Div(
                        style={"width": "100%", "overflow": "hidden"},
                        children=[
                            dp.DashPlayer(id="player", url="", controls=True,
                                          width="100%", height="120px"),
                        ],
                    ),
                ],
            ),

            # Distribution card
            html.Div(
                style={**CARD, "flex": "1 1 0", "minWidth": 0,
                       "overflow": "hidden", "position": "relative", "zIndex": 1},
                children=[
                    dcc.Graph(id="dist-plot", clear_on_unhover=True,
                              config={"displayModeBar": False},
                              style={"height": "390px"}),
                    dcc.Tooltip(id="dist-tooltip", direction="bottom"),
                ],
            ),

            # Scatter card (includes condition selectors)
            html.Div(
                style={**CARD, "flex": "0 0 28%", "minWidth": 0},
                children=[
                    # Condition selector row
                    html.Div(
                        style={
                            "display": "flex", "gap": "20px",
                            "marginBottom": "8px",
                            "paddingBottom": "8px",
                            "borderBottom": "1px solid #f0f0f0",
                        },
                        children=[
                            html.Div([
                                html.Div("Condition A  (x + y)", style={
                                    "fontSize": "11px", "fontWeight": "700",
                                    "color": HEX_COLORS["rt_km"],
                                    "letterSpacing": "0.4px",
                                    "marginBottom": "5px",
                                }),
                                dcc.RadioItems(
                                    ALL_COND_OPTIONS,
                                    id="cond1-select", value="rt_km",
                                    labelStyle={"display": "block",
                                                "marginBottom": "2px",
                                                "fontSize": "12px",
                                                "color": "#444"},
                                    inputStyle={"marginRight": "5px"}),
                            ]),
                            html.Div(style={"width": "1px",
                                            "background": "#f0f0f0",
                                            "flexShrink": 0}),
                            html.Div([
                                html.Div("Condition B  (x − y)", style={
                                    "fontSize": "11px", "fontWeight": "700",
                                    "color": HEX_COLORS["rt_eb"],
                                    "letterSpacing": "0.4px",
                                    "marginBottom": "5px",
                                }),
                                dcc.RadioItems(
                                    ALL_COND_OPTIONS,
                                    id="cond2-select", value="rt_eb",
                                    labelStyle={"display": "block",
                                                "marginBottom": "2px",
                                                "fontSize": "12px",
                                                "color": "#444"},
                                    inputStyle={"marginRight": "5px"}),
                            ]),
                        ],
                    ),
                    dcc.Graph(id="scatter-plot", clear_on_unhover=True,
                              config={"displayModeBar": False},
                              style={"height": "280px"}),
                    dcc.Tooltip(id="scatter-tooltip", direction="bottom"),
                ],
            ),
        ],
    ),

    # ── Row 2: full-width frame timeline ───────────────────────────────────
    html.Div(
        style={**CARD, "margin": "0 12px 12px 12px"},
        children=[
            dcc.Graph(id="frame-viz", config={"displayModeBar": False},
                      style={"width": "100%"}, clear_on_unhover=True),
            dcc.Tooltip(id="timeline-tooltip", direction="bottom"),
        ],
    ),

], style={
    "maxWidth": "1600px", "margin": "0 auto",
    "fontFamily": "'Arial', sans-serif",
    "backgroundColor": "#f4f5f7",
    "minHeight": "100vh",
})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(Output("player", "url"),
          Input("clip-dropdown", "value"),
          prevent_initial_call=True)
def update_player(clip_name):
    if not clip_name:
        raise PreventUpdate
    return f"{VIDEO_BASE}{clip_name}.mp4"


@callback(Output("dist-plot", "figure"),
          Input("clip-dropdown", "value"),
          Input("scale-type", "value"),
          Input("cond1-select", "value"),
          Input("cond2-select", "value"))
def update_dist(clip_name, scale_type, cond1, cond2):
    if not clip_name:
        raise PreventUpdate
    return make_dist_fig(clip_name, scale_type, cond1, cond2)


@callback(Output("scatter-plot", "figure"),
          Input("clip-dropdown", "value"),
          Input("cond1-select", "value"),
          Input("cond2-select", "value"),
          Input("scale-type", "value"))
def update_scatter(clip_name, cond1, cond2, scale_type):
    if not clip_name:
        raise PreventUpdate
    return make_scatter_fig(clip_name, cond1, cond2, scale_type)


@callback(Output("frame-viz", "figure"),
          Input("clip-dropdown", "value"),
          Input("cond1-select", "value"),
          Input("cond2-select", "value"),
          Input("view-type", "value"))
def update_frames(clip_name, cond1, cond2, view_type):
    if not clip_name:
        raise PreventUpdate
    if view_type == "Timeline":
        return make_timeline_fig(clip_name, cond1, cond2)
    return make_frames_fig(clip_name, cond1, cond2)


# Hover on distribution plot → tooltip image + seek video
@callback(
    Output("dist-tooltip", "show"),
    Output("dist-tooltip", "bbox"),
    Output("dist-tooltip", "children"),
    Output("player", "seekTo", allow_duplicate=True),
    Input("dist-plot", "hoverData"),
    Input("clip-dropdown", "value"),
    prevent_initial_call=True,
)
def dist_hover(hoverData, clip_name):
    if hoverData is None or not clip_name:
        return False, no_update, no_update, no_update

    pt     = hoverData["points"][0]
    bbox   = pt["bbox"]
    display_s = pt["x"]                        # raw clip time (10–190 s display)
    time_s    = display_s - TIME_OFFSET        # selected_frame_time (0–180 s)

    cd         = clip_data[clip_name]
    num_frames = cd["num_frames"]
    pdf_len    = cd["pdf_len"]

    peak_index = int(time_s * 1000)           # ms → index (1000 pts/s)
    peak_index = max(0, min(peak_index, int(pdf_len) - 1))

    url = frame_url(clip_name, peak_index, pdf_len, num_frames)

    children = html.Div([
        html.Img(src=url, style={"width": "220px", "display": "block",
                                 "borderRadius": "4px", "marginBottom": "4px"}),
        html.Span(fmt_ts(time_s),
                  style={"fontSize": "13px", "color": "#333",
                         "fontFamily": "Arial, sans-serif"}),
    ], style={"padding": "6px", "backgroundColor": "white",
              "border": "1px solid #ddd", "borderRadius": "6px",
              "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"})

    return True, bbox, children, display_s


# Hover on scatter plot → find closest peak of selected clip, highlight dist, seek video
@callback(
    Output("dist-plot", "figure", allow_duplicate=True),
    Output("scatter-tooltip", "show"),
    Output("scatter-tooltip", "bbox"),
    Output("scatter-tooltip", "children"),
    Output("player", "seekTo", allow_duplicate=True),
    Input("scatter-plot", "hoverData"),
    Input("clip-dropdown", "value"),
    Input("cond1-select", "value"),
    Input("cond2-select", "value"),
    Input("scale-type", "value"),
    prevent_initial_call=True,
)
def scatter_hover(hoverData, clip_name, cond1, cond2, scale_type):
    patched_fig = Patch()

    if hoverData is None or not clip_name:
        patched_fig["layout"]["shapes"][-1]["visible"] = False
        return patched_fig, False, no_update, no_update, no_update

    pt    = hoverData["points"][0]
    bbox  = pt["bbox"]
    hov_x = pt["x"]
    hov_y = pt["y"]

    cd         = clip_data[clip_name]
    c1_n       = normalize(cd["dist"][cond1], scale_type)
    c2_n       = normalize(cd["dist"][cond2], scale_type)
    pdf_len    = cd["pdf_len"]
    num_frames = cd["num_frames"]

    all_peak_idx = np.unique(np.concatenate([cd["peaks"][cond1], cd["peaks"][cond2]]))
    if len(all_peak_idx) == 0:
        patched_fig["layout"]["shapes"][-1]["visible"] = False
        return patched_fig, False, no_update, no_update, no_update

    sum_at_peaks  = c1_n[all_peak_idx] + c2_n[all_peak_idx]
    diff_at_peaks = c1_n[all_peak_idx] - c2_n[all_peak_idx]
    dists_sq      = (sum_at_peaks - hov_x) ** 2 + (diff_at_peaks - hov_y) ** 2
    closest       = all_peak_idx[np.argmin(dists_sq)]
    display_s     = int(closest) / 1000.0 + TIME_OFFSET

    max_y = max(c1_n.max(), c2_n.max())
    min_y = min(c1_n.min(), c2_n.min())
    x_lo  = max(display_s - 1.0, TIME_OFFSET)
    x_hi  = min(display_s + 1.0, cd["time_ms"].max() / 1000.0 + TIME_OFFSET)

    patched_fig["layout"]["shapes"][-1] = dict(
        type="rect",
        x0=x_lo, x1=x_hi,
        y0=min(0, min_y * 1.05), y1=max_y * 1.05,
        line=dict(color="black", width=0.5),
        fillcolor="rgba(0,0,0,0.35)",
        visible=True,
    )

    url = frame_url(clip_name, int(closest), pdf_len, num_frames)
    children = html.Div([
        html.Img(src=url, style={"width": "220px", "display": "block",
                                 "borderRadius": "4px", "marginBottom": "4px"}),
        html.Span(fmt_ts(display_s),
                  style={"fontSize": "13px", "color": "#333",
                         "fontFamily": "Arial, sans-serif"}),
    ], style={"padding": "6px", "backgroundColor": "white",
              "border": "1px solid #ddd", "borderRadius": "6px",
              "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"})

    return patched_fig, True, bbox, children, display_s


# Hover on timeline frame → enlarged image tooltip
@callback(
    Output("timeline-tooltip", "show"),
    Output("timeline-tooltip", "bbox"),
    Output("timeline-tooltip", "children"),
    Input("frame-viz", "hoverData"),
)
def timeline_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt    = hoverData["points"][0]
    bbox  = pt.get("bbox", no_update)
    cdata = pt.get("customdata")
    if not cdata:
        return False, no_update, no_update

    url, ts = cdata[0], cdata[1]

    children = html.Div([
        html.Img(src=url, style={
            "width": "420px", "display": "block",
            "borderRadius": "4px", "marginBottom": "4px",
        }),
        html.Span(ts, style={"fontSize": "13px", "color": "#333",
                             "fontFamily": "Arial, sans-serif"}),
    ], style={"padding": "6px", "backgroundColor": "white",
              "border": "1px solid #ddd", "borderRadius": "6px",
              "boxShadow": "0 2px 8px rgba(0,0,0,0.2)"})

    return True, bbox, children


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting on http://127.0.0.1:8051")
    app.run(debug=True, port=8051)
