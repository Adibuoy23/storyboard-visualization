import { normalizeArray, normalizeValue, normMax, frameUrl, fmtTs } from './dataUtils.js';

// All colours and labels come from appData.meta — nothing is hardcoded here.

// ── Distribution plot ─────────────────────────────────────────────────────────

export function buildDistFig(appData, clipName, scaleType, cond1, cond2) {
  const { meta }  = appData;
  const cd        = appData.data[clipName];
  const timeS     = cd.time_s;
  const active    = [cond1, cond2];

  const dists = {};
  for (const c of active) {
    dists[c] = normalizeArray(cd.distributions[c], scaleType, cd.norm_params[c]);
  }
  const maxY = Math.max(...active.map(c => Math.max(...dists[c])));

  const traces = active.map(c => ({
    x: timeS, y: dists[c], name: meta.labels[c],
    type: 'scatter', mode: 'lines', fill: 'tozeroy',
    fillcolor: meta.fill_colors[c],
    line: { color: meta.colors[c], width: 2 },
    hovertemplate: `%{x:.2f}s<extra>${meta.labels[c]}</extra>`,
  }));

  const shapes = [];
  for (const c of active) {
    for (const idx of cd.peaks[c].indices) {
      shapes.push({
        type: 'line',
        x0: idx / 1000.0 + meta.time_offset,
        x1: idx / 1000.0 + meta.time_offset,
        y0: 0, y1: maxY,
        line: { color: meta.colors[c], width: 1.5, dash: 'dash' },
      });
    }
  }
  // Hover-highlight placeholder — always the last shape
  shapes.push({
    type: 'rect', x0: 0, x1: 0, y0: 0, y1: 0,
    line: { color: 'rgba(0,0,0,0)', width: 0 },
    fillcolor: 'rgba(0,0,0,0)', visible: false,
  });

  const layout = {
    title: { text: 'Response Distributions', font: { size: 13, color: '#555' }, x: 0.5, xanchor: 'center' },
    paper_bgcolor: 'white', plot_bgcolor: 'white', height: 390,
    xaxis: {
      title: { text: 'Time (s)', font: { size: 12, color: '#555' } },
      showgrid: false, linecolor: '#d0d0d0', linewidth: 1.5, showline: true,
      zeroline: false, tickfont: { size: 10, color: '#666' },
      ticks: 'outside', ticklen: 4, tickcolor: '#ccc',
    },
    yaxis: {
      title: { text: 'Density', font: { size: 12, color: '#555' } },
      showgrid: true, gridcolor: 'rgba(0,0,0,0.05)', gridwidth: 1,
      linecolor: '#d0d0d0', linewidth: 1.5, showline: true,
      zeroline: false, tickfont: { size: 10, color: '#666' },
      ticks: 'outside', ticklen: 4, tickcolor: '#ccc',
    },
    legend: {
      orientation: 'h', y: 1.01, x: 0.5, xanchor: 'center',
      bgcolor: 'rgba(255,255,255,0.9)', bordercolor: 'rgba(0,0,0,0.08)',
      borderwidth: 1, font: { size: 11, color: '#333' },
    },
    margin: { l: 52, r: 12, t: 44, b: 48 },
    hovermode: 'x',
    shapes,
  };

  return { traces, layout };
}

// ── Scatter plot ──────────────────────────────────────────────────────────────

export function buildScatterFig(appData, clipName, cond1, cond2, scaleType) {
  const { meta }   = appData;
  const clips      = appData.clips;
  const label      = scaleType === 'norm' ? 'distribution' : 'z-score';
  const traces     = [];

  for (const [i, cn] of clips.entries()) {
    const cd         = appData.data[cn];
    const isSelected = cn === clipName;

    for (const cond of [cond1, cond2]) {
      const peakData = cd.peaks[cond];
      if (!peakData || peakData.indices.length === 0) continue;

      const xs = peakData.dist_at_peak[cond1].map((v, j) =>
        normalizeValue(v, scaleType, cd.norm_params[cond1]) +
        normalizeValue(peakData.dist_at_peak[cond2][j], scaleType, cd.norm_params[cond2])
      );
      const ys = peakData.dist_at_peak[cond1].map((v, j) =>
        normalizeValue(v, scaleType, cd.norm_params[cond1]) -
        normalizeValue(peakData.dist_at_peak[cond2][j], scaleType, cd.norm_params[cond2])
      );

      traces.push({
        x: xs, y: ys, type: 'scatter', mode: 'markers',
        name: i === 0 ? meta.labels[cond] : undefined,
        showlegend: i === 0,
        marker: {
          size: 10, color: meta.colors[cond],
          opacity: isSelected ? 0.85 : 0.25,
          line: { width: isSelected ? 2 : 0, color: 'black' },
        },
        hovertemplate: `${meta.labels[cond]}<br>sum=%{x:.4f}<br>diff=%{y:.4f}<extra></extra>`,
      });
    }
  }

  const annotations = [
    { x: 0.5, y: 1.07, xref: 'paper', yref: 'paper', showarrow: false,
      text: `▲ more ${meta.labels[cond1]}`, font: { size: 12, color: meta.hex_colors[cond1] }, xanchor: 'center' },
    { x: 0.5, y: -0.13, xref: 'paper', yref: 'paper', showarrow: false,
      text: `▼ more ${meta.labels[cond2]}`, font: { size: 12, color: meta.hex_colors[cond2] }, xanchor: 'center' },
    { x: 1.03, y: 0.5, xref: 'paper', yref: 'paper', showarrow: false,
      text: '▶ both', font: { size: 12, color: '#888' }, xanchor: 'left', yanchor: 'middle' },
  ];

  const layout = {
    title: { text: 'Peak Space', font: { size: 13, color: '#555' }, x: 0.5, xanchor: 'center' },
    paper_bgcolor: 'white', plot_bgcolor: 'white', height: 310,
    xaxis: {
      title: { text: `(${meta.labels[cond1]} + ${meta.labels[cond2]}) ${label}`, font: { size: 12, color: '#444' } },
      showgrid: true, gridcolor: 'rgba(0,0,0,0.05)', zeroline: true,
      zerolinecolor: 'rgba(0,0,0,0.18)', zerolinewidth: 1,
      linecolor: '#d0d0d0', linewidth: 1.5, showline: true,
      tickfont: { size: 12, color: '#555' }, ticks: 'outside',
    },
    yaxis: {
      title: { text: `(${meta.labels[cond1]} − ${meta.labels[cond2]}) ${label}`, font: { size: 12, color: '#444' } },
      showgrid: true, gridcolor: 'rgba(0,0,0,0.05)', zeroline: true,
      zerolinecolor: 'rgba(0,0,0,0.18)', zerolinewidth: 1,
      linecolor: '#d0d0d0', linewidth: 1.5, showline: true,
      tickfont: { size: 12, color: '#555' }, ticks: 'outside',
    },
    legend: {
      orientation: 'h', y: 1.01, x: 0.5, xanchor: 'center',
      bgcolor: 'rgba(255,255,255,0.9)', bordercolor: 'rgba(0,0,0,0.08)',
      borderwidth: 1, font: { size: 12, color: '#333' },
    },
    margin: { l: 55, r: 52, t: 44, b: 55 },
    annotations,
  };

  return { traces, layout };
}

// ── Timeline figure ───────────────────────────────────────────────────────────

export function buildTimelineFig(appData, clipName, cond1, cond2) {
  const { meta } = appData;
  const cd       = appData.data[clipName];
  const TO       = meta.time_offset;
  const duration = cd.pdf_len / 1000.0 + TO;
  const xMin     = TO - 4;
  const xMax     = duration + 4;

  const IMG_W       = 16.0;
  const IMG_H       = 0.75;
  const Y_TOP_CTR   =  1.22;
  const Y_BOT_CTR   = -1.22;
  const Y_TOP_ARROW = Y_TOP_CTR - IMG_H / 2;
  const Y_BOT_ARROW = Y_BOT_CTR + IMG_H / 2;
  const Y_TS_TOP    =  0.40;
  const Y_TS_BOT    = -0.40;

  const shapes = [
    { type: 'line', x0: xMin, x1: xMax, y0: 0, y1: 0,
      line: { color: 'rgba(30,30,30,0.85)', width: 3 } },
    { type: 'rect', x0: xMin, x1: xMax,
      y0: Y_TOP_CTR - IMG_H/2 - 0.08, y1: Y_TOP_CTR + IMG_H/2 + 0.08,
      fillcolor: meta.fill_colors[cond1], line: { width: 0 }, layer: 'below' },
    { type: 'rect', x0: xMin, x1: xMax,
      y0: Y_BOT_CTR - IMG_H/2 - 0.08, y1: Y_BOT_CTR + IMG_H/2 + 0.08,
      fillcolor: meta.fill_colors[cond2], line: { width: 0 }, layer: 'below' },
  ];

  const images      = [];
  const annotations = [];
  const traces      = [{
    x: [xMin, xMax], y: [-3, 3], type: 'scatter', mode: 'markers',
    marker: { opacity: 0 }, showlegend: false, hoverinfo: 'skip',
  }];

  const cfg = [
    { cond: cond1, yCtr: Y_TOP_CTR,  yTip: Y_TOP_ARROW,  yBase:  0.10, yTs: Y_TS_TOP },
    { cond: cond2, yCtr: Y_BOT_CTR,  yTip: Y_BOT_ARROW,  yBase: -0.10, yTs: Y_TS_BOT },
  ];

  for (const { cond, yCtr, yTip, yBase, yTs } of cfg) {
    const color    = meta.hex_colors[cond];
    const peakData = cd.peaks[cond];

    annotations.push({
      text: `<b> ${meta.labels[cond]} </b>`,
      xref: 'paper', yref: 'y',
      x: -0.004, y: yCtr, xanchor: 'right', yanchor: 'middle',
      showarrow: false,
      font: { color: 'white', size: 12, family: 'Arial' },
      bgcolor: color, bordercolor: color, borderwidth: 1, borderpad: 4, opacity: 0.9,
    });

    if (!peakData || peakData.indices.length === 0) continue;

    const peakTimes  = [];
    const customdata = [];

    for (let i = 0; i < peakData.indices.length; i++) {
      const pidx = peakData.indices[i];
      const t    = peakData.time_s[i];
      const url  = frameUrl(clipName, pidx, cd.num_frames, TO);
      const ts   = fmtTs(pidx / 1000.0);

      peakTimes.push(t);
      customdata.push([url, ts, t]);   // t = display seconds, used for dist highlight

      images.push({
        source: url, layer: 'above',
        x: t, y: yCtr + IMG_H / 2, xref: 'x', yref: 'y',
        xanchor: 'center', yanchor: 'top',
        sizex: IMG_W, sizey: IMG_H,
      });

      shapes.push({
        type: 'rect',
        x0: t - IMG_W/2, x1: t + IMG_W/2,
        y0: yCtr - IMG_H/2, y1: yCtr + IMG_H/2,
        xref: 'x', yref: 'y', layer: 'above',
        line: { color, width: 1.5 }, fillcolor: 'rgba(0,0,0,0)',
      });

      annotations.push({
        x: t, y: yTip, ax: t, ay: yBase,
        xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
        arrowhead: 4, arrowsize: 1.8, arrowwidth: 1.5, arrowcolor: color,
        showarrow: true, text: '',
      });

      annotations.push({
        x: t, y: yTs, xref: 'x', yref: 'y',
        showarrow: false, xanchor: 'center', yanchor: 'middle',
        text: `<span style="font-size:9px;color:${color};font-family:Arial">${ts}</span>`,
      });
    }

    traces.push({
      x: peakTimes,
      y: Array(peakTimes.length).fill(yCtr),
      type: 'scatter', mode: 'markers',
      marker: { size: 30, color, opacity: 0.0 },
      name: meta.labels[cond], showlegend: true,
      customdata,
      hovertemplate: `<b>${meta.labels[cond]}</b>  t = %{x:.1f} s<extra></extra>`,
    });
  }

  const layout = {
    xaxis: {
      title: { text: 'Time (s)', font: { size: 13 } },
      range: [xMin, xMax],
      showgrid: true, gridcolor: 'rgba(0,0,0,0.06)', gridwidth: 1,
      linecolor: 'rgba(30,30,30,0.4)', linewidth: 1.5, showline: true,
      zeroline: false, tickfont: { size: 11 },
    },
    yaxis: { range: [-3, 3], showticklabels: false, showgrid: false, zeroline: false },
    height: 600,
    plot_bgcolor: 'white', paper_bgcolor: 'white',
    font: { size: 13, color: '#222', family: 'Arial' },
    legend: {
      orientation: 'h', y: 1.04, x: 0.5, xanchor: 'center',
      font: { size: 12 }, bgcolor: 'rgba(255,255,255,0.8)',
      bordercolor: 'rgba(0,0,0,0.1)', borderwidth: 1,
    },
    margin: { l: 140, r: 20, t: 20, b: 55 },
    hovermode: 'closest',
    shapes, images, annotations,
  };

  return { traces, layout };
}
