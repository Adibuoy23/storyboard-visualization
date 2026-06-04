import { useRef, useEffect, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import { buildScatterFig } from '../figureBuilders.js';
import { normalizeValue, normMax, frameUrl, fmtTs } from '../dataUtils.js';

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

export default function ScatterPlot({
  appData, clipName, cond1, cond2, scaleType,
  setCond1, setCond2,
  distRef, onSeek,
}) {
  const divRef   = useRef(null);
  const propsRef = useRef({});
  propsRef.current = { appData, clipName, cond1, cond2, scaleType, distRef, onSeek };
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    const div = divRef.current;
    Plotly.newPlot(div, [], {}, PLOTLY_CONFIG);

    div.on('plotly_hover', (e) => {
      const { appData: ad, clipName: cn, cond1: c1, cond2: c2,
              scaleType: st, distRef: dr, onSeek: seek } = propsRef.current;
      if (!ad || !cn) return;
      const pt   = e.points[0];
      const hovX = pt.x, hovY = pt.y;
      const cd   = ad.data[cn];
      const TO   = ad.meta.time_offset;

      // Union of cond1 + cond2 peaks for the selected clip, deduplicated
      const seen     = new Set();
      const allPeaks = [];
      for (const cond of [c1, c2]) {
        const pd = cd.peaks[cond];
        for (let i = 0; i < pd.indices.length; i++) {
          if (seen.has(pd.indices[i])) continue;
          seen.add(pd.indices[i]);
          const n1 = normalizeValue(pd.dist_at_peak[c1][i], st, cd.norm_params[c1]);
          const n2 = normalizeValue(pd.dist_at_peak[c2][i], st, cd.norm_params[c2]);
          allPeaks.push({ sum: n1 + n2, diff: n1 - n2, idx: pd.indices[i], timeS: pd.time_s[i] });
        }
      }
      if (allPeaks.length === 0) return;

      let minDist = Infinity, closest = null;
      for (const p of allPeaks) {
        const d = (p.sum - hovX) ** 2 + (p.diff - hovY) ** 2;
        if (d < minDist) { minDist = d; closest = p; }
      }

      const url = frameUrl(cn, closest.idx, cd.num_frames, TO);

      if (dr?.current) {
        const maxY = Math.max(normMax(st, cd.norm_params[c1]), normMax(st, cd.norm_params[c2]));
        const x0   = Math.max(closest.timeS - 1.0, TO);
        const x1   = Math.min(closest.timeS + 1.0, cd.pdf_len / 1000.0 + TO);
        dr.current.updateHighlight(x0, x1, 0, maxY * 1.05);
      }

      if (seek) seek(closest.timeS);
      setTooltip({ url, ts: fmtTs(closest.idx / 1000.0), x: e.event.clientX, y: e.event.clientY });
    });

    div.on('plotly_unhover', () => {
      const { distRef: dr } = propsRef.current;
      if (dr?.current) dr.current.clearHighlight();
      setTooltip(null);
    });

    return () => { Plotly.purge(div); };
  }, []);

  useEffect(() => {
    const div = divRef.current;
    if (!div || !appData || !clipName) return;
    const { traces, layout } = buildScatterFig(appData, clipName, cond1, cond2, scaleType);
    Plotly.react(div, traces, layout, PLOTLY_CONFIG);
  }, [appData, clipName, cond1, cond2, scaleType]);

  if (!appData) return null;
  const { meta } = appData;
  const radioStyle = { display: 'block', marginBottom: 2, fontSize: 12, color: '#444', cursor: 'pointer' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Condition selectors — driven entirely by meta.conditions */}
      <div style={{
        display: 'flex', gap: 20, marginBottom: 8, paddingBottom: 8,
        borderBottom: '1px solid #f0f0f0',
      }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: meta.hex_colors[cond1],
                        letterSpacing: '0.4px', marginBottom: 5 }}>
            Condition A (x + y)
          </div>
          {meta.conditions.map(c => (
            <label key={c} style={radioStyle}>
              <input type="radio" name="cond1" value={c} checked={cond1 === c}
                onChange={() => setCond1(c)} style={{ marginRight: 5 }} />
              {meta.labels[c]}
            </label>
          ))}
        </div>
        <div style={{ width: 1, background: '#f0f0f0', flexShrink: 0 }} />
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: meta.hex_colors[cond2],
                        letterSpacing: '0.4px', marginBottom: 5 }}>
            Condition B (x − y)
          </div>
          {meta.conditions.map(c => (
            <label key={c} style={radioStyle}>
              <input type="radio" name="cond2" value={c} checked={cond2 === c}
                onChange={() => setCond2(c)} style={{ marginRight: 5 }} />
              {meta.labels[c]}
            </label>
          ))}
        </div>
      </div>

      <div style={{ position: 'relative' }}>
        <div ref={divRef} style={{ height: 280 }} />
        {tooltip && (
          <div style={{
            position: 'fixed', left: tooltip.x + 12, top: tooltip.y - 20,
            background: 'white', border: '1px solid #ddd', borderRadius: 6,
            padding: 6, boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
            zIndex: 9999, pointerEvents: 'none',
          }}>
            <img src={tooltip.url} style={{ width: 220, display: 'block', borderRadius: 4, marginBottom: 4 }} />
            <span style={{ fontSize: 13, color: '#333', fontFamily: 'Arial' }}>{tooltip.ts}</span>
          </div>
        )}
      </div>
    </div>
  );
}
