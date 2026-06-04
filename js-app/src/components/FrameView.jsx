import { useRef, useEffect, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import { buildTimelineFig } from '../figureBuilders.js';
import { frameUrl, fmtTs, normMax } from '../dataUtils.js';

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

// ── Timeline view (Plotly) ────────────────────────────────────────────────────

function TimelineView({ appData, clipName, cond1, cond2, scaleType, distRef }) {
  const divRef   = useRef(null);
  const propsRef = useRef({});
  propsRef.current = { appData, clipName, cond1, cond2, scaleType, distRef };
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    const div = divRef.current;
    Plotly.newPlot(div, [], {}, PLOTLY_CONFIG);

    div.on('plotly_hover', (e) => {
      const pt    = e.points[0];
      const cdata = pt.customdata;
      if (!cdata) return;
      const [url, ts, timeS] = cdata;
      setTooltip({ url, ts, x: e.event.clientX, y: e.event.clientY });

      const { appData: ad, clipName: cn, cond1: c1, cond2: c2, scaleType: st, distRef: dr } = propsRef.current;
      if (dr?.current && ad && cn && timeS != null) {
        const cd   = ad.data[cn];
        const TO   = ad.meta.time_offset;
        const maxY = Math.max(normMax(st, cd.norm_params[c1]), normMax(st, cd.norm_params[c2]));
        const x0   = Math.max(timeS - 1.0, TO);
        const x1   = Math.min(timeS + 1.0, cd.pdf_len / 1000.0 + TO);
        dr.current.updateHighlight(x0, x1, 0, maxY * 1.05);
      }
    });

    div.on('plotly_unhover', () => {
      setTooltip(null);
      const { distRef: dr } = propsRef.current;
      if (dr?.current) dr.current.clearHighlight();
    });

    return () => { Plotly.purge(div); };
  }, []);

  useEffect(() => {
    const div = divRef.current;
    if (!div || !appData || !clipName) return;
    const { traces, layout } = buildTimelineFig(appData, clipName, cond1, cond2);
    Plotly.react(div, traces, layout, PLOTLY_CONFIG);
  }, [appData, clipName, cond1, cond2]);

  return (
    <div style={{ position: 'relative' }}>
      <div ref={divRef} style={{ width: '100%' }} />
      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x + 12, top: tooltip.y - 20,
          background: 'white', border: '1px solid #ddd', borderRadius: 6,
          padding: 6, boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          zIndex: 9999, pointerEvents: 'none',
        }}>
          <img src={tooltip.url} style={{ width: 420, display: 'block', borderRadius: 4, marginBottom: 4 }} />
          <span style={{ fontSize: 13, color: '#333', fontFamily: 'Arial' }}>{tooltip.ts}</span>
        </div>
      )}
    </div>
  );
}

// ── Tile view (plain HTML) ────────────────────────────────────────────────────

function TileView({ appData, clipName, cond1, cond2 }) {
  if (!appData || !clipName) return null;
  const { meta } = appData;
  const cd       = appData.data[clipName];
  const TO       = meta.time_offset;
  const active   = [cond1, cond2];

  return (
    <div style={{ overflowX: 'auto' }}>
      {active.map(cond => {
        const pd = cd.peaks[cond];
        if (!pd || pd.indices.length === 0) return null;
        return (
          <div key={cond} style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
            <div style={{
              writingMode: 'vertical-rl', transform: 'rotate(180deg)',
              fontWeight: 700, fontSize: 13, color: meta.hex_colors[cond],
              marginRight: 8, whiteSpace: 'nowrap',
            }}>
              {meta.labels[cond]}
            </div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {pd.indices.map((idx, i) => {
                const url = frameUrl(clipName, idx, cd.num_frames, TO);
                const ts  = fmtTs(idx / 1000.0);
                return (
                  <div key={i} style={{ textAlign: 'center' }}>
                    <img src={url} title={ts} style={{
                      width: 120, height: 68, objectFit: 'cover',
                      border: `2px solid ${meta.hex_colors[cond]}`, borderRadius: 3,
                    }} />
                    <div style={{ fontSize: 9, color: meta.hex_colors[cond] }}>{ts}</div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Public component ──────────────────────────────────────────────────────────

export default function FrameView({ appData, clipName, cond1, cond2, viewType, scaleType, distRef }) {
  if (viewType === 'Timeline') {
    return <TimelineView appData={appData} clipName={clipName} cond1={cond1} cond2={cond2} scaleType={scaleType} distRef={distRef} />;
  }
  return <TileView appData={appData} clipName={clipName} cond1={cond1} cond2={cond2} />;
}
