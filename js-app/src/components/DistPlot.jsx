import { useRef, useEffect, useImperativeHandle, forwardRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import { buildDistFig } from '../figureBuilders.js';
import { frameUrl, fmtTs } from '../dataUtils.js';

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

const DistPlot = forwardRef(function DistPlot(
  { appData, clipName, scaleType, cond1, cond2, onSeek },
  ref
) {
  const divRef       = useRef(null);
  const baseShapesRef = useRef([]);   // shapes without highlight (rebuilt on every figure update)
  const [tooltip, setTooltip] = useState(null);

  // Refs for hover handler so it always sees current props without re-attaching
  const propsRef = useRef({});
  propsRef.current = { appData, clipName, onSeek };

  // Expose highlight/clearHighlight to parent (ScatterPlot calls these)
  useImperativeHandle(ref, () => ({
    updateHighlight(x0, x1, y0, y1) {
      const div = divRef.current;
      if (!div || !baseShapesRef.current.length) return;
      const shapes = baseShapesRef.current.map((s, i) =>
        i === baseShapesRef.current.length - 1
          ? { type: 'rect', x0, x1, y0, y1, line: { color: 'black', width: 0.5 }, fillcolor: 'rgba(0,0,0,0.35)' }
          : s
      );
      Plotly.relayout(div, { shapes });
    },
    clearHighlight() {
      const div = divRef.current;
      if (!div || !baseShapesRef.current.length) return;
      Plotly.relayout(div, { shapes: baseShapesRef.current });
    },
  }), []);

  // Initialize Plotly and attach hover listeners once
  useEffect(() => {
    const div = divRef.current;
    Plotly.newPlot(div, [], {}, PLOTLY_CONFIG);

    div.on('plotly_hover', (e) => {
      const { appData: ad, clipName: cn, onSeek: seek } = propsRef.current;
      if (!ad || !cn) return;
      const pt       = e.points[0];
      const displayS = pt.x;
      const TO       = ad.meta.time_offset;
      const timeS    = displayS - TO;
      const cd       = ad.data[cn];
      const peakIdx  = Math.max(0, Math.min(Math.round(timeS * 1000), cd.pdf_len - 1));
      const url      = frameUrl(cn, peakIdx, cd.num_frames, TO);
      if (seek) seek(displayS);
      setTooltip({ url, ts: fmtTs(timeS), x: e.event.clientX, y: e.event.clientY });
    });

    div.on('plotly_unhover', () => setTooltip(null));

    return () => { Plotly.purge(div); };
  }, []);

  // Re-render figure when props change
  useEffect(() => {
    const div = divRef.current;
    if (!div || !appData || !clipName) return;
    const { traces, layout } = buildDistFig(appData, clipName, scaleType, cond1, cond2);
    baseShapesRef.current = layout.shapes;
    Plotly.react(div, traces, layout, PLOTLY_CONFIG);
  }, [appData, clipName, scaleType, cond1, cond2]);

  return (
    <div style={{ position: 'relative' }}>
      <div ref={divRef} style={{ height: '390px' }} />
      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x + 12, top: tooltip.y - 20,
          background: 'white', border: '1px solid #ddd', borderRadius: '6px',
          padding: '6px', boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
          zIndex: 9999, pointerEvents: 'none',
        }}>
          <img src={tooltip.url} style={{ width: 220, display: 'block', borderRadius: 4, marginBottom: 4 }} />
          <span style={{ fontSize: 13, color: '#333', fontFamily: 'Arial' }}>{tooltip.ts}</span>
        </div>
      )}
    </div>
  );
});

export default DistPlot;
