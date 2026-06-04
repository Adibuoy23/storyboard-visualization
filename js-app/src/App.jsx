import { useState, useEffect, useRef } from 'react';
import DistPlot from './components/DistPlot.jsx';
import ScatterPlot from './components/ScatterPlot.jsx';
import FrameView from './components/FrameView.jsx';
import VideoPlayer from './components/VideoPlayer.jsx';
import { CARD } from './constants.js';

const SEP = { width: 1, height: 28, background: '#e0e0e0', flexShrink: 0 };

export default function App() {
  const [datasets,        setDatasets]        = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [appData,         setAppData]         = useState(null);
  const [clipName,        setClipName]        = useState(null);
  const [scaleType,       setScaleType]       = useState('norm');
  const [cond1,           setCond1]           = useState(null);
  const [cond2,           setCond2]           = useState(null);
  const [viewType,        setViewType]        = useState('Timeline');

  const distRef  = useRef(null);
  const videoRef = useRef(null);

  // Load dataset index once
  useEffect(() => {
    fetch('./data/datasets.json')
      .then(r => r.json())
      .then(list => {
        setDatasets(list);
        setSelectedDataset(list[0].file);
      });
  }, []);

  // Reload data whenever the selected dataset changes
  useEffect(() => {
    if (!selectedDataset) return;
    setAppData(null);   // triggers loading state
    fetch(`./data/${selectedDataset}`)
      .then(r => r.json())
      .then(d => {
        setAppData(d);
        setClipName(d.clips[0]);
        setCond1(d.meta.default_cond1);
        setCond2(d.meta.default_cond2);
      });
  }, [selectedDataset]);

  const seekVideo = (time) => videoRef.current?.seekTo(time);

  const loading = !appData || !cond1 || !cond2;

  return (
    <div style={{ maxWidth: 1600, margin: '0 auto', fontFamily: 'Arial, sans-serif', backgroundColor: '#f4f5f7', minHeight: '100vh' }}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 18,
        padding: '10px 16px', background: 'white',
        borderBottom: '1px solid #e8e8e8',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        flexWrap: 'wrap',
      }}>
        <span style={{ color: '#2c3e50', fontSize: 20, fontWeight: 700, letterSpacing: '-0.3px', whiteSpace: 'nowrap' }}>
          KM and EB Visualization
        </span>

        <div style={SEP} />

        {/* Dataset selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: '#777', fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap' }}>Dataset</span>
          <select
            value={selectedDataset ?? ''}
            onChange={e => setSelectedDataset(e.target.value)}
            style={{ fontSize: 13, padding: '4px 8px', borderRadius: 4, border: '1px solid #ccc' }}
          >
            {datasets.map(d => <option key={d.file} value={d.file}>{d.label}</option>)}
          </select>
        </div>

        <div style={SEP} />

        {/* Clip selector */}
        <select
          value={clipName ?? ''}
          onChange={e => setClipName(e.target.value)}
          disabled={loading}
          style={{ width: 340, fontSize: 13, padding: '4px 8px', borderRadius: 4, border: '1px solid #ccc' }}
        >
          {!loading && appData.clips.map(c => <option key={c} value={c}>{c}</option>)}
        </select>

        <div style={SEP} />

        <RadioGroup
          label="Scaling"
          name="scale"
          options={[{ label: 'Max → 1', value: 'norm' }, { label: 'Z-score', value: 'zscore' }]}
          value={scaleType}
          onChange={setScaleType}
        />

        <div style={SEP} />

        <RadioGroup
          label="Frame view"
          name="view"
          options={[{ label: 'Timeline', value: 'Timeline' }, { label: 'Tile', value: 'Tile' }]}
          value={viewType}
          onChange={setViewType}
        />
      </div>

      {/* ── Loading overlay ──────────────────────────────────────────────── */}
      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 56px)', fontSize: 18, color: '#777' }}>
          Loading data…
        </div>
      )}

      {/* ── Main content ─────────────────────────────────────────────────── */}
      {!loading && (<>

        {/* Row 1: video | dist | scatter */}
        <div style={{ display: 'flex', alignItems: 'stretch', gap: 12, padding: '12px 12px 6px 12px' }}>

          <div style={{ ...CARD, flex: '0 0 200px', width: 200, minWidth: 0, overflow: 'hidden' }}>
            <VideoPlayer ref={videoRef} clipName={clipName} />
          </div>

          <div style={{ ...CARD, flex: '1 1 0', minWidth: 0, overflow: 'hidden' }}>
            <DistPlot
              ref={distRef}
              appData={appData}
              clipName={clipName}
              scaleType={scaleType}
              cond1={cond1}
              cond2={cond2}
              onSeek={seekVideo}
            />
          </div>

          <div style={{ ...CARD, flex: '0 0 28%', minWidth: 0 }}>
            <ScatterPlot
              appData={appData}
              clipName={clipName}
              cond1={cond1}
              cond2={cond2}
              scaleType={scaleType}
              setCond1={setCond1}
              setCond2={setCond2}
              distRef={distRef}
              onSeek={seekVideo}
            />
          </div>

        </div>

        {/* Row 2: frame timeline */}
        <div style={{ ...CARD, margin: '0 12px 12px 12px' }}>
          <FrameView
            appData={appData}
            clipName={clipName}
            cond1={cond1}
            cond2={cond2}
            viewType={viewType}
            scaleType={scaleType}
            distRef={distRef}
          />
        </div>

      </>)}

    </div>
  );
}

function RadioGroup({ label, name, options, value, onChange }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ color: '#777', fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap' }}>{label}</span>
      {options.map(o => (
        <label key={o.value} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12, color: '#444', cursor: 'pointer', marginRight: 8 }}>
          <input type="radio" name={name} value={o.value} checked={value === o.value} onChange={() => onChange(o.value)} />
          {o.label}
        </label>
      ))}
    </div>
  );
}
