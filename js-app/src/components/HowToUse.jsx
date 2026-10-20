import { useState } from 'react';

const SECTIONS = [
  {
    title: 'What are KMs and EBs?',
    content: (
      <div>
        <p><strong>Key Moments (KM)</strong> — Participants are asked to select moments in the video that are important for conveying the underlying story. Key moments are related to the meaning of an event — they scaffold the semantic structure of narratives{' '}
        (<a href="https://osf.io/preprints/psyarxiv/dcfvw_v1" target="_blank" rel="noopener noreferrer" style={{ color: '#266a63' }}>Kim*, Upadhyayula* et al., <em>Key moments scaffold the semantic structure of narratives</em></a>).</p>
        <p><strong>Event Boundaries (EB)</strong> — Participants are asked to press a button whenever they feel one meaningful unit of activity has ended and a new one has begun. These correspond to perceived transitions between events in the narrative.</p>
        <p style={{ marginTop: 10 }}><strong>Realtime</strong> responses are collected while the participant is watching the video — they press a button at the moment they perceive a key moment or event boundary occurring.</p>
        <p><strong>Retrospective</strong> responses are collected after viewing — participants reflect on the video and select the moments they found most important.</p>
        <p>The distributions are estimated using a Gaussian kernel density estimator (KDE) with a bandwidth of 0.75 s (FWHM = 1.5 s), fit separately to each participant's responses and then averaged across participants.</p>
      </div>
    ),
  },
  {
    title: 'Video player',
    content: (
      <div>
        <p>Use the video player to watch the selected clip. Standard playback controls (play, pause, seek) work normally.</p>
        <p>Hovering anywhere on the <strong>distribution plot</strong> while the video is playing will jump the video to that exact timepoint — useful for scrubbing through the clip while watching the response curves in sync.</p>
      </div>
    ),
  },
  {
    title: 'Distribution plot',
    content: (
      <div>
        <p>Shows the continuous KM and EB response distributions over time for the selected clip. Dashed vertical lines mark detected peaks for each condition. Peaks are identified by selecting the top <em>k</em> local maxima from the distribution, where <em>k</em> is the median number of selections made by participants for that clip.</p>
        <p><strong>How to read the density:</strong> The y-axis reflects how many participants responded at each moment in the clip, smoothed into a continuous curve. A high density value at a given timepoint means many participants agreed — either selecting that moment as a Key Moment (important for the narrative) or pressing the boundary button at the end of a meaningful unit. Low density means few or no participants responded at that time.</p>
        <ul>
          <li>Hover anywhere on the curves to see the corresponding video frame in a tooltip and seek the video to that moment.</li>
          <li>Clicking and dragging within the plot creates a temporary box. Once you release the mouse, the chart automatically updates its axis ranges to "zoom in" on that specific area</li>
          <li>The two conditions displayed (A and B) can be changed using the radio selectors in the Scatter plot panel.</li>
          <li>Use <strong>Max → 1</strong> scaling to compare shapes across conditions; use <strong>Z-score</strong> to compare relative magnitudes.</li>
          <li><strong>Click a legend item</strong> to hide that trace; <strong>double-click</strong> to show it again.</li>
        </ul>
      </div>
    ),
  },
  {
    title: 'Scatter plot (Peak Space)',
    content: (
      <div>
        <p>Each point represents a detected response peak, plotted in a 2D space defined by the two selected conditions (A and B):</p>
        <ul>
          <li><strong>X-axis (sum A + B):</strong> points further right are strong peaks in <em>both</em> conditions simultaneously — timepoints where both KM and EB responses are high.</li>
          <li><strong>Y-axis (difference A − B):</strong> points above the midline are stronger in Condition A; points below are stronger in Condition B. Points near zero are balanced between the two.</li>
        </ul>
        <p><strong>Circles with black outlines</strong> are peaks from the currently selected clip. Faded circles are peaks from all other clips, giving context for how the selected clip compares across the dataset.</p>
        <p>Hovering over any point highlights the corresponding timepoint in the distribution plot and seeks the video to that moment.</p>
        <p>Use the <strong>Condition A / Condition B</strong> radio selectors to change which conditions define the axes.</p>
      </div>
    ),
  },
  {
    title: 'Timeline vs Tile view',
    content: (
      <div>
        <p><strong>Timeline view</strong> places peak frames along a horizontal time axis, with Condition A peaks above and Condition B peaks below the timeline. Arrows point to the exact timepoint of each peak. Hover over a frame to enlarge it and highlight the corresponding peak in the distribution plot.</p>
        <p><strong>Tile view</strong> displays the same peak frames as a simple grid, one row per condition — useful for quickly comparing all frames side by side without the temporal layout.</p>
      </div>
    ),
  },
  {
    title: 'Datasets',
    content: (
      <div>
        <p><strong>Storyboards fMRI</strong> — KM (retrospective) and EB (realtime) response distributions were collected separately within each coarse-grained event of the BBC series Sherlock, then applied to open-access fMRI data acquired while participants watched the same clips{' '}
        (<a href="https://doi.org/10.1038/nn.4450" target="_blank" rel="noopener noreferrer" style={{ color: '#266a63' }}>Chen et al., 2017, <em>Shared memories reveal shared structure in neural activity across individuals</em></a>).</p>
        <p style={{ marginLeft: 12, padding: '6px 10px', background: '#f8f8f8', borderRadius: 4, fontSize: 12, color: '#555', borderLeft: '3px solid #877243' }}>
          Upadhyayula et al. (2025). <em>Key moments in naturalistic events synchronize neural activity patterns and dominate memory reinstatement.</em>{' '}
          <a href="https://doi.org/10.1101/2025.08.30.673233" target="_blank" rel="noopener noreferrer"
             style={{ color: '#266a63' }}>doi: 10.1101/2025.08.30.673233</a>
        </p>
        <p style={{ marginTop: 12 }}><strong>2×2 KM &amp; EB</strong> — Responses were collected for clips of 3-minute duration (with 10-second overlap with the prior clip) to compare KM and EB responses on equal footing across four conditions: realtime/retrospective × KM/EB.</p>
        <p style={{ marginLeft: 12, padding: '6px 10px', background: '#f8f8f8', borderRadius: 4, fontSize: 12, color: '#555', borderLeft: '3px solid #266a63' }}>
          Upadhyayula et al. (in prep).
        </p>
        <p style={{ marginTop: 12 }}><strong>Mr. Bean</strong> — Responses were collected as part of the cued fMRI recall paradigm. They were collected within coarse grained events of a 14 minute excerpt clip from the movie Mr. Bean's holiday (with 10-second overlap with the prior clip) to compare KM and EB responses. </p>
        <p style={{ marginLeft: 12, padding: '6px 10px', background: '#f8f8f8', borderRadius: 4, fontSize: 12, color: '#555', borderLeft: '3px solid #576a26' }}>
          Upadhyayula et al. (in prep); Ding et al. (in prep).
        </p>
      </div>
    ),
  },
];

export default function HowToUse() {
  const [open,        setOpen]        = useState(false);
  const [openSection, setOpenSection] = useState(null);

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 5,
          background: open ? '#f0f4f8' : 'white',
          border: '1px solid #ddd', borderRadius: 6,
          padding: '5px 12px', fontSize: 12, fontWeight: 600,
          color: '#2c3e50', cursor: 'pointer', whiteSpace: 'nowrap',
          fontFamily: 'Arial, sans-serif',
        }}
      >
        How to use {open ? '▲' : '▼'}
      </button>

      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 6px)', right: 0,
          background: 'white', border: '1px solid #e0e0e0',
          borderRadius: 8, boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
          width: 460, maxHeight: '80vh', overflowY: 'auto',
          zIndex: 1000, fontFamily: 'Arial, sans-serif',
        }}>
          <div style={{
            padding: '10px 14px', borderBottom: '1px solid #f0f0f0',
            fontSize: 11, color: '#999', fontWeight: 600,
            letterSpacing: '0.6px', textTransform: 'uppercase',
            position: 'sticky', top: 0, background: 'white',
          }}>
            Tips &amp; reference
          </div>

          {SECTIONS.map((sec, i) => (
            <div key={i} style={{ borderBottom: i < SECTIONS.length - 1 ? '1px solid #f4f4f4' : 'none' }}>
              <button
                onClick={() => setOpenSection(openSection === i ? null : i)}
                style={{
                  width: '100%', textAlign: 'left',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '10px 14px', background: 'none', border: 'none',
                  fontSize: 13, fontWeight: 600, color: '#2c3e50',
                  cursor: 'pointer', fontFamily: 'Arial, sans-serif',
                }}
              >
                {sec.title}
                <span style={{ fontSize: 10, color: '#aaa', flexShrink: 0, marginLeft: 8 }}>
                  {openSection === i ? '▲' : '▼'}
                </span>
              </button>

              {openSection === i && (
                <div style={{
                  padding: '0 14px 14px 14px',
                  fontSize: 13, color: '#444', lineHeight: 1.65,
                }}>
                  {sec.content}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
