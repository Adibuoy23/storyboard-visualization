import { useRef, useImperativeHandle, forwardRef, useEffect } from 'react';
import { VIDEO_BASE } from '../constants.js';

const VideoPlayer = forwardRef(function VideoPlayer({ clipName }, ref) {
  const videoRef = useRef(null);

  useImperativeHandle(ref, () => ({
    seekTo(time) {
      if (videoRef.current) videoRef.current.currentTime = time;
    },
  }), []);

  // Update src when clip changes; keep paused and seek to start
  useEffect(() => {
    if (videoRef.current && clipName) {
      videoRef.current.load();
    }
  }, [clipName]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: '#999', letterSpacing: '0.8px', textTransform: 'uppercase', marginBottom: 8 }}>
        Video
      </div>
      <video
        ref={videoRef}
        controls
        style={{ width: '100%', borderRadius: 4 }}
        preload="metadata"
      >
        {clipName && <source src={`${VIDEO_BASE}${clipName}.mp4`} type="video/mp4" />}
      </video>
    </div>
  );
});

export default VideoPlayer;
