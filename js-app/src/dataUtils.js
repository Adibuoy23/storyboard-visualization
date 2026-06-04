import { IMAGE_BASE, CLIP_TOTAL_S } from './constants.js';

export function normalizeArray(arr, scaleType, params) {
  if (scaleType === 'zscore') {
    const { mean, std } = params;
    if (std === 0) return arr.map(() => 0);
    return arr.map(v => (v - mean) / std);
  }
  const { max } = params;
  if (max === 0) return arr.map(() => 0);
  return arr.map(v => v / max);
}

export function normalizeValue(v, scaleType, params) {
  if (scaleType === 'zscore') {
    const { mean, std } = params;
    return std > 0 ? (v - mean) / std : 0;
  }
  return params.max > 0 ? v / params.max : 0;
}

// Max of the normalised distribution — used for axis/shape bounds
export function normMax(scaleType, params) {
  if (scaleType === 'norm') return 1.0;
  return params.std > 0 ? (params.max - params.mean) / params.std : 0;
}

export function frameUrl(clipName, peakIndex, numFrames, timeOffset) {
  const rawS = peakIndex / 1000.0 + timeOffset;
  let f = Math.floor(rawS / CLIP_TOTAL_S * numFrames);
  f = Math.max(0, Math.min(f, numFrames - 1));
  return `${IMAGE_BASE}${clipName}/frames${String(f).padStart(4, '0')}.jpg`;
}

export function fmtTs(tS) {
  const m = Math.floor(tS / 60);
  const s = tS % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
}
