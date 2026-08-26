// JS port of the color_for_* / rgb_to_css / risk_to_rgba helpers in app_v28.py.
// color_for_hpc must hash identically to Python's hashlib.md5 so the same
// HPC id gets the same color in both UIs.
import md5 from "blueimp-md5";

function hsvToRgb(h, s, v) {
  // Same algorithm as Python's colorsys.hsv_to_rgb.
  if (s === 0) return [v, v, v];
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - s * f);
  const t = v * (1 - s * (1 - f));
  switch (i % 6) {
    case 0:
      return [v, t, p];
    case 1:
      return [q, v, p];
    case 2:
      return [p, v, t];
    case 3:
      return [p, q, v];
    case 4:
      return [t, p, v];
    default:
      return [v, p, q];
  }
}

// hashlib.md5(...).hexdigest() interpreted as a big integer, then % 360 —
// mirrored here via BigInt since the hex digest is 128 bits.
export function colorForHpc(hpcId) {
  if (hpcId === null || hpcId === undefined || Number.isNaN(hpcId)) {
    return [160, 160, 160];
  }
  const hex = md5(String(Math.trunc(hpcId)));
  const asInt = BigInt("0x" + hex);
  const hue = Number(asInt % 360n) / 360.0;
  const [r, g, b] = hsvToRgb(hue, 0.9, 1.0);
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export function colorForInflammation(label) {
  if (label === null || label === undefined || String(label).trim() === "") {
    return [160, 160, 160];
  }
  const mapping = {
    "none-sparse": [80, 200, 120],
    "mild-moderate": [255, 200, 0],
    marked: [255, 80, 80],
  };
  return mapping[String(label).trim().toLowerCase()] || [160, 160, 160];
}

export function colorForNecrosis(label) {
  if (label === null || label === undefined || String(label).trim() === "") {
    return [160, 160, 160];
  }
  const s = String(label).trim().toLowerCase();
  if (s === "none") return [60, 200, 120];
  if (s === "some") return [255, 165, 0];
  if (s === "universal") return [200, 40, 40];
  return [160, 160, 160];
}

export function rgbToCss([r, g, b]) {
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

export function rgbaToCss([r, g, b], alpha255) {
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${(alpha255 / 255).toFixed(3)})`;
}

// Positive score -> red (poor survival), negative -> blue (protective),
// near zero -> transparent. Returns [r, g, b, alpha(0-255)].
export function riskToRgba(scoreNorm, alphaMin = 20, alphaMax = 170) {
  let score = Number(scoreNorm);
  if (!Number.isFinite(score)) return [255, 255, 255, 0];
  score = Math.max(-1.0, Math.min(1.0, score));
  const intensity = Math.abs(score);
  if (intensity < 0.02) return [255, 255, 255, 0];
  const alpha = Math.round(alphaMin + (alphaMax - alphaMin) * intensity);
  if (score > 0) return [255, 0, 0, alpha];
  if (score < 0) return [0, 80, 255, alpha];
  return [255, 255, 255, 0];
}
