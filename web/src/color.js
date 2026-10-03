/** Pure sRGB/D65 color functions. No network or DOM dependencies. */
export const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
export function normalizeHex(value) {
  if (typeof value !== 'string') return null;
  let hex = value.trim().replace(/^#/, '');
  if (/^[\da-f]{3}$/i.test(hex)) hex = [...hex].map(c => c + c).join('');
  return /^[\da-f]{6}$/i.test(hex) ? `#${hex.toUpperCase()}` : null;
}
export function hexToRgb(hex) {
  const valid = normalizeHex(hex);
  if (!valid) throw new TypeError('올바른 HEX 색이 필요해. 예: #FF6B5C');
  return [1, 3, 5].map(i => parseInt(valid.slice(i, i + 2), 16));
}
export function rgbToHex(rgb) {
  if (!Array.isArray(rgb) || rgb.length !== 3 || !rgb.every(Number.isFinite)) {
    throw new TypeError('유한한 RGB 숫자 3개가 필요해.');
  }
  return '#' + rgb.map(v => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0')).join('').toUpperCase();
}
export const linear = v => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
export const encode = v => v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
export function rgbToLab(rgb) {
  const [r, g, b] = rgb.map(v => linear(v / 255));
  const xyz = [(r * .4124564 + g * .3575761 + b * .1804375) / .95047,
    r * .2126729 + g * .7151522 + b * .0721750,
    (r * .0193339 + g * .1191920 + b * .9503041) / 1.08883];
  const [x, y, z] = xyz.map(v => v > 216 / 24389 ? Math.cbrt(v) : (24389 / 27 * v + 16) / 116);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}
export const hexToLab = hex => rgbToLab(hexToRgb(hex));
const rad = degrees => degrees * Math.PI / 180;
const deg = radians => radians * 180 / Math.PI;
/** Original implementation of CIEDE2000 (kL=kC=kH=1). See docs/METHOD.md. */
export function deltaE00(first, second) {
  if (![first, second].every(v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))) {
    throw new TypeError('유한한 Lab 값 3개씩이 필요해.');
  }
  const [l1, a1, b1] = first, [l2, a2, b2] = second;
  const c1 = Math.hypot(a1, b1), c2 = Math.hypot(a2, b2), avgC = (c1 + c2) / 2;
  const g = .5 * (1 - Math.sqrt(avgC ** 7 / (avgC ** 7 + 25 ** 7)));
  const ap1 = (1 + g) * a1, ap2 = (1 + g) * a2;
  const cp1 = Math.hypot(ap1, b1), cp2 = Math.hypot(ap2, b2);
  const hp = (a, b) => a === 0 && b === 0 ? 0 : (deg(Math.atan2(b, a)) + 360) % 360;
  const h1 = hp(ap1, b1), h2 = hp(ap2, b2);
  const dl = l2 - l1, dc = cp2 - cp1;
  let dh = h2 - h1;
  if (cp1 * cp2 === 0) dh = 0;
  else if (dh > 180) dh -= 360;
  else if (dh < -180) dh += 360;
  const dH = 2 * Math.sqrt(cp1 * cp2) * Math.sin(rad(dh / 2));
  const lm = (l1 + l2) / 2, cm = (cp1 + cp2) / 2;
  let hm = h1 + h2;
  if (cp1 * cp2 !== 0) {
    if (Math.abs(h1 - h2) <= 180) hm /= 2;
    else hm = (hm + (hm < 360 ? 360 : -360)) / 2;
  }
  const t = 1 - .17 * Math.cos(rad(hm - 30)) + .24 * Math.cos(rad(2 * hm))
    + .32 * Math.cos(rad(3 * hm + 6)) - .20 * Math.cos(rad(4 * hm - 63));
  const sl = 1 + .015 * (lm - 50) ** 2 / Math.sqrt(20 + (lm - 50) ** 2);
  const sc = 1 + .045 * cm, sh = 1 + .015 * cm * t;
  const rt = -2 * Math.sqrt(cm ** 7 / (cm ** 7 + 25 ** 7))
    * Math.sin(rad(60 * Math.exp(-(((hm - 275) / 25) ** 2))));
  const L = dl / sl, C = dc / sc, H = dH / sh;
  return Math.sqrt(Math.max(0, L * L + C * C + H * H + rt * C * H));
}
export function luminance(hex) {
  const [r, g, b] = hexToRgb(hex).map(v => linear(v / 255));
  return .2126 * r + .7152 * g + .0722 * b;
}
export function contrast(a, b) {
  const [low, high] = [luminance(a), luminance(b)].sort((x, y) => x - y);
  return (high + .05) / (low + .05);
}
export const inkFor = hex => contrast(hex, '#000000') >= contrast(hex, '#FFFFFF') ? '#000000' : '#FFFFFF';
export function nearestColors(hex, colors, count = 3) {
  const lab = hexToLab(hex);
  return colors.map(color => ({ ...color, distance: deltaE00(lab, hexToLab(color.hex)) }))
    .sort((a, b) => a.distance - b.distance || a.hex.localeCompare(b.hex)).slice(0, count);
}
/** Linear-light, alpha-weighted average. Transparent pixels do not become black. */
export function averagePixels(data) {
  const sum = [0, 0, 0]; let weight = 0;
  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3] / 255;
    if (alpha < .125) continue;
    weight += alpha;
    for (let j = 0; j < 3; j++) sum[j] += linear(data[i + j] / 255) * alpha;
  }
  return weight ? rgbToHex(sum.map(v => encode(v / weight) * 255)) : null;
}
/** Deterministic, bounded palette extraction; all visible colors including neutrals count. */
export function extractPalette(data, count = 5) {
  const bins = new Map();
  const stride = Math.max(1, Math.ceil(data.length / 4 / 32000)) * 4;
  for (let i = 0; i < data.length; i += stride) {
    const weight = data[i + 3] / 255;
    if (weight < .125) continue;
    const key = ((data[i] >> 4) << 8) | ((data[i + 1] >> 4) << 4) | (data[i + 2] >> 4);
    if (!bins.has(key)) bins.set(key, { sum: [0, 0, 0], weight: 0 });
    const bin = bins.get(key); bin.weight += weight;
    for (let j = 0; j < 3; j++) bin.sum[j] += data[i + j] * weight;
  }
  const points = [...bins.values()].map(b => ({ rgb: b.sum.map(v => v / b.weight), weight: b.weight }));
  points.sort((a, b) => b.weight - a.weight);
  if (!points.length) return [];
  points.forEach(p => { p.lab = rgbToLab(p.rgb); });
  const distance = (a, b) => a.reduce((sum, v, j) => sum + (v - b[j]) ** 2, 0);
  const centers = [points[0].lab];
  while (centers.length < Math.min(count, points.length)) {
    let best = null, score = -1;
    for (const p of points) {
      const s = Math.min(...centers.map(c => distance(p.lab, c))) * Math.sqrt(p.weight);
      if (s > score) { score = s; best = p; }
    }
    if (score <= .00001) break;
    centers.push([...best.lab]);
  }
  let clusters;
  for (let n = 0; n < 10; n++) {
    clusters = centers.map(() => ({ lab: [0, 0, 0], rgb: [0, 0, 0], weight: 0 }));
    for (const p of points) {
      let best = 0;
      for (let j = 1; j < centers.length; j++) if (distance(p.lab, centers[j]) < distance(p.lab, centers[best])) best = j;
      const c = clusters[best]; c.weight += p.weight;
      for (let j = 0; j < 3; j++) { c.lab[j] += p.lab[j] * p.weight; c.rgb[j] += p.rgb[j] * p.weight; }
    }
    clusters.forEach((c, i) => { if (c.weight) centers[i] = c.lab.map(v => v / c.weight); });
  }
  const total = points.reduce((sum, p) => sum + p.weight, 0);
  return clusters.filter(c => c.weight).map(c => ({ hex: rgbToHex(c.rgb.map(v => v / c.weight)), share: c.weight / total }))
    .sort((a, b) => b.share - a.share);
}
