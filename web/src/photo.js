import { averagePixels, extractPalette, clamp } from './color.js';
export const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 24_000_000;
/** Inspect dimensions before decoding, including highly compressed oversized inputs. */
export function inspectImageHeader(bytes) {
  const d = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const text = (start, n) => String.fromCharCode(...bytes.slice(start, start + n));
  let width, height, type;
  if (bytes.length >= 24 && text(1, 3) === 'PNG' && d.getUint32(0) === 0x89504e47 && text(12, 4) === 'IHDR') {
    width = d.getUint32(16); height = d.getUint32(20); type = 'image/png';
  } else if (bytes.length >= 12 && text(0, 4) === 'RIFF' && text(8, 4) === 'WEBP') {
    type = 'image/webp';
    for (let i = 12; i + 8 <= bytes.length;) {
      const kind = text(i, 4), n = d.getUint32(i + 4, true), p = i + 8;
      if (p + n > bytes.length) break;
      if (kind === 'VP8X' && n >= 10) {
        width = 1 + bytes[p + 4] + (bytes[p + 5] << 8) + (bytes[p + 6] << 16);
        height = 1 + bytes[p + 7] + (bytes[p + 8] << 8) + (bytes[p + 9] << 16); break;
      }
      if (kind === 'VP8 ' && n >= 10 && bytes[p + 3] === 0x9d && bytes[p + 4] === 1 && bytes[p + 5] === 0x2a) {
        width = d.getUint16(p + 6, true) & 0x3fff; height = d.getUint16(p + 8, true) & 0x3fff; break;
      }
      if (kind === 'VP8L' && n >= 5 && bytes[p] === 0x2f) {
        const bits = d.getUint32(p + 1, true);
        width = (bits & 0x3fff) + 1; height = ((bits >>> 14) & 0x3fff) + 1; break;
      }
      i = p + n + (n % 2);
    }
  } else if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    type = 'image/jpeg';
    for (let i = 2; i + 4 <= bytes.length;) {
      if (bytes[i++] !== 0xff) break;
      while (i < bytes.length && bytes[i] === 0xff) i++;
      const marker = bytes[i++];
      if (marker === 0xda || marker === 0xd9) break;
      if (marker === 1 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (i + 2 > bytes.length) break;
      const n = d.getUint16(i);
      if (n < 2 || i + n > bytes.length) break;
      if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker) && n >= 7) {
        height = d.getUint16(i + 3); width = d.getUint16(i + 5); break;
      }
      i += n;
    }
  }
  if (!width || !height) throw new Error('JPG·PNG·WebP 사진을 골라 줘. HEIC는 JPG로 변환해 줘.');
  if (width * height > MAX_IMAGE_PIXELS || Math.max(width, height) > 12000) {
    throw new Error('사진이 너무 커. 2,400만 화소 이하로 줄여서 다시 골라 줘.');
  }
  return { width, height, type };
}
export function canvasPoint(x, y, rect, width, height) {
  return { x: clamp(Math.floor((x - rect.left) / rect.width * width), 0, width - 1),
    y: clamp(Math.floor((y - rect.top) / rect.height * height), 0, height - 1) };
}
async function decode(file) {
  if (typeof createImageBitmap === 'function') return createImageBitmap(file, { imageOrientation: 'from-image' });
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url; await image.decode(); return image;
  } finally { URL.revokeObjectURL(url); }
}
export class PhotoController {
  constructor(canvas, callbacks) {
    this.canvas = canvas; this.callbacks = callbacks; this.generation = 0; this.source = null;
    this.worker = null; this.timer = null; this.start = null; this.point = null;
    canvas.addEventListener('pointerdown', e => {
      if (!this.source || e.button !== 0) return;
      this.start = this.location(e); this.point = this.start;
      canvas.setPointerCapture(e.pointerId); this.paint();
    });
    canvas.addEventListener('pointermove', e => {
      if (this.start && this.source) { this.point = this.location(e); this.paint(); }
    });
    canvas.addEventListener('pointerup', e => {
      if (!this.start || !this.source) return;
      this.point = this.location(e); this.pick(this.start, this.point); this.start = null;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      this.paint();
    });
    canvas.addEventListener('pointercancel', () => { this.start = null; this.paint(); });
    canvas.addEventListener('keydown', e => {
      if (!this.source) return;
      const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      const p = this.point || { x: Math.floor(canvas.width / 2), y: Math.floor(canvas.height / 2) };
      if (directions[e.key]) {
        e.preventDefault(); const step = e.shiftKey ? 10 : 1;
        const [dx, dy] = directions[e.key];
        this.point = { x: clamp(p.x + dx * step, 0, canvas.width - 1), y: clamp(p.y + dy * step, 0, canvas.height - 1) };
        this.paint();
      } else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.pick(p, p); }
    });
  }
  location(e) { return canvasPoint(e.clientX, e.clientY, this.canvas.getBoundingClientRect(), this.canvas.width, this.canvas.height); }
  stopWorker() { this.worker?.terminate(); this.worker = null; clearTimeout(this.timer); }
  clear() {
    this.generation++; this.stopWorker(); this.source = null; this.start = this.point = null;
    this.canvas.width = this.canvas.height = 1;
    this.callbacks.status('empty'); this.callbacks.colors([]);
  }
  async load(file) {
    const id = ++this.generation; this.stopWorker(); this.start = this.point = null;
    this.callbacks.colors([]); this.callbacks.status('loading');
    let bitmap;
    try {
      if (!file || !file.size || file.size > MAX_IMAGE_BYTES) throw new Error('12MB 이하의 사진 한 장을 골라 줘.');
      inspectImageHeader(new Uint8Array(await file.arrayBuffer()));
      if (id !== this.generation) return;
      bitmap = await decode(file);
      if (id !== this.generation) return;
      const w = bitmap.width || bitmap.naturalWidth, h = bitmap.height || bitmap.naturalHeight;
      if (!w || !h || w * h > MAX_IMAGE_PIXELS) throw new Error('사진 크기를 줄여서 다시 시도해 줘.');
      const ratio = Math.min(1, 960 / Math.max(w, h));
      const source = document.createElement('canvas');
      source.width = Math.max(1, Math.round(w * ratio)); source.height = Math.max(1, Math.round(h * ratio));
      source.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' }).drawImage(bitmap, 0, 0, source.width, source.height);
      this.source = source; this.canvas.width = source.width; this.canvas.height = source.height;
      this.paint(); this.callbacks.status('ready');
      const pixels = source.getContext('2d').getImageData(0, 0, source.width, source.height).data;
      const fallback = () => {
        if (id !== this.generation) return; this.stopWorker();
        const original = this.source.getContext('2d').getImageData(0, 0, source.width, source.height).data;
        this.callbacks.colors(extractPalette(original));
      };
      try {
        this.worker = new Worker(new URL('./palette-worker.js', import.meta.url), { type: 'module' });
        this.worker.onmessage = ({ data }) => {
          if (id !== this.generation || data.id !== id) return;
          this.stopWorker();
          if (data.error) fallback(); else this.callbacks.colors(data.colors);
        };
        this.worker.onerror = fallback;
        this.timer = setTimeout(fallback, 6000);
        this.worker.postMessage({ id, pixels: pixels.buffer }, [pixels.buffer]);
      } catch { fallback(); }
    } catch (error) {
      if (id === this.generation) {
        this.callbacks.status(this.source ? 'ready' : 'empty'); this.callbacks.error(error.message || '사진을 읽지 못했어.');
      }
    } finally { bitmap?.close?.(); }
  }
  pick(a, b) {
    const source = this.source; if (!source) return;
    const click = Math.abs(a.x - b.x) < 4 && Math.abs(a.y - b.y) < 4;
    const x = click ? clamp(b.x - 2, 0, source.width - 1) : Math.min(a.x, b.x);
    const y = click ? clamp(b.y - 2, 0, source.height - 1) : Math.min(a.y, b.y);
    const w = Math.min(source.width - x, click ? 5 : Math.abs(a.x - b.x) + 1);
    const h = Math.min(source.height - y, click ? 5 : Math.abs(a.y - b.y) + 1);
    const hex = averagePixels(source.getContext('2d').getImageData(x, y, w, h).data);
    if (hex) this.callbacks.pick(hex); else this.callbacks.error('투명한 영역이야. 색이 있는 부분을 골라 줘.');
  }
  paint() {
    if (!this.source) return;
    const ctx = this.canvas.getContext('2d'); ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.source, 0, 0);
    if (!this.point) return;
    const p = this.point, s = this.start;
    for (const [color, width] of [['#000000', 4], ['#FFFFFF', 2]]) {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      if (s) ctx.rect(s.x, s.y, p.x - s.x, p.y - s.y);
      else { ctx.moveTo(p.x - 10, p.y); ctx.lineTo(p.x + 10, p.y); ctx.moveTo(p.x, p.y - 10); ctx.lineTo(p.x, p.y + 10); }
      ctx.stroke();
    }
  }
}
