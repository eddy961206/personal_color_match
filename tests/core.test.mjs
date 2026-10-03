import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { normalizeHex, hexToRgb, rgbToHex, rgbToLab, hexToLab, deltaE00, contrast, inkFor, nearestColors, averagePixels, extractPalette } from '../web/src/color.js';
import { PALETTES, ALL_COLORS } from '../web/src/palettes.js';
import { readLink, linkHash, initialState, parseCollection, mergeCollections, loadCollection, writeCollection, serializeCollection } from '../web/src/state.js';
import { inspectImageHeader, canvasPoint } from '../web/src/photo.js';
const near = (a, b, e = .0001) => assert.ok(Math.abs(a - b) < e, `${a} != ${b}`);
const reference = readFileSync(new URL('./ciede2000.txt', import.meta.url), 'utf8').trim().split('\n');
// Numeric reference values: Sharma, Wu & Dalal (2005), supplementary test data. See docs/METHOD.md.
reference.forEach((row, i) => test(`CIEDE2000 published reference ${i + 1}`, () => {
  const n = row.split(/\s+/).map(Number), a = n.slice(0, 3), b = n.slice(3, 6);
  near(deltaE00(a, b), n[6]); near(deltaE00(b, a), n[6]);
}));
test('HEX normalization is strict and expands shorthand', () => {
  assert.equal(normalizeHex(' abc '), '#AABBCC'); assert.equal(normalizeHex('#123456'), '#123456');
  for (const bad of ['', '#12', 'zzzzzz', '#ffffffff', '#123456;', null, 42, '<img onerror=alert(1)>']) assert.equal(normalizeHex(bad), null);
});
test('RGB conversion clamps finite values and rejects invalid inputs', () => {
  assert.deepEqual(hexToRgb('#FF0080'), [255, 0, 128]); assert.equal(rgbToHex([256, -1, 127.6]), '#FF0080');
  assert.throws(() => rgbToHex([NaN, 1, 2])); assert.throws(() => hexToRgb('red'));
});
test('D65 white, black and red reference coordinates', () => {
  const white = rgbToLab([255, 255, 255]), black = rgbToLab([0, 0, 0]);
  near(white[0], 100, .001); near(white[1], 0, .001); near(white[2], 0, .001);
  assert.deepEqual(black, [0, 0, 0]); near(rgbToLab([255, 0, 0])[0], 53.2408, .001);
});
test('all 12 palettes have validated unique named base/accent colors', () => {
  assert.equal(PALETTES.length, 12); assert.equal(new Set(PALETTES.map(p => p.id)).size, 12);
  for (const p of PALETTES) {
    assert.ok(p.colors.length >= 10); assert.equal(new Set(p.colors.map(c => c.hex)).size, p.colors.length);
    for (const c of p.colors) { assert.equal(normalizeHex(c.hex), c.hex); assert.ok(c.name); }
    for (const c of p.colors) near(nearestColors(c.hex, p.colors, 1)[0].distance, 0);
  }
});
test('deltaE is finite, symmetric and zero at identity over a deterministic RGB grid', () => {
  for (let r = 0; r <= 255; r += 51) for (let g = 0; g <= 255; g += 51) for (let b = 0; b <= 255; b += 51) {
    const a = rgbToLab([r, g, b]), c = rgbToLab([b, r, g]);
    near(deltaE00(a, a), 0); near(deltaE00(a, c), deltaE00(c, a)); assert.ok(Number.isFinite(deltaE00(a, c)));
  }
  assert.throws(() => deltaE00([1, NaN, 2], [0, 0, 0]));
});
test('black/white contrast and swatch text reach at least 4.5:1', () => {
  near(contrast('#000', '#fff'), 21); near(contrast('#333', '#333'), 1);
  for (const c of ALL_COLORS) assert.ok(contrast(c.hex, inkFor(c.hex)) >= 4.5);
});
test('linear alpha-weighted average excludes transparent pixels', () => {
  assert.equal(averagePixels(new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 255, 0])), '#FF0000');
  assert.equal(averagePixels(new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255])), '#BCBCBC');
  assert.equal(averagePixels(new Uint8ClampedArray([255, 0, 0, 0])), null);
});
test('palette extraction retains black, white, grey and excludes transparency', () => {
  const pixels = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255, 128, 128, 128, 255, 255, 0, 0, 0]);
  const colors = extractPalette(pixels);
  assert.deepEqual(new Set(colors.map(c => c.hex)), new Set(['#000000', '#FFFFFF', '#808080']));
  near(colors.reduce((a, c) => a + c.share, 0), 1); assert.deepEqual(extractPalette(new Uint8ClampedArray()), []);
  assert.deepEqual(extractPalette(pixels), extractPalette(pixels));
});
test('solid-color extraction and nearest sort are deterministic', () => {
  assert.equal(extractPalette(new Uint8ClampedArray([255, 0, 0, 255]))[0].hex, '#FF0000');
  const n = nearestColors('#123456', ALL_COLORS); assert.ok(n[0].distance <= n[1].distance && n[1].distance <= n[2].distance);
});
test('URL fragment round trip does not contain photos or saved collection', () => {
  const state = { color: '#123456', compare: '#AABBCC', tone: 'free', context: 'accent' };
  assert.deepEqual(readLink(linkHash(state)), state); assert.deepEqual(readLink('#v=7&c=foo'), initialState());
  assert.equal(readLink('#v=2&t=__proto__&c=%3Cscript%3E').tone, 'spring-bright');
});
test('collection imports validate atomically and remove duplicates/unknown fields', () => {
  const items = parseCollection(JSON.stringify({ version: 2, items: [{ hex: '#abc', tone: 'free', name: 'shirt', photo: 'PRIVATE' }, { hex: '#AABBCC', tone: 'free' }] }));
  assert.deepEqual(items, [{ hex: '#AABBCC', tone: 'free', name: 'shirt' }]);
  assert.throws(() => parseCollection('{')); assert.throws(() => parseCollection('a'.repeat(65537)));
  assert.throws(() => parseCollection(JSON.stringify({ version: 2, items: [{ hex: 'badHEX', tone: 'free' }] })));
  assert.throws(() => parseCollection(JSON.stringify({ version: 2, items: [{ hex: '#abc', tone: 'injected' }] })));
});
test('storage denial and corrupt data do not throw or claim persistence', () => {
  const blocked = { getItem() { throw Error('denied'); }, setItem() { throw Error('denied'); } };
  assert.ok(loadCollection(blocked).error); assert.equal(writeCollection(blocked, []), false);
  assert.ok(loadCollection({ getItem: () => 'bad' }).error);
  const store = new Map(); const storage = { getItem: k => store.get(k), setItem: (k, v) => store.set(k, v) };
  const items = [{ hex: '#123456', tone: 'free', name: 'hat' }];
  assert.equal(writeCollection(storage, items), true); assert.deepEqual(loadCollection(storage).items, items);
});
test('merge preserves new additions on undo and enforces limits', () => {
  const a = { hex: '#123456', tone: 'free', name: '' }, b = { hex: '#654321', tone: 'free', name: '' };
  assert.equal(mergeCollections([a], [b, a]).length, 2);
  assert.throws(() => mergeCollections([], Array.from({ length: 49 }, (_, i) => ({ ...a, hex: rgbToHex([i, 0, 0]) }))));
  assert.deepEqual(parseCollection(serializeCollection([a])), [a]);
});
function pngHeader(w, h) {
  const bytes = new Uint8Array(24), view = new DataView(bytes.buffer); view.setUint32(0, 0x89504e47);
  bytes.set([73, 72, 68, 82], 12); view.setUint32(16, w); view.setUint32(20, h); return bytes;
}
test('image headers reject oversized/corrupt/unsupported data before decode', () => {
  assert.deepEqual(inspectImageHeader(pngHeader(640, 480)), { width: 640, height: 480, type: 'image/png' });
  assert.throws(() => inspectImageHeader(pngHeader(20000, 20000)));
  assert.throws(() => inspectImageHeader(new Uint8Array([1, 2, 3])));
  assert.throws(() => inspectImageHeader(pngHeader(0, 10)));
});
test('JPEG SOF and WebP extended headers report dimensions', () => {
  const jpg = new Uint8Array([255, 216, 255, 192, 0, 11, 8, 1, 224, 2, 128, 1, 1, 17, 0]);
  assert.deepEqual(inspectImageHeader(jpg), { width: 640, height: 480, type: 'image/jpeg' });
  const webp = new Uint8Array(30); webp.set([...Buffer.from('RIFF')], 0); webp.set([...Buffer.from('WEBPVP8X')], 8);
  new DataView(webp.buffer).setUint32(16, 10, true); webp[24] = 127; webp[25] = 2; webp[27] = 223; webp[28] = 1;
  assert.deepEqual(inspectImageHeader(webp), { width: 640, height: 480, type: 'image/webp' });
});
test('pointer coordinates scale to the actual canvas instead of the center', () => {
  const rect = { left: 10, top: 20, width: 200, height: 100 };
  assert.deepEqual(canvasPoint(60, 45, rect, 400, 200), { x: 100, y: 50 });
  assert.deepEqual(canvasPoint(-100, 1000, rect, 400, 200), { x: 0, y: 199 });
});
test('runtime has no API clients, unsafe DOM interpolation or external imports', () => {
  const files = readdirSync(new URL('../web/src/', import.meta.url)).filter(f => f.endsWith('.js'));
  for (const name of files) {
    const text = readFileSync(new URL('../web/src/' + name, import.meta.url), 'utf8');
    assert.doesNotMatch(text, /\bfetch\s*\(|XMLHttpRequest|WebSocket\s*\(|innerHTML|insertAdjacentHTML|\beval\s*\(/);
    assert.doesNotMatch(text, /from\s+['"]https?:/);
  }
});
