import test from 'node:test';
import assert from 'node:assert/strict';
import { PhotoController } from '../web/src/photo.js';
function fakeCanvas() {
  const events = {}, samples = [];
  const ctx = { drawImage() {}, clearRect() {}, beginPath() {}, rect() {}, moveTo() {}, lineTo() {}, stroke() {},
    getImageData(x, y, w, h) { samples.push({ x, y, w, h }); return { data: new Uint8ClampedArray([255, 0, 0, 255]) }; } };
  return { width: 100, height: 60, events, samples, addEventListener(name, fn) { events[name] = fn; },
    getContext: () => ctx, getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 120 }),
    setPointerCapture() {}, hasPointerCapture: () => false, releasePointerCapture() {} };
}
function fixture() {
  const canvas = fakeCanvas(), picks = [], errors = [], statuses = [], colors = [];
  const controller = new PhotoController(canvas, { pick: x => picks.push(x), error: x => errors.push(x), status: x => statuses.push(x), colors: x => colors.push(x) });
  return { canvas, controller, picks, errors, statuses, colors };
}
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
function file() {
  const bytes = new Uint8Array(24), d = new DataView(bytes.buffer); d.setUint32(0, 0x89504e47);
  bytes.set([73, 72, 68, 82], 12); d.setUint32(16, 100); d.setUint32(20, 60);
  return { size: 24, arrayBuffer: async () => bytes.buffer };
}
test('photo click samples actual top-left neighborhood, not image center or overlay', () => {
  const f = fixture(); f.controller.source = fakeCanvas();
  f.canvas.events.pointerdown({ button: 0, pointerId: 1, clientX: 30, clientY: 40 });
  f.canvas.events.pointerup({ pointerId: 1, clientX: 30, clientY: 40 });
  assert.deepEqual(f.controller.source.samples, [{ x: 8, y: 8, w: 5, h: 5 }]);
  assert.deepEqual(f.picks, ['#FF0000']); assert.deepEqual(f.canvas.samples, []);
});
test('drag supports reversed directions and clips edge sampling', () => {
  const f = fixture(); f.controller.source = fakeCanvas();
  f.controller.pick({ x: 90, y: 50 }, { x: 10, y: 5 });
  f.controller.pick({ x: 99, y: 59 }, { x: 99, y: 59 });
  assert.deepEqual(f.controller.source.samples, [{ x: 10, y: 5, w: 81, h: 46 }, { x: 97, y: 57, w: 3, h: 3 }]);
});
test('keyboard selection uses moved coordinates and prevents default scrolling', () => {
  const f = fixture(); f.controller.source = fakeCanvas(); let prevented = 0;
  f.canvas.events.keydown({ key: 'ArrowLeft', shiftKey: true, preventDefault: () => prevented++ });
  f.canvas.events.keydown({ key: 'Enter', preventDefault: () => prevented++ });
  assert.equal(prevented, 2); assert.equal(f.controller.source.samples[0].x, 38);
});
test('clear invalidates requests, terminates workers, drops decoded source', () => {
  const f = fixture(); let terminated = false;
  f.controller.source = fakeCanvas(); f.controller.worker = { terminate() { terminated = true; } };
  f.controller.clear(); assert.equal(terminated, true); assert.equal(f.controller.source, null);
  assert.equal(f.canvas.width, 1); assert.equal(f.controller.generation, 1); assert.deepEqual(f.statuses, ['empty']);
});
test('cancel during file read never invokes a decoder or restores a photo', async () => {
  const f = fixture(), pending = deferred();
  const loading = f.controller.load({ size: 24, arrayBuffer: () => pending.promise });
  f.controller.clear(); pending.resolve(await file().arrayBuffer()); await loading;
  assert.equal(f.controller.source, null); assert.deepEqual(f.statuses, ['loading', 'empty']);
});
test('out-of-order decode closes stale bitmaps and preserves the latest photo', async t => {
  const f = fixture(), old = deferred(), decodes = [];
  const oldBitmap = { width: 50, height: 50, close() { this.closed = true; } };
  const currentBitmap = { width: 100, height: 60, close() { this.closed = true; } };
  t.mock.method(globalThis, 'setTimeout', () => 0);
  const previous = { bitmap: globalThis.createImageBitmap, document: globalThis.document, Worker: globalThis.Worker };
  globalThis.createImageBitmap = () => { decodes.push(1); return decodes.length === 1 ? old.promise : Promise.resolve(currentBitmap); };
  globalThis.document = { createElement: () => fakeCanvas() };
  globalThis.Worker = class { postMessage() {} terminate() {} };
  try {
    const first = f.controller.load(file()); await new Promise(r => setImmediate(r));
    await f.controller.load(file()); const currentSource = f.controller.source;
    old.resolve(oldBitmap); await first;
    assert.equal(f.controller.source, currentSource); assert.equal(oldBitmap.closed, true); assert.equal(currentBitmap.closed, true);
    assert.deepEqual(f.statuses, ['loading', 'loading', 'ready']);
  } finally { Object.assign(globalThis, { createImageBitmap: previous.bitmap, document: previous.document, Worker: previous.Worker }); f.controller.clear(); }
});
test('stale worker error cannot terminate a newer worker', async t => {
  const f = fixture(), workers = [];
  t.mock.method(globalThis, 'setTimeout', () => 0);
  const previous = { bitmap: globalThis.createImageBitmap, document: globalThis.document, Worker: globalThis.Worker };
  globalThis.createImageBitmap = async () => ({ width: 100, height: 60, close() {} });
  globalThis.document = { createElement: () => fakeCanvas() };
  globalThis.Worker = class { constructor() { workers.push(this); } postMessage() {} terminate() { this.terminated = true; } };
  try {
    await f.controller.load(file()); const lateError = workers[0].onerror;
    await f.controller.load(file()); lateError();
    assert.equal(workers[1].terminated, undefined); assert.equal(f.controller.worker, workers[1]);
  } finally { Object.assign(globalThis, { createImageBitmap: previous.bitmap, document: previous.document, Worker: previous.Worker }); f.controller.clear(); }
});
test('oversized file rejection preserves existing source and reports an actionable error', async () => {
  const f = fixture(), old = fakeCanvas(); f.controller.source = old;
  await f.controller.load({ size: 13 * 1024 * 1024 });
  assert.equal(f.controller.source, old); assert.match(f.errors[0], /12MB/); assert.equal(f.statuses.at(-1), 'ready');
});
