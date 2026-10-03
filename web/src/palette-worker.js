import { extractPalette } from './color.js';
self.onmessage = ({ data }) => {
  try { self.postMessage({ id: data.id, colors: extractPalette(new Uint8ClampedArray(data.pixels)) }); }
  catch { self.postMessage({ id: data.id, error: '대표색을 추출하지 못했어. 사진에서 직접 색을 골라 봐.' }); }
};
