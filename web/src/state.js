import { normalizeHex } from './color.js';
import { getPalette, DEFAULT_COLOR } from './palettes.js';
export const STORAGE_KEY = 'personal-color-match:v2:collection';
export const MAX_SAVED = 48;
export const validTone = value => value === 'free' || Boolean(getPalette(value));
export const initialState = () => ({ color: DEFAULT_COLOR, tone: 'spring-bright', compare: '#103A64', context: 'top' });
export function readLink(hash) {
  const state = initialState();
  const p = new URLSearchParams(String(hash).replace(/^#/, ''));
  if (p.get('v') !== '2') return state;
  state.color = normalizeHex(p.get('c')) || state.color;
  state.compare = normalizeHex(p.get('b')) || state.compare;
  if (validTone(p.get('t'))) state.tone = p.get('t');
  if (['top', 'accent', 'bottom'].includes(p.get('u'))) state.context = p.get('u');
  return state;
}
export function linkHash(state) {
  return '#' + new URLSearchParams({ v: '2', c: state.color, b: state.compare, t: state.tone, u: state.context });
}
/** Strict import boundary: only known fields can reach storage or the DOM. */
export function parseCollection(text) {
  if (typeof text !== 'string' || text.length > 65536) throw new Error('백업 파일은 64KB 이하만 불러올 수 있어.');
  let parsed;
  try { parsed = JSON.parse(text); } catch { throw new Error('올바른 JSON 백업 파일이 아니야.'); }
  if (parsed?.version !== 2 || !Array.isArray(parsed.items) || parsed.items.length > MAX_SAVED) {
    throw new Error('이 앱에서 내보낸 v2 백업인지 확인해 줘. 최대 48색을 지원해.');
  }
  const items = []; const keys = new Set();
  for (const value of parsed.items) {
    const hex = normalizeHex(value?.hex);
    if (!hex || !validTone(value?.tone) || (value.name != null && typeof value.name !== 'string')) {
      throw new Error('백업에 잘못된 색이나 톤이 들어 있어. 기존 컬렉션은 그대로야.');
    }
    const key = `${value.tone}:${hex}`;
    if (!keys.has(key)) { items.push({ hex, tone: value.tone, name: (value.name || '').slice(0, 40) }); keys.add(key); }
  }
  return items;
}
export const serializeCollection = items => JSON.stringify({ version: 2, items }, null, 2);
export function loadCollection(storage) {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    return { items: raw ? parseCollection(raw) : [], error: null };
  } catch { return { items: [], error: '기기 저장소를 읽지 못했어. 저장소 접근 설정이나 백업 파일을 확인해 줘.' }; }
}
export function writeCollection(storage, items) {
  try { storage.setItem(STORAGE_KEY, serializeCollection(items)); return true; } catch { return false; }
}
export function mergeCollections(current, imported) {
  const values = [...new Map([...current, ...imported].map(c => [`${c.tone}:${c.hex}`, c])).values()];
  if (values.length > MAX_SAVED) throw new Error('합치면 48색을 넘어. 일부 색을 지운 뒤 다시 불러와 줘.');
  return values;
}
