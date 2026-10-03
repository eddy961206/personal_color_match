import { normalizeHex, hexToRgb, rgbToHex, hexToLab, deltaE00, nearestColors, contrast, inkFor } from './color.js';
import { PALETTES, getPalette, ALL_COLORS } from './palettes.js';
import { readLink, linkHash, loadCollection, writeCollection, parseCollection, serializeCollection, mergeCollections, MAX_SAVED } from './state.js';
import { PhotoController } from './photo.js';
const $ = id => document.getElementById(id);
const element = (tag, className, text) => {
  const node = document.createElement(tag); if (className) node.className = className;
  if (text != null) node.textContent = text; return node;
};
const paint = (node, hex) => { node.style.backgroundColor = hex; node.style.color = inkFor(hex); };
let state = readLink(location.hash), storage = null;
try { storage = localStorage; } catch { /* Private/restricted browser: keep a session-only collection. */ }
const loaded = loadCollection(storage); let collection = loaded.items, sessionOnly = Boolean(loaded.error);
const tabs = [...document.querySelectorAll('[role=tab]')];
function notify(message, undo) {
  $('toast-text').textContent = message; $('toast').hidden = false;
  $('toast-action').hidden = !undo;
  $('toast-action').onclick = undo ? () => { try { undo(); } catch (e) { notify(e.message); } } : null;
}
$('toast-close').onclick = () => { $('toast').hidden = true; };
function setView(id, focus = false) {
  if (!tabs.some(t => t.dataset.view === id)) return;
  tabs.forEach(tab => {
    const active = tab.dataset.view === id;
    tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
    $(tab.dataset.view).hidden = !active; if (focus && active) tab.focus();
  });
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => setView(tab.dataset.view));
  tab.addEventListener('keydown', e => {
    const offsets = { ArrowLeft: -1, ArrowRight: 1 };
    let next;
    if (e.key in offsets) next = (index + offsets[e.key] + tabs.length) % tabs.length;
    if (e.key === 'Home') next = 0; if (e.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { e.preventDefault(); setView(tabs[next].dataset.view, true); }
  });
});
document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => setView(button.dataset.go, true)));
$('tone').append(new Option('아직 몰라 · 자유 탐색', 'free'));
for (const family of ['봄', '여름', '가을', '겨울']) {
  const group = element('optgroup'); group.label = family;
  PALETTES.filter(p => p.family === family).forEach(p => group.append(new Option(p.name, p.id)));
  $('tone').append(group);
}
function colorButton(color, subtitle) {
  const button = element('button', 'swatch-button');
  button.type = 'button'; button.title = `${color.name || color.hex} ${color.hex}`;
  const fill = element('span', 'swatch-fill', color.hex); paint(fill, color.hex);
  const info = element('span', 'swatch-info');
  info.append(element('strong', '', color.name || color.hex));
  if (subtitle) info.append(element('small', '', subtitle));
  button.append(fill, info); button.addEventListener('click', () => setColor(color.hex)); return button;
}
function setColor(value) {
  const hex = normalizeHex(value);
  if (!hex) { $('hex-error').textContent = 'HEX 3자리 또는 6자리를 입력해 줘. 예: #FF6B5C'; $('hex-input').setAttribute('aria-invalid', 'true'); return; }
  state.color = hex; $('hex-error').textContent = ''; $('hex-input').removeAttribute('aria-invalid'); renderColor();
}
function renderColor() {
  const focused = document.activeElement;
  const parent = focused?.parentElement;
  const focusList = ['nearest-colors', 'palette-grid', 'outfit'].includes(parent?.id) ? parent.id : null;
  const focusIndex = focusList ? [...parent.children].indexOf(focused) : -1;
  const palette = getPalette(state.tone), references = palette?.colors || ALL_COLORS;
  const nearest = nearestColors(state.color, references);
  const [l, a, b] = hexToLab(state.color), rgb = hexToRgb(state.color);
  paint($('hero-swatch'), state.color); $('hero-hex').textContent = state.color;
  $('hero-rgb').textContent = `RGB ${rgb.join(' · ')}`;
  $('hex-input').value = state.color; $('color-picker').value = state.color;
  ['red', 'green', 'blue'].forEach((id, i) => { $(id).value = rgb[i]; $(id + '-value').textContent = rgb[i]; });
  $('tone').value = state.tone; $('use-context').value = state.context;
  $('match-title').textContent = palette ? `${palette.name}와 비교해 보면` : '취향부터 찾아도 괜찮아';
  $('tone-description').textContent = palette?.description || '가까운 예시 색을 보여줄 뿐, 너의 톤을 판정하지 않아.';
  $('nearest-name').textContent = nearest[0].name; $('nearest-hex').textContent = nearest[0].hex;
  $('nearest-distance').textContent = nearest[0].distance.toFixed(1);
  $('nearest-colors').replaceChildren(...nearest.map(c => colorButton(c, `색 차이 ${c.distance.toFixed(1)}`)));
  $('lightness').textContent = `${l.toFixed(0)} / 100 L*`; $('chroma').textContent = `${Math.hypot(a, b).toFixed(0)} C*`;
  const browse = palette?.colors || PALETTES.flatMap(p => [p.colors[0], p.colors.at(-1)]);
  $('palette-count').textContent = `${browse.length}색 예시`;
  $('palette-grid').replaceChildren(...browse.map(c => {
    const button = element('button', 'palette-item'); button.title = `${c.name} ${c.hex}`;
    button.setAttribute('aria-pressed', String(c.hex === state.color));
    const chip = element('span', 'chip'); paint(chip, c.hex);
    button.append(chip, element('span', 'chip-label', c.name));
    button.addEventListener('click', () => setColor(c.hex)); return button;
  }));
  const bases = references.filter(c => c.role === 'base' && c.hex !== state.color);
  const base = ($('mood').value === 'bold' ? [...bases].sort((a, b) => contrast(state.color, b.hex) - contrast(state.color, a.hex)) : bases)[0] || references[0];
  const accents = references.filter(c => c.role === 'accent' && c.hex !== state.color && c.hex !== base.hex);
  const accent = $('mood').value === 'bold' ? [...accents].sort((a, b) => deltaE00(hexToLab(state.color), hexToLab(b.hex)) - deltaE00(hexToLab(state.color), hexToLab(a.hex)))[0] : nearestColors(state.color, accents, 1)[0];
  const colors = [{ hex: state.color, label: '좋아하는 색' }, { hex: base.hex, label: '바탕색' }, { hex: (accent || base).hex, label: '포인트' }];
  $('outfit').replaceChildren(...colors.map(c => {
    const button = element('button'); paint(button, c.hex); button.title = `${c.label} ${c.hex}`;
    button.append(element('span', '', c.label), element('small', '', c.hex));
    button.addEventListener('click', () => setColor(c.hex)); return button;
  }));
  $('outfit').style.gridTemplateColumns = state.context === 'accent' ? '.7fr 1.3fr 1fr' : '1.3fr 1fr .7fr';
  $('styling-tip').textContent = {
    top: '상의로 입고 싶다면, 바탕색을 이너나 겉옷으로 함께 놓아 봐. 실제 옷을 얼굴 가까이에 대고 네가 편안하게 느끼는 조합을 골라.',
    accent: '좋아하는 색을 가방이나 작은 소품으로 먼저 시도해 봐. 전체 옷장을 바꾸지 않고도 새로운 분위기를 즐길 수 있어.',
    bottom: '하의에는 좋아하는 색을 그대로 두고, 상의에는 손이 자주 가는 바탕색을 놓아 봐. 팔레트와 다르다는 이유로 옷을 포기할 필요는 없어.',
  }[state.context];
  paint($('compare-a'), state.color); paint($('compare-b'), state.compare);
  $('compare-a-code').textContent = state.color; $('compare-b-code').textContent = state.compare; $('compare-picker').value = state.compare;
  $('compare-details').textContent = `두 색의 차이 ΔE00 ${deltaE00(hexToLab(state.color), hexToLab(state.compare)).toFixed(1)} · 화면 글자 대비 ${contrast(state.color, state.compare).toFixed(2)}:1. 색 차이는 작을수록 비슷하고, 대비율은 클수록 밝기 구분이 뚜렷해.`;
  paint($('photo-selected'), state.color); $('photo-selected-code').textContent = state.color;
  $('save-color').textContent = collection.some(c => c.hex === state.color && c.tone === state.tone) ? '이미 저장한 색 ✓' : '이 색 저장하기';
  if (focusList) $(focusList).children[focusIndex]?.focus({ preventScroll: true });
}
$('hex-form').addEventListener('submit', e => { e.preventDefault(); setColor($('hex-input').value); });
$('color-picker').addEventListener('input', e => setColor(e.target.value));
['red', 'green', 'blue'].forEach(id => $(id).addEventListener('input', () => setColor(rgbToHex(['red', 'green', 'blue'].map(key => Number($(key).value))))));
$('tone').addEventListener('change', e => { state.tone = e.target.value; renderColor(); });
$('use-context').addEventListener('change', e => { state.context = e.target.value; renderColor(); });
$('mood').addEventListener('change', renderColor);
$('compare-picker').addEventListener('input', e => { state.compare = normalizeHex(e.target.value) || state.compare; renderColor(); });
$('pin-color').addEventListener('click', () => { state.compare = state.color; renderColor(); notify('비교 색을 고정했어. 다른 색을 골라 나란히 비교해 봐.'); });
$('share-color').addEventListener('click', async () => {
  const url = new URL(location.href); url.hash = linkHash(state);
  try {
    if (!navigator.clipboard?.writeText) throw new Error('clipboard unavailable');
    await navigator.clipboard.writeText(url.href); notify('색 설정 링크를 복사했어. 사진과 컬렉션은 포함되지 않아.');
  } catch { $('share-url').value = url.href; $('share-dialog').showModal(); $('share-url').focus(); $('share-url').select(); }
});
$('share-close').onclick = () => $('share-dialog').close();
window.addEventListener('hashchange', () => { if (new URLSearchParams(location.hash.slice(1)).get('v') === '2') { state = readLink(location.hash); renderColor(); } });
function persist(next) {
  collection = next; sessionOnly = !writeCollection(storage, collection);
  renderCollection(); renderColor();
}
function savedMessage(text) { return sessionOnly ? `${text} 다만 기기 저장이 막혀 있어. 탭을 닫기 전에 JSON 백업을 해 줘.` : text; }
$('save-color').addEventListener('click', () => {
  if (collection.some(c => c.hex === state.color && c.tone === state.tone)) return notify('이미 컬렉션에 저장한 색이야.');
  if (collection.length >= MAX_SAVED) return notify('컬렉션은 최대 48색이야. 일부 색을 지우고 다시 저장해 줘.');
  persist([...collection, { hex: state.color, tone: state.tone, name: '' }]); notify(savedMessage('마음이 가는 색 하나를 저장했어.'));
});
function renderCollection() {
  $('saved-count').textContent = collection.length; $('collection-total').textContent = `${collection.length} / 48색`;
  $('collection-empty').hidden = collection.length > 0;
  $('storage-warning').textContent = sessionOnly ? '기기 저장소를 사용할 수 없거나 읽지 못했어. 지금 변경은 이 탭에서만 유지돼. JSON 백업을 사용해 줘.' : '';
  ['export-json', 'export-card', 'clear-collection'].forEach(id => { $(id).disabled = collection.length === 0; });
  $('collection-grid').replaceChildren(...collection.map((c, i) => {
    const card = element('article', 'saved-card'), swatch = element('button', 'saved-swatch', c.hex); paint(swatch, c.hex);
    swatch.setAttribute('aria-label', `${c.name || c.hex} 스튜디오에서 비교`);
    swatch.onclick = () => { state.tone = c.tone; setColor(c.hex); setView('studio', true); };
    const body = element('div', 'saved-body'), label = element('label', '', '내가 붙인 이름'); label.htmlFor = `saved-name-${i}`;
    const input = element('input'); input.id = label.htmlFor; input.value = c.name; input.maxLength = 40; input.placeholder = '예: 자주 입는 셔츠';
    input.addEventListener('change', () => {
      // Do not replace the focused card: change fires before a following button click.
      const next = { ...c, name: input.value.slice(0, 40) };
      collection = collection.map(item => item === c ? next : item); c = next;
      sessionOnly = !writeCollection(storage, collection);
      swatch.setAttribute('aria-label', `${c.name || c.hex} 스튜디오에서 비교`);
      remove.setAttribute('aria-label', `${c.name || c.hex} 삭제`);
      $('storage-warning').textContent = sessionOnly ? '지금 변경은 이 탭에서만 유지돼. JSON 백업을 해 줘.' : '';
      if (sessionOnly) notify(savedMessage('이름을 바꿨어.'));
    });
    const bottom = element('div', 'saved-bottom'), remove = element('button', '', '삭제');
    remove.setAttribute('aria-label', `${c.name || c.hex} 삭제`);
    remove.onclick = () => { persist(collection.filter(item => item !== c)); notify('색을 컬렉션에서 뺐어.', () => { persist(mergeCollections(collection, [c])); notify(savedMessage('색을 되돌렸어.')); }); };
    bottom.append(element('span', '', getPalette(c.tone)?.name || '자유 탐색'), remove); body.append(label, input, bottom); card.append(swatch, body); return card;
  }));
}
function download(blob, name) {
  const url = URL.createObjectURL(blob), a = element('a'); a.href = url; a.download = name;
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
}
$('export-json').onclick = () => download(new Blob([serializeCollection(collection)], { type: 'application/json' }), 'color-me-collection.json');
$('import-json').onclick = () => $('import-input').click();
$('import-input').addEventListener('change', async e => {
  const file = e.target.files[0]; e.target.value = ''; if (!file) return;
  try {
    if (file.size > 65536) throw new Error('백업 파일은 64KB 이하만 불러올 수 있어.');
    const imported = parseCollection(await file.text()); persist(mergeCollections(collection, imported));
    notify(savedMessage('백업을 기존 컬렉션에 합쳤어. 중복 색은 한 번만 저장해.'));
  } catch (error) { notify(error.message); }
});
$('clear-collection').onclick = () => $('confirm-dialog').showModal();
$('confirm-cancel').onclick = () => $('confirm-dialog').close();
$('confirm-delete').onclick = () => {
  const removed = collection; persist([]); $('confirm-dialog').close();
  notify('컬렉션을 비웠어.', () => { persist(mergeCollections(collection, removed)); notify(savedMessage('컬렉션을 되돌렸어.')); });
};
$('export-card').onclick = () => {
  const cols = Math.min(collection.length, 4), rows = Math.ceil(collection.length / cols);
  const canvas = document.createElement('canvas'); canvas.width = 1000; canvas.height = 175 + rows * 190;
  const ctx = canvas.getContext('2d'); ctx.fillStyle = '#F7F7F3'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#243B35'; ctx.font = 'bold 38px Georgia, serif'; ctx.fillText('color, me', 40, 65);
  ctx.font = '20px sans-serif'; ctx.fillText('나다운 색 컬렉션 · 개인 진단 결과가 아닌 저장한 색', 40, 108);
  const width = (canvas.width - 80) / cols;
  collection.forEach((c, i) => {
    const x = 40 + (i % cols) * width, y = 145 + Math.floor(i / cols) * 190;
    ctx.fillStyle = c.hex; ctx.fillRect(x, y, width - 15, 103);
    ctx.fillStyle = inkFor(c.hex); ctx.font = '20px monospace'; ctx.fillText(c.hex, x + 12, y + 82);
    ctx.fillStyle = '#243B35'; ctx.font = '18px sans-serif';
    let label = c.name || (getPalette(c.tone)?.name || '자유 탐색');
    while (ctx.measureText(label).width > width - 25 && label.length) label = label.slice(0, -1);
    ctx.fillText(label, x, y + 133); ctx.fillStyle = '#5B6A64'; ctx.font = '14px sans-serif'; ctx.fillText(getPalette(c.tone)?.name || '자유 탐색', x, y + 158);
  });
  canvas.toBlob(blob => { if (blob) download(blob, 'color-me-card.png'); else notify('이미지 저장에 실패했어. JSON 백업을 사용해 줘.'); }, 'image/png');
};
const photo = new PhotoController($('photo-canvas'), {
  pick(hex) { setColor(hex); notify(`${hex} 색을 골랐어. 컬러 스튜디오에서 비교하거나 저장해 봐.`); },
  error(message) { $('photo-error').textContent = message; },
  status(status) {
    $('photo-error').textContent = ''; $('canvas-wrap').hidden = status !== 'ready'; $('cancel-photo').hidden = status !== 'loading';
    $('photo-status').textContent = { empty: '', loading: '사진을 기기 안에서 읽고 있어…', ready: '사진에서 지점을 누르거나 영역을 드래그해 봐.' }[status];
  },
  colors(colors) {
    if (!colors.length) $('extracted-colors').replaceChildren(element('p', 'muted', '사진을 고르면 대표색이 나타나. 투명한 영역은 제외해.'));
    else $('extracted-colors').replaceChildren(...colors.map(c => colorButton(c, `약 ${(c.share * 100).toFixed(0)}%`)));
  },
});
$('cancel-photo').onclick = () => { photo.clear(); notify('사진 읽기를 취소했어.'); };
$('choose-photo').onclick = () => $('photo-input').click();
$('photo-input').addEventListener('change', e => { const file = e.target.files[0]; e.target.value = ''; if (file) photo.load(file); });
$('remove-photo').onclick = () => { photo.clear(); notify('사진과 추출 결과를 메모리에서 지웠어. 저장한 색은 그대로야.'); };
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, e => { e.preventDefault(); $('dropzone').classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, e => { e.preventDefault(); $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', e => {
  if (e.dataTransfer.files.length !== 1) return notify('사진은 한 번에 한 장씩 골라 줘.');
  photo.load(e.dataTransfer.files[0]);
});
window.addEventListener('pagehide', () => photo.clear());
window.addEventListener('storage', e => {
  if (e.key && e.key !== 'personal-color-match:v2:collection') return;
  const value = loadCollection(storage); if (!value.error) { collection = value.items; sessionOnly = false; renderCollection(); renderColor(); }
});
function networkStatus() { $('offline-status').textContent = navigator.onLine ? '로컬 색 분석 · API 비용 없음' : '오프라인에서도 색 비교와 사진 분석을 사용할 수 있어'; }
window.addEventListener('online', networkStatus); window.addEventListener('offline', networkStatus);
renderCollection(); renderColor(); networkStatus();
const localDev = ['localhost', '127.0.0.1'].includes(location.hostname);
if ('serviceWorker' in navigator && window.isSecureContext && (!localDev || new URLSearchParams(location.search).has('offline'))) {
  window.addEventListener('load', () => { navigator.serviceWorker.register('./sw.js').catch(() => { /* Online functionality remains available. */ }); });
}
