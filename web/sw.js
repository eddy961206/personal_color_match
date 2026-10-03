/* Cache only the application shell. Never cache user photos or arbitrary requests. */
const PREFIX = `color-me:${self.registration.scope}:`;
const CACHE = PREFIX + '__BUILD_ID__';
const SHELL = ['./', './index.html', './style.css', './icon.svg', './app.webmanifest',
  './src/app.js', './src/color.js', './src/palettes.js', './src/state.js', './src/photo.js', './src/palette-worker.js'];
const ALLOWED = new Set(SHELL.map(path => new URL(path, self.registration.scope).href));
self.addEventListener('install', event => { event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL))); });
self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith(PREFIX) && key !== CACHE).map(key => caches.delete(key))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  // Navigation query strings do not create separate caches or bypass the shell offline.
  if (event.request.mode === 'navigate') url.search = '';
  if (event.request.method !== 'GET' || !ALLOWED.has(url.href)) return;
  event.respondWith(caches.open(CACHE).then(async cache => {
    const cached = await cache.match(url.href);
    if (cached) return cached;
    try { return await fetch(event.request); } catch { return Response.error(); }
  }));
});
