import http from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import { resolve, sep, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
const project = fileURLToPath(new URL('../', import.meta.url));
const root = await realpath(resolve(project, process.argv.includes('--dist') ? 'dist' : 'web'));
const base = process.env.BASE_PATH || '/';
if (!/^\/(?:[a-zA-Z0-9_-]+\/)*$/.test(base)) throw new Error('BASE_PATH must be an absolute directory URL ending in /.');
const port = Number(process.env.PORT || 4173);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.webmanifest': 'application/manifest+json', '.json': 'application/json' };
const csp = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'self'; form-action 'none'; frame-ancestors 'none'";
const server = http.createServer(async (req, res) => {
  res.setHeader('X-Content-Type-Options', 'nosniff'); res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Content-Security-Policy', csp); res.setHeader('Cache-Control', 'no-store');
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405, { Allow: 'GET, HEAD' }); return res.end(); }
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (!pathname.startsWith(base) || pathname.includes('\0') || pathname.includes('\\')) throw new Error('Invalid path');
    const relative = pathname.slice(base.length) || 'index.html';
    const target = await realpath(resolve(root, relative));
    if (!target.startsWith(root + sep) || !(await stat(target)).isFile()) throw new Error('Outside web root');
    const buffer = await readFile(target); res.writeHead(200, { 'Content-Type': types[extname(target)] || 'application/octet-stream', 'Content-Length': buffer.length });
    res.end(req.method === 'HEAD' ? undefined : buffer);
  } catch { res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }); res.end('찾을 수 없는 파일이야.'); }
});
server.listen(port, '127.0.0.1', () => console.log(`color, me: http://127.0.0.1:${port}${base}`));
server.on('error', error => { console.error(error.message); process.exitCode = 1; });
