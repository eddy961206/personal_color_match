import { readdir, readFile, writeFile, cp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
const root = fileURLToPath(new URL('../', import.meta.url));
async function walk(directory) {
  const files = [];
  for (const item of await readdir(directory, { withFileTypes: true })) {
    if (item.isSymbolicLink()) throw new Error('Symlinks are not permitted in web/.');
    const path = join(directory, item.name);
    files.push(...(item.isDirectory() ? await walk(path) : [path]));
  }
  return files.sort();
}
const files = await walk(join(root, 'web')); const hash = createHash('sha256'); let raw = 0, gzip = 0;
for (const file of files) {
  const bytes = await readFile(file); raw += bytes.length; gzip += gzipSync(bytes).length;
  hash.update(file.slice(root.length)); hash.update(bytes);
  if (file.endsWith('.js')) execFileSync(process.execPath, ['--check', file], { stdio: 'inherit' });
}
if (raw > 300000 || gzip > 100000) throw new Error('Static application exceeded its 300KB raw / 100KB gzip budget.');
const revision = hash.digest('hex').slice(0, 12), out = join(root, 'dist');
await rm(out, { recursive: true, force: true }); await cp(join(root, 'web'), out, { recursive: true });
const sw = await readFile(join(out, 'sw.js'), 'utf8'); await writeFile(join(out, 'sw.js'), sw.replace('__BUILD_ID__', revision));
await writeFile(join(out, '.nojekyll'), '');
await writeFile(join(out, 'build-info.json'), JSON.stringify({ version: '2.0.0', revision, files: files.length, rawBytes: raw, gzipBytes: gzip }, null, 2));
console.log(JSON.stringify({ revision, files: files.length, rawBytes: raw, gzipBytes: gzip }));
