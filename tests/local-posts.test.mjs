import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { createPostStore } from '../src/dev/post-store.mjs';
import { parsePostDocument, serializePostDocument } from '../src/lib/post-document.mjs';
import { GET, POST } from '../src/dev/posts.ts';

const sample = overrides => serializePostDocument({ title: '연구 기록', slug: 'test-note', date: '2026-11-07', category: 'Research', tags: ['3DGS'], draft: true, custom: 'preserve me', body: '본문\n\n$$x^2$$\n', ...overrides });
async function fixture(run) {
  const base = resolve(tmpdir());
  const root = await mkdtemp(join(base, 'oweixx-post-store-'));
  try { await run(createPostStore(root), root); }
  finally {
    assert(root.startsWith(base + sep) && root.includes('oweixx-post-store-'));
    await rm(root, { recursive: true, force: true });
  }
}
test('create and edit preserve Unicode, unknown metadata and the original filename', async () => fixture(async (store, root) => {
  const created = await store.save({ markdown: sample() });
  assert.equal(created.url, '/blog/2026/test-note/');
  const loaded = await store.read(created.file);
  assert.equal(loaded.post.custom, 'preserve me');
  assert.equal(loaded.post.title, '연구 기록');
  const saved = await store.save({ ...loaded, markdown: sample({ title: '수정한 제목', slug: 'new-url' }) });
  assert.equal(saved.file, created.file);
  assert.equal(saved.url, '/blog/2026/new-url/');
  assert.equal(parsePostDocument(await readFile(join(root, saved.file), 'utf8')).title, '수정한 제목');
  assert.equal((await store.list()).length, 1);
}));
test('stale editors and simultaneous saves cannot overwrite newer content', async () => fixture(async (store, root) => {
  const created = await store.save({ markdown: sample() });
  const attempts = await Promise.allSettled([
    store.save({ ...created, markdown: sample({ body: 'first writer' }) }),
    store.save({ ...created, markdown: sample({ body: 'second writer' }) }),
  ]);
  assert.equal(attempts.filter(r => r.status === 'fulfilled').length, 1);
  assert.equal(attempts.find(r => r.status === 'rejected').reason.status, 409);
  const current = await store.read(created.file);
  await writeFile(join(root, created.file), sample({ body: 'external edit' }));
  await assert.rejects(store.save({ ...current, markdown: sample() }), error => error.status === 409);
  assert.match(await readFile(join(root, created.file), 'utf8'), /external edit/);
}));
test('new files cannot replace existing files or claim an existing post URL', async () => fixture(async (store, root) => {
  await store.save({ markdown: sample() });
  await assert.rejects(store.save({ markdown: sample({ body: 'overwrite' }) }), error => error.status === 409);
  await assert.rejects(store.save({ markdown: sample({ date: '2026-11-08' }) }), error => error.status === 409);
  assert.equal((await readdir(root)).length, 1);
}));
test('invalid dates and paths do not write any files', async () => fixture(async (store, root) => {
  for (const file of ['../outside.md', 'C:/outside.md', 'nested/../../outside.md', 'bad.txt']) {
    await assert.rejects(store.save({ file, revision: 'x', markdown: sample() }));
  }
  await assert.rejects(store.save({ markdown: sample({ date: '2026-02-31' }) }));
  await assert.rejects(store.save({ markdown: sample({ draft: 'false' }) }));
  await assert.rejects(store.save({ markdown: sample({ updated: '2026-01-01' }) }));
  assert.deepEqual(await readdir(root), []);
}));
test('draft defaults and markdown round trips do not accumulate whitespace', () => {
  const source = sample({ draft: undefined });
  const post = parsePostDocument(source);
  assert.equal(post.draft, true);
  const once = serializePostDocument(post);
  assert.equal(serializePostDocument(parsePostDocument(once)), once);
});
test('post endpoints reject remote hosts and cross-origin or unauthenticated writes', async () => {
  const context = (url, headers = {}) => ({ request: new Request(url, { method: 'POST', headers, body: '{}' }) });
  assert.equal((await POST(context('https://example.com/write/posts/', { Origin: 'https://example.com' }))).status, 403);
  assert.equal((await POST(context('http://localhost:4321/write/posts/', { Origin: 'https://other.test' }))).status, 403);
  assert.equal((await POST(context('http://localhost:4321/write/posts/'))).status, 403);
  assert.equal((await GET({ request: new Request('http://localhost:4321/write/posts/', { headers: { Origin: 'https://other.test' } }) })).status, 403);
});
