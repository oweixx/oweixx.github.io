import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, rm } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { POST } from '../src/dev/images.ts';

test('local image uploads preserve bytes, avoid filename collisions and reject unsafe requests', async () => {
  const folder = `qa-upload-${crypto.randomUUID()}`;
  const root = resolve('assets/blog');
  const directory = resolve(root, folder);
  assert(directory.startsWith(root + sep));
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6kZQAAAAASUVORK5CYII=', 'base64');
  const upload = (files, slug = folder, origin = 'http://localhost:4321') => {
    const form = new FormData();
    form.set('slug', slug);
    files.forEach(file => form.append('images', file));
    return POST({ request: new Request('http://localhost:4321/write/images/', { method: 'POST', headers: { Origin: origin }, body: form }) });
  };
  try {
    const result = await upload([new File([png], '같은 사진.png'), new File([png], '같은 사진.png')]);
    assert.equal(result.status, 200);
    const { images } = await result.json();
    assert.equal(images.length, 2);
    assert.notEqual(images[0].url, images[1].url);
    for (const image of images) {
      assert(image.url.startsWith(`/assets/blog/${folder}/`));
      assert.equal(image.name, '같은 사진');
      assert.deepEqual(await readFile(resolve('.' + image.url)), png);
    }
    assert.equal((await upload([new File([png], 'photo.png')], folder, 'https://other.test')).status, 403);
    assert.equal((await upload([new File([png], 'photo.png')], '../../escape')).status, 400);
    assert.equal((await upload([new File(['<script>alert(1)</script>'], 'fake.png')])).status, 400);
  } finally {
    // Delete only this test's generated directory under the image folder.
    assert(directory.startsWith(root + sep) && directory.endsWith(folder));
    await rm(directory, { recursive: true, force: true });
  }
});
