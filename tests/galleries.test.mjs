import { test } from 'node:test';
import assert from 'node:assert/strict';
import { unified as astroMarkdownProcessor } from '@astrojs/markdown-remark';
import { previewMarkdown, remarkPlugins, rehypePlugins } from '../src/lib/markdown.mjs';

const gallery = ':::gallery\n\n![첫 사진](/assets/blog/first.png "첫 번째 설명")\n\n![두 번째 사진](https://example.com/second.jpg)\n\n:::';

test('preview and published Markdown render the same multi-image gallery', async () => {
  const preview = await previewMarkdown(gallery);
  const renderer = await astroMarkdownProcessor({ remarkPlugins, rehypePlugins, smartypants: false, remarkRehype: { allowDangerousHtml: false } }).createRenderer({ syntaxHighlight: false });
  const published = await renderer.render(gallery);
  assert.equal(published.code, preview);
  assert.match(preview, /class="image-gallery"/);
  assert.equal((preview.match(/class="gallery-slide"/g) ?? []).length, 2);
  assert.match(preview, /<figcaption>첫 번째 설명<\/figcaption>/);
  assert.match(preview, /<figcaption>두 번째 사진<\/figcaption>/);
  assert.match(preview, /loading="lazy"/);
});

test('multiple galleries stay separate, image references and normal-image captions work', async () => {
  const html = await previewMarkdown(`${gallery}\n\n:::gallery\n![참조 사진][photo]\n:::\n\n[photo]: /assets/blog/reference.jpg\n\n![단일 사진](/assets/blog/single.png "별도 설명")`);
  assert.equal((html.match(/class="image-gallery"/g) ?? []).length, 2);
  assert.match(html, /src="\/assets\/blog\/reference.jpg"/);
  assert.match(html, /class="post-image"/);
  assert.match(html, /<figcaption>별도 설명<\/figcaption>/);
});

test('gallery captions and image URLs cannot execute HTML or script', async () => {
  const html = await previewMarkdown(':::gallery{onclick="alert(1)"}\n![<script>alert(1)</script>](javascript:alert%281%29)\n:::');
  // Literal angle brackets are valid inside a quoted alt attribute.
  assert.doesNotMatch(html.replace(/alt="[^"]*"/g, ''), /<script|javascript:|onclick=/);
  assert.match(html, /<figcaption>(?:&#x3C;|&lt;)script>/);
});

test('invalid gallery content gives an author-facing error and code examples remain code', async () => {
  await assert.rejects(previewMarkdown(':::gallery\n글만 있습니다\n:::'), /Markdown 이미지/);
  await assert.rejects(previewMarkdown(':::gallery\n:::'), /이미지를 한 장 이상/);
  const example = await previewMarkdown('```text\n:::gallery\n![사진](/image.png)\n:::\n```');
  assert.doesNotMatch(example, /class="image-gallery"/);
});
