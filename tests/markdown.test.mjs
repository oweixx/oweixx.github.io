import { test } from 'node:test';
import assert from 'node:assert/strict';
import { previewMarkdown } from '../src/lib/markdown.mjs';

test('writer renders Korean, math, highlighted code, GFM tables and heading links', async () => {
  const html = await previewMarkdown('## 연구 기록\n\n수식 $x^2$\n\n$$\nx^2 + y^2 = 1\n$$\n\n```python\nprint("hello")\n```\n\n| 항목 | 결과 |\n| --- | --- |\n| 실험 | 성공 |');
  assert.match(html, /연구 기록/);
  assert.match(html, /id="연구-기록"/);
  assert.match(html, /class="katex"/);
  assert.match(html, /class="katex-display"/);
  assert.match(html, /hljs-built_in/);
  assert.match(html, /<table>/);
});

test('raw HTML is not executed by the writer', async () => {
  const html = await previewMarkdown('<script>alert("bad")</script>\n\n[link](javascript:alert(1))');
  assert.doesNotMatch(html, /<script/);
  assert.doesNotMatch(html, /javascript:/);
});

test('writer source offsets identify complete Unicode paragraphs, code, math, images and galleries', async () => {
  const blocks = [
    '## 연구 기록 👋',
    '줄이 긴 한국어 문단입니다.\n다음 줄도 같은 문단입니다.',
    '```python\nprint("hello")\n```',
    '$$\nx^2 + y^2 = 1\n$$',
    '![사진](/assets/profile.png "설명")',
    ':::gallery\n![첫 사진](/assets/geometry.gif)\n![두 번째 사진](/assets/edgar.gif)\n:::',
    '| 항목 | 결과 |\n| --- | --- |\n| 실험 | 성공 |',
    '> 인용문\n> 두 번째 줄',
  ];
  const markdown = blocks.join('\n\n');
  const html = await previewMarkdown(markdown, { sourceMap: true });
  const sources = [...html.matchAll(/data-source-start="(\d+)" data-source-end="(\d+)"/g)]
    .map(([, start, end]) => markdown.slice(Number(start), Number(end)));
  assert.deepEqual(sources, blocks);
  // The synchronization metadata never changes the actual rendered content.
  assert.equal(html.replace(/ data-source-(?:start|end)="\d+"/g, ''), await previewMarkdown(markdown));
});
