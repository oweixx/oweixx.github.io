import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
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

test('the two migrated research posts preserve their original bodies', async () => {
  for (const name of ['2026-02-01-Research_rebuttal.md', '2026-02-21-Research_decision.md']) {
    const original = await readFile(new URL(`../legacy/_posts/${name}`, import.meta.url), 'utf8');
    const migrated = await readFile(new URL(`../src/content/blog/${name}`, import.meta.url), 'utf8');
    const body = (text) => text.replaceAll('\r\n', '\n').split(/^---\s*$/m).slice(2).join('---').trim();
    assert.equal(body(migrated), body(original));
  }
});
