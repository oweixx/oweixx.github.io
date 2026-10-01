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
