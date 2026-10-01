import { mkdir, writeFile } from 'node:fs/promises';
import { stringify } from 'yaml';

const [slug, title = slug] = process.argv.slice(2);
if (!slug || !/^[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*$/.test(slug)) {
  console.error('Usage: npm run new -- my-post "글 제목"');
  process.exit(1);
}
const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
const directory = new URL('../src/content/blog/', import.meta.url);
const path = new URL(`${date}-${slug}.md`, directory);
await mkdir(directory, { recursive: true });
try {
  await writeFile(path, `---\n${stringify({ title, date, slug, description: '', category: 'Research', tags: [], draft: true })}---\n\n## 시작하며\n\n`, { flag: 'wx' });
  console.log(`Created ${path.pathname}\nPreview: http://127.0.0.1:4321/blog/${date.slice(0, 4)}/${slug}/`);
} catch (error) {
  console.error(error.code === 'EEXIST' ? 'A post with this filename already exists.' : error.message);
  process.exit(1);
}
