import { mkdir, readFile, readdir, realpath, lstat, writeFile, rename, unlink } from 'node:fs/promises';
import { resolve, sep, dirname, basename } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { parsePostDocument, documentUrl } from '../lib/post-document.mjs';

export class PostError extends Error {
  constructor(message, status = 400) { super(message); this.status = status; }
}
const revisionOf = markdown => createHash('sha256').update(markdown).digest('hex');

export function createPostStore(directory) {
  const root = resolve(directory);
  let queue = Promise.resolve();
  async function safePath(file, mustExist = true) {
    if (typeof file !== 'string' || !/^[\w-]+(?:\/[\w-]+)*\.md$/i.test(file)) throw new PostError('허용되지 않는 파일 경로입니다.');
    const path = resolve(root, file);
    if (!path.startsWith(root + sep)) throw new PostError('허용되지 않는 파일 경로입니다.');
    const rootReal = await realpath(root);
    const parent = await realpath(dirname(path));
    if (parent !== rootReal && !parent.startsWith(rootReal + sep)) throw new PostError('블로그 폴더 안의 파일만 열 수 있습니다.');
    try {
      const stat = await lstat(path);
      if (stat.isSymbolicLink() || !stat.isFile()) throw new PostError('일반 Markdown 파일만 열 수 있습니다.');
    } catch (error) {
      if (error.code !== 'ENOENT' || mustExist) throw error;
    }
    return path;
  }
  async function files(folder = root, prefix = '') {
    const found = [];
    for (const entry of await readdir(folder, { withFileTypes: true })) {
      if (entry.isSymbolicLink()) continue;
      const name = prefix + entry.name;
      if (entry.isDirectory()) found.push(...await files(resolve(folder, entry.name), name + '/'));
      else if (entry.isFile() && name.endsWith('.md')) found.push(name);
    }
    return found;
  }
  async function read(file) {
    const markdown = await readFile(await safePath(file), 'utf8');
    const post = parsePostDocument(markdown, basename(file));
    return { file, markdown, revision: revisionOf(markdown), post, url: documentUrl(post) };
  }
  async function list() {
    await mkdir(root, { recursive: true });
    const posts = [];
    for (const file of await files()) {
      try {
        const { post, url } = await read(file);
        posts.push({ file, title: post.title, date: post.date, draft: post.draft, url });
      } catch {
        // One malformed file must not hide the rest of the library.
        posts.push({ file, title: file, date: '', draft: true, error: true });
      }
    }
    return posts.sort((a, b) => b.date.localeCompare(a.date) || a.file.localeCompare(b.file));
  }
  async function write({ file, revision, markdown }) {
    if (typeof markdown !== 'string' || Buffer.byteLength(markdown) > 2 * 1024 * 1024) throw new PostError('글은 2MB 이하의 Markdown이어야 합니다.', 413);
    await mkdir(root, { recursive: true });
    const post = parsePostDocument(markdown, typeof file === 'string' ? basename(file) : '');
    const target = file || `${post.date}-${post.slug}.md`;
    const path = await safePath(target, Boolean(file));
    if (file) {
      const current = await readFile(path, 'utf8');
      if (!revision || revisionOf(current) !== revision) throw new PostError('다른 곳에서 수정된 파일입니다. 작성 내용을 다운로드한 뒤 글을 다시 열어 주세요.', 409);
    }
    for (const other of await list()) {
      if (other.file !== target && other.url === documentUrl(post)) throw new PostError('같은 연도와 글 주소를 사용하는 글이 이미 있습니다.', 409);
    }
    if (file) {
      const temp = resolve(dirname(path), `.writer-${randomUUID()}.tmp`);
      try {
        await writeFile(temp, markdown, { flag: 'wx' });
        await rename(temp, path);
      } finally { await unlink(temp).catch(error => { if (error.code !== 'ENOENT') throw error; }); }
    } else {
      try { await writeFile(path, markdown, { flag: 'wx' }); }
      catch (error) { if (error.code === 'EEXIST') throw new PostError('파일이 이미 있습니다. 기존 글 목록에서 열어 주세요.', 409); throw error; }
    }
    return { file: target, revision: revisionOf(markdown), url: documentUrl(post) };
  }
  return {
    list, read,
    save: input => {
      const result = queue.then(() => write(input));
      queue = result.catch(() => {});
      return result;
    },
  };
}
