import type { APIRoute } from 'astro';
import { createPostStore, PostError } from './post-store.mjs';
export const prerender = false;
const store = createPostStore('src/content/blog');

function allowed(request: Request, write = false) {
  const url = new URL(request.url);
  const origin = request.headers.get('Origin');
  return ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
    && (!origin || origin === url.origin) && (!write || origin === url.origin)
    && request.headers.get('Sec-Fetch-Site') !== 'cross-site';
}
const json = (value: unknown, status = 200) => Response.json(value, { status, headers: { 'Cache-Control': 'no-store' } });
function failure(error: unknown) {
  if (error instanceof PostError) return json({ error: error.message }, error.status);
  if ((error as NodeJS.ErrnoException).code === 'ENOENT') return json({ error: '파일을 찾을 수 없습니다.' }, 404);
  return json({ error: error instanceof Error ? error.message : '글을 처리하지 못했습니다.' }, 400);
}
export const GET: APIRoute = async ({ request }) => {
  if (!allowed(request)) return json({ error: '로컬 글쓰기 화면에서만 사용할 수 있습니다.' }, 403);
  try {
    const file = new URL(request.url).searchParams.get('file');
    return json(file ? await store.read(file) : { posts: await store.list() });
  } catch (error) { return failure(error); }
};
export const POST: APIRoute = async ({ request }) => {
  if (!allowed(request, true)) return json({ error: '로컬 글쓰기 화면에서만 저장할 수 있습니다.' }, 403);
  if (!request.headers.get('Content-Type')?.startsWith('application/json')) return json({ error: 'JSON 요청이 필요합니다.' }, 415);
  try {
    const raw = await request.text();
    if (Buffer.byteLength(raw) > 3 * 1024 * 1024) return json({ error: '요청이 너무 큽니다.' }, 413);
    const input = JSON.parse(raw);
    if (!input || typeof input !== 'object' || Array.isArray(input)) return json({ error: '글 정보를 확인해 주세요.' }, 400);
    return json(await store.save(input));
  } catch (error) { return failure(error); }
};
