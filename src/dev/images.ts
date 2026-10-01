import type { APIRoute } from 'astro';
import { mkdir, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { Buffer } from 'node:buffer';
import { resolve, sep } from 'node:path';

export const prerender = false;

function imageExtension(bytes: Uint8Array): string | undefined {
  const signature = Buffer.from(bytes.subarray(0, 64));
  if (signature.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return 'png';
  if (signature[0] === 255 && signature[1] === 216 && signature[2] === 255) return 'jpg';
  if (['GIF87a', 'GIF89a'].includes(signature.toString('ascii', 0, 6))) return 'gif';
  if (signature.toString('ascii', 0, 4) === 'RIFF' && signature.toString('ascii', 8, 12) === 'WEBP') return 'webp';
  if (signature.toString('ascii', 4, 8) === 'ftyp' && /avif|avis/.test(signature.toString('ascii', 8))) return 'avif';
}

export const POST: APIRoute = async ({ request }) => {
  const source = new URL(request.url);
  const origin = request.headers.get('Origin');
  if (!['localhost', '127.0.0.1', '[::1]'].includes(source.hostname) || origin && origin !== source.origin) {
    return Response.json({ error: '로컬 글쓰기 화면에서만 이미지를 저장할 수 있습니다.' }, { status: 403 });
  }
  const maxTotal = 80 * 1024 * 1024;
  if (Number(request.headers.get('Content-Length')) > maxTotal) return Response.json({ error: '한 번에 80MB 이하로 선택해 주세요.' }, { status: 413 });
  try {
    const form = await request.formData();
    const files = form.getAll('images');
    const slug = String(form.get('slug') || 'uploads');
    if (!/^[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*$/.test(slug)) return Response.json({ error: '글 주소는 영문·숫자·하이픈·밑줄로 입력해 주세요.' }, { status: 400 });
    if (!files.length || files.length > 20) return Response.json({ error: '이미지를 1~20장 선택해 주세요.' }, { status: 400 });
    let total = 0;
    const images = [];
    for (const file of files) {
      if (!(file instanceof File) || !file.size || file.size > 20 * 1024 * 1024) return Response.json({ error: '각 이미지는 20MB 이하의 파일이어야 합니다.' }, { status: 400 });
      total += file.size;
      if (total > maxTotal) return Response.json({ error: '한 번에 80MB 이하로 선택해 주세요.' }, { status: 413 });
      const bytes = new Uint8Array(await file.arrayBuffer());
      const extension = imageExtension(bytes);
      if (!extension) return Response.json({ error: 'PNG, JPEG, GIF, WebP, AVIF 이미지를 선택해 주세요.' }, { status: 400 });
      const name = file.name.replace(/\.[^.]+$/, '').replace(/[\r\n]/g, ' ').slice(0, 120) || '이미지';
      const stem = name.normalize('NFKD').replace(/[^a-zA-Z0-9_-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'image';
      images.push({ name, filename: `${stem}-${randomUUID().slice(0, 8)}.${extension}`, bytes });
    }
    const root = resolve('assets/blog');
    const directory = resolve(root, slug);
    if (!directory.startsWith(root + sep)) throw new Error('Invalid image directory');
    await mkdir(directory, { recursive: true });
    const saved = [];
    for (const image of images) {
      await writeFile(resolve(directory, image.filename), image.bytes, { flag: 'wx' });
      saved.push({ name: image.name, url: `/assets/blog/${slug}/${image.filename}` });
    }
    return Response.json({ images: saved }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    console.error('Local image upload failed:', error);
    return Response.json({ error: '이미지를 저장하지 못했습니다. 파일과 로컬 폴더의 쓰기 권한을 확인해 주세요.' }, { status: 500 });
  }
};
