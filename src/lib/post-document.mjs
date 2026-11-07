import { parse, stringify } from 'yaml';

export const validSlug = /^[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*$/;
const validDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
  && !Number.isNaN(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;

export function parsePostDocument(markdown, filename = '') {
  const source = markdown.replace(/^\uFEFF/, '').replaceAll('\r\n', '\n');
  const match = source.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
  if (!match) throw new Error('Markdown 앞부분에 글 정보가 필요합니다.');
  const metadata = parse(match[1]);
  if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) throw new Error('글 정보를 확인해 주세요.');
  const slug = metadata.slug ?? filename.replace(/\.md$/i, '').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  if (typeof metadata.title !== 'string' || !metadata.title.trim()) throw new Error('제목을 입력해 주세요.');
  if (typeof slug !== 'string' || !validSlug.test(slug)) throw new Error('글 주소는 영문·숫자·하이픈·밑줄로 입력해 주세요.');
  if (!validDate(metadata.date)) throw new Error('올바른 작성일을 입력해 주세요.');
  if (metadata.updated !== undefined && (!validDate(metadata.updated) || metadata.updated < metadata.date)) throw new Error('수정일은 작성일 이후여야 합니다.');
  if (metadata.category !== undefined && !['Research', 'Papers', 'Notes', 'Life'].includes(metadata.category)) throw new Error('카테고리를 확인해 주세요.');
  if (metadata.description !== undefined && typeof metadata.description !== 'string') throw new Error('설명을 확인해 주세요.');
  if (metadata.tags !== undefined && (!Array.isArray(metadata.tags) || metadata.tags.some(tag => typeof tag !== 'string'))) throw new Error('태그는 문자열 목록이어야 합니다.');
  if (metadata.draft !== undefined && typeof metadata.draft !== 'boolean') throw new Error('초안 설정을 확인해 주세요.');
  return { ...metadata, slug, draft: metadata.draft ?? true, body: source.slice(match[0].length).replace(/^\n/, '') };
}

export function serializePostDocument(post) {
  const { body = '', ...metadata } = post;
  return `---\n${stringify(metadata)}---\n\n${body}${body.endsWith('\n') ? '' : '\n'}`;
}

export const documentUrl = post => `/blog/${post.date.slice(0, 4)}/${post.slug}/`;
