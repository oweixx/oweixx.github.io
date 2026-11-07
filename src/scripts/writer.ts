import { previewMarkdown } from '../lib/markdown.mjs';
import { parse } from 'yaml';
import { serializePostDocument } from '../lib/post-document.mjs';
import { attachCopyButtons } from './copy-code';
import { attachGalleries } from './galleries';

const input = (id: string) => document.getElementById(id) as HTMLInputElement;
const button = (id: string) => document.getElementById(id) as HTMLButtonElement;
const text = (id: string, value: string) => { document.getElementById(id)!.textContent = value; };
const category = document.getElementById('post-category') as HTMLSelectElement;
const body = document.getElementById('post-body') as HTMLTextAreaElement;
const library = document.getElementById('existing-post') as HTMLSelectElement;
const previewLink = document.getElementById('saved-preview') as HTMLAnchorElement;
const key = 'oweixx-blog-writer-v1';
const today = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
let extraMetadata: Record<string, unknown> = {};
let activeFile: string | null = null;
let revision: string | null = null;
let savedSnapshot = '';
let renderSequence = 0;
let timer: ReturnType<typeof setTimeout>;
let cleanUpGalleries: (() => void) | undefined;
let busy = false;

function values() {
  return {
    ...extraMetadata,
    title: input('post-title').value.trim(), slug: input('post-slug').value.trim(), date: input('post-date').value,
    category: category.value, tags: input('post-tags').value.split(',').map(tag => tag.trim()).filter(Boolean),
    description: input('post-description').value.trim(), draft: input('post-draft').checked, body: body.value,
  };
}
const snapshot = () => JSON.stringify(values());
const dirty = () => snapshot() !== savedSnapshot;

function persist() {
  try { localStorage.setItem(key, JSON.stringify({ post: values(), activeFile, revision, savedSnapshot })); }
  catch { text('save-status', '브라우저 임시 저장 불가 · 파일에 저장해 주세요'); }
}
function setBusy(value: boolean) {
  busy = value;
  for (const id of ['save-post', 'new-post', 'open-post', 'insert-images', 'insert-gallery']) button(id).disabled = value;
  input('import-file').disabled = value;
  library.disabled = value;
}
function updateFileStatus() {
  text('file-status', activeFile ? `${activeFile}${dirty() ? ' · 저장하지 않은 변경 있음' : ' · 저장됨'}` : '새 글 · 파일에 저장하면 글 목록에 추가됩니다');
}
function restore(post: Record<string, unknown>) {
  const { title, slug, date, category: postCategory, tags, description, draft, body: postBody, ...extra } = post;
  extraMetadata = extra;
  input('post-title').value = String(title ?? '');
  input('post-slug').value = String(slug ?? '');
  input('post-date').value = String(date ?? today).slice(0, 10);
  category.value = ['Research', 'Papers', 'Notes', 'Life'].includes(String(postCategory)) ? String(postCategory) : 'Notes';
  input('post-tags').value = Array.isArray(tags) ? tags.join(', ') : String(tags ?? '');
  input('post-description').value = String(description ?? '');
  input('post-draft').checked = draft !== false;
  body.value = String(postBody ?? '');
}
async function renderPreview() {
  const sequence = ++renderSequence;
  const post = values();
  text('preview-title', post.title || '글 제목'); text('preview-date', post.date.replaceAll('-', '.'));
  text('preview-category', post.category); text('preview-description', post.description);
  document.getElementById('preview-draft')!.hidden = !post.draft;
  document.getElementById('preview-tags')!.replaceChildren(...post.tags.map(tag => {
    const el = document.createElement('span'); el.textContent = `#${tag}`; return el;
  }));
  try {
    const html = await previewMarkdown(post.body);
    if (sequence !== renderSequence) return;
    const preview = document.getElementById('preview-body')!;
    cleanUpGalleries?.();
    preview.innerHTML = html || '<p class="preview-placeholder">왼쪽에 글을 입력하면 여기에 표시됩니다.</p>';
    attachCopyButtons(preview); cleanUpGalleries = attachGalleries(preview);
    text('render-status', '미리보기 반영됨');
  } catch (error) { if (sequence === renderSequence) text('render-status', error instanceof Error ? error.message : '문법을 확인해 주세요'); }
}
function changed() {
  text('save-status', '브라우저 임시 저장됨'); persist(); updateFileStatus();
  renderSequence++; clearTimeout(timer); timer = setTimeout(renderPreview, 180);
}
function canSwitch() {
  return !busy && (!dirty() || window.confirm('파일에 저장하지 않은 변경이 있습니다. 저장하지 않고 다른 글로 이동할까요?'));
}
function showPreview(url: string) { previewLink.href = url; previewLink.hidden = false; }
async function responseData(response: Response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '요청을 처리하지 못했습니다.');
  return data;
}
async function refreshLibrary() {
  const data = await responseData(await fetch('/write/posts/'));
  library.replaceChildren(new Option('기존 글을 선택하세요', ''));
  for (const post of data.posts) library.add(new Option(`${post.draft ? '[초안] ' : ''}${post.title}${post.error ? ' (글 정보 확인 필요)' : ''}`, post.file));
  library.value = activeFile || '';
}
async function savePost() {
  if (busy) return;
  for (const id of ['post-title', 'post-slug', 'post-date']) if (!input(id).reportValidity()) return;
  const post = values();
  const savingSnapshot = snapshot();
  setBusy(true); text('save-status', '파일 저장 중…');
  try {
    const data = await responseData(await fetch('/write/posts/', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: activeFile, revision, markdown: serializePostDocument(post) }),
    }));
    activeFile = data.file; revision = data.revision; savedSnapshot = savingSnapshot;
    persist(); updateFileStatus(); showPreview(data.url);
    text('save-status', dirty() ? '파일 저장됨 · 이후 입력은 다시 저장해 주세요' : '파일에 저장됨');
    await refreshLibrary().catch(() => { text('file-status', `${activeFile} · 파일 저장 완료, 목록 갱신 실패`); });
  } catch (error) { text('save-status', error instanceof Error ? error.message : '저장하지 못했습니다.'); }
  finally { setBusy(false); }
}
button('save-post').addEventListener('click', () => void savePost());
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); void savePost(); }
});
button('open-post').addEventListener('click', async () => {
  if (!library.value || !canSwitch()) return;
  setBusy(true);
  try {
    const data = await responseData(await fetch(`/write/posts/?file=${encodeURIComponent(library.value)}`));
    restore(data.post); activeFile = data.file; revision = data.revision; savedSnapshot = snapshot();
    changed(); showPreview(data.url); text('save-status', '기존 글을 열었습니다');
  } catch (error) { text('save-status', error instanceof Error ? error.message : '글을 열지 못했습니다.'); }
  finally { setBusy(false); }
});
button('new-post').addEventListener('click', () => {
  if (!canSwitch()) return;
  restore({ category: 'Research', draft: true }); activeFile = null; revision = null;
  savedSnapshot = snapshot(); library.value = ''; previewLink.hidden = true;
  changed(); text('save-status', '새 글을 작성하세요'); input('post-title').focus();
});

const imageFiles = input('image-files');
let imageMode: 'single' | 'gallery' = 'single';
for (const id of ['insert-images', 'insert-gallery']) button(id).addEventListener('click', () => {
  imageMode = id === 'insert-gallery' ? 'gallery' : 'single'; imageFiles.click();
});
async function uploadImages(files: File[], gallery = false) {
  if (!files.length || busy) return;
  const previousBody = body.value, start = body.selectionStart, end = body.selectionEnd;
  setBusy(true); text('image-status', `${files.length}장 저장 중…`);
  try {
    const form = new FormData(); form.set('slug', input('post-slug').value.trim() || 'uploads');
    files.forEach(file => form.append('images', file));
    const result = await responseData(await fetch('/write/images/', { method: 'POST', body: form }));
    const lines = result.images.map((image: { name: string; url: string }) => `![${image.name.replace(/[\\\[\]]/g, '\\$&')}](${image.url})`).join('\n\n');
    const markdown = `\n\n${gallery ? `:::gallery\n\n${lines}\n\n:::` : lines}\n\n`;
    if (body.value === previousBody) body.setRangeText(markdown, start, end, 'end');
    else body.setRangeText(markdown, body.selectionStart, body.selectionStart, 'end');
    changed(); body.focus(); text('image-status', `${files.length}장 추가됨 · 글도 파일에 저장해 주세요`);
  } catch (error) { text('image-status', error instanceof Error ? error.message : '이미지를 저장하지 못했습니다.'); }
  finally { setBusy(false); imageFiles.value = ''; }
}
imageFiles.addEventListener('change', () => void uploadImages(Array.from(imageFiles.files ?? []), imageMode === 'gallery'));
body.addEventListener('paste', event => {
  const files = Array.from(event.clipboardData?.files ?? []).filter(file => file.type.startsWith('image/'));
  if (files.length) { event.preventDefault(); void uploadImages(files); }
});
body.addEventListener('dragover', event => { if (event.dataTransfer?.types.includes('Files')) event.preventDefault(); });
body.addEventListener('drop', event => {
  const files = Array.from(event.dataTransfer?.files ?? []).filter(file => file.type.startsWith('image/'));
  if (files.length) { event.preventDefault(); void uploadImages(files); }
});
input('import-file').addEventListener('change', async () => {
  const file = input('import-file').files?.[0];
  if (!file || !canSwitch()) { input('import-file').value = ''; return; }
  try {
    const markdown = (await file.text()).replace(/^\uFEFF/, '').replaceAll('\r\n', '\n');
    const match = markdown.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
    const metadata = match ? parse(match[1]) : {};
    if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) throw new Error('Invalid metadata');
    const filenameSlug = file.name.replace(/\.m(?:arkdown|d)$/i, '').replace(/^\d{4}-\d{2}-\d{2}-/, '');
    restore({ ...metadata, slug: metadata.slug ?? filenameSlug, body: match ? markdown.slice(match[0].length) : markdown });
    activeFile = null; revision = null; savedSnapshot = ''; library.value = ''; previewLink.hidden = true;
    changed(); text('save-status', '가져온 글 · 저장 시 새 파일로 추가됩니다');
  } catch { text('save-status', '파일의 Markdown / YAML 형식을 확인해 주세요'); }
  input('import-file').value = '';
});
button('download').addEventListener('click', () => {
  for (const id of ['post-title', 'post-slug', 'post-date']) if (!input(id).reportValidity()) return;
  const post = values();
  const url = URL.createObjectURL(new Blob([serializePostDocument(post)], { type: 'text/markdown;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = `${post.date}-${post.slug}.md`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000); text('save-status', 'Markdown 다운로드됨');
});
window.addEventListener('beforeunload', event => { if (dirty()) event.preventDefault(); });

restore({ category: 'Research', draft: true }); savedSnapshot = snapshot();
try {
  const raw = localStorage.getItem(key);
  if (raw) {
    const saved = JSON.parse(raw);
    restore(saved.post ?? saved);
    if (saved.post) { activeFile = saved.activeFile ?? null; revision = saved.revision ?? null; savedSnapshot = saved.savedSnapshot ?? ''; }
    text('save-status', '임시 저장한 글을 복원했습니다');
    if (activeFile) showPreview(`/blog/${input('post-date').value.slice(0, 4)}/${input('post-slug').value}/`);
  } else text('save-status', '작성 준비됨');
} catch { text('save-status', '임시 저장을 복원하지 못했습니다'); }
document.querySelector('.editor-fields')!.addEventListener('input', changed);
(document.querySelector('.editor-fields') as HTMLFieldSetElement).disabled = false;
button('download').disabled = false; setBusy(false); updateFileStatus(); void renderPreview();
void refreshLibrary().catch(error => text('file-status', `글 목록을 불러오지 못했습니다: ${error.message}`));
