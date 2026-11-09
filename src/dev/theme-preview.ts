import { palettes, defaultPalette, type Palette } from './theme-palettes';

const frame = document.querySelector<HTMLIFrameElement>('#site-preview')!;
const stage = document.querySelector<HTMLElement>('#preview-stage')!;
const shell = document.querySelector<HTMLElement>('#frame-shell')!;
const sectionSelect = document.querySelector<HTMLSelectElement>('#preview-section')!;
const params = new URLSearchParams(location.search);
let selected: Palette = palettes.find(palette => palette.id === params.get('theme')) ?? defaultPalette;
let emphasis = params.get('emphasis') === 'headings' ? 'headings' : 'subtle';
let viewport = params.get('viewport') === 'mobile' ? 'mobile' : 'desktop';
const sections: Record<string, string> = { top: '.name', about: '.about', news: '[aria-labelledby="news-title"]', publications: '[aria-labelledby="publications-title"]', posts: '.recent-posts' };
sectionSelect.value = params.get('section') && sections[params.get('section')!] ? params.get('section')! : 'top';

function previewCss(palette: Palette) {
  if (palette.id === 'current') return '';
  return `
    #color-preview-page { --preview-color: ${palette.color}; --preview-hover: ${palette.hover}; --preview-tint: ${palette.tint}; --preview-border: ${palette.border}; --blue: var(--preview-color); --accent-hover: var(--preview-hover); --accent-tint: var(--preview-tint); --accent-border: var(--preview-border); --profile-link-color: var(--preview-color); --publication-link-color: var(--preview-color); }
    #color-preview-page a { color: var(--preview-color); }
    #color-preview-page a:hover { color: var(--preview-hover); }
    #color-preview-page .pub-title a { color: inherit; }
    #color-preview-page .pub-authors a { color: #000; }
    #color-preview-page .pub-title a:hover, #color-preview-page .pub-title a:focus-visible,
    #color-preview-page .pub-authors a:hover, #color-preview-page .pub-authors a:focus-visible,
    #color-preview-page .profile-links a { color: var(--preview-color); }
    #color-preview-page .profile-links a:hover { color: var(--preview-hover); }
    #color-preview-page .pub-links a, #color-preview-page .citation summary { color: var(--preview-color); border-color: var(--preview-color); }
    #color-preview-page .pub-links a:hover, #color-preview-page .pub-links a:focus-visible,
    #color-preview-page .citation summary:hover, #color-preview-page .citation summary:focus-visible { background: var(--preview-color); border-color: var(--preview-color); color: #fff; }
    #color-preview-page .recent-posts h2 a { color: var(--ink); }
    #color-preview-page .recent-posts h2 a:hover { color: var(--preview-color); }
    #color-preview-page .pub:focus-within { border-color: var(--preview-border); background: var(--preview-tint); }
    @media (hover: hover) and (pointer: fine) {
      #color-preview-page .pub:hover { border-color: var(--preview-border); background: var(--preview-tint); }
    }
    ${emphasis === 'headings' ? `
      #color-preview-page .name strong, #color-preview-page .section-title, #color-preview-page .news strong { color: var(--preview-color); }
      #color-preview-page .section-title { border-color: var(--preview-border); }
    ` : ''}
  `;
}

function applyTheme() {
  const doc = frame.contentDocument;
  if (!doc?.querySelector('.pub')) return;
  let style = doc.querySelector<HTMLStyleElement>('#theme-demo-colors');
  if (!style) {
    style = doc.createElement('style');
    style.id = 'theme-demo-colors';
    doc.head.append(style);
  }
  doc.documentElement.id = 'color-preview-page';
  doc.documentElement.toggleAttribute('data-color-preview', selected.id !== 'current');
  style.textContent = '#color-preview-page .profile-links .writer-link { display: none; }' + previewCss(selected);
}

function resizePreview() {
  const padding = parseFloat(getComputedStyle(stage).paddingLeft) * 2;
  const width = viewport === 'mobile' ? 390 : 1360;
  const scale = Math.min(1, Math.max(1, stage.clientWidth - padding) / width);
  const height = Math.max(1, Math.floor((stage.clientHeight - padding) / scale));
  frame.style.width = `${width}px`;
  frame.style.height = `${height}px`;
  frame.style.transform = `scale(${scale})`;
  shell.style.width = `${width * scale}px`;
  shell.style.height = `${height * scale}px`;
  document.querySelector('#viewport-label')!.textContent = `${viewport === 'mobile' ? '모바일' : 'PC'} · ${width}px`;
}

function jumpToSection() {
  const doc = frame.contentDocument;
  const target = doc?.querySelector<HTMLElement>(sections[sectionSelect.value]);
  if (target && frame.contentWindow) {
    const y = sectionSelect.value === 'top' ? 0 : target.getBoundingClientRect().top + frame.contentWindow.scrollY - 24;
    frame.contentWindow.scrollTo({ top: y, behavior: 'instant' });
  }
}

function syncControls() {
  document.querySelectorAll<HTMLButtonElement>('[data-palette]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.palette === selected.id)));
  document.querySelectorAll<HTMLButtonElement>('[data-emphasis]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.emphasis === emphasis));
    button.disabled = selected.id === 'current';
  });
  document.querySelectorAll<HTMLButtonElement>('[data-viewport]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.viewport === viewport)));
  document.querySelector('#selected-name')!.textContent = selected.name;
  document.querySelector('#selected-color')!.textContent = selected.color.toUpperCase();
  const url = new URL(location.href);
  url.search = new URLSearchParams({ theme: selected.id, emphasis, viewport, section: sectionSelect.value }).toString();
  history.replaceState(null, '', url);
}

document.querySelectorAll<HTMLButtonElement>('[data-palette]').forEach(button => button.addEventListener('click', () => {
  selected = palettes.find(palette => palette.id === button.dataset.palette)!;
  applyTheme();
  syncControls();
}));
document.querySelectorAll<HTMLButtonElement>('[data-emphasis]').forEach(button => button.addEventListener('click', () => {
  emphasis = button.dataset.emphasis!;
  applyTheme();
  syncControls();
}));
document.querySelectorAll<HTMLButtonElement>('[data-viewport]').forEach(button => button.addEventListener('click', () => {
  viewport = button.dataset.viewport!;
  resizePreview();
  jumpToSection();
  syncControls();
}));
sectionSelect.addEventListener('change', () => { jumpToSection(); syncControls(); });

function ready() {
  const doc = frame.contentDocument;
  if (!doc?.querySelector('.pub')) return;
  applyTheme();
  resizePreview();
  // Keep links usable without navigating away from the comparison.
  doc.querySelectorAll<HTMLAnchorElement>('a[href]:not([href^="#"])').forEach(link => { link.target = '_blank'; link.rel = 'noopener'; });
  document.querySelector<HTMLElement>('#preview-loading')!.hidden = true;
  void doc.fonts.ready.then(jumpToSection);
}
frame.addEventListener('load', ready);
new ResizeObserver(resizePreview).observe(stage);
syncControls();
if (frame.contentDocument?.readyState === 'complete') ready();
