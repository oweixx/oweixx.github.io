import { badgeVariants, badgeCss, defaultBadgeVariant, type BadgeVariant } from './badge-variants';

const frame = document.querySelector<HTMLIFrameElement>('#badge-frame')!;
const stage = document.querySelector<HTMLElement>('#badge-stage')!;
const shell = document.querySelector<HTMLElement>('#badge-shell')!;
const compare = document.querySelector<HTMLButtonElement>('#compare-original')!;
const params = new URLSearchParams(location.search);
let selected: BadgeVariant = badgeVariants.find(variant => variant.id === params.get('style')) ?? defaultBadgeVariant;
let viewport = params.get('viewport') === 'mobile' ? 'mobile' : 'desktop';
let comparing = false;

function applyBadge() {
  const doc = frame.contentDocument;
  if (!doc?.querySelector('.pub-badge')) return;
  if (!doc.querySelector('#badge-preview-font')) {
    const font = document.querySelector<HTMLLinkElement>('#badge-preview-font');
    if (font) doc.head.append(font.cloneNode(true));
  }
  doc.documentElement.id = 'badge-preview-page';
  let style = doc.querySelector<HTMLStyleElement>('#badge-demo-style');
  if (!style) {
    style = doc.createElement('style');
    style.id = 'badge-demo-style';
    doc.head.append(style);
  }
  const variant = comparing ? badgeVariants[0] : selected;
  style.textContent = badgeCss('#badge-preview-page .pub-badge', variant);
}

function jumpToPublications() {
  const target = frame.contentDocument?.querySelector<HTMLElement>('[aria-labelledby="publications-title"]');
  if (!target || !frame.contentWindow) return;
  frame.contentWindow.scrollTo({ top: target.getBoundingClientRect().top + frame.contentWindow.scrollY - 24, behavior: 'instant' });
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

function syncControls() {
  document.querySelectorAll<HTMLButtonElement>('[data-variant]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.variant === selected.id)));
  document.querySelectorAll<HTMLButtonElement>('[data-viewport]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.viewport === viewport)));
  const displayed = comparing ? badgeVariants[0] : selected;
  document.querySelector('#selected-number')!.textContent = displayed.number;
  document.querySelector('#selected-name')!.textContent = displayed.name;
  compare.setAttribute('aria-pressed', String(comparing));
  compare.textContent = comparing ? '선택한 스타일로' : '이전 배지와 비교';
  compare.disabled = selected.id === 'current';
  const url = new URL(location.href);
  url.search = new URLSearchParams({ style: selected.id, viewport }).toString();
  history.replaceState(null, '', url);
}

document.querySelectorAll<HTMLButtonElement>('[data-variant]').forEach(button => button.addEventListener('click', () => {
  selected = badgeVariants.find(variant => variant.id === button.dataset.variant)!;
  comparing = false;
  applyBadge();
  syncControls();
}));
document.querySelectorAll<HTMLButtonElement>('[data-viewport]').forEach(button => button.addEventListener('click', () => {
  viewport = button.dataset.viewport!;
  resizePreview();
  jumpToPublications();
  syncControls();
}));
compare.addEventListener('click', () => { comparing = !comparing; applyBadge(); syncControls(); });

function ready() {
  const doc = frame.contentDocument;
  if (!doc?.querySelector('.pub-badge')) return;
  doc.querySelectorAll<HTMLElement>('.pub-badge').forEach(badge => {
    const match = badge.textContent?.trim().match(/^(.*) (\d{4})$/);
    if (!match) return;
    const venue = doc.createElement('span');
    venue.className = 'badge-venue';
    venue.textContent = match[1];
    const year = doc.createElement('span');
    year.className = 'badge-year';
    year.textContent = match[2];
    badge.replaceChildren(venue, doc.createTextNode(' '), year);
  });
  doc.querySelectorAll<HTMLAnchorElement>('a[href]:not([href^="#"])').forEach(link => { link.target = '_blank'; link.rel = 'noopener'; });
  applyBadge();
  resizePreview();
  jumpToPublications();
  document.querySelector<HTMLElement>('#badge-loading')!.hidden = true;
  void doc.fonts.ready.then(jumpToPublications);
}

frame.addEventListener('load', ready);
new ResizeObserver(() => { resizePreview(); jumpToPublications(); }).observe(stage);
syncControls();
if (frame.contentDocument?.readyState === 'complete') ready();
