import { imageCaption } from '../lib/galleries.mjs';

let openImage: ((image: HTMLImageElement) => void) | undefined;

function initialiseViewer() {
  if (openImage) return;
  const dialog = document.querySelector<HTMLDialogElement>('#image-viewer');
  if (!dialog || typeof dialog.showModal !== 'function') return;
  const image = dialog.querySelector<HTMLImageElement>('.viewer-image')!;
  const canvas = dialog.querySelector<HTMLElement>('.viewer-canvas')!;
  const caption = dialog.querySelector<HTMLElement>('.viewer-caption')!;
  const position = dialog.querySelector<HTMLElement>('.viewer-position')!;
  const original = dialog.querySelector<HTMLAnchorElement>('.viewer-original')!;
  const zoom = dialog.querySelector<HTMLButtonElement>('.viewer-zoom')!;
  const previous = dialog.querySelector<HTMLButtonElement>('.viewer-previous')!;
  const next = dialog.querySelector<HTMLButtonElement>('.viewer-next')!;
  const error = dialog.querySelector<HTMLElement>('.viewer-error')!;
  let images: HTMLImageElement[] = [];
  let index = 0;
  let opener: HTMLImageElement | undefined;
  let zoomed = false;
  let touch: { x: number; y: number; id: number } | undefined;
  let suppressClick = false;

  function sizeToCanvas() {
    if (!dialog!.open || zoomed || !image.complete || !image.naturalWidth) return;
    const scale = Math.min(canvas.clientWidth / image.naturalWidth, canvas.clientHeight / image.naturalHeight);
    image.style.width = `${image.naturalWidth * scale}px`;
    image.style.height = `${image.naturalHeight * scale}px`;
  }
  function fit() {
    zoomed = false;
    canvas.classList.remove('is-zoomed');
    image.style.width = image.style.height = '';
    zoom.textContent = '확대';
    zoom.setAttribute('aria-pressed', 'false');
    sizeToCanvas();
    canvas.scrollTo({ left: 0, top: 0, behavior: 'instant' });
  }
  function toggleZoom() {
    if (zoom.disabled) return;
    if (zoomed) { fit(); return; }
    const bounds = image.getBoundingClientRect();
    zoomed = true;
    canvas.classList.add('is-zoomed');
    image.style.width = `${bounds.width * 2}px`;
    image.style.height = `${bounds.height * 2}px`;
    zoom.textContent = '화면에 맞추기';
    zoom.setAttribute('aria-pressed', 'true');
    canvas.scrollTo({ left: (canvas.scrollWidth - canvas.clientWidth) / 2, top: (canvas.scrollHeight - canvas.clientHeight) / 2, behavior: 'instant' });
  }
  function show(nextIndex: number) {
    index = Math.max(0, Math.min(images.length - 1, nextIndex));
    const source = images[index];
    fit();
    error.hidden = true;
    image.hidden = false;
    zoom.disabled = true;
    image.alt = source.alt;
    image.src = source.currentSrc || source.src;
    original.href = image.src;
    caption.textContent = source.closest('figure')?.querySelector('figcaption')?.textContent || imageCaption(source.alt, source.title);
    position.textContent = `${index + 1} / ${images.length}`;
    previous.disabled = index === 0;
    next.disabled = index === images.length - 1;
    dialog!.querySelector('.viewer-footer')!.classList.toggle('is-single', images.length === 1);
    if (image.complete && image.naturalWidth) zoom.disabled = false;
  }
  openImage = (source) => {
    if (dialog.open) return;
    opener = source;
    const gallery = source.closest('.image-gallery');
    images = gallery ? Array.from(gallery.querySelectorAll<HTMLImageElement>('.gallery-slide img[data-image-zoom]')) : [source];
    show(images.indexOf(source));
    document.documentElement.classList.add('image-viewer-open');
    dialog.showModal();
    sizeToCanvas();
  };

  image.addEventListener('load', () => { zoom.disabled = false; sizeToCanvas(); });
  image.addEventListener('error', () => { image.hidden = true; error.hidden = false; zoom.disabled = true; });
  zoom.addEventListener('click', toggleZoom);
  previous.addEventListener('click', () => show(index - 1));
  next.addEventListener('click', () => show(index + 1));
  dialog.querySelector('.viewer-close')!.addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    document.documentElement.classList.remove('image-viewer-open');
    fit();
    image.removeAttribute('src');
    images = [];
    if (opener?.isConnected) opener.focus({ preventScroll: true });
    opener = undefined;
  });
  dialog.addEventListener('keydown', (event) => {
    if (event.key === 'Tab') {
      const focusable = Array.from(dialog.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], [tabindex="0"]'))
        .filter(element => element.getClientRects().length && getComputedStyle(element).visibility !== 'hidden');
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first || !event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        (event.shiftKey ? last : first)?.focus();
      }
      return;
    }
    if (zoomed || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      show(index + (event.key === 'ArrowLeft' ? -1 : 1));
    }
  });
  canvas.addEventListener('click', (event) => {
    if (suppressClick) { suppressClick = false; return; }
    if (event.target === image) toggleZoom();
    else if (event.target === canvas) dialog.close();
  });
  canvas.addEventListener('pointerdown', (event) => {
    suppressClick = false;
    if (event.pointerType === 'touch' && event.isPrimary && !zoomed) touch = { x: event.clientX, y: event.clientY, id: event.pointerId };
    else touch = undefined;
  });
  canvas.addEventListener('pointerup', (event) => {
    if (!touch || touch.id !== event.pointerId || zoomed) return;
    const dx = event.clientX - touch.x;
    const dy = event.clientY - touch.y;
    touch = undefined;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) {
      suppressClick = true;
      show(index + (dx < 0 ? 1 : -1));
    }
  });
  canvas.addEventListener('pointercancel', () => { touch = undefined; });
  window.addEventListener('resize', () => { if (dialog.open) fit(); });
  new ResizeObserver(sizeToCanvas).observe(canvas);

  // Delegation also handles images replaced by the live Markdown preview.
  document.addEventListener('click', (event) => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (event.target instanceof HTMLImageElement && event.target.matches('.prose img[data-image-zoom]')) openImage?.(event.target);
  });
  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLImageElement && event.target.matches('.prose img[data-image-zoom]') && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      openImage?.(event.target);
    }
  });
}

export function prepareImageViewer(root: ParentNode) {
  initialiseViewer();
  if (!openImage) return;
  root.querySelectorAll<HTMLImageElement>('.prose img:not([data-image-zoom])').forEach(image => {
    // An explicitly linked image keeps its author-selected destination.
    if (!image.getAttribute('src') || image.closest('a, button')) return;
    image.dataset.imageZoom = '';
    image.tabIndex = 0;
    image.setAttribute('role', 'button');
    image.setAttribute('aria-haspopup', 'dialog');
    image.setAttribute('aria-label', `이미지 크게 보기${image.alt ? `: ${image.alt}` : ''}`);
  });
}
