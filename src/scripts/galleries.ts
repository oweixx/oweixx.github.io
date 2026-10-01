export function attachGalleries(root: ParentNode) {
  const cleanups: Array<() => void> = [];
  root.querySelectorAll<HTMLElement>('.image-gallery:not(.gallery-ready)').forEach((gallery) => {
    const track = gallery.querySelector<HTMLElement>('.gallery-track');
    const slides = Array.from(gallery.querySelectorAll<HTMLElement>('.gallery-slide'));
    if (!track || slides.length < 2) return;
    gallery.classList.add('gallery-ready');
    gallery.setAttribute('role', 'region');
    gallery.setAttribute('aria-roledescription', '이미지 갤러리');
    gallery.setAttribute('aria-label', `이미지 갤러리: ${slides[0].querySelector('img')?.alt || `${slides.length}장`}`);
    track.tabIndex = 0;
    track.setAttribute('role', 'group');
    track.setAttribute('aria-label', '이미지 영역 · 좌우 방향키로 이동');
    slides.forEach((slide, index) => {
      slide.setAttribute('role', 'group');
      slide.setAttribute('aria-label', `${index + 1} / ${slides.length}`);
      const image = slide.querySelector('img');
      if (image) image.draggable = false;
    });
    const controls = document.createElement('div');
    controls.className = 'gallery-controls';
    const button = (label: string, value: string) => {
      const element = document.createElement('button');
      element.type = 'button';
      element.setAttribute('aria-label', label);
      element.textContent = value;
      return element;
    };
    const previous = button('이전 이미지', '←');
    const next = button('다음 이미지', '→');
    const status = document.createElement('span');
    status.className = 'gallery-position';
    status.setAttribute('aria-live', 'polite');
    status.setAttribute('aria-atomic', 'true');
    controls.append(previous, status, next);
    gallery.appendChild(controls);
    let current = 0;
    let frame = 0;
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    function update(index: number) {
      current = Math.max(0, Math.min(slides.length - 1, index));
      status.textContent = `${current + 1} / ${slides.length}`;
      previous.disabled = current === 0;
      next.disabled = current === slides.length - 1;
      slides.forEach((slide, index) => slide.setAttribute('aria-hidden', String(index !== current)));
    }
    function move(index: number) {
      index = Math.max(0, Math.min(slides.length - 1, index));
      track!.scrollTo({ left: slides[index].offsetLeft, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
    }
    previous.addEventListener('click', () => move(current - 1));
    next.addEventListener('click', () => move(current + 1));
    track.addEventListener('keydown', (event) => {
      const destinations: Record<string, number> = { ArrowLeft: current - 1, ArrowRight: current + 1, Home: 0, End: slides.length - 1 };
      if (!(event.key in destinations)) return;
      event.preventDefault();
      move(destinations[event.key]);
    });
    track.addEventListener('scroll', () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const closest = slides.reduce((best, slide, index) => Math.abs(slide.offsetLeft - track.scrollLeft) < Math.abs(slides[best].offsetLeft - track.scrollLeft) ? index : best, 0);
        update(closest);
      });
    }, { passive: true });
    let width = track.getBoundingClientRect().width;
    const observer = new ResizeObserver(() => {
      if (width === track.getBoundingClientRect().width) return;
      width = track.getBoundingClientRect().width;
      track.scrollTo({ left: slides[current].offsetLeft, behavior: 'instant' });
    });
    observer.observe(track);
    update(0);
    cleanups.push(() => { observer.disconnect(); cancelAnimationFrame(frame); });
  });
  return () => cleanups.forEach((cleanup) => cleanup());
}
