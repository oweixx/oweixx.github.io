export function attachGalleries(root: ParentNode) {
  const cleanups: Array<() => void> = [];
  root.querySelectorAll<HTMLElement>('.image-gallery:not(.gallery-ready)').forEach((gallery) => {
    const track = gallery.querySelector<HTMLElement>('.gallery-track');
    const slides = Array.from(gallery.querySelectorAll<HTMLElement>('.gallery-slide'));
    if (!track || slides.length < 2) return;
    const events = new AbortController();
    const { signal } = events;
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
    function fitHeight() {
      const image = slides[current].querySelector('img');
      // Let unloaded images establish their size before fixing the track height.
      track!.style.height = image?.complete && image.naturalHeight
        ? `${Math.ceil(slides[current].getBoundingClientRect().height)}px` : '';
    }
    function update(index: number) {
      current = Math.max(0, Math.min(slides.length - 1, index));
      status.textContent = `${current + 1} / ${slides.length}`;
      previous.disabled = current === 0;
      next.disabled = current === slides.length - 1;
      slides.forEach((slide, index) => {
        slide.setAttribute('aria-hidden', String(index !== current));
        slide.inert = index !== current;
      });
      fitHeight();
    }
    function move(index: number) {
      index = Math.max(0, Math.min(slides.length - 1, index));
      track!.scrollTo({ left: slides[index].offsetLeft, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
    }
    previous.addEventListener('click', () => move(current - 1), { signal });
    next.addEventListener('click', () => move(current + 1), { signal });
    track.addEventListener('keydown', (event) => {
      const destinations: Record<string, number> = { ArrowLeft: current - 1, ArrowRight: current + 1, Home: 0, End: slides.length - 1 };
      if (!(event.key in destinations)) return;
      event.preventDefault();
      move(destinations[event.key]);
    }, { signal });
    track.addEventListener('scroll', () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const closest = slides.reduce((best, slide, index) => Math.abs(slide.offsetLeft - track.scrollLeft) < Math.abs(slides[best].offsetLeft - track.scrollLeft) ? index : best, 0);
        if (closest !== current) update(closest);
      });
    }, { passive: true, signal });
    track.addEventListener('load', fitHeight, { capture: true, signal });
    let width = track.getBoundingClientRect().width;
    const observer = new ResizeObserver(() => {
      const nextWidth = track.getBoundingClientRect().width;
      if (width !== nextWidth) {
        width = nextWidth;
        track.scrollTo({ left: slides[current].offsetLeft, behavior: 'instant' });
      }
      fitHeight();
    });
    // Observe slides, not the height we change, including late image/font loads.
    slides.forEach(slide => observer.observe(slide));
    update(0);
    cleanups.push(() => {
      events.abort(); observer.disconnect(); cancelAnimationFrame(frame);
      controls.remove(); gallery.classList.remove('gallery-ready'); track.style.height = '';
      slides.forEach(slide => { slide.inert = false; slide.removeAttribute('aria-hidden'); });
    });
  });
  return () => cleanups.forEach((cleanup) => cleanup());
}
