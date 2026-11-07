type Pane = 'editor' | 'preview';
type Anchor = { editor: number; preview: number };

// Measure wrapped textarea text so images, code and galleries do not skew syncing.
export function connectWriterScroll(editor: HTMLTextAreaElement, preview: HTMLElement, content: HTMLElement, toggle: HTMLInputElement) {
  const mirror = document.createElement('div');
  mirror.setAttribute('aria-hidden', 'true');
  Object.assign(mirror.style, { position: 'fixed', left: '-10000px', top: '0', visibility: 'hidden', pointerEvents: 'none', border: '0', margin: '0', height: 'auto', boxSizing: 'border-box' });
  document.body.append(mirror);

  const panes = { editor, preview };
  const expectedScroll = new Map<HTMLElement, number>();
  let anchors: Anchor[] = [];
  let renderedMarkdown: string | null = null;
  let leader: Pane = 'editor';
  let frame = 0;
  let needsMeasure = true;
  let followCaret = false;
  let sourceOffsetTop = (_offset: number) => 0;
  const preferenceKey = 'oweixx-writer-scroll-sync';
  try { toggle.checked = localStorage.getItem(preferenceKey) !== 'off'; } catch { /* Storage is optional. */ }

  function measure() {
    needsMeasure = false;
    const style = getComputedStyle(editor);
    for (const property of ['font-family', 'font-size', 'font-weight', 'font-style', 'font-stretch', 'line-height', 'letter-spacing', 'word-spacing', 'tab-size', 'text-indent', 'text-transform', 'direction', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left', 'word-break', 'overflow-wrap']) {
      mirror.style.setProperty(property, style.getPropertyValue(property));
    }
    mirror.style.whiteSpace = 'pre-wrap';
    mirror.style.width = `${editor.clientWidth}px`;
    // The final marker gives a measurable caret position after a trailing newline.
    const source = document.createTextNode(editor.value + '\u200b');
    mirror.replaceChildren(source);
    const range = document.createRange();
    const mirrorTop = mirror.getBoundingClientRect().top;
    const lineHeight = parseFloat(style.lineHeight);
    function sourceTop(offset: number) {
      const index = Math.max(0, Math.min(offset, editor.value.length));
      range.setStart(source, index);
      range.setEnd(source, index + 1);
      // Glyph bounds sit inside the textarea's line box.
      const rect = range.getBoundingClientRect();
      return rect.top - mirrorTop - (lineHeight - rect.height) / 2;
    }
    sourceOffsetTop = sourceTop;
    const previewTop = preview.getBoundingClientRect().top + preview.clientTop - preview.scrollTop;
    const points: Anchor[] = [{ editor: 0, preview: 0 }];
    for (const element of content.querySelectorAll<HTMLElement>('[data-source-start][data-source-end]')) {
      const start = Number(element.dataset.sourceStart);
      const end = Number(element.dataset.sourceEnd);
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
      const rect = element.getBoundingClientRect();
      points.push({ editor: sourceTop(start), preview: rect.top - previewTop });
      points.push({ editor: sourceTop(Math.max(start, end - 1)) + lineHeight, preview: rect.bottom - previewTop });
    }
    points.push({ editor: editor.scrollHeight, preview: preview.scrollHeight });
    // Adjacent Markdown blocks can share a visual line; retain monotonic anchors.
    anchors = [];
    for (const point of points.sort((a, b) => a.editor - b.editor || a.preview - b.preview)) {
      const last = anchors.at(-1);
      if (!last || point.editor > last.editor + .5 && point.preview > last.preview + .5) anchors.push(point);
    }
  }

  function sync(from: Pane) {
    if (!toggle.checked || renderedMarkdown !== editor.value) return;
    const to: Pane = from === 'editor' ? 'preview' : 'editor';
    const source = panes[from], target = panes[to];
    const sourceMax = source.scrollHeight - source.clientHeight;
    const targetMax = Math.max(0, target.scrollHeight - target.clientHeight);
    if (anchors.length < 2) return;
    let position = source.scrollTop;
    let caretInset: number | null = null;
    if (from === 'editor' && followCaret && editor.selectionStart === editor.selectionEnd) {
      const caret = sourceOffsetTop(editor.selectionStart);
      const inset = caret - editor.scrollTop;
      if (inset >= 0 && inset < editor.clientHeight) {
        position = caret;
        // Leave room around the active paragraph, especially for a large heading.
        caretInset = Math.max(24, Math.min(preview.clientHeight - 96, inset * preview.clientHeight / editor.clientHeight));
      }
    }
    // Short source text can still drive the preview through its caret.
    if (sourceMax <= 0 && caretInset === null) return;
    const atTextEnd = from === 'editor' && caretInset !== null && editor.selectionEnd === editor.value.length;
    let destination = 0;
    if (atTextEnd || caretInset === null && position >= sourceMax - 1) destination = targetMax;
    else if (position > 0) {
      let lower = anchors[0], upper = anchors.at(-1)!;
      for (let index = 1; index < anchors.length; index++) {
        if (anchors[index][from] >= position) { lower = anchors[index - 1]; upper = anchors[index]; break; }
      }
      const fraction = (position - lower[from]) / Math.max(1, upper[from] - lower[from]);
      destination = lower[to] + fraction * (upper[to] - lower[to]);
    }
    if (caretInset !== null && !atTextEnd) destination -= caretInset;
    const next = Math.max(0, Math.min(targetMax, destination));
    if (Math.abs(target.scrollTop - next) > 1) {
      target.scrollTop = next;
      expectedScroll.set(target, target.scrollTop);
    }
  }

  function schedule(remeasure = false) {
    needsMeasure ||= remeasure;
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (renderedMarkdown !== editor.value) return;
      if (needsMeasure) measure();
      sync(leader);
    });
  }
  for (const name of ['editor', 'preview'] as const) {
    panes[name].addEventListener('scroll', event => {
      // Ignore horizontal scrolling inside galleries and code blocks.
      if (event.target !== panes[name]) return;
      const expected = expectedScroll.get(panes[name]);
      expectedScroll.delete(panes[name]);
      if (expected !== undefined && Math.abs(panes[name].scrollTop - expected) < 1) return;
      leader = name;
      if (name === 'preview') followCaret = false;
      schedule();
    }, { passive: true });
  }
  function trackCaret() { leader = 'editor'; followCaret = true; schedule(); }
  editor.addEventListener('input', () => { needsMeasure = true; trackCaret(); });
  editor.addEventListener('keyup', trackCaret);
  editor.addEventListener('wheel', () => { followCaret = false; }, { passive: true });
  editor.addEventListener('pointerdown', () => { followCaret = false; });
  editor.addEventListener('click', event => {
    // Clicking the scrollbar should keep following the viewport, not the old caret.
    const rect = editor.getBoundingClientRect();
    if (event.clientX < rect.left + editor.clientLeft + editor.clientWidth) trackCaret();
  });
  toggle.addEventListener('change', () => {
    try { localStorage.setItem(preferenceKey, toggle.checked ? 'on' : 'off'); } catch { /* Storage is optional. */ }
    schedule(true);
  });
  const resizeObserver = new ResizeObserver(() => schedule(true));
  resizeObserver.observe(editor);
  resizeObserver.observe(preview);
  resizeObserver.observe(content);
  preview.addEventListener('load', () => schedule(true), true);
  void document.fonts.ready.then(() => schedule(true));

  return {
    update(markdown: string) {
      if (markdown !== renderedMarkdown) { leader = 'editor'; followCaret = true; }
      renderedMarkdown = markdown;
      schedule(true);
    },
    reset() {
      renderedMarkdown = null;
      leader = 'editor';
      followCaret = false;
      editor.scrollTop = preview.scrollTop = 0;
      expectedScroll.clear();
      expectedScroll.set(editor, 0);
      expectedScroll.set(preview, 0);
      needsMeasure = true;
    },
  };
}
