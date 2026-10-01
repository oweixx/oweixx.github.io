export function attachCopyButtons(root: ParentNode) {
  root.querySelectorAll<HTMLElement>('.prose pre').forEach((pre) => {
    if (pre.querySelector('.copy-code')) return;
    const code = pre.querySelector('code');
    if (!code) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = '복사';
    button.setAttribute('aria-label', '코드 복사');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent ?? '');
        button.textContent = '복사됨';
      } catch {
        button.textContent = '복사 실패';
      }
      setTimeout(() => { button.textContent = '복사'; }, 1600);
    });
    pre.appendChild(button);
  });
}
