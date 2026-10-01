import { CounterError, readCount, readDailyCount, recentDates } from '../lib/counters.mjs';

const memory = new Map<string, Promise<number>>();
let queue: Promise<unknown> = Promise.resolve();
let lastRequest = 0;

export function getCount(site: string, path: string, start?: string, end?: string): Promise<number> {
  const key = `oweixx-count:${site}:${path}:${start ?? ''}:${end ?? ''}`;
  const existing = memory.get(key);
  if (existing) return existing;
  const promise = queue.catch(() => undefined).then(async () => {
    try {
      const cached = JSON.parse(sessionStorage.getItem(key) ?? 'null');
      if (cached && Date.now() - cached.time < 15 * 60_000 && Number.isSafeInteger(cached.value) && cached.value >= 0) return cached.value as number;
    } catch { /* Storage is optional. */ }
    const pause = Math.max(0, 350 - (Date.now() - lastRequest));
    if (pause) await new Promise((resolve) => setTimeout(resolve, pause));
    lastRequest = Date.now();
    const value = await readCount(site, path, start, end);
    try { sessionStorage.setItem(key, JSON.stringify({ time: Date.now(), value })); } catch { /* Storage is optional. */ }
    return value;
  });
  queue = promise;
  memory.set(key, promise);
  void promise.catch(() => { memory.delete(key); });
  return promise;
}

export async function loadViewCounts(root: ParentNode = document) {
  await Promise.all(Array.from(root.querySelectorAll<HTMLElement>('[data-counter-path]')).map(async (element) => {
    const output = element.querySelector<HTMLElement>('[data-counter-value]');
    if (!output || !element.dataset.counterSite || !element.dataset.counterPath) return;
    try {
      const today = element.dataset.counterPeriod === 'today';
      const count = today
        ? await readDailyCount(element.dataset.counterSite, recentDates(1)[0], getCount)
        : await getCount(element.dataset.counterSite, element.dataset.counterPath);
      output.textContent = count.toLocaleString('ko-KR');
      element.title = today ? '오늘 전체 페이지 방문 · UTC 기준(한국 시간 오전 9시에 날짜 변경) · 최대 4시간 지연' : '같은 세션의 반복 방문을 제외한 누적 방문 수 · 최대 4시간 지연';
    } catch (error) {
      output.textContent = '—';
      element.title = error instanceof CounterError && error.status === 404 ? '아직 집계되지 않은 페이지입니다.' : '방문 통계를 불러올 수 없습니다.';
    }
  }));
}
