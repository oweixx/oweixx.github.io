import { getCount } from './view-counts';
import { recentDates, readDailyCount, CounterError } from '../lib/counters.mjs';

type Point = { date: string; count: number | null };
const svgNamespace = 'http://www.w3.org/2000/svg';

function svgElement(name: string, attributes: Record<string, string | number>, content?: string) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (content) element.textContent = content;
  return element;
}

function showChart(points: Point[]) {
  const chart = document.getElementById('daily-chart')!;
  chart.replaceChildren();
  chart.setAttribute('aria-busy', 'false');
  const hasData = points.some((point) => point.count !== null);
  if (!hasData) {
    const message = document.createElement('p');
    message.className = 'empty-chart';
    message.textContent = '방문 기록을 불러올 수 없습니다. 잠시 후 다시 확인해주세요.';
    chart.appendChild(message);
    document.getElementById('daily-details')!.hidden = true;
    return;
  }
  const svg = svgElement('svg', { viewBox: '0 0 640 210', role: 'img', 'aria-label': `최근 ${points.length}일의 페이지 방문 추이` });
  const max = Math.max(1, ...points.map((point) => point.count ?? 0));
  const spacing = 580 / points.length;
  svg.appendChild(svgElement('line', { x1: 45, y1: 174, x2: 625, y2: 174, stroke: '#e9edf0' }));
  svg.appendChild(svgElement('line', { x1: 45, y1: 26, x2: 625, y2: 26, stroke: '#f1f3f5' }));
  svg.appendChild(svgElement('text', { x: 34, y: 30, fill: '#999', 'font-size': 11, 'text-anchor': 'end' }, String(max)));
  svg.appendChild(svgElement('text', { x: 34, y: 177, fill: '#999', 'font-size': 11, 'text-anchor': 'end' }, '0'));
  points.forEach((point, index) => {
    const x = 45 + spacing * index;
    const height = 148 * (point.count ?? 0) / max;
    const bar = svgElement('rect', { x: x + spacing * .22, y: point.count === null ? 169 : 174 - height, width: spacing * .56, height: point.count === null ? 5 : height, rx: 2, fill: point.count === null ? '#c9c9c9' : '#87b5dd' });
    bar.appendChild(svgElement('title', {}, `${point.date}: ${point.count === null ? '불러오기 실패' : point.count.toLocaleString('ko-KR') + '회'}`));
    svg.appendChild(bar);
    if (points.length === 7 || index === 0 || index === points.length - 1 || index % 7 === 0) {
      svg.appendChild(svgElement('text', { x: x + spacing / 2, y: 198, fill: '#999', 'font-size': 11, 'text-anchor': 'middle' }, point.date.slice(5).replace('-', '.')));
    }
  });
  chart.appendChild(svg);
  const table = document.getElementById('daily-values')!;
  table.replaceChildren(...points.map((point) => {
    const row = document.createElement('tr');
    const date = document.createElement('td');
    const count = document.createElement('td');
    date.textContent = point.date;
    count.textContent = point.count === null ? '불러오기 실패' : point.count.toLocaleString('ko-KR');
    row.append(date, count);
    return row;
  }));
  document.getElementById('daily-details')!.hidden = false;
}

export async function initialiseStats() {
  const root = document.getElementById('statistics');
  const site = root?.dataset.site;
  if (!root || !site) return;
  let generation = 0;
  const status = document.getElementById('stats-status')!;

  root.querySelectorAll<HTMLElement>('[data-total]').forEach(async (element) => {
    try { element.textContent = (await getCount(site, element.dataset.total!)).toLocaleString('ko-KR'); }
    catch { element.textContent = '—'; }
  });

  const ranking = document.getElementById('post-ranking')!;
  const rows = Array.from(ranking.querySelectorAll<HTMLElement>('[data-post-path]'));
  void Promise.all(rows.map(async (row) => {
    try {
      const count = await getCount(site, row.dataset.postPath!);
      row.querySelector('.rank-value')!.textContent = `${count.toLocaleString('ko-KR')}회`;
      return { row, count };
    } catch {
      row.querySelector('.rank-value')!.textContent = '불러오기 실패';
      return { row, count: null };
    }
  })).then((counts) => {
    const maximum = Math.max(1, ...counts.map((entry) => entry.count ?? 0));
    counts.sort((a, b) => (b.count ?? -1) - (a.count ?? -1)).forEach(({ row, count }) => {
      (row.querySelector('.rank-fill') as HTMLElement).style.width = `${100 * (count ?? 0) / maximum}%`;
      ranking.appendChild(row);
    });
  });

  async function loadDays(days: number) {
    const current = ++generation;
    status.textContent = '방문 기록을 불러오는 중입니다.';
    document.getElementById('daily-chart')!.setAttribute('aria-busy', 'true');
    let forbidden = false;
    let completed = 0;
    const points = await Promise.all(recentDates(days).map(async (date: string) => {
      try { return { date, count: await readDailyCount(site!, date, getCount) }; }
      catch (error) {
        if (error instanceof CounterError && error.status === 403) forbidden = true;
        return { date, count: null };
      }
      finally {
        completed++;
        if (current === generation) status.textContent = `방문 기록을 불러오는 중입니다. (${completed}/${days})`;
      }
    }));
    if (current !== generation) return;
    showChart(points);
    status.textContent = forbidden ? '공개 방문 통계를 아직 불러올 수 없습니다.' : points.some((point) => point.count === null) ? '일부 날짜의 기록을 불러올 수 없습니다. · UTC 기준' : '전체 페이지 방문 · 일별 집계 · UTC 기준';
  }

  root.querySelectorAll<HTMLButtonElement>('[data-days]').forEach((button) => {
    button.addEventListener('click', () => {
      root.querySelectorAll('[data-days]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      void loadDays(Number(button.dataset.days));
    });
  });
  await loadDays(7);
}
