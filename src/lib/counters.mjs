export class CounterError extends Error {
  constructor(status) {
    super(`Counter unavailable (${status})`);
    this.status = status;
  }
}

export function counterUrl(site, path, start, end) {
  const origin = new URL(site);
  if (origin.protocol !== 'https:' || !/^[a-z0-9-]+\.goatcounter\.com$/.test(origin.hostname) || origin.username || origin.password || origin.port) {
    throw new Error('Invalid GoatCounter site address');
  }
  const url = new URL(`/counter/${encodeURIComponent(path)}.json`, origin.origin);
  for (const [key, value] of [['start', start], ['end', end]]) {
    if (value !== undefined) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error('Invalid counter date');
      url.searchParams.set(key, value);
    }
  }
  return url.href;
}

export function parseCount(value) {
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return value;
  if (typeof value !== 'string' || !/^(?:\d+|\d{1,3}(?:[, .\u00a0\u202f]\d{3})+)$/.test(value)) {
    throw new Error('Invalid counter response');
  }
  const number = Number(value.replace(/[, .\u00a0\u202f]/g, ''));
  if (!Number.isSafeInteger(number)) throw new Error('Counter is too large');
  return number;
}

export async function readCount(site, path, start, end, fetcher = fetch) {
  const response = await fetcher(counterUrl(site, path, start, end), { signal: AbortSignal.timeout(8000) });
  // GoatCounter returns a JSON zero with 404 for a path with no recorded visits.
  if (response.status === 404) {
    try {
      const body = await response.json();
      if (parseCount(body.count) === 0 && parseCount(body.count_unique) === 0) return 0;
    } catch { /* Other 404 responses are unavailable, not zero visits. */ }
  }
  if (!response.ok) throw new CounterError(response.status);
  return parseCount((await response.json()).count);
}

export function recentDates(days, now = new Date()) {
  // The public TOTAL endpoint uses UTC, regardless of dashboard timezone.
  const today = now.toISOString().slice(0, 10);
  return Array.from({ length: days }, (_, index) => {
    const date = new Date(`${today}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() - (days - index - 1));
    return date.toISOString().slice(0, 10);
  });
}

export async function readDailyCount(site, date, reader = readCount, now = new Date()) {
  const today = now.toISOString().slice(0, 10);
  if (date === today) return reader(site, 'TOTAL', date);
  const next = new Date(`${date}T00:00:00Z`);
  next.setUTCDate(next.getUTCDate() + 1);
  const end = next.toISOString().slice(0, 10);
  // The end boundary is inclusive of the next day's midnight hour.
  // Subtract that hour so each graph bar covers exactly one UTC day.
  const [inclusive, boundary] = await Promise.all([
    reader(site, 'TOTAL', date, end),
    reader(site, 'TOTAL', end, end),
  ]);
  if (boundary > inclusive) throw new Error('Counter snapshots are inconsistent');
  return inclusive - boundary;
}
