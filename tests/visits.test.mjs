import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseSnapshot, recentDates } from '../src/lib/visits.mjs';

test('confirmed snapshots preserve site, daily and individual page counts', () => {
  const state = { total: 12, today: 3, day: '2026-10-01', timeZone: 'Asia/Seoul', pages: { '/': 9, '/blog/': 3 }, days: { '2026-10-01': 3 } };
  assert.equal(parseSnapshot(state), state);
  for (const count of [-1, 1.5, NaN, '12', Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => parseSnapshot({ ...state, total: count }));
    assert.throws(() => parseSnapshot({ ...state, pages: { '/': count } }));
  }
  assert.throws(() => parseSnapshot({ ...state, days: null }));
});

test('chart dates follow the server Korean date across month, year and leap day boundaries', () => {
  assert.deepEqual(recentDates(2, '2026-10-01'), ['2026-09-30', '2026-10-01']);
  assert.deepEqual(recentDates(2, '2027-01-01'), ['2026-12-31', '2027-01-01']);
  assert.deepEqual(recentDates(3, '2024-03-01'), ['2024-02-28', '2024-02-29', '2024-03-01']);
});
