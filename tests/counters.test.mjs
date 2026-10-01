import { test } from 'node:test';
import assert from 'node:assert/strict';
import { counterUrl, parseCount, readCount, readDailyCount, recentDates, CounterError } from '../src/lib/counters.mjs';

const site = 'https://oweixx.goatcounter.com';

test('counter requests preserve post paths and formatted service counts', () => {
  const url = new URL(counterUrl(site, '/blog/2026/Research_rebuttal/', '2026-09-30', '2026-10-01'));
  assert.equal(decodeURIComponent(url.pathname), '/counter//blog/2026/Research_rebuttal/.json');
  assert.equal(url.searchParams.get('start'), '2026-09-30');
  for (const value of ['1,234', '1.234', '1\u202f234', '1234', 1234]) assert.equal(parseCount(value), 1234);
  for (const value of ['1,23', '-1', 'NaN', null, 1.5, '9007199254740992']) assert.throws(() => parseCount(value));
  assert.throws(() => counterUrl('https://goatcounter.com.evil.test', 'TOTAL'));
});

test('known empty paths are zero; denied, malformed and failed responses remain errors', async () => {
  const response = (status, body) => async () => new Response(body, { status });
  assert.equal(await readCount(site, '/', undefined, undefined, response(404, '{"count":"0","count_unique":"0"}')), 0);
  assert.equal(await readCount(site, 'TOTAL', undefined, undefined, response(200, '{"count":"1,234"}')), 1234);
  await assert.rejects(readCount(site, '/', undefined, undefined, response(403, 'Forbidden')), (error) => error instanceof CounterError && error.status === 403);
  await assert.rejects(readCount(site, '/', undefined, undefined, response(404, '<html>Not found</html>')), (error) => error.status === 404);
  await assert.rejects(readCount(site, '/', undefined, undefined, response(200, '{"count":"broken"}')));
  await assert.rejects(readCount(site, '/', undefined, undefined, async () => { throw new Error('Network failed'); }));
});

test('graph dates follow the public endpoint UTC boundary across months and leap days', () => {
  assert.deepEqual(recentDates(2, new Date('2026-10-02T01:00:00+09:00')), ['2026-09-30', '2026-10-01']);
  assert.deepEqual(recentDates(3, new Date('2024-03-01T12:00:00Z')), ['2024-02-28', '2024-02-29', '2024-03-01']);
});

test('historical daily counts exclude the inclusive next-midnight hour', async () => {
  const requests = [];
  const reader = async (...args) => {
    requests.push(args);
    // Includes 3 views in the next day's midnight hour and 12 during this day.
    return args[2] === '2026-09-30' ? 15 : 3;
  };
  const now = new Date('2026-10-01T08:00:00Z');
  assert.equal(await readDailyCount(site, '2026-09-30', reader, now), 12);
  assert.deepEqual(requests, [[site, 'TOTAL', '2026-09-30', '2026-10-01'], [site, 'TOTAL', '2026-10-01', '2026-10-01']]);
  await assert.rejects(readDailyCount(site, '2026-09-30', async (_site, _path, start) => start === '2026-09-30' ? 1 : 2, now), /inconsistent/);
});

test('today starts at UTC midnight and uses one request without an end boundary', async () => {
  const requests = [];
  const count = await readDailyCount(site, '2026-10-01', async (...args) => { requests.push(args); return 8; }, new Date('2026-10-01T08:00:00Z'));
  assert.equal(count, 8);
  assert.deepEqual(requests, [[site, 'TOTAL', '2026-10-01']]);
});
