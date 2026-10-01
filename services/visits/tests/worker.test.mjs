import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

test('saved counts are immediate, concurrent loads survive, retries deduplicate and sockets receive updates', async () => {
  const origin = 'https://oweixx.github.io';
  const mf = new Miniflare(convertV4MiniflareOptions({
    modules: true,
    scriptPath: new URL('../src/index.js', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'),
    compatibilityDate: '2026-10-01',
    durableObjects: { VISITS: { className: 'VisitCounter', useSQLite: true } },
    bindings: { SITE_ORIGIN: origin },
  }));
  const visit = (path, id = crypto.randomUUID(), source = origin) => mf.dispatchFetch('https://counter.test/visit', { method: 'POST', headers: { Origin: source, 'Content-Type': 'application/json' }, body: JSON.stringify({ path, id }) });
  try {
    assert.equal((await (await mf.dispatchFetch('https://counter.test/counts')).json()).total, 0);
    const response = await mf.dispatchFetch('https://counter.test/live', { headers: { Origin: origin, Upgrade: 'websocket' } });
    const socket = response.webSocket;
    assert(socket);
    socket.accept();
    const receive = () => new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Socket update timed out')), 3000);
      socket.addEventListener('message', (event) => { clearTimeout(timer); resolve(JSON.parse(event.data)); }, { once: true });
    });
    assert.equal((await receive()).total, 0);
    const pushed = receive();
    const id = crypto.randomUUID();
    const first = await visit('/index.html', id);
    assert.equal(first.headers.get('Cache-Control'), 'no-store');
    const firstState = await first.json();
    assert.equal(firstState.total, 1);
    assert.equal(firstState.pages['/'], 1);
    assert.equal(firstState.today, 1);
    assert.equal(firstState.timeZone, 'Asia/Seoul');
    assert.equal((await pushed).total, 1);
    assert.equal((await (await visit('/', id)).json()).total, 1);
    assert.equal((await (await visit('/')).json()).total, 2, 'Reload with a new event ID counts once');
    const post = '/blog/2026/Research_rebuttal/';
    await Promise.all(Array.from({ length: 25 }, () => visit(post)));
    const state = await (await mf.dispatchFetch('https://counter.test/counts')).json();
    assert.equal(state.total, 27);
    assert.equal(state.today, 27);
    assert.equal(state.pages[post], 25);
    assert.equal(state.pages['/'], 2);
    assert.equal(state.day, new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date()));
    assert.equal((await mf.dispatchFetch('https://counter.test/counts')).status, 200);
    assert.equal((await (await mf.dispatchFetch('https://counter.test/counts')).json()).total, 27, 'Reads never add visits');
    assert.equal((await visit(post, crypto.randomUUID(), 'https://other.test')).status, 403);
    assert.equal((await visit('/write/')).status, 400);
    assert.equal((await visit('/assets/profile.png')).status, 400);
    assert.equal((await mf.dispatchFetch('https://counter.test/visit', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: 'broken' })).status, 400);
    socket.close();
  } finally { await mf.dispose(); }
});
