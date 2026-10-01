import { DurableObject } from 'cloudflare:workers';

const timezone = 'Asia/Seoul';
const dayKey = (date = new Date()) => new Intl.DateTimeFormat('en-CA', { timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit' }).format(date);

function canonicalPath(path) {
  if (typeof path !== 'string') return null;
  path = path.replace(/\/index\.html$/, '/');
  if (!path.endsWith('/')) path += '/';
  return /^(?:\/|\/blog\/|\/stats\/|\/blog\/\d{4}\/[A-Za-z0-9_-]+\/)$/.test(path) ? path : null;
}

function json(value, status = 200, origin) {
  return Response.json(value, { status, headers: {
    'Cache-Control': 'no-store',
    'Access-Control-Allow-Origin': origin ?? '*',
    'Vary': 'Origin',
    'X-Content-Type-Options': 'nosniff',
  } });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const origin = request.headers.get('Origin');
    if (request.method !== 'GET' && origin && origin !== env.SITE_ORIGIN) return json({ error: 'Origin not allowed' }, 403);
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: {
        'Access-Control-Allow-Origin': env.SITE_ORIGIN,
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Vary': 'Origin',
      } });
    }
    const counter = env.VISITS.getByName('oweixx.github.io');
    if (request.method === 'GET' && url.pathname === '/counts') {
      return json(await counter.snapshot(), 200, origin);
    }
    if (request.method === 'GET' && url.pathname === '/live' && request.headers.get('Upgrade')?.toLowerCase() === 'websocket') {
      if (origin !== env.SITE_ORIGIN) return json({ error: 'Origin required' }, 403);
      return counter.fetch(request);
    }
    if (request.method !== 'POST' || url.pathname !== '/visit') return json({ error: 'Not found' }, 404, origin);
    if (origin !== env.SITE_ORIGIN) return json({ error: 'Origin required' }, 403);
    if (!request.headers.get('Content-Type')?.startsWith('application/json')) return json({ error: 'JSON required' }, 415, origin);
    if (Number(request.headers.get('Content-Length')) > 1024) return json({ error: 'Request too large' }, 413, origin);
    let data;
    try {
      const body = await request.text();
      if (body.length > 1024) return json({ error: 'Request too large' }, 413, origin);
      data = JSON.parse(body);
    } catch { return json({ error: 'Invalid JSON' }, 400, origin); }
    const path = canonicalPath(data?.path);
    if (!path || typeof data?.id !== 'string' || !/^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i.test(data.id)) {
      return json({ error: 'Invalid visit' }, 400, origin);
    }
    // No IP addresses, user identifiers, or referrers are stored.
    return json(await counter.record(path, data.id), 200, origin);
  },
};

export class VisitCounter extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    this.sql.exec('CREATE TABLE IF NOT EXISTS counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, path TEXT NOT NULL, recorded_at INTEGER NOT NULL)');
    this.sql.exec('CREATE INDEX IF NOT EXISTS events_time ON events(recorded_at)');
  }

  snapshot() {
    const today = dayKey();
    const entries = this.sql.exec('SELECT key, value FROM counters').toArray();
    const pages = {};
    const days = {};
    let total = 0;
    for (const { key, value } of entries) {
      if (key === 'total') total = value;
      else if (key.startsWith('page:')) pages[key.slice(5)] = value;
      else if (key.startsWith('day:')) days[key.slice(4)] = value;
    }
    return { total, today: days[today] ?? 0, day: today, timeZone: timezone, pages, days };
  }

  record(path, id) {
    let changed = false;
    this.ctx.storage.transactionSync(() => {
      const existing = this.sql.exec('SELECT path FROM events WHERE id = ?', id).toArray()[0];
      if (existing) return;
      const now = Date.now();
      this.sql.exec('INSERT INTO events (id, path, recorded_at) VALUES (?, ?, ?)', id, path, now);
      for (const key of ['total', `day:${dayKey()}`, `page:${path}`]) {
        this.sql.exec('INSERT INTO counters (key, value) VALUES (?, 1) ON CONFLICT(key) DO UPDATE SET value = value + 1', key);
      }
      changed = true;
    });
    const state = this.snapshot();
    if (changed) {
      if (state.total % 100 === 0) this.sql.exec('DELETE FROM events WHERE recorded_at < ?', Date.now() - 86400_000);
      const message = JSON.stringify(state);
      for (const socket of this.ctx.getWebSockets()) {
        try { socket.send(message); } catch { socket.close(1011, 'Reconnect'); }
      }
    }
    return state;
  }

  fetch() {
    const [client, server] = Object.values(new WebSocketPair());
    this.ctx.acceptWebSocket(server);
    server.send(JSON.stringify(this.snapshot()));
    return new Response(null, { status: 101, webSocket: client });
  }

  webSocketClose(socket, code, reason) { socket.close(code, reason); }
  webSocketError(socket) { socket.close(1011, 'Reconnect'); }
}
