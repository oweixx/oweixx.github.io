import { visitsEndpoint, trackedHostname } from '../config/analytics';
import { parseSnapshot } from '../lib/visits.mjs';

export type VisitSnapshot = { total: number; today: number; day: string; timeZone: string; pages: Record<string, number>; days: Record<string, number> };
let current: VisitSnapshot | undefined;
let initial: Promise<VisitSnapshot> | undefined;
let socket: WebSocket | undefined;
let reconnect: ReturnType<typeof setTimeout> | undefined;
let failures = 0;

function applySnapshot(snapshot: VisitSnapshot) {
  if (current && (snapshot.total < current.total || snapshot.total === current.total && snapshot.day < current.day)) return;
  current = snapshot;
  document.querySelectorAll<HTMLElement>('[data-counter-path]').forEach((element) => {
    const output = element.querySelector('[data-counter-value]');
    if (!output) return;
    const today = element.dataset.counterPeriod === 'today';
    const count = today ? snapshot.today : element.dataset.counterPath === 'TOTAL' ? snapshot.total : snapshot.pages[element.dataset.counterPath!] ?? 0;
    output.textContent = count.toLocaleString('en-US');
    element.title = today ? 'Visits today · Resets at midnight in Korea' : 'Page visits · Each page load counts once';
  });
  document.dispatchEvent(new CustomEvent('visits:update', { detail: snapshot }));
}

async function requestSnapshot(path: string, body?: { path: string; id: string }) {
  const response = await fetch(`${visitsEndpoint}${path}`, {
    method: body ? 'POST' : 'GET', cache: 'no-store', credentials: 'omit',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(10000),
  });
  if (!response.ok) throw new Error(`Visits unavailable (${response.status})`);
  return parseSnapshot(await response.json()) as VisitSnapshot;
}

function connectLive() {
  if (document.hidden || socket && socket.readyState < WebSocket.CLOSING) return;
  const url = new URL(`${visitsEndpoint}/live`);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  let connection: WebSocket;
  try { connection = new WebSocket(url); }
  catch { return; }
  socket = connection;
  connection.addEventListener('open', () => { failures = 0; });
  connection.addEventListener('message', (event) => {
    try { applySnapshot(parseSnapshot(JSON.parse(event.data))); } catch { /* Keep the last confirmed value. */ }
  });
  connection.addEventListener('close', () => {
    if (socket !== connection) return;
    socket = undefined;
    if (!document.hidden) reconnect = setTimeout(connectLive, Math.min(30000, 1000 * 2 ** failures++));
  });
}

export function initialiseVisits(): Promise<VisitSnapshot> {
  if (initial) return initial;
  initial = (async () => {
    if (!visitsEndpoint) throw new Error('Visits service is not configured');
    const marker = document.getElementById('visit-context');
    const tracked = !!marker && location.hostname === trackedHostname;
    let snapshot: VisitSnapshot;
    if (tracked) {
      // New ID per page load; retries reuse it to avoid double counting.
      const body = { path: marker.dataset.path!, id: crypto.randomUUID() };
      let result: VisitSnapshot | undefined;
      for (let attempt = 0; attempt < 3; attempt++) {
        try { result = await requestSnapshot('/visit', body); break; }
        catch (error) {
          if (attempt === 2) throw error;
          await new Promise((resolve) => setTimeout(resolve, 400 * 2 ** attempt));
        }
      }
      snapshot = result!;
    } else snapshot = await requestSnapshot('/counts');
    applySnapshot(snapshot);
    if (tracked) {
      connectLive();
      document.addEventListener('visibilitychange', () => {
        if (reconnect) clearTimeout(reconnect);
        if (document.hidden) socket?.close();
        else {
          void requestSnapshot('/counts').then(applySnapshot).catch(() => undefined);
          connectLive();
        }
      });
      function scheduleMidnight() {
        const now = new Date();
        const midnight = new Date(now.getTime());
        midnight.setUTCHours(15, 0, 0, 0);
        if (midnight <= now) midnight.setUTCDate(midnight.getUTCDate() + 1);
        setTimeout(() => {
          void requestSnapshot('/counts').then(applySnapshot).catch(() => undefined);
          scheduleMidnight();
        }, midnight.getTime() - now.getTime() + 1000);
      }
      scheduleMidnight();
    }
    return snapshot;
  })();
  return initial;
}

export async function loadViewCounts() {
  try { await initialiseVisits(); }
  catch {
    document.querySelectorAll<HTMLElement>('[data-counter-path]').forEach((element) => {
      element.title = 'Visit counts are temporarily unavailable';
      const output = element.querySelector('[data-counter-value]');
      if (output) output.textContent = '—';
    });
  }
}
