// Public endpoint; authentication credentials stay in Cloudflare's tooling.
export const visitsEndpoint = import.meta.env.PUBLIC_VISITS_ENDPOINT ?? 'https://oweixx-visits.oweixx-personal-site.workers.dev';
export const trackedHostname = 'oweixx.github.io';
