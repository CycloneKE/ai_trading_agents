// Resolves the trading API base URL.
// Priority: NEXT_PUBLIC_API_URL (baked in at build time) → same hostname the
// dashboard was loaded from, on the API port (so LAN users reach the server,
// not their own machine) → localhost during SSR.
const API_PORT = process.env.NEXT_PUBLIC_API_PORT || '5001';

export function getApiBase() {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== 'undefined') {
    // Served over HTTPS means a reverse proxy (deploy/Caddyfile) fronts both
    // the dashboard and the API — use same-origin so /api routes through it.
    if (window.location.protocol === 'https:') return '';
    return `${window.location.protocol}//${window.location.hostname}:${API_PORT}`;
  }
  return `http://localhost:${API_PORT}`;
}
