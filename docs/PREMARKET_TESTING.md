# Premarket Testing Runbook

How to run a supervised testing session where several traders use the
dashboard against the paper-trading backend — e.g. during US premarket
(4:00–9:30 AM ET).

## What traders get

- The **AEGIS dashboard** in their browser at `http://<host-ip>:3001`, each
  logging in with their own username/password (JWT, 24-hour expiry).
- Live view of portfolio, positions, performance, risk metrics, news
  sentiment, and NSE (Kenya) market data.
- **No real money is at risk**: the configured brokers are the internal paper
  broker and Alpaca with `"paper": true`. Coinbase is disabled. Verify this in
  `config/config.json` → `brokers` before every session.

## One-time setup (host machine)

1. `.env` must exist at the repo root with at least:
   - `SECRET_KEY` — JWT signing key. **The API refuses all protected routes
     without it** (fails closed with 503).
   - `TRADING_FMP_API_KEY`, `TRADING_ALPHA_VANTAGE_API_KEY`,
     `TRADING_FINNHUB_API_KEY` — market data.
   - `TRADING_ALPACA_API_KEY` / `TRADING_ALPACA_API_SECRET` — Alpaca *paper*
     keys.
   - Optional: `MONITORING_PASSWORD` (basic-auth password for the :8080
     monitoring endpoints; falls back to `SECRET_KEY`).
2. Allow the ports through Windows Firewall (admin shell):

   ```
   netsh advfirewall firewall add rule name="TradingTest" dir=in action=allow protocol=tcp localport=3001,5001
   ```

3. Install frontend deps once: `cd frontend && npm install`.

## Create trader accounts

```powershell
.\.venv\Scripts\python.exe scripts\add_traders.py alice bob carol
```

Each account gets a random 16-character password, printed **once** — hand them
out privately. `--list` shows existing users, `--reset <name>` regenerates a
password. Accounts live in `users.json` (bcrypt hashes only).

## Start a session

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_premarket.ps1
```

The script builds the frontend, starts the backend (`src.agent.main`) and the
dashboard (`next start` on 3001, bound to 0.0.0.0), sets
`CORS_ALLOWED_ORIGINS` for the host's LAN IP, and prints the URLs to give the
traders. Use `-SkipBuild` to reuse the previous frontend build, and
`-Config config\<other>.json` to run an alternate configuration.

Traders then browse to `http://<host-ip>:3001` and log in.

> The dashboard calls the API on the **same hostname it was loaded from**,
> port 5001 (see `frontend/utils/apiBase.js`). To point it elsewhere — e.g. a
> reverse proxy — set `NEXT_PUBLIC_API_URL` before `npm run build`.

### Docker alternative

`docker compose up --build` starts backend (5001/8080), frontend (3001), and
Redis. Set `CORS_ALLOWED_ORIGINS=http://<host-ip>:3001` in `.env` first —
the compose file passes `.env` through, but nothing computes the LAN IP for
you as the PowerShell script does.

## Health checks before traders join

| Check | Command / URL | Expect |
|---|---|---|
| API up | `curl http://localhost:5001/api/health` | `{"status": "ok"}` |
| Auth works | POST `/api/login` with a trader credential | token returned |
| Monitoring | `http://localhost:8080/health` (basic auth, any user + `MONITORING_PASSWORD`) | `"ready": true` |
| Full smoke test | `.\.venv\Scripts\python.exe -m pytest tests\test_agent_smoke.py -q` | 1 passed |

## Premarket notes and limits

- Market data refreshes on `data_manager.update_interval` (60s in
  `config/config.json`); the dashboard polls the API every 20–30s. Quotes come
  from FMP/Finnhub/Alpha Vantage and may be delayed or sparse before the US
  open — that is a data-vendor property, not a fault.
- The trading loop runs every `trading_loop_interval` (60s). Alpaca paper
  orders placed outside regular hours follow Alpaca's extended-hours rules.
- All traders see the **same shared portfolio** — accounts control access,
  not per-trader books. Treat the session as collaborative observation and
  control of one paper account.
- Rate limit: 100 API requests/minute per client IP (`API_RATE_LIMIT` to
  change). A room of traders behind one NAT IP can hit this; raise it for the
  session if dashboards start showing 429s.

## Stopping

Close the two windows the start script opened (backend and `next start`), or
stop the compose stack with `docker compose down`. Paper-broker state persists
in `data/` between sessions.

## Known limitations (testing scope)

- HTTP only — do not expose these ports to the public internet. LAN or VPN
  only. Put a TLS reverse proxy (Caddy/nginx) in front before any wider
  exposure.
- No per-user roles yet: every login has the same (full) API access.
- `/api/strategy-performance` returns mock numbers pending database
  integration.
