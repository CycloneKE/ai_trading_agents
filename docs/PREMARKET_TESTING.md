# Premarket Testing Runbook

How to run a supervised testing session where several traders use the
dashboard against the paper-trading backend — e.g. during US premarket
(4:00–9:30 AM ET).

## What traders get

- The **AEGIS dashboard** in their browser at `http://<host-ip>:3001`, each
  logging in with their own username/password (JWT, 24-hour expiry).
- Live view of portfolio, positions, performance, risk metrics, news
  sentiment, and NSE (Kenya) market data.
- A **market session clock** in the header (ET time, premarket/open/after-hours
  state, countdown to the next transition) and a **Getting Started guide**
  that opens automatically on first login — tab overview, metric glossary
  (VaR, Sharpe, drawdown...), and session ground rules. Reopenable via the
  `?` header button.
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

## Preflight (automatic, or run it yourself)

`start_premarket.ps1` runs `scripts/preflight.py` before starting anything and
aborts on blocking failures. You can also run it directly:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py          # static checks
.\.venv\Scripts\python.exe scripts\preflight.py --live   # + probe running stack
```

It verifies env keys, config sanity, **that every enabled broker is
paper-mode**, trader accounts, the frontend build, and (with `--live`) API and
monitoring health.

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

### Automated startup (no 4 AM ops)

Register a Windows scheduled task so the stack is already running when
traders arrive (times are local machine time — 4:00 AM ET premarket is
11:00 AM in Nairobi during EDT):

```powershell
# from an elevated PowerShell
powershell -ExecutionPolicy Bypass -File scripts\schedule_premarket.ps1 -At 10:45
powershell ... schedule_premarket.ps1 -Status    # check next/last run
powershell ... schedule_premarket.ps1 -Remove    # unregister
```

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

## Strategy execution modes

`strategy_execution_mode` in `config/config.json` (default `"blend"`):

- `blend` — strategies vote, one ensemble order per symbol (tagged
  `ensemble` in attribution).
- `parallel` — every strategy agreeing with the validated ensemble
  direction places its own tagged order and keeps an independent P&L book
  on the symbol (see Strategy Attribution on the dashboard). Dissenting
  strategies skip, so the account never trades against itself in a cycle;
  the per-symbol dollar cap is split between the agreeing strategies.

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

## TLS (recommended)

`deploy/Caddyfile` puts a TLS reverse proxy in front of both the dashboard
and the API — traders browse to `https://<host-ip>` and JWTs stop crossing
the LAN in cleartext:

```powershell
caddy run --config deploy\Caddyfile
```

When the dashboard is served through Caddy it calls the API on the same
origin, so set `NEXT_PUBLIC_API_URL=https://<host-ip>` before `npm run build`
(the API is proxied under `/api` on port 443).

## Login protection

- 5 failed logins for a username within 15 minutes lock the account for
  15 minutes (`LOGIN_MAX_FAILURES` / `LOGIN_LOCKOUT_SECONDS` to tune).
  Failures and lockouts are written to the agent log.
- Tokens expire after 8 hours (`TOKEN_TTL_HOURS` to change).

## Known limitations (testing scope)

- Without the Caddy proxy the stack is HTTP-only — do not expose these
  ports to the public internet. LAN or VPN only.
- No per-user roles yet: every login has the same (full) API access.
- `/api/strategy-performance` returns mock numbers pending database
  integration.
