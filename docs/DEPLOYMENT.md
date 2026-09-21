# Deployment runbook

The single starting point for getting this trading agent running on a server
and knowing it is actually working. Written for someone who is not a
developer; every command can be copied as it appears.

If you only read one section, read **Verify it is actually working**. A
running container and a working agent are different things, and most of the
faults found in this system looked healthy from the outside.

## Which path

| Your situation | Go to |
|---|---|
| You have a VPS with Coolify | **Path A** |
| You have a VPS with Docker but no Coolify | **Path B** |
| You want to try it on your own machine first | **Path C** |

All three run the same three containers: the Python trading agent (backend),
the Next.js dashboard (frontend), and Redis. All three trade **paper money**.

## Before you start, whichever path

**A server.** Two CPU cores and 4 GB of memory is comfortable. Compiling the
dashboard is the most memory-hungry moment; 2 GB works but is tight.

**Your keys.** See `.env.example` for the full list. At minimum you want your
Alpaca paper key and secret, plus one market data key.

**Three random strings.** Generate them yourself and keep them in a password
manager. Never let a tool or an assistant invent them for you:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Run it three times, for `SECRET_KEY`, `REDIS_PASSWORD` and
`MONITORING_PASSWORD`.

**A domain** pointing at the server, if you want HTTPS. Set the DNS record
before you deploy: certificate issuance fails if the name does not yet
resolve.

---

## Path A: Coolify

Follow **[COOLIFY_DEPLOYMENT.md](COOLIFY_DEPLOYMENT.md)**, which covers the
resource, the environment variables, the domains, the persistent storage and
the scheduled jobs in order.

If you would rather have an assistant click through it, paste the prompt in
**[COOLIFY_DEPLOY_PROMPT.md](COOLIFY_DEPLOY_PROMPT.md)** into Claude for
Chrome with Coolify open. That prompt is written to ask you for every secret
rather than inventing any.

One thing people get wrong: point Coolify at **`docker-compose.coolify.yml`**,
not the default `docker-compose.yml`. The default publishes ports to
`127.0.0.1`, which is correct for Path B and invisible to Coolify's proxy, so
the app would build and then be unreachable.

Then come back here for **Verify it is actually working**.

---

## Path B: plain Docker on a VPS

```bash
git clone https://github.com/CycloneKE/ai_trading_agents.git
cd ai_trading_agents
cp .env.example .env
```

Edit `.env` and fill in at least `SECRET_KEY`, `REDIS_PASSWORD`,
`MONITORING_PASSWORD`, your Alpaca paper keys, and
`CORS_ALLOWED_ORIGINS` set to the address you will reach the dashboard on,
including `https://` and with no trailing slash.

```bash
docker compose up -d --build
docker compose ps
```

The compose file binds every port to `127.0.0.1` on purpose, so nothing is
exposed to the internet directly. Put TLS in front of it:

```bash
caddy run --config deploy/Caddyfile.public
```

`deploy/Caddyfile.public` obtains a real certificate automatically for a
public domain. `deploy/Caddyfile` is the LAN equivalent with a local
certificate authority, for testing without a public name.

Then continue to **Verify it is actually working**.

---

## Path C: on your own machine

For trying it out, not for a real run. It has no TLS and no supervision.

```bash
cp .env.example .env     # fill in SECRET_KEY at minimum
pip install -r requirements.txt
python main.py --config config/config.json
```

The dashboard is a separate process:

```bash
cd frontend && npm install && npm run dev
```

Reach it at `http://localhost:3001`. Use `localhost`, not `127.0.0.1`: the
API's allowed-origins list contains the former, and a mismatch gives you a
dashboard that loads and shows nothing.

---

## Verify it is actually working

Do these in order. Each one answers a question the previous one cannot.

**1. Is the process alive?**

```bash
curl -fsS -u ":$MONITORING_PASSWORD" https://<your-health-domain>/health
```

**2. Can you log in?** Open the dashboard. If the login page refuses you, see
the troubleshooting table below; it is almost always `SECRET_KEY`.

**3. Will it actually trade?** This is the one that matters, and a running
container does not answer it:

```bash
python scripts/start_paper_run.py --days 90 --write-manifest
```

Under Coolify, run this from the resource's Terminal. It checks that the
strategies are real rather than placeholders, that the price feed returns
genuine quotes rather than generated ones, and that the configuration would
place a trade at all. Lines marked `[BLOCK]` stop the run being worth
starting; `[warn]` lines shape how you read the results.

It also records the run's start date, which is what later lets the weekly
report tell "nothing in three weeks" apart from "started an hour ago".

**4. Are the prices real?**

```bash
python scripts/check_data_freshness.py
```

Expect this to fail on a fresh install, and that is correct: the historical
store seeds itself with generated prices marked `source=synthetic`, and the
check refuses to call those real. It should go green once the scraper has
fetched genuine bars. If it never does, the scraper cannot reach the
exchange, and any Nairobi analysis is describing invented data.

---

## Day two

**Updating.** Push to your branch; Coolify rebuilds, or on Path B run
`git pull && docker compose up -d --build`. Your journals survive because
they live in named volumes rather than in the image. The agent restarts, and
its startup reconciliation checks every unresolved order against the broker
before trading resumes, so a redeploy mid-order does not lose track of it.

**The scheduled jobs.** Five of them, listed with exact cron expressions in
[COOLIFY_DEPLOYMENT.md](COOLIFY_DEPLOYMENT.md). The two that matter most are
the daily backup and the liveness check: one protects the record, the other
tells you the agent stopped.

**Reading the run.**

```bash
python scripts/paper_run_report.py --since 7
```

[PAPER_RUN_RUNBOOK.md](PAPER_RUN_RUNBOOK.md) explains what each section means
and what to do about each answer.

---

## Troubleshooting by symptom

Every row here is a fault that was actually found in this system. They share
a shape: something fails, something else reports success, and nothing says so.

| What you see | What it is | Fix |
|---|---|---|
| Login page refuses you; no obvious error | `SECRET_KEY` unset. The API starts, reports healthy, and answers every protected route with 503 | Set `SECRET_KEY` and redeploy |
| Dashboard loads but shows no data | `CORS_ALLOWED_ORIGINS` does not match the address you are using | Match it exactly, with `https://` and no trailing slash |
| Dashboard is blank or 404 | The frontend image did not build or is not running | Check the `frontend` container; the build now fails loudly rather than shipping an empty image |
| Journals are empty after an update | No volume mounted at `/app/data` | Confirm `trading_data` exists; without it every redeploy wipes the record |
| Container exits with `FATAL: /app/data is not writable` | A host directory was bind-mounted and is owned by root | Use a named volume, or `chown -R 10001:10001 <path>` |
| Agent runs for days and never trades | Could be correct, could be broken | Run `start_paper_run.py`, then read the "Why it holds" table in the weekly report. Reasons marked DEGRADED are faults; "by design" is the strategy being selective |
| `/health` returns 404 | Monitoring disabled in config | Check `monitoring.enabled`. This used to happen when the optional `psutil` library was missing; that no longer disables the endpoint |
| Freshness check fails on synthetic bars | The scraper cannot reach the exchange | Check outbound network from the server. Until fixed, NSE results are not real |
| Capacity check fails with "over LLM quota" | The universe outgrew the validation quota | Trim symbols, or raise the quota. Left alone, validation silently switches off |

---

## Rolling back

Coolify keeps previous deployments; redeploy an earlier one from its
interface. On Path B:

```bash
git log --oneline -10          # find the commit you want
git checkout <commit>
docker compose up -d --build
```

Your journals are unaffected either way, because they live in volumes rather
than in the image. That is deliberate: a rollback should undo the code, not
the record of what the agent did.

---

## What this deployment does not do

It trades **paper money**. `brokers.alpaca_broker.paper` is `true`, and the
readiness check refuses to start if any enabled broker is set to trade real
money.

`alpaca-trade-api` is deliberately not installed in the container. It is a
heavy package with strict version pins that conflict with the rest of the
stack, and it is untested here. Without it the agent falls back to its own
internal paper broker, which simulates fills locally rather than using your
Alpaca paper account. That is honest behaviour but it is not the same thing.

Library versions in the container are not pinned, so a rebuild months from now
may install different ones. The weekly CI run exists to catch that drift. If
you want a deployment reproducible for a full ninety days, freeze them once
after a good deploy with `pip freeze > scripts/requirements-lock.txt` and
install that file instead.

**Before any real money**, one thing still needs settling and it is not a
software problem: the Nairobi Securities Exchange transaction costs in
`config/config.json` are a placeholder at 1.7% per side, pending confirmation
from AIB-AXYS. On that exchange the figure is large enough to decide whether
a strategy is profitable at all.

---

## Older documents

These predate the current setup and contradict it in places. They are kept
for history; do not follow them for a deployment:

- `ENHANCED_DEPLOYMENT_GUIDE.md`
- `README_STARTUP.md`
- `PRODUCTION_READINESS_REPORT.md`
- `PAPER_TRADING_GUIDE.md`

The current set is this file, [COOLIFY_DEPLOYMENT.md](COOLIFY_DEPLOYMENT.md),
[COOLIFY_DEPLOY_PROMPT.md](COOLIFY_DEPLOY_PROMPT.md) and
[PAPER_RUN_RUNBOOK.md](PAPER_RUN_RUNBOOK.md).
