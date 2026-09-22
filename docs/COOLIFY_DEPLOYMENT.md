# Deploying on a VPS with Coolify

How to get the trading agent running on your own server, with the dashboard
reachable over HTTPS and the trading records surviving every future update.

Written for someone who is not a developer. Copy the commands exactly as they
appear.

## What you are deploying

Three containers that Coolify builds and runs for you:

| Container | What it is | Port |
|---|---|---|
| `backend` | The trading agent, its REST API and its health endpoint | 8080 health, 5001 API |
| `frontend` | The Next.js dashboard you log in to | 3001 |
| `redis` | A short-term cache the agent uses between cycles | internal only |

Coolify puts its own reverse proxy in front and obtains HTTPS certificates
automatically, so you do not need the Caddy configuration in `deploy/`. That
folder remains for people running without Coolify.

## Before you start

**A VPS.** Two CPU cores and 4 GB of memory is comfortable. The build step
compiles the dashboard, which is the most memory-hungry moment; 2 GB works but
can be tight.

**Coolify installed** on that VPS, and a domain pointing at the server's IP
address. If you want to use a subdomain such as `trading.murzak.tech`, create
an A record for it pointing at the VPS before you begin, because certificate
issuance fails if the name does not yet resolve.

**Your API keys to hand.** See `.env.example` for the full list.

## Step 1: create the resource in Coolify

In Coolify, choose **New Resource**, then **Docker Compose**, and connect it to
this GitHub repository.

When it asks for the compose file path, enter:

```
docker-compose.coolify.yml
```

This matters. The default `docker-compose.yml` publishes its ports to
`127.0.0.1`, which is correct for a plain VPS but wrong under Coolify: Coolify's
proxy reaches each container over the internal Docker network, and a port
published only to the host's loopback interface is invisible to it. The Coolify
file uses `expose` instead, which is what the proxy expects.

## Step 2: set the environment variables

Open the **Environment Variables** tab and add the following. The deployment
refuses to start without the first two, deliberately.

| Variable | What to put |
|---|---|
| `SECRET_KEY` | A long random string. This signs your dashboard login tokens. |
| `REDIS_PASSWORD` | A different long random string. |
| `CORS_ALLOWED_ORIGINS` | The full dashboard address, for example `https://trading.murzak.tech` |
| `TRADING_ALPACA_API_KEY` | Your Alpaca paper trading key |
| `TRADING_ALPACA_API_SECRET` | Your Alpaca paper trading secret |
| `TRADING_FINNHUB_API_KEY` | Market data |
| `TRADING_ALPHA_VANTAGE_API_KEY` | Market data |
| `MONITORING_PASSWORD` | A third random string. The health endpoint is authenticated, so you need this to check on the run. |

To generate the two random strings, run this on any machine and copy the
output:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

`SECRET_KEY` deserves particular attention. If it is missing, the API still
starts and reports itself healthy, but its authentication layer fails to load
and every protected route returns the error code 503. The symptom you would see
is a login page that never lets you in, with nothing obviously wrong in the
logs. Set it before the first deploy and this never happens.

`CORS_ALLOWED_ORIGINS` must match the dashboard address exactly, including
`https://` and with no trailing slash. A mismatch produces a dashboard that
loads but shows no data, because the browser blocks its calls to the API.

## Step 3: confirm the persistent storage

This is the step people skip and regret.

Coolify rebuilds the container on every push to your repository. Anything not
stored in a named volume is destroyed each time. The agent's order journal and
decision journal are SQLite database files under `/app/data`, and they are the
only record of what the agent did. Losing them resets your ninety-day
experiment to zero while everything still appears to work.

The compose file declares the volumes for you:

```yaml
volumes:
  - trading_data:/app/data
  - trading_logs:/app/logs
```

In Coolify's **Storages** tab you should see `trading_data` and `trading_logs`
listed after the first deploy. Confirm they are there.

One caution if you ever change these to a bind mount, meaning a path on the
host such as `/srv/trading/data`. The container runs as a non-root user with ID
10001, and Docker creates bind-mounted host directories owned by root, which
that user cannot write to. The container detects this and stops immediately
with an explanation rather than trading without a record. If you hit it, run:

```bash
sudo chown -R 10001:10001 /srv/trading/data
```

Named volumes, which is what the file uses by default, do not have this problem.

## Step 4: assign the domains

In the **Domains** section, map your domain to the `frontend` service on port
`3001`. That is the dashboard.

Optionally, map a second domain or subdomain to the `backend` service on port
`8080`. That is the health endpoint, and it is the cheapest way to find out
your agent died on day three rather than on day ninety. Anything that can make
an HTTP request can watch it.

Do not expose port 5001 publicly. The dashboard calls the API through Coolify's
proxy on the internal network; it does not need a public address.

## Step 5: deploy and check

Press **Deploy** and watch the build log. The first build takes several minutes
because it installs Python packages and compiles the dashboard.

Three things now fail loudly that used to fail silently, so a green build
genuinely means something:

- If the dashboard does not compile, the build stops. It no longer produces a
  working-looking image that serves a blank page.
- If the authentication libraries are missing from the image, the build stops.
- If the data directory is not writable, the container refuses to start and
  says exactly why.

Once it is up, check these in order.

**Is the agent alive?** Visit the health endpoint, or run:

```bash
curl -u :$MONITORING_PASSWORD https://<your-health-domain>/health
```

**Can you log in?** Open the dashboard domain. If the login page rejects you,
`SECRET_KEY` is missing or changed.

**Is it actually going to trade?** This is the question that matters, and a
running container does not answer it. Open a terminal into the backend
container from Coolify and run:

```bash
python scripts/start_paper_run.py --days 90 --write-manifest
```

That checks the strategies are real rather than placeholders, that the price
feed returns genuine quotes rather than synthetic ones, and that the
configuration would actually place a trade. It also records the run's start
time, which is what lets the weekly report tell "nothing for three weeks" apart
from "started an hour ago".

## Scheduled jobs

Three things should run on a schedule. Two run inside the container and go in
Coolify's **Scheduled Tasks** tab for the backend resource; the third has to
run from outside and lives in GitHub Actions.

Coolify scheduled tasks run a command inside the running container, which is
what these need: the container's Python environment and the mounted data
volume. Host cron would have to exec into the container and would break every
time the container is recreated.

### 1. Back up the journals, daily

In Coolify, **Scheduled Tasks** on the `backend` resource, add:

| Field | Value |
|---|---|
| Name | `Daily journal backup` |
| Command | `python scripts/backup_state.py --keep-days 30` |
| Frequency | `0 2 * * *` |
| Container | `backend` |

That cron expression means 02:00 every day, in UTC. Nairobi is UTC+3, so it
runs at 05:00 your time, comfortably outside both the Nairobi and New York
sessions.

The backup lands in `data/backups/`, inside the mounted volume, so it survives
redeploys. It uses SQLite's online backup interface, so it is safe to run while
the agent is trading, and it reads every copy back afterwards to confirm the
file is usable. A backup that cannot be opened is not a backup.

Be clear about what this protects. Backups sitting beside the journals survive
a redeploy, a bad configuration change and an accidental deletion, which are
the realistic failures. They do not survive losing the volume or the server.
For that, copy `data/backups/` off the machine periodically as well.

To check the most recent backup by hand:

```bash
python scripts/backup_state.py --verify-only
```

### 2. Weekly report, Monday morning

| Field | Value |
|---|---|
| Name | `Weekly paper run report` |
| Command | `python scripts/paper_run_report.py --since 7 --out data/reports/week-$(date +%Y-%m-%d).md` |
| Frequency | `0 4 * * 1` |
| Container | `backend` |

04:00 UTC on Mondays, which is 07:00 in Nairobi. Note the output path is under
`data/`, deliberately: anywhere else and the report is destroyed by the next
redeploy.

### 3. Liveness check, nominally every fifteen minutes

This one cannot run inside the container, because a container that has stopped
cannot report that it has stopped. The agent does have an internal dead-man's
switch, but it runs on a thread inside the process and dies with it.

`.github/workflows/liveness.yml` polls the health endpoint from GitHub's
runners and fails the workflow when the agent does not answer, which sends you
GitHub's normal failure notification. It retries three times with backoff
first, so a redeploy or a momentary network problem does not cry wolf.

To switch it on, add two repository secrets under **Settings, Secrets and
variables, Actions**:

| Secret | Value |
|---|---|
| `HEALTH_URL` | `https://<your health domain>/health` |
| `MONITORING_PASSWORD` | the same value the agent runs with |

Until both are set the workflow reports "not configured" and passes, rather
than failing every quarter of an hour.

This requires the health endpoint to be reachable from the internet, which is
why step 4 above suggested mapping a domain to the backend on port 8080. It
stays password protected.

**How much to trust it: measured, not assumed.**

The workflow asks for a run every fifteen minutes. Over the first 7.8 hours
after it was set up, GitHub delivered **two runs**, against roughly 31
requested. That is a 6% delivery rate, an average gap of 4.7 hours, and the
runs did not even land on the requested minutes.

| | Requested | Actually delivered |
|---|---|---|
| Interval | 15 minutes | ~4.7 hours |
| Runs in 7.8 hours | ~31 | 2 |
| Fire times | :07, :22, :37, :52 | :01, :41 |

An earlier version on `*/15` produced no scheduled runs at all in its first
hour, which is why the schedule now avoids the quarter hours. That change
helped, in that scheduled runs happen at all now. It did not make the
schedule dependable.

**So treat this workflow as a backstop, not as your monitor.** If the agent
dies at one in the morning you may not hear until five. For a ninety-day
unattended run that is probably tolerable; for anything you actually care
about it is not.

**Use a dedicated uptime service as the primary alarm.** Point it at the same
health endpoint. Free tiers from the usual providers check every one to five
minutes and actually honour that. The GitHub workflow costs nothing to leave
running alongside, so keep both: the external service tells you promptly, and
the workflow is a second pair of eyes if that service itself fails.

### 4. Capacity check, weekly

| Field | Value |
|---|---|
| Name | `Weekly capacity check` |
| Command | `python scripts/monitor_capacity.py --csv-dir data/nse_historical --strict` |
| Frequency | `30 4 * * 1` |
| Container | `backend` |

Catches the language model quota ceiling before validation quietly switches
itself off. This matters most right after you add symbols, which is exactly
when you would forget to check.

`--strict` is important. Without it the script only fails when a cycle
overruns its time budget, and a quota overrun leaves the cycle well inside
its budget, so a scheduled job would report success while the thing you are
watching for is happening.

### 5. Data freshness check, daily after the close

| Field | Value |
|---|---|
| Name | `Daily data freshness` |
| Command | `python scripts/check_data_freshness.py` |
| Frequency | `0 15 * * 1-5` |
| Container | `backend` |

15:00 UTC is 18:00 in Nairobi, after the NSE close. It checks three things
and exits non-zero on any of them, so the job goes visibly red:

- **Stale bars.** The scraper logs `0/9 symbols updated` when its sources are
  unreachable and carries on regardless. The threshold is four days by
  default, which absorbs a weekend plus a public holiday without crying wolf.
- **Synthetic bars.** The historical store seeds itself with generated prices
  so a fresh install has something to work with, and those rows are marked
  `source=synthetic`. Nothing had ever compared that marker against reality.
  Recency is not credibility: a generated bar dated today is worse than a
  real bar from last week, because it looks fine.
- **A dead live feed.** When the data layer falls back to synthetic quotes
  the agent correctly refuses to trade and records `fallback_price`. A run
  that held four thousand times because it could not see prices has not
  tested your strategy at all, and from the outside that is indistinguishable
  from a strategy that found nothing.

On a fresh install that has not scraped yet, `--allow-synthetic` stops it
failing on the seed data. Take that off once the scraper is working, or the
check stops telling you anything.

### 6. Repository automation, already running

Two things run in GitHub rather than on your server and need no setup:

- **Dependabot** opens grouped weekly pull requests for Python and frontend
  dependencies, and monthly ones for the workflow actions. This matters more
  here than usual because the container installs version ranges rather than
  pins, so what it gets drifts over time.
- **A weekly CI run on `main`**, Monday 05:00 UTC, with nothing changed. A
  green history plus a red scheduled run localises dependency drift
  immediately: the code did not change, so something underneath it did. It
  runs the tests, both image builds and the startup smoke test, and
  deliberately does not deploy or publish anything.

## Reading the run each week

From a terminal in the backend container:

```bash
python scripts/paper_run_report.py --since 7
```

`docs/PAPER_RUN_RUNBOOK.md` explains what the report says and what to do about
each answer.

## Updating the code later

Push to your branch and Coolify rebuilds. Your journals survive, because they
live in the `trading_data` volume rather than in the image.

Two things worth knowing:

**The agent restarts.** Any position it holds stays open at the broker. The
startup reconciliation checks every unresolved order against the broker before
trading resumes, so a redeploy mid-order does not lose track of it.

**Library versions are not pinned in the runtime image.** A rebuild three
months from now could install newer versions of pandas or numpy than the ones
your run started with. If you want a deployment that is reproducible for the
full ninety days, freeze them once after a successful deploy:

```bash
pip freeze > scripts/requirements-lock.txt
```

and change the Dockerfile to install that file instead. This is a
recommendation, not something the current setup does.

## One thing this deployment does not do

It trades paper money. `brokers.alpaca_broker.paper` is `true` in
`config/config.json`, and the readiness check refuses to start if any enabled
broker is set to trade real money.

Note also that `alpaca-trade-api` is deliberately not installed in the
container. It is a heavy package with strict version pins that conflict with
the rest of the stack, and it is untested here. Without it the agent falls back
to its own internal paper broker, which simulates fills locally rather than
using your Alpaca paper account. The behaviour is honest but it is not the same
thing, so if you specifically want Alpaca to handle the simulation, add it to
`scripts/requirements-runtime.txt` and test the build before relying on it.

Before any real money is involved, one thing still needs settling, and it is
not a software problem:

- The Nairobi Securities Exchange transaction costs in `config/config.json` are
  a placeholder at 1.7% per side, pending confirmation from AIB-AXYS. On the
  NSE that figure is large enough to turn an apparently profitable strategy
  into a losing one.

## How positions are sized

Worth understanding before you read any results, because three settings work
together and it is easy to misread one for another.

`trading.risk_per_trade` (0.5%) is the authority. It is how much of your
equity you lose if a position's stop is hit, not how large the position is.
The position size follows from it and from how volatile the instrument is:

    position = equity x risk_per_trade / (stop distance)

where the stop distance is `risk_limits.stop_loss_atr_mult` (2.5) multiplied by
the instrument's recent average daily range. A calm stock gets a large
position because its stop is close; a volatile one gets a small position
because its stop has to be far away. Both risk the same 0.5%.

`risk_limits.max_position_size` (5%) is a hard ceiling on the resulting
position, so no single name dominates the book regardless of how calm it looks.
Below roughly 4% daily range this ceiling is what decides the size.

`cash_policy.max_risk_multiplier` (2.5) stretches the risk budget, up to 2.5
times, when the portfolio is sitting on idle cash and a signal has real
conviction. It never shrinks a position and never overrides the 5% ceiling.

The live agent and the backtest now use this identical formula. Until
recently they did not: the live loop sized every position at a flat fraction
of equity and ignored volatility entirely, which meant backtest results
described a sizing method the running agent never used.
