# Deployment prompt for Claude for Chrome

Paste everything in the fenced block below into Claude for Chrome, with your
Coolify dashboard open in the active tab.

Before you start, have these ready. The assistant will ask for each one and
must never invent them:

- Your domain or subdomain for the dashboard, with DNS already pointing at the
  VPS. Certificate issuance fails if the name does not yet resolve.
- Your Alpaca **paper** API key and secret.
- Your market data keys (Finnhub, Alpha Vantage, FMP as applicable).
- Three long random strings. Generate them yourself on any machine:

  ```bash
  python3 -c "import secrets; print(secrets.token_urlsafe(48))"
  ```

  Run it three times: one for `SECRET_KEY`, one for `REDIS_PASSWORD`, one for
  `MONITORING_PASSWORD`. Keep them in your password manager.

---

```text
You are helping me deploy a self-hosted trading agent to my own VPS using
Coolify, which is open in the active tab. Work through the browser only.

## What we are deploying

A GitHub repository, CycloneKE/ai_trading_agents, as a Docker Compose
resource. It runs three containers: a Python trading agent (backend), a
Next.js dashboard (frontend), and Redis. It trades PAPER money only.

## Rules, in priority order

1. NEVER invent, guess, or auto-generate a secret, password, API key, domain
   name or URL. Ask me for each value and wait. If I have not given you a
   value, stop and ask rather than filling the field with a placeholder.
2. NEVER repeat a secret value back in the chat, and never type one into any
   site other than this Coolify instance. When confirming a field is filled,
   say "set" rather than showing the value.
3. NEVER click anything labelled Delete, Destroy, Remove, Reset or Danger
   Zone. If a step seems to require one, stop and ask me.
4. If a screen does not match what is described here, because Coolify's
   layout differs by version, describe what you actually see and ask, rather
   than clicking the nearest similar-looking thing.
5. After each step, confirm the result on screen before moving on. Report what
   you observed, not what you expected.

## Step 1: create the resource

Create a new resource of type Docker Compose, connected to the GitHub
repository CycloneKE/ai_trading_agents, branch main.

When it asks for the Docker Compose file path, enter exactly:

    docker-compose.coolify.yml

This is important and easy to get wrong. Do NOT use the default
docker-compose.yml. That file publishes its ports to 127.0.0.1, which is
correct for a plain server but invisible to Coolify's proxy, so the app would
build and then be unreachable. The .coolify.yml file uses `expose` instead,
which is what the proxy expects.

Confirm on screen that the compose path is set to docker-compose.coolify.yml
before continuing.

## Step 2: environment variables

Open the Environment Variables tab. Ask me for each value below, one at a
time, and enter them. Do not proceed to the next until I have answered.

Required, the deployment refuses to start without the first two:

- SECRET_KEY               (long random string, ask me)
- REDIS_PASSWORD           (a different long random string, ask me)
- MONITORING_PASSWORD      (a third long random string, ask me)
- CORS_ALLOWED_ORIGINS     (ask me; it must be the full dashboard address
                            including https:// and with NO trailing slash)

Broker and market data, ask me for each; if I say I do not have one, skip it
and note which ones were skipped:

- TRADING_ALPACA_API_KEY
- TRADING_ALPACA_API_SECRET
- TRADING_FINNHUB_API_KEY
- TRADING_ALPHA_VANTAGE_API_KEY
- TRADING_FMP_API_KEY

Two of these have consequences worth knowing, so tell me if I skip them:

- Without SECRET_KEY the API still starts and reports itself healthy, but its
  authentication layer fails to load and every protected page returns error
  503. The symptom is a login page that silently refuses you.
- If CORS_ALLOWED_ORIGINS does not match the dashboard address exactly, the
  dashboard loads but shows no data, because the browser blocks its calls to
  the API.

When finished, confirm the list of variable NAMES that are now set. Do not
show any values.

## Step 3: domains

In the Domains section:

- Map my domain to the `frontend` service on port 3001. This is the dashboard.
- Ask me whether I also want a second domain or subdomain mapped to the
  `backend` service on port 8080. That is the health endpoint, and it is what
  lets an external monitor tell me the agent has stopped. Recommend it.

Do NOT expose port 5001 publicly. The dashboard reaches the API over Coolify's
internal network and does not need a public address for it.

Confirm the domain mappings on screen before continuing.

## Step 4: deploy

Press Deploy and watch the build log. The first build takes several minutes:
it installs Python packages and compiles the dashboard.

Three things fail loudly by design, so if the build goes red, read the error
and tell me which of these it is:

- The dashboard failing to compile now stops the build, rather than producing
  an image that serves a blank page.
- Missing authentication libraries now stop the build.
- An unwritable data directory stops the container from starting, with a
  message explaining why.

Report the outcome. If it failed, quote the actual error text.

## Step 5: confirm persistent storage

Open the Storages tab. Confirm that two volumes appear:

    trading_data   mounted at /app/data
    trading_logs   mounted at /app/logs

This matters more than it sounds. Coolify rebuilds the image on every push to
the repository, and anything not in a named volume is destroyed each time.
Those volumes hold the trading journals, which are the only record of what the
agent did. If they are missing, stop and tell me: the deployment will appear
to work while silently resetting its records on every update.

## Step 6: scheduled tasks

Open the Scheduled Tasks tab for the `backend` resource and add these four,
exactly as written. All times are UTC.

1. Name: Daily journal backup
   Command: python scripts/backup_state.py --keep-days 30
   Frequency: 0 2 * * *

2. Name: Weekly paper run report
   Command: python scripts/paper_run_report.py --since 7 --out data/reports/week-$(date +%Y-%m-%d).md
   Frequency: 0 4 * * 1

3. Name: Weekly capacity check
   Command: python scripts/monitor_capacity.py --csv-dir data/nse_historical --strict
   Frequency: 30 4 * * 1

4. Name: Daily data freshness
   Command: python scripts/check_data_freshness.py
   Frequency: 0 15 * * 1-5

Confirm all four appear in the list, and tell me if the container field needs
setting to `backend` for any of them.

## Step 7: verify it is actually working

A running container is not the same as a working agent. Check these in order
and report each result.

a) If Coolify offers a Terminal or Execute Command feature for the backend
   container, run:

       python scripts/start_paper_run.py --days 90 --write-manifest

   This checks that the strategies are real rather than placeholders, that the
   price feed returns genuine quotes, and that the configuration would place a
   trade at all. Report the full output. Lines marked [BLOCK] are problems;
   lines marked [warn] are worth knowing but do not stop the run.

   If Coolify has no terminal feature, say so and I will run it myself.

b) Open the dashboard domain in a new tab and confirm the login page loads.
   Do not attempt to log in with credentials I have not given you.

c) If we mapped a health domain, confirm it responds. It is password
   protected, so it will prompt for credentials; do not enter any.

## When you are done

Give me a short summary containing:

- Whether the deployment succeeded, and the live URLs.
- Which environment variables are set, by NAME only, and which I skipped.
- Whether both storage volumes exist.
- Whether all four scheduled tasks were created.
- The output of the readiness check, or a note that I need to run it myself.
- Anything you could not complete and why.

Do not tell me the deployment is finished if any step was skipped or
uncertain. Say plainly what is outstanding.
```

---

## After the browser work

Two things remain that Coolify cannot do, both in GitHub, under
**Settings, Secrets and variables, Actions**:

| Secret | Value |
|---|---|
| `HEALTH_URL` | `https://<your health domain>/health` |
| `MONITORING_PASSWORD` | the same value you gave Coolify |

These switch on the liveness workflow, which checks that the agent is still
answering and emails you when it stops. Until both are set the workflow runs,
reports "not configured", and passes without watching anything.

It asks GitHub for a check every fifteen minutes and GitHub delivers roughly
one every five hours, so treat it as a backstop rather than your alarm. For
prompt notice, point a dedicated uptime service at the same health endpoint;
see the "How much to trust it" note in
[COOLIFY_DEPLOYMENT.md](COOLIFY_DEPLOYMENT.md).

Then read the run weekly:

```bash
python scripts/paper_run_report.py --since 7
```

`docs/PAPER_RUN_RUNBOOK.md` explains what each section of that report means.
