# Connecting the Alpaca paper sandbox: prompt for Claude for Chrome

Use this after the broker-routing fix is merged and deployed. It finishes the
job `docs/COOLIFY_DEPLOY_PROMPT.md` started, by pointing the agent at a real
broker sandbox instead of the internal simulator.

## Why this step exists

`config/config.json` has always marked Alpaca as the `primary` broker, meaning
the venue that executes trades. It never actually ran. The connector imported
an SDK that was not installed in the image, so the broker type was never
registered, the broker was dropped with one warning, and orders were filled by
the built-in paper simulator instead. The readiness gate read the same config
file, saw Alpaca marked as paper mode, and reported READY.

That is now fixed three ways: the connector needs no SDK, a primary broker that
cannot be built stops the agent instead of being replaced, and the gate checks
what was actually created rather than what the file asked for. What remains is
supplying the credentials.

## Before you start

Have these ready. The assistant will ask and must never invent them.

- **Alpaca paper API key and secret.** Generate them yourself at
  <https://app.alpaca.markets/paper/dashboard/overview>, under *API Keys* in
  the right-hand panel, using **Generate New Key**. Make sure the page says
  *Paper* and not *Live*. The secret is shown once only, so copy it before
  closing the dialog.
- **Your Coolify dashboard open in the active tab**, signed in.
- Your domain, which is `murzaktech.tech`, with `trading.` and `health.`
  subdomains already resolving.

---

```text
You are helping me finish configuring a self-hosted trading agent on my own
VPS using Coolify, which is open in the active tab. Work through the browser
only.

## Context

The application is already deployed and running. The backend starts cleanly.
What is missing is the Alpaca paper trading credentials, without which the
agent has a broker object but no authenticated session, so it cannot place
orders.

The repository is CycloneKE/ai_trading_agents, deployed as a Docker Compose
resource with three containers: a Python trading agent (backend), a Next.js
dashboard (frontend), and Redis.

This trades PAPER money only. Alpaca's paper endpoint is
https://paper-api.alpaca.markets and the code refuses to use any other host
unless the config explicitly turns paper mode off. If at any point you see
the live endpoint https://api.alpaca.markets in a log or a config field,
STOP and tell me immediately.

## Rules, in priority order

1. NEVER invent, guess, or auto-generate an API key, secret, password,
   domain or URL. Ask me for each value and wait for my answer. If I have
   not given you a value, stop and ask rather than filling the field with a
   placeholder.
2. NEVER repeat a secret value back in the chat, and never type one into any
   site other than this Coolify instance. In particular, never type the
   Alpaca secret into the trading dashboard itself.
3. NEVER enable, suggest, or switch on live trading. If you find a setting
   that would do so, leave it alone and tell me.
4. Report what you actually observe. If a step fails, say so and show me the
   error text. Do not describe an action as successful unless you have seen
   the confirmation in the page. If a browser tool is unavailable or a page
   will not render, say so rather than inferring the result from network
   activity.
5. Work one step at a time and wait for my confirmation before moving on.

## Step 1: Redeploy onto the merged fix

In Coolify, open the application and check which commit the running
deployment is on. If it predates the broker-routing fix, trigger a redeploy
from the latest main branch. Use the rebuild-without-cache option if one is
offered, because the change includes a dependency file.

Wait for the deployment to finish and tell me the result before continuing.

## Step 2: Add the two environment variables

In the backend application's Configuration, open Environment Variables. Add
these two, exactly as spelled, marked as secret or build-time-hidden if
Coolify offers that option:

  TRADING_ALPACA_API_KEY
  TRADING_ALPACA_API_SECRET

Ask me for each value in turn. I will paste them. Do not echo them back.

Check first whether either variable already exists with an empty or
placeholder value, and update rather than duplicate it. Two variables of the
same name is a silent trap: which one wins is not defined.

Tell me when both are saved, confirming only that each field is now
non-empty. Do not display the values.

## Step 3: Redeploy and read the logs

Redeploy the backend, then open its logs and look for this line:

  Successfully connected to Alpaca at https://paper-api.alpaca.markets
  (paper=True, account status=ACTIVE)

Report to me exactly which of the following you see:

- That line, with paper=True  ->  success, continue to step 4.
- "Alpaca API keys not provided"  ->  the variables did not reach the
  container. Check spelling and that they are attached to the backend
  service, not the frontend.
- A 401 or 403 from /v2/account  ->  the key and secret are valid syntax but
  rejected. Most often this means live keys were pasted into a paper setup,
  or the key was revoked. Ask me to regenerate them.
- "Broker(s) marked primary could not be created"  ->  the deployment is
  still on the old image. Go back to step 1.
- The live URL https://api.alpaca.markets anywhere  ->  STOP and tell me.

## Step 4: Run the readiness gate

Open a terminal against the backend container in Coolify and run:

  python scripts/start_paper_run.py --probe-broker --probe-data

Paste the whole output back to me. I am looking specifically at these four
lines, which must all read [ok]:

  [ok] all enabled brokers are paper mode
  [ok] every enabled broker can be created
  [ok] primary broker is the configured one    alpaca_broker
  [ok] broker connected

The third line must say alpaca_broker. If it names paper_broker instead,
the routing fix is not live and we are back to step 1.

Lines marked [warn] are acceptable and expected. Lines marked [BLOCK] are
not. Report any [BLOCK] line verbatim and stop.

## Step 5: Check the public endpoints

Visit each of these and tell me what actually renders on the page, not what
the network log implies:

  https://trading.murzaktech.tech        -> the dashboard login page
  https://health.murzaktech.tech/health  -> a JSON health response

If a certificate warning appears on the first attempt, wait sixty seconds and
retry once: Coolify requests certificates on demand and the first request to
a new hostname can arrive before issuance completes. If it still fails,
report the exact error.

## Step 6: Start the run

Only once every [BLOCK] is cleared, run:

  python scripts/start_paper_run.py --probe-broker --probe-data --write-manifest

Report the manifest path it prints. Then confirm to me, in one line each:

- the primary broker named in the output
- the number of blockers
- the manifest path

Do not pass --force under any circumstances. If the gate blocks, the run
would produce no usable evidence, and I would rather know that now.
```

## After the assistant is done

Once a US market session has opened and the agent has placed something,
confirm the venue two ways. Both matter, because the internal simulator also
issues UUID order IDs, so an ID on its own proves nothing.

1. **Open Alpaca's own paper dashboard** at
   <https://app.alpaca.markets/paper/dashboard/overview> and look at the
   order history. An order that reaches Alpaca appears there. One filled by
   the internal simulator never will. This is the unambiguous check.
2. **Check the `broker_name` recorded against the order** in the journal or
   the dashboard. It reads `alpaca` for a real sandbox fill and `paper` for a
   simulated one.
