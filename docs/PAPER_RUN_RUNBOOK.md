# Paper run runbook

How to run the agent against live prices with fake money for ninety days, and
how to tell each week whether it is working.

This is written for someone who is not a developer. Every command below is
meant to be copied and pasted exactly as it appears, from a terminal opened in
the project folder.

## Why this runbook exists

A paper run is only worth ninety days if it produces evidence at the end. This
system has three documented ways of running for months and producing none:

1. The strategies quietly load as `MockStrategy`, which returns "hold" forever.
   The agent starts, reports healthy, and never places a trade.
2. The dashboard's API fails to import, so there is no way to watch the run.
3. The price feed falls back to synthetic data, which the agent correctly
   refuses to trade on. The vendor being down looks exactly like the strategy
   finding nothing.

None of these announce themselves. Each one produces a running process, a
clean log, and an empty result. The two scripts below exist to make that
impossible: one refuses to start a run that would produce nothing, the other
tells you every week what the run has actually done.

## Before you start

You need three things in place.

**A `.env` file with your keys.** Copy `.env.example` to `.env` and fill in the
values. At minimum you need `SECRET_KEY` (any long random string; it signs the
dashboard's login tokens) and a market data key. Without `SECRET_KEY` the
dashboard starts but nobody can log in.

**Dependencies installed.**

```bash
pip install -r requirements.txt
```

**Historical price files**, one CSV per symbol named `<SYMBOL>.csv`, in a
folder. The repository ships some in `data/nse_historical`. These are not used
for trading; they are used to prove the strategies would actually act before
you commit ninety days to finding out.

## Step 1: check the run is worth starting

```bash
python scripts/start_paper_run.py --csv-dir data/nse_historical
```

This prints a list of checks. Three outcomes matter:

- `[ok]` means the check passed.
- `[warn]` means something is worth knowing but does not stop the run. A
  warning usually means a question could not be answered, not that the answer
  was good.
- `[BLOCK]` means starting now would produce a run with no evidence in it.

The script exits with an error code when anything is blocked, so it will not
let you start by accident.

The most important line is **"strategies produce actionable signals"**. It runs
the real strategies over your historical bars and counts how many would have
cleared the live trading gate, which needs all three of a direction, confidence
above 0.1, and a position size above zero. A configuration can satisfy two of
those and still place no orders for three months. If this check blocks, nothing
else is worth doing until it passes.

### Checking against live vendors and your broker

The checks above run offline. To test the parts that need the network:

```bash
python scripts/start_paper_run.py --csv-dir data/nse_historical \
    --probe-data --probe-broker
```

`--probe-data` starts the price feed briefly and checks that real quotes come
back rather than synthetic ones. `--probe-broker` connects to your broker and
checks it is in paper mode and accepting orders. Both take under a minute.

## Step 2: start the run

If you want the script to launch the agent for you:

```bash
python scripts/start_paper_run.py --csv-dir data/nse_historical \
    --probe-data --probe-broker --days 90 --start
```

If you prefer to run the agent yourself, in Docker, under systemd, or in a
terminal you leave open, record the run's start first and then launch it your
own way:

```bash
python scripts/start_paper_run.py --days 90 --write-manifest
python main.py --config config/config.json
```

Either way a **run manifest** is written to `data/paper_runs/`. It records when
the run started, which commit it is running, which symbols are in the universe,
and any warnings that were live at the start. Without it, a report cannot tell
a run that has traded nothing in three weeks from a run that started an hour
ago. Those are very different problems.

If a check blocks and you want to start anyway, add `--force`. The manifest
records that you did, so a later report can say the run was started knowing it
might produce nothing.

## Step 3: read the run each week

```bash
python scripts/paper_run_report.py
```

Both journals are opened read-only, so this is safe to run while the agent is
trading. To write the report to a file you can keep:

```bash
python scripts/paper_run_report.py --since 7 --out reports/week1.md
```

The report answers four questions in order.

**Is it trading?** Decisions recorded, orders placed, and how long since the
last of each. If nothing has been recorded for more than a day, the report says
the agent has most likely stopped rather than leaving you to guess: a live run
writes a heartbeat row per symbol every thirty cycles even when nothing changes.

**Why does it hold?** A table of reasons, each one labelled. "by design" means
the strategy looked and found nothing, which is a normal and mostly correct
state for a trading system. "DEGRADED" means something was broken and the trade
never got a fair chance. The distinction is the whole point of the table: a run
that held four thousand times because the price feed was down has not tested
your strategy at all.

**What did it make?** Closed round trips matched first in, first out, broken
down by symbol and by strategy. Attribution follows the order that *opened* the
position, not the one that closed it, so a trailing stop is not blamed for an
entry it did not choose. If most of the profit comes from a single symbol, the
report says so and tells you what the rest of the book made without it. One
symbol over one period is an anecdote, not an edge.

**Is anything silently degraded?** Orders that never reached a final status,
orders the broker rejected, decisions with no LLM verdict attached (which is
what running out of quota looks like from the journal), and decisions that saw
synthetic prices.

## Where the agent stands against where it needs to be

The weekly report says what happened. The scorecard says whether that is good
enough, measure by measure, each with a target and a status:

```bash
python scripts/agent_scorecard.py
```

It is also on the dashboard, under Risk & System, in the Scorecard tab.

Each line is marked `[PASS]`, `[WATCH]`, `[FAIL]`, `[early]` (not enough
evidence yet, and it says how much more is needed) or `[info]`. It covers six
questions: is the agent healthy, has it traded enough for a result to mean
anything, do its closed trades make money after costs, is the account ahead of
simply holding the S&P 500 with drawdown inside its limit, do its buy and sell
signals move the price the right way by more than chance, and is it learning
from its own results.

The forecast check judges every signal, traded or not, by what the price did
5 and 20 trading days later, against how often that symbol simply went up
anyway. A buy in a rising market is not counted as skill just because the
market rose. Add `--offline` to skip the checks that need the internet.

The targets are proposals, kept in `config/config.json` under
`scorecard.targets`. Change them there if you disagree with a number.

## Email alerts and self-repair

The agent emails you when it needs a person, and repairs what is safe to
repair by itself. It emails when:

- trading is halted (unless you pressed the button yourself);
- a background worker dies and is restarted, or keeps dying;
- the trading loop fails three times in a row, and again at ten;
- a symbol has had no real price for half an hour;
- the agent was killed rather than stopped (a crash, or out of memory) and
  has started again;
- a risk limit is blocking every new trade.

It repairs: dead background workers (restarted, up to six times an hour), and
one kind of halt, the one it causes itself when its loop freezes. That halt is
lifted once the loop and the broker have been healthy for ten minutes, at most
three times a day. It never lifts any other halt (a loss limit, or your own
button); those wait for you to press Resume. To turn that off, set
`self_healing.auto_resume.enabled` to `false`.

To switch email on, set these in Coolify's Environment Variables tab (they are
the mail account's details, the same ones you gave Coolify for its own
notifications) and redeploy:

| Variable | What to put |
|---|---|
| `ALERT_EMAIL_TO` | the address to email (several allowed, separated by commas) |
| `SMTP_HOST` | the mail server, for example `smtp.gmail.com` |
| `SMTP_PORT` | `587` (the usual), or `465` for SSL |
| `SMTP_USER` | the mail account's login, often its address |
| `SMTP_PASSWORD` | its password; Gmail needs an "app password" |
| `SMTP_FROM` | optional; defaults to `SMTP_USER` |

To check it works, run this from the project folder (in Coolify, the
backend container's terminal) and look for the email:

```bash
python scripts/send_test_alert.py
```

It says exactly what is missing, or what the mail server replied if it refused.
The Scorecard's Health section shows "Email alerts: on" once it is working.
Every alert is also recorded in `data/alerts.jsonl`, sent or not. The same
alert is not repeated within an hour (15 minutes for a critical one).

## What the numbers mean

Two cautions the report repeats, because both are easy to get wrong.

**Realised profit and loss is calculated from actual fill prices.** Broker
commission and the slippage already inside those fills are not subtracted a
second time, because that would double-count them. Before calling a result an
edge, compare each strategy's average return against the round-trip cost of
trading in that market, which lives in `src/agent/cost_model.py`. On the NSE
that hurdle is large, and the figures currently in the config are flagged
placeholders until the real AIB-AXYS schedule is confirmed.

**Fewer than about thirty closed round trips per strategy tells you nothing.**
The report refuses to give a verdict below twenty, and says so. Win rate and
profit factor only start separating skill from luck once there are enough
trades for luck to average out.

## If the run goes quiet

Run the report first. It will tell you which of these it is:

- **No decisions at all**: the agent is not running, or `data/` is not
  writable. Check the process and the folder permissions.
- **Decisions but no orders, all "by design"**: the configuration is being
  selective. Not a fault, but if it persists for weeks the confidence gate or
  the regime filter is probably too strict for the current market.
- **Decisions but no orders, reasons marked DEGRADED**: something is broken.
  `fallback_price` means the vendor feed is down. `halted` means the kill
  switch is engaged. `no_account_info` means the broker is not answering.
- **Orders stuck at "intent" or "submitted"**: they never reached a final
  status, which means either the broker never answered or startup
  reconciliation is not running.

## Watching it day to day

The agent exposes a health endpoint on the port set in `config/config.json`
under `monitoring.port`, 8080 by default. Anything that can make an HTTP
request can watch it, which is the cheapest possible way to find out the
process died on day three rather than on day ninety.

The dashboard is a separate, richer view. If it starts but will not let you log
in, `SECRET_KEY` is missing from your `.env`. The readiness check in step 1
reports this as a warning, because the journals keep recording either way and
the weekly report reads them directly, not through the API.
