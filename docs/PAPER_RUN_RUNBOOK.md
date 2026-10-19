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

## Is the agent buying and selling the same thing over and over?

A strategy that trades daily bars should hold a position for days. If the same
symbol is bought and sold again and again within minutes, every pass pays the
spread and the fees and earns nothing. In the first weeks of the paper run a
trailing stop closed a crypto position, the strategy still said buy, the agent
bought it back at nearly the same price, the new position inherited the old
one's trailing-stop peak, and it was stopped out again within minutes: hundreds
of round trips, about 15% winners, a loss of 2.7% of the account.

Three things now guard against it:

- a position the agent closes no longer leaves a trailing-stop peak behind;
- a symbol sold by a strategy or a stop is not bought back for 24 hours
  (`trading.reentry_cooldown_hours`; the Notifications page shows such a signal
  as held back, and the kill switch's flatten does not count);
- the pattern is watched whatever the cause: the scorecard's Health section
  shows how many positions were sold within an hour of buying, and the agent
  emails you (critical) when one symbol does it three times in a day.

To look at it yourself:

```bash
python scripts/churn_report.py
python scripts/churn_report.py --hours 72 --symbol BTC-USD
```

It lists, per symbol, how many round trips closed within an hour of opening,
how long they were held, what they made and what closed them.

## Learning from every signal, not only closed trades

The agent is long-only and selective, so trades close slowly; early in a run
there may be none, and until a strategy has closed five trades its weight is
not adjusted from results. So the agent also learns from every buy or sell
signal it produces, traded or not, by each strategy on its own:

- It records what the price did 5 and 20 trading days later, in the signal's
  direction, and how often that symbol simply went up anyway. Outcomes are
  recorded once known, never change, and are kept in `data/signal_outcomes.db`,
  so a redeploy cannot lose them. It refreshes every six hours.
- The scorecard's Learning section shows, for each strategy in each market,
  how often its signals moved the right way against how often chance would
  have, and a "z" for how many standard errors that is. About 2 or more is a
  real result; a row with under 50 signals is greyed as too few to say.

By default it only measures and shows. Nothing about how the agent trades
changes. Once a strategy shows a clear result in a market, you can let it act
on it by setting `learning.signal_skill_tilt.enabled` to `true` in
`config/config.json`. The strategy's vote in that market is then scaled from
0.5 to 1.5 times, never more, never off, and only after 50 signals with a known
outcome. Review the table first; the "weight scale" column shows what each
strategy would get.

This measures direction, before costs, on each strategy's raw signal. It is
evidence of skill, not of profit.

## Alerts on the dashboard

Every alert the agent raises appears on the dashboard's Notifications page
(the bell in the header), whether or not email is set up:

- **What is wrong right now** (a background worker down, the trading loop
  failing, symbols with no price) sits under "Needs your attention" and clears
  by itself when the agent has fixed it.
- **New alerts** (a halt, a restart, a crash) stay there, and on the bell's
  count, until you press "Mark as read". Which alerts you have read is
  remembered in your browser only, so another device starts afresh.
- **Recent agent alerts** lists everything from the last week, with filters
  for critical, warning and info. The operator sees the detail of each one;
  a viewer sees the headline only.

Without email you only see these while the dashboard is open, so it is worth
looking at the bell each morning until email is set up.

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

## What the agent does with each uploaded document

The Research page reads a PDF in this order, and the upload result says which it
found:

1. **AIB-AXYS Market Pulse**: read by rules; prices, earnings, dividends, rates
   and announcements are stored and used (the dividend sleeve, the price history,
   the AI's trade review).
2. **Bond auction note**: read by rules and kept for you; the agent trades no bonds.
3. **Analyst note with ratings**: an AI finds each buy, sell or hold; stocks the
   agent trades are followed or queued for your approval.
4. **Anything else** (an IPO briefing, a results announcement, a central bank or
   tax notice, a sector or commodity report): kept in the **Research Library**.

For the library, an AI writes a digest: the kind of document, its date, a short
summary, the stocks and markets it concerns, the figures and dates that matter,
and the risks it names. Every figure and date has to come with the exact words it
was taken from. Rules then check that those words are really in the document
and that the figure is really in them; anything that fails is dropped and listed
as "Not used", so a wrong or invented figure cannot reach the agent.

How it is used later: for about 90 days (longer if the document names a later key
date, until 14 days after it), the AI sees a short line from each document that
names a stock when it reviews a trade in that stock, or one document about the
stock's whole market if none names it. It is background only: the AI can confirm,
weaken or veto a signal, never create one, and a document never places a trade
or changes a setting. You can remove a document from the library at any time,
and it stops being shown.

**Links inside a document.** Most documents point to fuller evidence (audited
accounts, AGM results, announcements). The library lists every link a document
carries, and nothing is downloaded until you press **Fetch** on one. Only links
the document itself contains can be fetched. The server checks each address
before and after every redirect: https only, standard port, and never a private
or internal address; at most 3 MB and 15 seconds; web pages, plain text and
PDFs only. The page is then read and checked exactly like the document: a
figure is kept only with a quote that is really on the page and a number that
is really in the quote, and it is stored with its source address and shown to
the AI together with the document. One gap remains: the address is checked just
before the request and looked up again by it, so a hostile name server could in
theory answer differently the second time; that is why fetching is one link at a
time on your click, and everything that comes back is treated as untrusted text.

If the upload says "no AI is switched on", add `GROQ_API_KEY` or `GEMINI_API_KEY`
(both free) in Coolify and upload the document again; without an AI only the
text is kept. Documents that are pictures are read only if they are rating sheets.

## Watch-only markets: commodities and four stock markets

The agent watches metals, oil and farm produce and learns their patterns without
trading them (`src/agent/commodity_watch.py`; settings under `watch_only` in
`config/config.json`). It has no broker connection and places no orders, and the
commodities are deliberately not in `data_manager.symbols`.

Three patterns are scored: price crossing its 200-day average (`trend_200d`), the
past 12 months' direction on the first trading day of a month (`momentum_12m`),
and the same calendar month's record over at least five earlier years
(`seasonal`). Each signal is scored against the price 20 trading days later and
against how often that commodity simply rose anyway.

Two kinds of evidence are kept apart:

- **Backfill** is the history replayed over years of prices. It is quick, but a
  pattern found by trying several ideas can look good by luck, and the
  commodities move together, so the count of independent signals is lower than it
  looks. A pattern is called a CANDIDATE only when it beats chance by two standard
  errors in all the history and in both its earlier and later half.
- **Live** is every signal dated after the watch began (the date is kept in
  `data/watch_state.json`). Only this confirms a pattern, and it arrives slowly:
  a signal's outcome is known 20 trading days later.

**Stock markets.** The same three patterns are scored on the UK (FTSE 100), Japan
(Nikkei 225 and TOPIX), France (CAC 40) and South Africa (FTSE/JSE) markets: for
each, the index, an index fund and several of the largest companies, from Yahoo
Finance. No broker reaches these markets, so nothing is ever traded. Points to
keep in mind when reading the results:

- Each market keeps its own live start day (in `data/watch_state.json`), set the
  first time it returns prices, so a market added later is never credited with
  "live" days before it was watched.
- Prices are in each market's own currency and unit: London quotes in pence and
  Johannesburg in cents. Yahoo sometimes serves such a price in pounds or rand for
  a stretch; a jump of about 100 times between neighbouring days is undone before
  scoring, and the dashboard marks any instrument where that happened.
- Results are the move in local currency, before the exchange rate a
  dollar-based investor would also bear.
- The companies listed are today's large ones, which flatters their history
  (survivorship). The index and index-fund rows are the fairer test.
- Each market has an assumed round-trip cost (`watch_only.equity_markets.costs_pct`),
  the market's tax on purchases plus an allowance for spread and commission.
  Checked against public sources on 18 October 2026:
  - **UK 0.7%**: stamp duty reserve tax is 0.5% on purchases (HMRC 2026/27 table),
    plus a PTM levy of GBP 1.50 on trades over GBP 10,000. The tax is due to be
    replaced in 2027 by a securities transfer tax; draft legislation was published
    in July 2026 and I did not find its rate.
  - **France 0.6%**: the financial transaction tax is **0.4%** (raised from 0.3% in
    2025) on purchases of shares of French companies worth over EUR 1 billion
    (about 121 companies on the 2026 list). Index funds are not subject to it, so
    the fund row's cost is slightly overstated.
  - **South Africa 0.7%**: securities transfer tax 0.25% (SARS, unchanged for
    2026), plus settlement and levy fees of about 0.006% (an undated JSE page).
    Brokerage carries VAT; I found no current schedule.
  - **Japan 0.3%**: I believe there is no securities transaction tax, and that
    non-residents without a permanent establishment are generally not taxed on
    ordinary share sales, but I could not confirm either from a source. Some
    online brokers charge no commission on certain venues (Japannext X-Market for
    retail brokers).
  - **The allowance for commission and spread (about 0.1% a side, more in
    Johannesburg) is an assumption, not any broker's schedule.** Interactive
    Brokers' published European tiers run from 0.05% of trade value down to
    0.015% with a EUR 1.25 minimum, but I could not confirm its rates for each of
    these four exchanges. Replace the figures with your own broker's.
- A ticker that returns no prices is skipped; the dashboard shows which
  instruments loaded. Not every ticker has been confirmed against the live feed.

**Longer horizons.** Each signal is judged 20, 60 and 120 trading days later (about
one, three and six months); the dashboard has a selector and the study script
prints all three (`--horizons 20,60,120,250` to choose others). The patterns
signal about once a month, so a 60 or 120 day outcome window overlaps the next
two to five signals. Counting each as independent would count the same stretch of
price several times and make luck look like skill, so for each instrument and
pattern a signal whose window overlaps an earlier counted one is set aside. That is
why the counts fall as the horizon grows, and it applies to the 20 day figures too
(a trend signal that whipsaws is counted once a window). Trying three horizons gives
three chances for luck: a pattern that is a candidate at one horizon only deserves
doubt; one that holds at neighbouring horizons deserves a closer look. The seasonal
pattern is a one-month pattern, so longer horizons dilute it by design.

To replay the history now, from the backend container's Terminal:

    python scripts/commodity_study.py
    python scripts/commodity_study.py --only metals --kind futures

The dashboard's **Insights, Watched Markets** tab shows both records and what each
pattern says today (it is separate from the Scorecard, so it appears even while the
scorecard is slow to load). Two cautions: Yahoo's futures series jump when a contract rolls over, and
the oil and gas funds (USO, UNG) lose value from rolling contracts, so judge a
fund by its own price. Switching a pattern into real trading is a separate
decision, and not one this watch makes.

## Kenyan commodity markets

Reviewed on 18 October 2026 from public sources; confirm before relying on it.

- **No tradable price feed exists for it that I could find.** The Nairobi
  Securities Exchange lists shares, not commodities. The Kenya National
  Multi-Commodities Exchange (KOMEX), a state corporation, is running an
  over-the-counter pilot (a food supply delivery pilot in Kakamega) and its 2026
  budget is about KES 43 million; a 2026 target of 30,000 tonnes traded is on the
  national budget portal, and I could not verify the actual figure. I found no
  commodity futures contract in Kenya, and an older profile of the Kenya
  Agricultural Commodity Exchange says futures were never traded there.
- **Auction prices do exist for the two big exports.** The Nairobi Coffee Exchange
  auction (the Kenya National Bureau of Statistics reported July 2026 at an
  average of USD 7.19 per kg on 2.6 thousand tonnes) and the weekly Mombasa tea
  auctions run by EATTA (reported in the press as a weekly average in dollars per
  kg with packages offered and sold).
- **What that allows.** The agent could record these auction prices, weekly for
  tea and coffee, as a watch-only series and score the same patterns on them.
  Two limits: there is no market to trade in, and a weekly series is too short on
  history and too slow to reach the 30 signals needed for a verdict for a long
  time. The sources and their format also need confirming before any reader is
  built. Maize and other staples are not on any exchange feed I found.

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
