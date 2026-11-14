# Making a review pack

A review pack is one zip file holding everything needed to judge the agent:

- a **backtest of the technical strategies**: how the momentum, mean-reversion
  and RSI rules would have done over the last six years on the symbols they
  trade, after costs, against simply buying and holding;
- a **backtest of the momentum sleeve**: the same for the monthly 12-month
  momentum rotation on its own funds, plus a check that the result does not
  rest on one lucky fund;
- the **scorecard**: how the paper run is doing against its targets so far.

It holds reports only: no keys, no account numbers, no trade journals.

## Steps (Coolify or Docker)

The agent runs in a Docker container that Coolify rebuilds from the main
branch whenever it changes. The container has no `git`; it always holds the
code of the last deployment.

1. Make sure the latest code is deployed: in Coolify, open the application
   and check the last deployment finished after the change you want (or
   press **Redeploy**).
2. Open the container's terminal: in Coolify, the application's
   **Terminal** tab (or `docker exec -it <container> sh` on the server). The
   prompt shows `$` and the folder is `/app`.
3. Make the pack and show it on screen:

   ```bash
   python scripts/review_pack.py --print
   ```

   It downloads six years of daily prices (a few minutes), runs both
   backtests and the scorecard, then prints the summary and the three
   reports. If a step fails, the others still run; the summary says which
   failed, and its log is in `data/reports/review_pack_<date>/logs/`.
4. Copy everything from the first `==== SUMMARY.md ====` line to the end
   and paste it into the conversation.

The pack is also saved as a zip in `data/reports/`, on the data volume, so a
redeploy does not delete it.

To run it again without downloading prices again:

```bash
python scripts/review_pack.py --print --skip-download
```

## Steps (a plain copy of the project)

From the project folder: `git pull`, then `python scripts/review_pack.py`.
The zip is in `data/reports/`.

## When to make one

- **Now**, for the backtests: they show whether the rules have an edge after
  costs at all, before any more paper trading time is spent on them.
- **After three to four weeks of paper trading**, for the scorecard: by then
  there are enough closed trades and scored signals for most measures to
  move past "too early".

## Reading it yourself

In each backtest report, the **Verdict** section is the short answer. The
two numbers to look at first:

- **Excess over risk-free** against **Buy & hold**: the strategy should beat
  simply holding the same symbols, after costs, or it is not earning its
  risk.
- **Max drawdown**: the worst fall from a high. The account's limit is 10%.

In the scorecard, every measure is marked PASS, WATCH, FAIL or early. FAIL
lines come first and say what is short of target.
