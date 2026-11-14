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

## Steps

Run these on the server where the agent runs, from the project folder (the
one holding `config/` and `scripts/`).

1. Pull the latest code, so the scripts are there:

   ```bash
   git pull
   ```

2. Make the pack:

   ```bash
   python scripts/review_pack.py
   ```

   It downloads six years of daily prices (a few minutes), runs both
   backtests and the scorecard, and prints where the zip is, for example:

   ```
   Review pack: reports/review_pack_20261205_0910.zip
   ```

   If a step fails, the others still run; the pack says which failed and
   keeps its log in `logs/`.

3. Share the zip, or open `SUMMARY.md` and the three reports inside and
   paste them into the conversation.

To run it again without downloading prices again:

```bash
python scripts/review_pack.py --skip-download
```

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
