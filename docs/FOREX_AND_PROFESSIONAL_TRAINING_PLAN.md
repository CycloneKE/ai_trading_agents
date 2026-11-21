# Forex inactivity, and a plan to train the agent like a professional

Written 21 November 2026, from tests on ten years of real daily prices
(Yahoo Finance) and official interest rates (FRED, the US Federal Reserve's
data service). Every number below can be reproduced with
`python scripts/forex_factor_study.py` and `scripts/run_backtest.py`.

## 1. Summary

- **Forex is quiet by design, not because something is broken.** The agent's
  forex rules trade about once every five weeks across all three pairs, the
  positions are small, and the market is closed every weekend.
- **When the current rules do trade forex, they lose money.** Over ten years
  the trading itself lost money after costs (profit factor 0.88; every dollar
  of profit in the backtest came from interest on idle cash).
- **The biggest problem is that the forex book is long only.** It can only
  bet that the US dollar will fall. Every long-only version of the
  professional forex approaches tested here lost money; the same rules
  allowed to go both ways made money.
- **The approach professionals use on currencies, carry plus trend, is the
  best candidate.** On five major pairs it earned a Sharpe ratio of 0.47,
  stayed positive in both halves of the period, and its worst fall was 7%.
  The edge is real but modest; it is not a money machine.

## 2. Why forex shows no activity

Four reasons, in order of size.

1. **The rules fire rarely.** The agent's own backtest of its forex rules
   (momentum, mean reversion, RSI and forex trend) on EUR/USD, GBP/USD and
   AUD/USD from 2016 to 2026 produced 99 round trips in ten years: about ten
   a year for all three pairs together, each held about 43 days on average.
   Weeks without a single forex trade are normal.
2. **Positions are small, so a trade barely moves the account.** A position
   is capped at 5% of the book: about $500 of a $10,000 forex book. A
   currency pair moves about 0.5% on a typical day, so a held position moves
   the account by about $2.50 a day. On the dashboard that looks like nothing.
3. **Long only.** The book can buy euros, pounds and Australian dollars with
   US dollars, but never the reverse. When the dollar is strong (much of
   2021 to 2024), the rules correctly see no reason to buy, so they sit out.
4. **The market is closed from Friday 17:00 to Sunday 17:00 New York time**
   (Saturday 01:00 to Monday 01:00 Nairobi time). The agent correctly
   refuses forex orders then.

One thing checked and ruled out: Yahoo reports zero volume for every
currency pair (forex has no central exchange counting volume). The
strategies ignore zero volume, so it does not block trades.

## 3. What "training like a professional" means for this agent

An important point in plain language: **the agent does not learn by reading
tutorials.** Books and courses teach *you* (and whoever writes the code)
which approaches have worked and how to test them honestly. The agent then
gets better in only three ways:

1. **Better rules**: strategies taken from published, peer-reviewed research
   and written into the code, such as carry and trend for currencies.
2. **Honest testing**: each rule is tested on years of history it was not
   designed on, after costs, before it trades even on paper.
3. **Learning from its own results**: the agent already does this. It
   scores every signal against what the price did 5 and 20 days later and
   gives more weight to strategies that beat chance after costs
   (`src/agent/signal_ledger.py`, `learning` in `config/config.json`).

The AI models the agent already uses (Groq, Gemini, optionally Claude)
review trades; feeding them more trading material would not give them an
edge, because they cannot predict prices from text knowledge.

## 4. Review of the research you shared

The research is sound and matches how quantitative firms work. Notes on
fitting it to this agent:

| Point in the research | Verdict for this agent |
|---|---|
| Data, features, model, backtest pipeline | Correct. The agent already has the data and the backtest, after costs. |
| Pitfalls: look-ahead bias, overfitting, ignoring costs | Correct and the most valuable part. The agent's tuner already tests changes on held-out data; the new study script has a test proving it cannot see the future. |
| Gradient boosted trees (XGBoost, LightGBM) | Useful later, as a **filter** on top of a tested rule ("meta-labelling", Lopez de Prado chapter 3), not to predict prices directly. Ten years of daily data on a few pairs is only a few thousand rows, which is too little to train a price-predicting model without overfitting. |
| LSTMs and Transformers | Little evidence of an edge on daily prices for a firm of this size. Skip for now. |
| Reinforcement learning for execution | Matters for funds moving millions; irrelevant at this account size. |
| Lopez de Prado and Jansen books | Both excellent. Read de Prado chapters 3 (labelling), 7 (cross-validation) and 11 to 14 (backtesting) first. |
| Coursera (NYIF and Google Cloud), Quantra | Good for learning concepts. They will not supply a working edge. |
| QuantConnect | Good for research, but its data cannot be taken out; an idea found there must be re-tested here before it trades. |
| MetaTrader 5 with Python | **Caution:** the MetaTrader 5 Python library runs on Windows only. This agent runs in a Linux Docker container on Coolify, so MT5 would need a separate Windows server as a bridge. A broker with a web API (the repository already has an OANDA connector) fits much better. Check that any broker is licensed by Kenya's Capital Markets Authority and offers a demo account with an API. |

Additional trusted reading, all directly relevant to forex and
commodities:

- Robert Carver, *Systematic Trading* and *Leveraged Trading*: how a
  small trader runs trend and carry on currencies and futures, with
  position sizing by volatility. The closest book to what this agent needs.
- Moskowitz, Ooi and Pedersen, "Time Series Momentum" (2012), and Koijen,
  Moskowitz, Pedersen and Vrugt, "Carry" (2018): the research behind the
  two approaches tested here. Both are freely available as working papers.
- Ernest Chan, *Algorithmic Trading*: practical testing of currency and
  futures strategies.

## 5. What the tests show

Five major pairs against the US dollar (EUR, GBP, AUD, NZD, JPY), November
2017 to November 2026. Textbook settings only, nothing tuned. Each pair
sized to 10% yearly volatility, about one pip of cost per trade, overnight
interest counted.

| Approach | Yearly return | Sharpe | Worst fall | Sharpe, first half / second half |
|---|---|---|---|---|
| Trend 12 months, long and short | 1.88% | 0.31 | -12.8% | 0.50 / 0.10 |
| Trend 12 months, long only | -1.15% | -0.22 | -16.0% | -0.17 / -0.27 |
| Moving average 50/200, long and short | -0.90% | -0.10 | -27.2% | 0.28 / -0.48 |
| Carry, long and short | 1.21% | 0.38 | -10.1% | 0.08 / 0.82 |
| **Carry + trend, long and short** | **1.64%** | **0.47** | **-7.1%** | **0.53 / 0.41** |
| Carry + trend, long only | -0.35% | -0.11 | -6.8% | -0.03 / -0.19 |
| Buy and hold the currencies | -4.46% | -0.51 | -39.3% | -0.66 / -0.37 |

How to read it: **Sharpe** is return per unit of risk; above 0.5 is
respectable for a single asset class. A real edge should be positive in
both halves, not just in one lucky stretch. Only "carry + trend, long and
short" passes both tests. The returns look small because each position is
sized cautiously; doubling the size roughly doubles both the return and the
worst fall.

## 6. Recommended plan

**Phase 1: rebuild the forex book on carry and trend (recommended next).**
- Allow the forex paper book to sell a pair as well as buy it (selling
  EUR/USD means holding dollars against euros), with leverage capped at 3x.
- Add a carry-plus-trend strategy using the settings above, rebalanced
  weekly, and make it forex's only voter (as `crypto_trend` is for crypto),
  since the current fast rules lost money on forex.
- Charge and pay overnight interest (rollover) in the paper book, as a real
  broker does; carry trades depend on it.
- Add USD/JPY and NZD/USD, so there are five pairs to diversify across.
- Success measure: positive return after costs over at least three months
  of paper trading, with the loss limits unchanged.

**Phase 2: commodities.** Commodities are watched but never traded today
(no broker reaches commodity futures). Trend following has historically
worked better on commodities than on major currencies. Run the same study
on commodity funds Alpaca can trade (gold, oil, broad commodities) and on
the watch list's live scores, and promote only what passes.

**Phase 3: machine learning as a filter.** Once the forex and commodity
books have a few hundred scored signals, train a small gradient-boosted
model to decide which signals to take and how large, using de Prado's
labelling and purged cross-validation. Install the libraries in
`scripts/requirements-ml.txt` first.

**Phase 4: a real broker's demo account.** Only after the paper scorecard
passes, connect a regulated broker's practice account through its API.

## 7. How to re-run the tests

From the container terminal in Coolify (see `docs/REVIEW_PACK.md` for how
to open it):

```bash
python scripts/forex_factor_study.py
```

It downloads the prices and interest rates (about a minute), then prints
the table in section 5 with the latest data. To re-run the agent's own
forex backtest:

```bash
python scripts/fetch_history.py --symbols EUR_USD,GBP_USD,AUD_USD --years 10 --out data/history/forex
```

```bash
python scripts/run_backtest.py --csv-dir data/history/forex --capital 10000 --out data/reports/forex_bt
```

These are historical results, not forecasts, and not investment advice.
