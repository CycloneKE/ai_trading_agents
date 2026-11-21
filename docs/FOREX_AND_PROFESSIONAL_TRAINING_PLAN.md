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
  The edge is modest, and smaller once timing luck and broker costs are
  counted (sections 5 and 8); it is not a money machine.
- **Update, 21 November 2026: Phase 1 is built and running on paper.**
  The forex book is now a margin account that can sell short, traded only
  by a weekly carry and trend plan. See section 8 for what changed, section 9
  for the wider search for better rules, and sections 10 and 11 for the
  questions about a "money machine" and about reading the books.

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

| Approach | Yearly return | Sharpe | Worst fall | Sharpe, first half / second half | Sharpe, 1-5 day delays (mean, worst) |
|---|---|---|---|---|---|
| Trend 12 months, long and short | 1.88% | 0.31 | -12.8% | 0.50 / 0.10 | 0.20, 0.12 |
| Trend 12 months, long only | -1.15% | -0.22 | -16.0% | -0.17 / -0.27 | -0.29, -0.34 |
| Moving average 50/200, long and short | -0.90% | -0.10 | -27.2% | 0.28 / -0.48 | -0.10, -0.12 |
| Carry, long and short | 1.21% | 0.38 | -10.1% | 0.08 / 0.82 | 0.36, 0.34 |
| **Carry + trend, long and short** | **1.64%** | **0.47** | **-7.1%** | **0.53 / 0.41** | **0.35, 0.28** |
| Carry + trend, long only | -0.35% | -0.11 | -6.8% | -0.03 / -0.19 | -0.11, -0.14 |
| Buy and hold the currencies | -4.46% | -0.51 | -39.3% | -0.66 / -0.37 | -0.50, -0.51 |

**Timing luck** (last column): the same rule acted on one to five days
later. A slow signal whose result swings with a day's delay is partly
noise. Carry holds steady; the 12-month trend does not, so its 0.31 above
flatters it.

How to read it: **Sharpe** is return per unit of risk; above 0.5 is
respectable for a single asset class. A real edge should be positive in
both halves, not just in one lucky stretch. Only "carry + trend, long and
short" passes both tests. The returns look small because each position is
sized cautiously; doubling the size roughly doubles both the return and the
worst fall.

## 6. Recommended plan

**Phase 1: rebuild the forex book on carry and trend. DONE, see section 8.**
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

**Phase 2: the strongest candidates from the wider search (section 9).**
Commodities are watched but never traded today (no broker reaches commodity
futures), and the search in section 9 found that trend rules on commodity
funds did not beat simply holding them. The candidates that did pass every
check are on the S&P 500: a 10-month average filter (crash protection) and
short pullback buys. Each should run on paper on its own small share of
the account, so the scorecard can judge it separately.

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

To replay the live forex book itself over the same history (after the
study has downloaded the data):

```bash
python scripts/forex_book_replay.py
```

To run the wider search for better rules (section 9):

```bash
python scripts/rule_research.py
```

## 8. Phase 1: what changed (built 21 November 2026)

**The new forex book.** `brokers.forex_paper` is now a forex margin account
(`src/connectors/forex_margin_broker.py`), still paper money only, starting
at $10,000. Unlike the old book it can:

- **sell a pair short** (selling EUR/USD means holding dollars against euros);
- **use leverage up to 3 times equity**, refusing any order that would pass
  that (an order that reduces a position is never refused);
- **charge and pay overnight interest** (rollover) at each currency's
  3-month interest rate, less a 0.5% a year broker markup (an estimate:
  check your broker's figure), and about one pip of spread on every trade;
- **halt itself** if it falls 15% from its highest value: it closes every
  position, emails you (if email alerts are set up) and trades nothing more
  until you clear `halted` in `data/forex_margin_state.json` and restart.

**The only strategy that trades it** is the weekly carry and trend plan
(`src/agent/forex_carry_trend.py`, settings in `forex_carry_trend` in
`config/config.json`). In the first forex session of each week (Monday
morning Nairobi time) it gives each of five pairs (EUR/USD, GBP/USD,
AUD/USD, NZD/USD, USD/JPY) a signal of -1, -0.5, 0, +0.5 or +1, sizes each
by its volatility and trades only the changes worth making. The old fast
rules no longer trade forex: they lost money on it.

**What it would hold on Monday 23 November 2026** (a dry run on the real
prices and rates of 21 November): short the euro, short the yen, long the
Australian dollar, a small short in the New Zealand dollar, nothing in
sterling; gross exposure 1.12 times equity, positions of about $1,500 to
$4,300 each. The old book's positions were capped at about $500.

**Where to see it.** The dashboard's Forex paper account page shows the new
book: its value, each position (a short shows a negative quantity), its own
trade list and its closed trades, long and short. It has no stop-loss
levels, by design: the weekly plan and the 15% halt are its exits, as the
research tested it.

**Honest expectations.** A replay of the live code itself over 2018 to
2026, trading weekly with spread and the 0.5% rollover markup
(`scripts/forex_book_replay.py`), made 0.48% a year with a Sharpe ratio of
0.15 and a worst fall of 6.9%: about 36 trades a year. The research
version (daily, no markup) scored 0.47, but the same rule acted on a day or
two later scored between 0.28 and 0.42 (section 5, "timing luck"), and nine
years of data cannot tell a Sharpe of 0.2 from zero with confidence. So the
forex book will now be visibly active, but expect small results in either
direction. That is exactly why it runs on paper: three months of live paper
results will say more than any further backtest.

## 9. Exploring better rules

`scripts/rule_research.py` tested well-documented rules on 16 funds the US
account can trade (stock markets, bonds, gold, silver, oil, commodities and
currency funds), December 2007 to November 2026, each at its published
settings, after 0.05% costs a side, with returns measured over cash.

| Rule | Yearly return | Sharpe | Worst fall | Sharpe by half | Sharpe, 1-5 day delays |
|---|---|---|---|---|---|
| **S&P 500 above its 10-month average, else cash** (Faber 2007) | 8.88% | **0.64** | **-20.1%** | 0.64 / 0.65 | 0.52 to 0.65 |
| S&P 500 buy and hold | 10.58% | 0.52 | -53.4% | 0.44 / 0.63 | 0.52 |
| **S&P 500 two-day RSI pullbacks** (Connors and Alvarez 2008) | 4.74% | **0.50** | -14.9% | 0.50 / 0.50 | 0.35 to 0.50 |
| Cross-asset trend, long or cash (Moskowitz, Ooi and Pedersen 2012) | 2.89% | 0.36 | -6.7% | 0.39 / 0.34 | 0.27 to 0.39 |
| Equal-risk buy and hold of the 16 funds | 2.90% | 0.28 | -13.4% | 0.34 / 0.22 | 0.25 to 0.34 |
| Cross-asset trend, long and short | 2.70% | 0.25 | -8.5% | 0.23 / 0.28 | 0.15 to 0.25 |
| S&P 500 turn of the month (McConnell and Xu 2008) | 2.54% | 0.14 | -15.6% | 0.09 / 0.21 | 0.03 to 0.14 |
| Trend blend 1/3/12 months, long and short (Hurst, Ooi and Pedersen 2017) | 2.11% | 0.14 | -6.4% | 0.24 / 0.03 | -0.06 to 0.14 |

What it says, in plain language:

- **The S&P 500 10-month filter is the most robust rule found.** It beat
  simply holding the index on return per unit of risk, in both halves and
  at every delay, and cut the worst fall from 53% to 20%. Its whole benefit
  is stepping aside in long bear markets (2008, 2022); in calm years it
  earns a little less than holding. It is a candidate for the core index
  funds, which today are held through every fall.
- **Short pullback buys on the S&P 500 (RSI of 2 days below 10, while above
  the 200-day average) are a good complement:** a similar Sharpe to holding,
  while invested only 11% of the time, so it could run beside other rules.
- **Trend following across many markets is a diversifier, not a return
  engine, in this period.** Professional trend funds also had a weak decade
  from 2011 to 2019. The long-or-cash version beat holding the same funds
  with half the worst fall.
- **Rules that failed** (turn of the month, the 1/3/12-month trend blend)
  are recorded here so nobody spends time on them again.

Nothing here is changed in the agent yet. These are candidates for Phase 2,
each to be paper traded on its own before any is trusted.

## 10. Can this become a money machine?

No honest system is a money machine, and anyone selling one is selling
something else. What exists, and what professional funds actually do:

1. **Small, real edges, stacked.** The best rules found here have a Sharpe
   ratio of about 0.5 to 0.65 each. Large funds reach 1.0 or more only by
   running many such edges that do not move together (stocks, bonds,
   commodities, currencies; trend, carry, value) and by keeping costs tiny.
   Each extra *uncorrelated* edge helps more than making one rule cleverer.
2. **Leverage turns risk into return, not into free money.** Doubling the
   forex book's size doubles both its expected profit and its worst fall.
   The edge (Sharpe) stays the same; only the stakes change.
3. **The real money is in survival and compounding.** A rule that makes
   9% a year with a 20% worst fall can be held for decades; one that makes
   15% with a 55% fall usually gets abandoned at the bottom. The loss limits
   and the 15% forex halt exist for this.
4. **Costs decide most outcomes.** Every rule tested here pays spread and
   slippage; the fast technical rules the agent ran on forex lost money
   mostly because of them.
5. **Capital matters more than cleverness at the start.** A 6% a year edge
   on $10,000 is $600 a year. For a business like Murzak Technologies, the
   agent's realistic value is as disciplined, tested, low-cost management
   of savings, not as an income replacement, at least until a long paper
   and then small-money record says otherwise.

The path to the best realistic outcome: (a) paper trade each tested rule
for at least three months; (b) keep only those whose paper results match
their backtests; (c) combine the survivors, each with its own share of
capital; (d) go live with a small amount at a regulated broker; (e) scale
only with a record.

## 11. Can Claude read the books to implement better rules?

What is possible, and what is not:

- **What Claude already knows.** The books and papers named in section 4
  were published before Claude's training, so their methods (de Prado's
  triple-barrier labelling and purged cross-validation, Jansen's pipelines,
  Carver's volatility sizing and buffering, the carry and trend research)
  are already known in detail. The rules in sections 8 and 9 were
  implemented from that knowledge and then checked against real data,
  which matters more than any book: a rule earns its place by passing the
  tests, not by being in print.
- **Paid books cannot be downloaded.** Claude will not fetch pirated
  copies. If you own a book as a PDF (bought from the publisher, for
  example), you can upload it to a Claude session and ask for specific
  chapters to be read and compared with the code. Keep it out of the
  repository: it is copyrighted, and the repository is shared.
- **Free papers can be read directly.** The research papers (Moskowitz,
  Ooi and Pedersen; Koijen et al.; Hurst, Ooi and Pedersen; Faber) are free
  on SSRN and AQR's website. This session's network allows only Yahoo
  Finance and FRED; to have Claude read papers live, allow `papers.ssrn.com`
  and `www.aqr.com` in the environment settings.
- **Tutorials and videos.** Many free tutorials contain the very mistakes
  section 4 warns about (look-ahead bias, no costs). They are useful for
  ideas; every idea still has to pass `scripts/rule_research.py` or
  `scripts/forex_factor_study.py` before it trades, even on paper.

These are historical results, not forecasts, and not investment advice.
