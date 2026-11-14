# NSE Dividend History (Operator-Maintained)

`nse_dividends.json` is the source of truth for a symbol's dividend history
and payout quality. It ships **empty** — the sleeve treats an unlisted
symbol as "no fundamentals available" and excludes it from ranking, which is
the safe default (see `FundamentalsStore.get`).

`years_consecutive_paid` and `eps_trend` are judgment calls that can't be
reliably scraped from a single page snapshot — populate them from the
company's actual dividend/earnings history before adding a symbol here.

## Schema (per symbol)

```json
{
  "SCOM": {
    "yield_ttm_pct": 6.56,
    "dividend_per_share_kes": 2.30,
    "eps_kes": 2.3863,
    "years_consecutive_paid": 8,
    "eps_trend": "positive",
    "last_updated": "2026-07-11"
  }
}
```

- `yield_ttm_pct` / `dividend_per_share_kes` / `eps_kes`: fallback values.
  When the live afx.kwayisi.org scrape succeeds, its values win for every
  field it returns — even a scraped `0.0` (e.g. a suspended dividend), which
  correctly excludes the symbol rather than falling back to a stale operator
  number. The operator values here are only used when the whole scrape fails
  (returns nothing) or when a specific field is absent (scraped as `None`)
  from an otherwise successful scrape.
- `years_consecutive_paid`: consecutive years the company has paid a
  dividend without a cut. Research this from the company's investor-relations
  page or annual reports — it is not scraped.
- `eps_trend`: one of `"positive"`, `"flat"`, `"negative"` — your own
  assessment of the recent earnings trajectory.
- `last_updated`: ISO date you last verified this entry. Entries older than
  the configured `stale_days` (default 400) are flagged as stale in ticket
  rationale, but are still scored — stale data degrades to a visible flag,
  never a block.
- `payout_ratio_override`: optional — only used if EPS is zero/unavailable
  and the payout ratio can't be computed from `dividend_per_share_kes / eps_kes`.

The `SCOM` figures above (yield, DPS, EPS) are real values fetched from
afx.kwayisi.org on 2026-07-11 as a worked example of the expected shape —
`years_consecutive_paid` and `eps_trend` still need your own research before
you'd actually rely on them for a real accumulation decision.

## Dividend events (`nse_dividend_events.json`)

The NSE paper account is paid the dividends listed here, so its returns
include income, not only price moves. Add each company announcement:

```json
{
  "events": [
    {"symbol": "SCOM", "dividend_kes": 0.85, "book_closure": "2026-07-31",
     "payment_date": "2026-08-29"}
  ]
}
```

- `dividend_kes`: gross dividend per share, as announced.
- `ex_date`: the first day a buyer is no longer entitled, if the notice gives
  it. Otherwise give `book_closure` and the account takes the ex-date as three
  trading days earlier, since NSE trades settle T+3.
- `payment_date`: the day the cash arrives; the account credits it then.

Shares held the day before the ex-date qualify. The account deducts 5%
withholding tax (the final rate for resident individuals on NSE-listed
dividends) and shows gross, tax and net on the Paper Account page. The rate
is `nse_paper_trading.dividend_withholding_pct` in `config.json`.

## Bonus issues and splits (`nse_corporate_actions.json`)

Add each bonus issue or share split the company announces:

```json
{
  "events": [
    {"symbol": "KCB", "type": "bonus", "ratio": "1:10", "ex_date": "2026-05-20"},
    {"symbol": "EQTY", "type": "split", "ratio": "2:1", "ex_date": "2026-03-02"}
  ]
}
```

- `type`: `bonus` or `split`.
- `ratio`: for a bonus, new shares for every shares held (`"1:10"` is one new
  share for every ten, so 1,000 shares become 1,100). For a split, shares
  after for every share before (`"2:1"` doubles the share count; `"1:5"` is a
  consolidation).
- `ex_date`: the first day the shares trade without the entitlement.

What the agent does with these, and with the dividends above
(`src/agent/nse_corporate_actions.py`):

- **Prices are adjusted.** Prices before each ex-date are scaled so the
  event does not read as a fall, the way data vendors adjust a price series:
  a bonus or split by shares before / shares after, a dividend by
  (last close - dividend) / last close. This covers the history the strategies,
  the regime filter, the volatility behind the stops and the weekly screen
  read. An event taking effect while the agent runs is applied to the prices
  it holds on its ex-date.
- **The paper account receives the shares.** Holdings at the close before the
  ex-date grow by the ratio, in whole shares, at no cost, so the cost per share
  falls. A buy on the ex-date gets nothing, and the free shares are not a
  purchase for the "add to winners" limits.
- **Stops are not tripped by the event.** On an ex-date the trailing stop's
  peak is rescaled once (recorded, so a restart never does it twice), and for
  a dividend the stop-loss measures from the cost scaled by the same factor.

Without an entry here, a bonus issue looks like a 9% fall (1-for-10) and can
trip the 8% stop-loss. Add the announcement before the ex-date when you can.

