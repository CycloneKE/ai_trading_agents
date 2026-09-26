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
