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

- `yield_ttm_pct` / `dividend_per_share_kes` / `eps_kes`: fallback values used
  only when the live afx.kwayisi.org scrape fails or omits a field. When the
  scrape succeeds, its values take precedence.
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
