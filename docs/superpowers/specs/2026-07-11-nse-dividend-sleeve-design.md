# NSE Dividend-Led Long-Term Sleeve — Design

## Problem

The agent today runs a single trading book: short-to-medium horizon decisions, sell logic
tuned for that horizon, capital sized for that horizon. There is no way to hold a position
for years, no dividend awareness anywhere in the system, and no separate capital bucket for
a buy-and-accumulate strategy. The user wants to run a dividend-led, long-term investing
sleeve alongside the existing trading agent, on the NSE, using the execution rails that
already exist (the operator-approved NSE order-ticket queue).

## Goals

- A long-term investment sleeve that accumulates dividend-paying NSE equities on a
  recurring cadence, sized from a fixed percentage of total portfolio capital.
- Stock selection driven by a deterministic yield + quality score (not LLM judgment).
- An LLM-based news/red-flag check that can only flag or veto a candidate — never pick,
  score, or size a position — so the numeric decision stays auditable.
- Sleeve holdings are structurally isolated from the trading agent's positions, so the
  trading agent's sell/rebalance logic can never touch or liquidate a long-term holding.
- Reuse existing execution (`nse_order_queue`), pricing (`nse_connector`), and LLM
  (`llm_orchestrator`, with its provider-failover/caching) infrastructure rather than
  building parallel plumbing.
- Every failure mode (missing fundamentals, stale data, LLM outage) degrades to
  "proceed with reduced confidence, visibly flagged" — never to blocking the whole
  cycle or guessing.

## Non-Goals (v1)

- No selling or trimming logic for the sleeve — this is an accumulation-only strategy.
  Rebalancing/trimming is a natural v2 once the accumulation cycle has run for a while.
- No US equities / cross-market sleeve — NSE only, matching the existing execution path.
- No automated dividend-payment detection — NSE dividend confirmations aren't reliably
  scrapable, so receipt is recorded by the operator.
- No fully automatic order placement — sleeve tickets go through the same
  operator-approval queue as trading tickets. No sleeve order reaches a broker unattended.
- No dynamic/LLM-driven capital allocation between sleeves — the trading/sleeve split is
  a fixed, operator-configured percentage.

## Architecture

New, self-contained module family, deliberately parallel to (not entangled with) the
existing trading loop:

```
src/agent/sleeve/
├── fundamentals_store.py   # NSE fundamentals: scraped + operator-maintained
├── dividend_scorer.py      # deterministic yield+quality scoring (pure function)
├── sleeve_manager.py       # holdings book, capital split, accumulation cycle
├── llm_veto.py             # red-team news check (flag/veto only, no picking/sizing)
└── dividend_ledger.py      # dividends declared/received, DRIP cash
```

Integration points with the existing system:

- **Execution**: `sleeve_manager` emits buy-only tickets into the existing
  `nse_order_queue`, tagged `book: "long_term"`. These appear in the same
  operator-approval flow as trading tickets, visually distinguished by the tag.
- **Position isolation**: sleeve holdings are stored in their own state/table, entirely
  separate from the trading agent's positions. The trading agent's sell/rebalance logic
  requires no changes and cannot see or act on sleeve holdings — isolation by
  construction, not by a filter that could be bypassed or forgotten.
- **Prices**: reuses `nse_connector` quotes as-is; no new price plumbing.
- **LLM veto**: routes through the existing `llm_orchestrator`, inheriting its
  provider-failover and caching behavior. One new prompt template, one new call site.
- **Scheduling**: a monthly accumulation cycle triggered from the main agent loop via a
  date check (first trading day of the month) — no new scheduler infrastructure.

Boundary invariant: the scorer never calls the LLM, and the LLM veto never sees or
changes numbers. Score → rank → size is a pure function of stored fundamentals and
prices, so every ticket is reproducible from data alone.

## Data & Scoring

### Fundamentals store (`fundamentals_store.py`)

Two merged inputs:

- **Scraped**: extends `nse_scraper.py` with an `afx.kwayisi.org` company-page parser
  pulling EPS, P/E, and last-reported dividend yield where published. Runs on the same
  cadence as existing price scraping.
- **Operator-maintained**: `config/nse_dividends.yml` — per-symbol dividend history
  (declared date, amount, frequency) and payout ratio. Scraped coverage will be
  incomplete/lagged, so this file is the source of truth for dividend history; the
  operator updates it a few times a year from company announcements. The system never
  blocks on it being current — it scores with whatever is present and flags stale
  entries (older than a configurable threshold, e.g. 400 days).

Per-symbol record: `{yield_ttm, years_consecutive_paid, payout_ratio, eps_trend,
avg_daily_volume, last_updated}`.

### Dividend scorer (`dividend_scorer.py`)

Pure function — no network calls, no LLM:

- **Yield score**: TTM dividend yield, penalized/capped above a configurable threshold
  (default 12%) to suppress yield traps (yield inflated by a collapsed price).
- **Quality score**: consecutive years paid without a cut, payout ratio within a healthy
  band (default 30–70%), non-negative EPS trend.
- **Liquidity gate**: hard exclusion below a minimum average daily volume — a name that
  can't be exited in size shouldn't be accumulated in size.
- **Combined score**: weighted blend of yield and quality scores (config-driven weights,
  default 50/50) → ranked list → top N eligible names.

### Sleeve manager (`sleeve_manager.py`)

Monthly cycle:

1. Compute sleeve capital = fixed % of total portfolio value, plus any new
   contributions and reinvested dividend cash since the last cycle.
2. Compute target weights across the top N eligible names (equal-weight or
   score-weighted, config-driven).
3. Diff target weights against current sleeve holdings.
4. Generate buy-only tickets sized to close the gap, subject to the LLM veto check
   (below) before finalization.

No selling logic in v1 (see Non-Goals).

## LLM Veto (`llm_veto.py`)

Runs once per candidate, after scoring/sizing and before the ticket is finalized —
never during scoring. The LLM receives only the symbol, company name, and an
instruction to check for red flags (using its own knowledge and any injected recent
news snippet from `regional_news.py`): recent dividend cut/suspension, profit warning,
rights issue, delisting risk, governance/fraud concerns.

Output is schema-constrained to `{flag: bool, reason: str}`. The LLM cannot return a
score, a buy/sell decision, or a position size — it is structurally incapable of
picking or sizing.

- `flag=true`: the ticket still generates and enters the operator approval queue, but
  carries the flag and reason visibly, so the operator sees it before approving.
- LLM call fails / all providers exhausted: the ticket proceeds unflagged, carrying a
  `veto_unavailable: true` marker, so the operator knows the check didn't run rather
  than assuming a clean pass. This mirrors the fail-open philosophy in Goals — a stalled
  cycle is worse than an unflagged ticket the operator can still catch at approval time.

## Dividend Ledger (`dividend_ledger.py`)

Tracks declared and received dividends per sleeve holding. Entries are added manually or
via the operator-maintained YAML (NSE dividend payment confirmation isn't reliably
scrapable). Received dividends accumulate as sleeve cash and are swept into the next
month's accumulation cycle — a manual DRIP achieved by feeding more capital into
`sleeve_manager`'s existing allocation step, with no separate reinvestment code path.

## Error Handling Summary

Every failure mode degrades to *skip this candidate or proceed with a visible flag* —
never to guessing or blocking the whole cycle:

| Failure | Behavior |
|---|---|
| Missing/stale fundamentals for a symbol | Symbol scored with available fields; flagged stale if `last_updated` exceeds threshold; excluded from ranking if core fields (yield, volume) are absent |
| Price lookup miss | Symbol skipped for this cycle |
| LLM veto unavailable | Ticket proceeds unflagged, marked `veto_unavailable: true` |
| Zero eligible candidates after liquidity/yield gates | Cycle produces zero tickets; logged, not treated as an error |

A partial cycle with 3 tickets beats a stalled cycle with zero.

## Dashboard

Extends the existing frontend rather than introducing a new app — a "Long-Term Sleeve"
panel alongside the current Agent Operations board, following the `/api/agent-focus`
pattern:

- Sleeve capital (target vs. deployed)
- Current holdings with yield/quality scores
- Pending accumulation tickets, with veto flags surfaced
- Cumulative dividend income received

## Testing

- `dividend_scorer.py`: unit tests on scoring/ranking/liquidity-gate edge cases (zero
  volume, missing fields, yield above cap, payout ratio out of band).
- `sleeve_manager.py`: unit tests on target-weight diffing and ticket sizing given a
  mocked holdings state and capital figure.
- `llm_veto.py`: tests mock the LLM orchestrator call — flag, no-flag, and unavailable
  paths must all produce a valid ticket (never block the pipeline).
- Integration test: a full monthly cycle end-to-end against a fixture universe (fake
  fundamentals + fake prices), asserting the right tickets land in `nse_order_queue`
  with correct tags, sizes, and veto flags.

## Configuration (new)

- `config/nse_dividends.yml` — operator-maintained dividend history and payout ratios.
- `config/sleeve_config.yml` — capital split %, scoring weights, yield cap, payout
  ratio band, liquidity minimum, top-N universe size, stale-data threshold. A new file
  alongside the existing `config/` conventions, rather than extending an unrelated
  config module.

## Open Questions for Implementation Planning

- Whether the top-N universe is a fixed symbol list or dynamically derived from
  liquidity + sector diversification rules. Defaulting to a fixed, operator-curated
  list in v1 (simplest, matches the manual nature of `nse_dividends.yml`) unless the
  implementation plan finds a reason to derive it dynamically.
