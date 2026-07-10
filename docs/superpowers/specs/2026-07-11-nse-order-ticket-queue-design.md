# NSE Order-Ticket Queue (Approach A) — Design

## Problem

The agent's execution path routes all orders to a broker connector (Alpaca
for US stocks). Alpaca does not support the Nairobi Securities Exchange, and
the user's NSE broker (AIB-AXYS) is portal-only with no programmatic API.
Today NSE symbols (`config.data_manager.nse_symbols`) are scraped for prices
and fed into research/escalation, but the trading loop only iterates
`config.data_manager.symbols` (US), so NSE symbols never receive a trade
decision or an order.

Goal: let the agent make real NSE trade *decisions* using the existing
strategy/LLM/risk pipeline, and surface them as reviewable **order tickets**
a human keys into the AIB-AXYS portal — with a clean seam to swap in a real
broker connector later if AIB-AXYS ever grants API access.

## Non-goals (deferred to v2)

- Unified KES→USD consolidated equity/P&L. v1 keeps NSE P&L in KES on the
  NSE tab. The KES/USD rate is already fetched for a later unification.
- Sophisticated ticket auto-expiry. v1 uses a simple age rule.
- Auto-placing orders via a real NSE broker API (Approach C).
- Browser automation of the AIB-AXYS portal (rejected: ToS/credential risk).

## Reused, not rebuilt

- **Decision pipeline** is symbol-agnostic: `StrategyManager.generate_signals`
  → `LLMOrchestrator.validate_trade` → risk check. Same code path as US.
- **NSE prices** already flow via the `nse` connector (`get_all_quotes()`),
  refreshed ~every 30 min by the periodic scraper, with 730 days of history
  in the DB.
- **Operator-approval pattern**: `EscalationManager` (SQLite/WAL) +
  `/api/operator/escalations` + `/resolve` + a dashboard panel with
  approve/reject controls is the exact shape to mirror for order tickets.
- **`order_journal`** records executions (client_order_id, fills) for any
  symbol.

## Components

### 1. NSE decision pass — `src/agent/main.py`

New method `_evaluate_nse_symbols()`, invoked from the main loop but
**rate-limited to the NSE scrape cadence** (default: at most once per
`nse_eval_interval` seconds, config-defaulted to 1800 = 30 min). Guards:

- **Market hours**: only evaluate when `nse_connector.is_market_open()` is
  true (09:00–15:00 EAT, already implemented).
- **Price freshness**: skip a symbol whose latest quote timestamp/price is
  unchanged since its last evaluation. Feeding a stale repeated price into
  the strategies' rolling buffers would distort SMA/RSI. Track last-seen
  price per symbol in memory.

For each fresh, in-hours NSE symbol it runs the same
`generate_signals → validate_trade → risk` pipeline as US symbols. A
non-hold, risk-approved signal is handed to the order-ticket sink (below)
instead of a broker. Every evaluated NSE symbol also writes a normal
decision-journal row (so drill-down / agent-activity work for NSE too).

**Strategy warm-start**: NSE symbols' per-strategy price buffers must be
seeded from NSE historical bars at startup (US symbols already get this;
NSE currently does not — otherwise NSE strategies are blind for
`lookback_period` scrape cycles ≈ many days). Seed from the NSE historical
data already in the DB, mirroring the existing US warm-start.

### 2. Order-ticket queue — `src/agent/nse_order_queue.py` (new)

Small class mirroring `EscalationManager`'s SQLite/WAL/threading-lock style.
New table in the existing `escalations.db`:

```
CREATE TABLE IF NOT EXISTS nse_order_tickets (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    side                  TEXT NOT NULL,          -- 'buy' | 'sell'
    quantity              INTEGER NOT NULL,       -- NSE trades in whole shares
    suggested_limit_price REAL,                   -- KES, from the signal's price
    rationale             TEXT,
    ensemble_confidence   REAL,
    llm_reasoning         TEXT,
    status                TEXT DEFAULT 'pending', -- pending|placed|filled|cancelled|expired
    fill_price            REAL,                   -- KES, operator-entered
    fill_quantity         INTEGER,
    fill_at               TEXT,
    operator_notes        TEXT,
    resolved_by           TEXT
);
```

Methods: `create_ticket(...)`, `get_pending(limit)`, `mark_placed(id, by)`,
`mark_filled(id, fill_price, fill_qty, by, notes)`, `cancel(id, by, notes)`,
`expire_stale(max_age_hours)`, `recent_fills(limit)`, `positions()`
(net qty per symbol from filled tickets), `close()`.

The agent's decision pass calls `create_ticket`. This is the **swap-in
seam**: a future real-broker connector would implement the same "place this
order" call and skip the queue.

On `mark_filled`, also record the fill in `order_journal` with a
deterministic client_order_id (`nse-<symbol>-<side>-<ticket_id>`) so there's
one auditable execution record across US and NSE.

### 3. Operator API — `src/api/api_server.py`

Mirror the escalation endpoints, operator-role-gated:

- `GET  /api/operator/nse-tickets` → pending tickets (+ recent fills).
- `POST /api/operator/nse-tickets/<id>/place`  → status `placed`.
- `POST /api/operator/nse-tickets/<id>/fill`   → body `{fill_price, fill_quantity, notes?}`; status `filled`; writes order_journal.
- `POST /api/operator/nse-tickets/<id>/cancel` → body `{notes?}`; status `cancelled`.

### 4. Dashboard panel — `frontend/components/AdvancedDashboard.js` (NSE tab)

A "NSE Order Tickets" panel on the existing NSE tab, operator-only (mirrors
how the Agent Log / escalations gate). Lists pending tickets: symbol, side,
quantity, suggested limit price (KES), confidence, rationale. Controls per
ticket: **Mark Placed** → **Mark Filled** (prompt for actual fill price +
quantity) / **Cancel**. A small "Recent NSE fills" list beneath shows the
executed history. Non-operators see an "Operator view only" placeholder.

### 5. NSE positions/P&L (KES, v1)

The NSE tab's existing "YOUR POSITION" column currently reads broker
positions (which never include NSE). Add an NSE-positions source derived
from `nse_order_queue.positions()` (net filled qty × current scraped price,
in KES) so the NSE tab reflects real NSE holdings and unrealized P&L in KES.

## Data flow

```
NSE scraper (~30m) ─▶ nse_connector.get_all_quotes()
                          │  (fresh + in-hours guard)
                          ▼
       _evaluate_nse_symbols()  ── generate_signals → validate_trade → risk
                          │  non-hold, approved
                          ▼
        nse_order_queue.create_ticket()  ─▶  escalations.db (pending)
                          │
   dashboard NSE panel ◀──┤  GET /api/operator/nse-tickets
        operator places order in AIB-AXYS portal (manual)
        operator: Mark Placed → Mark Filled(price, qty)
                          ▼
        nse_order_queue.mark_filled() ─▶ order_journal + NSE position tally
```

## Error handling

- No NSE connector / no quotes → skip the pass, log at debug, no ticket.
- LLM unavailable (e.g. 429) → existing `validate_trade` fallback returns the
  base signal (does not block); NSE follows the same behavior as US.
- Duplicate tickets: don't create a new pending ticket for a symbol+side that
  already has one pending (dedupe in `create_ticket`).
- `mark_filled` with bad/missing price or qty → 400, no journal write.
- All DB access under the connection lock, WAL mode (same as EscalationManager).

## Testing

- Unit: `nse_order_queue` create → get_pending → mark_placed → mark_filled
  round-trip; dedupe of duplicate pending symbol+side; `positions()` net
  calc; `expire_stale`.
- Unit: `_evaluate_nse_symbols` freshness guard (stale price → no eval) and
  market-hours guard (closed → no eval), using a stubbed NSE connector.
- API: nse-tickets endpoints return 403 for viewer, 200 for operator; fill
  with bad body → 400.
- Manual (browser preview): with the backend live, confirm the NSE panel
  renders pending tickets, and a Mark Filled updates the recent-fills list
  and NSE positions. (Real signals are sparse; a seeded/test ticket may be
  used to exercise the UI.)

## Rollout

- Additive: new table in existing DB, new module, new endpoints, new panel.
  No change to the US execution path.
- Config: `data_manager.nse_eval_interval` (default 1800),
  `nse_order_tickets.max_age_hours` (default 24) — both optional with
  defaults, no required config changes.
