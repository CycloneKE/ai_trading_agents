# NSE Dividend-Led Long-Term Sleeve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained, buy-only accumulation sleeve that scores NSE stocks on dividend yield + quality, runs an LLM red-flag veto that can only flag (never pick or size), and emits monthly accumulation tickets into the existing NSE order-ticket queue — fully isolated from the trading agent's own positions.

**Architecture:** A new `src/agent/sleeve/` package (fundamentals store, deterministic scorer, LLM veto, dividend ledger, orchestrating manager) plugs into two existing systems: `NseOrderQueue` (gets a new `book` tag so sleeve tickets/positions never mix with trading tickets/positions) and `LLMOrchestrator` (via its existing generic `propose_json` hook). A new `_run_sleeve_cycle()` method on the main trading loop calls the sleeve manager at most once per calendar month; a new `/api/sleeve` endpoint and `SleeveDashboard` panel expose it in the dashboard.

**Tech Stack:** Python (stdlib `sqlite3`, `dataclasses`, `re`), pytest, existing Flask API server, existing React/Next.js frontend (plain `useState`/`useEffect`, no new libraries).

## Global Constraints

- No selling/trimming logic — this sleeve only ever creates `side='buy'` tickets (design non-goal).
- No sleeve order reaches a broker unattended — sleeve tickets go through the same operator-approval queue (`nse_order_queue`) as trading tickets.
- The deterministic scorer (`dividend_scorer.py`) must never call the LLM or the network. The LLM veto (`llm_veto.py`) must never return a score, action, or size — only `{flag, reason}`.
- Every failure mode (missing fundamentals, stale data, LLM unavailable, price lookup miss) must degrade to *skip this candidate or proceed with a visible flag* — never raise out of the monthly cycle or block the whole run.
- Follow existing patterns: SQLite+WAL with a `threading.Lock` for any new persistent store (mirrors `NseOrderQueue`/`EscalationManager`), config read via `config.get('<key>', {})` sub-dicts (mirrors `CashDeploymentPolicy`), tests are flat files under `tests/` using pytest fixtures (mirrors `test_nse_order_queue.py`), no manual `sys.path.insert` in new test files (conftest.py already puts `src/agent` etc. on the path).
- Spec reference: `docs/superpowers/specs/2026-07-11-nse-dividend-sleeve-design.md`.

---

## File Structure

```
src/agent/nse_order_queue.py        # MODIFY: add `book` tag (migration-safe)
src/connectors/nse_scraper.py       # MODIFY: add afx company-page fundamentals scrape
src/agent/sleeve/__init__.py        # CREATE: empty package marker
src/agent/sleeve/fundamentals_store.py  # CREATE
src/agent/sleeve/dividend_scorer.py     # CREATE
src/agent/sleeve/llm_veto.py            # CREATE
src/agent/sleeve/dividend_ledger.py     # CREATE
src/agent/sleeve/sleeve_manager.py      # CREATE
src/agent/main.py                   # MODIFY: wire components + monthly cycle hook
src/api/sleeve_view.py              # CREATE
src/api/api_server.py               # MODIFY: add /api/sleeve route
frontend/components/SleeveDashboard.js   # CREATE
frontend/components/AdvancedDashboard.js # MODIFY: mount SleeveDashboard
config/config.json                  # MODIFY: add "sleeve" key
config/nse_dividends.json           # CREATE: empty operator-maintained store
config/nse_dividends.README.md      # CREATE: schema + worked example

tests/test_nse_order_queue.py       # MODIFY: add book-tag tests
tests/test_nse_company_scraper.py   # CREATE
tests/test_fundamentals_store.py    # CREATE
tests/test_dividend_scorer.py       # CREATE
tests/test_llm_veto.py              # CREATE
tests/test_dividend_ledger.py       # CREATE
tests/test_sleeve_manager.py        # CREATE
tests/test_sleeve_integration.py    # CREATE
tests/test_sleeve_view.py           # CREATE
```

Each `sleeve/` file has one job: `fundamentals_store` gets data, `dividend_scorer` ranks it (pure), `llm_veto` red-teams a candidate (pure w.r.t. state), `dividend_ledger` tracks cash, `sleeve_manager` is the only file that talks to all of them plus `nse_order_queue`.

---

### Task 1: Tag NSE order tickets with a `book` (trading vs. long_term)

Without this, `NseOrderQueue.positions()` blends sleeve and trading fills together, breaking the isolation the design requires. This is an additive, migration-safe change to a file already in production (existing `escalations.db` files must keep working).

**Files:**
- Modify: `src/agent/nse_order_queue.py`
- Test: `tests/test_nse_order_queue.py`

**Interfaces:**
- Produces: `NseOrderQueue.create_ticket(..., book: str = 'trading') -> Optional[int]` (new optional kwarg, default preserves existing callers' behavior)
- Produces: `NseOrderQueue.positions(book: Optional[str] = None) -> Dict[str, Dict[str, Any]]` (new optional kwarg; `None` = unfiltered, same as today)
- Every row from `get_pending()` / `recent_fills()` now includes a `'book'` key (both already `SELECT *`, no code change needed there).

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_nse_order_queue.py` (add `import sqlite3` to its imports at the top):

```python
def test_default_book_is_trading(queue):
    queue.create_ticket('SCOM', 'buy', 100, suggested_limit_price=15.0)
    pending = queue.get_pending()
    assert pending[0]['book'] == 'trading'


def test_long_term_book_does_not_collide_with_trading_dup_check(queue):
    trading_id = queue.create_ticket('SCOM', 'buy', 100, suggested_limit_price=15.0, book='trading')
    sleeve_id = queue.create_ticket('SCOM', 'buy', 50, suggested_limit_price=15.0, book='long_term')
    assert trading_id is not None
    assert sleeve_id is not None
    books = {t['book'] for t in queue.get_pending()}
    assert books == {'trading', 'long_term'}


def test_positions_filtered_by_book(queue):
    trading_id = queue.create_ticket('SCOM', 'buy', 1000, book='trading')
    queue.mark_filled(trading_id, 15.0, 1000)
    sleeve_id = queue.create_ticket('SCOM', 'buy', 200, book='long_term')
    queue.mark_filled(sleeve_id, 16.0, 200)

    trading_only = queue.positions(book='trading')
    sleeve_only = queue.positions(book='long_term')
    everything = queue.positions()

    assert trading_only['SCOM']['quantity'] == 1000
    assert sleeve_only['SCOM']['quantity'] == 200
    assert everything['SCOM']['quantity'] == 1200


def test_migrates_existing_db_without_book_column(tmp_path):
    path = str(tmp_path / 'old_schema.db')
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE nse_order_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            symbol TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL,
            suggested_limit_price REAL, rationale TEXT, ensemble_confidence REAL,
            llm_reasoning TEXT, status TEXT DEFAULT 'pending', fill_price REAL,
            fill_quantity INTEGER, fill_at TEXT, operator_notes TEXT, resolved_by TEXT
        );
    """)
    conn.execute(
        "INSERT INTO nse_order_tickets (created_at, symbol, side, quantity, status)"
        " VALUES ('2026-01-01T00:00:00', 'OLD', 'buy', 10, 'pending')")
    conn.commit()
    conn.close()

    q = NseOrderQueue(path)
    pending = q.get_pending()
    assert len(pending) == 1
    assert pending[0]['book'] == 'trading'  # DEFAULT backfilled onto the pre-existing row
    q.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nse_order_queue.py -v -k "book or migrates"`
Expected: FAIL — `sqlite3.OperationalError: no such column: book` (or `KeyError: 'book'`), and `positions() got an unexpected keyword argument 'book'`.

- [ ] **Step 3: Implement the migration-safe schema + `book`-aware methods**

In `src/agent/nse_order_queue.py`, change `_SCHEMA` (add the column to fresh installs):

```python
_SCHEMA = """
CREATE TABLE IF NOT EXISTS nse_order_tickets (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    side                  TEXT NOT NULL,          -- 'buy' | 'sell'
    quantity              INTEGER NOT NULL,       -- NSE trades in whole shares
    suggested_limit_price REAL,                   -- KES
    rationale             TEXT,
    ensemble_confidence   REAL,
    llm_reasoning         TEXT,
    status                TEXT DEFAULT 'pending', -- pending|placed|filled|cancelled|expired
    fill_price            REAL,                   -- KES, operator-entered
    fill_quantity         INTEGER,
    fill_at               TEXT,
    operator_notes        TEXT,
    resolved_by           TEXT,
    book                  TEXT DEFAULT 'trading'  -- 'trading' | 'long_term'
);
CREATE INDEX IF NOT EXISTS idx_nse_tickets_status ON nse_order_tickets(status, created_at);
CREATE INDEX IF NOT EXISTS idx_nse_tickets_symbol ON nse_order_tickets(symbol);
"""
```

In `__init__`, right after `self._conn.executescript(_SCHEMA)` / `self._conn.commit()`, add the migration for pre-existing DB files:

```python
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate_add_book_column()
        logger.info(f"NseOrderQueue initialized at {db_path}")

    def _migrate_add_book_column(self) -> None:
        """CREATE TABLE IF NOT EXISTS doesn't add columns to a table that
        already existed before `book` was introduced — do that explicitly so
        pre-existing escalations.db files keep working."""
        cols = [row[1] for row in self._conn.execute(
            "PRAGMA table_info(nse_order_tickets)").fetchall()]
        if 'book' not in cols:
            self._conn.execute(
                "ALTER TABLE nse_order_tickets ADD COLUMN book TEXT DEFAULT 'trading'")
            self._conn.commit()
```

Update `create_ticket` to accept and store `book`, and scope the duplicate-check to it (so a pending trading ticket for SCOM/buy never blocks a pending sleeve ticket for SCOM/buy, or vice versa):

```python
    def create_ticket(self, symbol: str, side: str, quantity: int,
                      suggested_limit_price: Optional[float] = None,
                      rationale: str = '', ensemble_confidence: Optional[float] = None,
                      llm_reasoning: str = '', book: str = 'trading') -> Optional[int]:
        """Create a pending order ticket. Returns the ticket id, or None if an
        identical pending ticket (same symbol+side+book) already exists — the
        agent re-proposes the same trade every scrape/cycle while the signal
        persists, and we don't want a pile of duplicates waiting for the
        operator. `book` scopes the dedupe so a trading and a long_term
        ticket for the same symbol/side never collide."""
        symbol = symbol.upper()
        side = side.lower()
        if side not in ('buy', 'sell') or quantity <= 0:
            return None
        with self._lock:
            dup = self._conn.execute(
                "SELECT id FROM nse_order_tickets WHERE symbol = ? AND side = ?"
                " AND status = 'pending' AND book = ? LIMIT 1", (symbol, side, book)).fetchone()
            if dup:
                return None
            cur = self._conn.execute(
                "INSERT INTO nse_order_tickets (created_at, symbol, side, quantity,"
                " suggested_limit_price, rationale, ensemble_confidence, llm_reasoning,"
                " status, book) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (datetime.utcnow().isoformat(), symbol, side, int(quantity),
                 suggested_limit_price, rationale, ensemble_confidence, llm_reasoning, book))
            self._conn.commit()
            return cur.lastrowid
```

Update `positions` to accept an optional `book` filter:

```python
    def positions(self, book: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Net position per symbol from filled tickets: signed quantity and
        the volume-weighted average entry price (KES). Pass `book` to scope
        to just 'trading' or 'long_term' fills; omit for the blended view."""
        with self._lock:
            query = ("SELECT symbol, side, fill_quantity, fill_price FROM nse_order_tickets"
                     " WHERE status = 'filled' AND fill_quantity > 0")
            params: Tuple[Any, ...] = ()
            if book is not None:
                query += " AND book = ?"
                params = (book,)
            query += " ORDER BY fill_at ASC"
            rows = self._conn.execute(query, params).fetchall()
        book_map: Dict[str, Dict[str, Any]] = {}
        for symbol, side, qty, price in rows:
            b = book_map.setdefault(symbol, {'quantity': 0, 'buy_qty': 0, 'buy_cost': 0.0})
            b['quantity'] += qty if side == 'buy' else -qty
            if side == 'buy':
                b['buy_qty'] += qty
                b['buy_cost'] += qty * price
        out = {}
        for symbol, b in book_map.items():
            avg = round(b['buy_cost'] / b['buy_qty'], 2) if b['buy_qty'] > 0 else 0.0
            out[symbol] = {'quantity': b['quantity'], 'avg_entry_price_kes': avg}
        return out
```

(Only the local variable `book` inside `positions` was renamed to `book_map` to avoid shadowing the new `book` parameter — the aggregation logic itself is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nse_order_queue.py -v`
Expected: PASS (all tests, including the pre-existing ones — confirms the migration didn't break current behavior).

- [ ] **Step 5: Commit**

```bash
git add src/agent/nse_order_queue.py tests/test_nse_order_queue.py
git commit -m "feat(nse): tag order tickets with a book (trading/long_term)"
```

---

### Task 2: Scrape EPS / P/E / dividend fields from afx.kwayisi.org company pages

`nse_scraper.py` currently only scrapes daily price bars. The sleeve needs per-symbol dividend yield, EPS, and P/E, which afx.kwayisi.org's company pages publish as `<tr><td>Label</td><td>Value</td></tr>` rows inside a "Growth & Valuation" table (verified by fetching `https://afx.kwayisi.org/nse/scom.html` directly).

**Files:**
- Modify: `src/connectors/nse_scraper.py`
- Test: `tests/test_nse_company_scraper.py`

**Interfaces:**
- Produces: `scrape_afx_company_page(symbol: str) -> Optional[Dict[str, Any]]` — network call, returns `{'symbol', 'eps', 'pe_ratio', 'dividend_per_share', 'dividend_yield_pct'}` or `None`.
- Produces: `_parse_company_page(html: str, symbol: str) -> Optional[Dict[str, Any]]` — pure parser, used directly by tests (no network).
- Consumes: existing `_parse_num(s: str) -> float` (already in this file — strips commas/`%`, never raises).

- [ ] **Step 1: Write the failing test**

Create `tests/test_nse_company_scraper.py`:

```python
"""Regression fixture is the real HTML snippet from
https://afx.kwayisi.org/nse/scom.html (fetched 2026-07-11) — verifies the
parser against afx's actual "Growth & Valuation" table markup, not a
guessed structure."""
from src.connectors.nse_scraper import _parse_company_page

REAL_SCOM_SNIPPET = (
    '<table><thead><tr><th colspan="2">Growth &amp; Valuation</th></tr></thead>'
    '<tbody><tr><td>Earnings Per Share</td><td class="hi">2.3863</td></tr>'
    '<tr><td>Price/Earning Ratio</td><td class="hi">14.69</td></tr>'
    '<tr><td>Dividend Per Share</td><td>2.30</td></tr>'
    '<tr><td>Dividend Yield</td><td>6.56%</td></tr>'
    '<tr><td>Shares Outstanding</td><td>40.1B</td></tr>'
    '<tr><td>Market Capitalization</td><td>1.4T</td></tr></tbody></table>'
)


def test_parses_real_afx_company_page_snippet():
    result = _parse_company_page(REAL_SCOM_SNIPPET, 'scom')
    assert result == {
        'symbol': 'SCOM', 'eps': 2.3863, 'pe_ratio': 14.69,
        'dividend_per_share': 2.30, 'dividend_yield_pct': 6.56,
    }


def test_missing_growth_valuation_table_returns_none():
    assert _parse_company_page('<html><body>no data here</body></html>', 'XXXX') is None


def test_partial_fields_missing_yield_returns_none():
    html = ('<table><thead><tr><th colspan="2">Growth &amp; Valuation</th></tr></thead>'
            '<tbody><tr><td>Earnings Per Share</td><td>1.0</td></tr></tbody></table>')
    assert _parse_company_page(html, 'XXXX') is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nse_company_scraper.py -v`
Expected: FAIL with `ImportError: cannot import name '_parse_company_page'`.

- [ ] **Step 3: Implement the scraper**

In `src/connectors/nse_scraper.py`, add `import re` to the imports at the top of the file, then add near `scrape_afx_kwayisi` (after its closing, before `_parse_num`):

```python
_COMPANY_PAGE_FIELDS = {
    'Earnings Per Share': 'eps',
    'Price/Earning Ratio': 'pe_ratio',
    'Dividend Per Share': 'dividend_per_share',
    'Dividend Yield': 'dividend_yield_pct',
}


def scrape_afx_company_page(symbol: str) -> Optional[Dict[str, Any]]:
    """Scrape EPS / P-E / dividend fields from a company's afx.kwayisi.org
    page. Returns None on any network error, non-200, or unrecognized page
    layout — callers must treat that as "no fundamentals available", never
    guess a value."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = f"https://afx.kwayisi.org/nse/{symbol.lower()}.html"
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return None
        html_text = resp.text
    except Exception as e:
        logger.warning(f"AFX company page scrape error ({symbol}): {e}")
        return None
    return _parse_company_page(html_text, symbol)


def _parse_company_page(html_text: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Pure parser for the "Growth & Valuation" table afx renders as
    <tr><td>Label</td><td>Value</td></tr> rows. No network, safe to unit test
    directly against a captured HTML fixture."""
    marker = html_text.find('Growth &amp; Valuation')
    if marker == -1:
        marker = html_text.find('Growth & Valuation')
    if marker == -1:
        return None
    section = html_text[marker:marker + 2000]
    row_re = re.compile(r'<td>([^<]+)</td>\s*<td[^>]*>([^<]*)</td>', re.IGNORECASE)
    values: Dict[str, str] = {}
    for label, value in row_re.findall(section):
        label = label.strip()
        if label in _COMPANY_PAGE_FIELDS:
            values[_COMPANY_PAGE_FIELDS[label]] = value.strip()
    if 'eps' not in values or 'dividend_yield_pct' not in values:
        return None
    return {
        'symbol': symbol.upper(),
        'eps': _parse_num(values.get('eps', '0')),
        'pe_ratio': _parse_num(values.get('pe_ratio', '0')),
        'dividend_per_share': _parse_num(values.get('dividend_per_share', '0')),
        'dividend_yield_pct': _parse_num(values.get('dividend_yield_pct', '0')),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nse_company_scraper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/connectors/nse_scraper.py tests/test_nse_company_scraper.py
git commit -m "feat(nse): scrape EPS/P-E/dividend fields from afx company pages"
```

---

### Task 3: Fundamentals store

Merges the scraped afx fields, the operator-maintained dividend-history JSON, and locally stored NSE price history (for average daily volume) into one per-symbol record. A symbol with no operator entry, or no usable yield, is excluded entirely — never scored with guessed data.

**Files:**
- Create: `src/agent/sleeve/__init__.py` (empty)
- Create: `src/agent/sleeve/fundamentals_store.py`
- Create: `config/nse_dividends.json`
- Create: `config/nse_dividends.README.md`
- Test: `tests/test_fundamentals_store.py`

**Interfaces:**
- Produces: `@dataclass Fundamentals` with fields `symbol: str, yield_ttm_pct: float, dividend_per_share_kes: float, eps_kes: float, payout_ratio: Optional[float], years_consecutive_paid: int, eps_trend: str, avg_daily_volume: int, last_updated: str`, plus method `is_stale(stale_days: int) -> bool`.
- Produces: `FundamentalsStore.get(symbol: str) -> Optional[Fundamentals]` and `FundamentalsStore.get_all(symbols: List[str]) -> List[Fundamentals]` (skips symbols that return `None`, logs at debug level).
- Consumes: `src.connectors.nse_scraper.scrape_afx_company_page`, `src.connectors.nse_scraper.load_csv`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamentals_store.py`:

```python
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from src.agent.sleeve.fundamentals_store import Fundamentals, FundamentalsStore


def test_is_stale_true_past_threshold():
    old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
    f = Fundamentals('SCOM', 6.5, 2.3, 2.4, 0.48, 8, 'positive', 100000, old)
    assert f.is_stale(400) is True


def test_is_stale_false_within_threshold():
    recent = datetime.now(timezone.utc).isoformat()
    f = Fundamentals('SCOM', 6.5, 2.3, 2.4, 0.48, 8, 'positive', 100000, recent)
    assert f.is_stale(400) is False


@pytest.fixture
def dividends_file():
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    os.remove(path)


def test_returns_none_for_symbol_with_no_operator_entry(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    assert store.get('SCOM') is None


def test_merges_scraped_and_operator_data(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'years_consecutive_paid': 8, 'eps_trend': 'positive',
                            'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr(
        'src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
        lambda s: {'symbol': 'SCOM', 'eps': 2.3863, 'pe_ratio': 14.69,
                   'dividend_per_share': 2.30, 'dividend_yield_pct': 6.56})
    monkeypatch.setattr(
        'src.agent.sleeve.fundamentals_store.load_csv', lambda s: [])

    f = store.get('SCOM')
    assert f is not None
    assert f.symbol == 'SCOM'
    assert f.yield_ttm_pct == 6.56
    assert f.years_consecutive_paid == 8
    assert f.eps_trend == 'positive'
    assert round(f.payout_ratio, 4) == round(2.30 / 2.3863, 4)


def test_scrape_failure_falls_back_to_operator_fields(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'yield_ttm_pct': 6.0, 'dividend_per_share_kes': 2.0,
                            'eps_kes': 2.0, 'years_consecutive_paid': 5,
                            'eps_trend': 'flat', 'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.load_csv', lambda s: [])

    f = store.get('SCOM')
    assert f is not None
    assert f.yield_ttm_pct == 6.0
    assert f.payout_ratio == 1.0


def test_avg_daily_volume_from_local_history(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'yield_ttm_pct': 6.0, 'dividend_per_share_kes': 2.0,
                            'eps_kes': 2.0, 'years_consecutive_paid': 5,
                            'eps_trend': 'flat', 'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file, volume_window=3)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    monkeypatch.setattr(
        'src.agent.sleeve.fundamentals_store.load_csv',
        lambda s: [{'volume': '100'}, {'volume': '200'}, {'volume': '300'}, {'volume': '9999'}])

    f = store.get('SCOM')
    # volume_window=3 keeps the last 3 rows [200, 300, 9999]; mean = 3499.67,
    # int() truncates to 3499.
    assert f.avg_daily_volume == 3499


def test_get_all_skips_missing_symbols(dividends_file, monkeypatch):
    with open(dividends_file, 'w') as f:
        json.dump({'SCOM': {'yield_ttm_pct': 6.0, 'dividend_per_share_kes': 2.0,
                            'eps_kes': 2.0, 'years_consecutive_paid': 5,
                            'eps_trend': 'flat', 'last_updated': '2026-07-01'}}, f)
    store = FundamentalsStore(dividends_path=dividends_file)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.scrape_afx_company_page',
                        lambda s: None)
    monkeypatch.setattr('src.agent.sleeve.fundamentals_store.load_csv', lambda s: [])

    result = store.get_all(['SCOM', 'UNKNOWN'])
    assert [f.symbol for f in result] == ['SCOM']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fundamentals_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.sleeve'`.

- [ ] **Step 3: Implement the store**

Create `src/agent/sleeve/__init__.py` (empty file).

Create `src/agent/sleeve/fundamentals_store.py`:

```python
"""Merges scraped afx fundamentals, operator-maintained dividend history
(config/nse_dividends.json), and locally stored NSE price history into
per-symbol scoring inputs for the dividend sleeve.

A symbol with no operator entry, or with no usable dividend yield after
merging, returns None from `get()` — the caller excludes it from ranking
rather than scoring it on guessed data.
"""
import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.connectors.nse_scraper import load_csv, scrape_afx_company_page
from src.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_DIVIDENDS_PATH = PROJECT_ROOT / 'config' / 'nse_dividends.json'


@dataclass
class Fundamentals:
    symbol: str
    yield_ttm_pct: float
    dividend_per_share_kes: float
    eps_kes: float
    payout_ratio: Optional[float]
    years_consecutive_paid: int
    eps_trend: str  # 'positive' | 'flat' | 'negative'
    avg_daily_volume: int
    last_updated: str  # ISO date/datetime string

    def is_stale(self, stale_days: int) -> bool:
        try:
            updated = datetime.fromisoformat(self.last_updated)
        except ValueError:
            return True
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - updated).days
        return age_days > stale_days


class FundamentalsStore:
    def __init__(self, dividends_path: Path = DEFAULT_DIVIDENDS_PATH,
                volume_window: int = 20):
        self.dividends_path = Path(dividends_path)
        self.volume_window = volume_window

    def _load_operator_data(self) -> Dict[str, Dict[str, Any]]:
        if not self.dividends_path.exists():
            return {}
        try:
            with open(self.dividends_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {self.dividends_path}: {e}")
            return {}

    def _avg_daily_volume(self, symbol: str) -> int:
        bars = load_csv(symbol)
        if not bars:
            return 0
        recent = bars[-self.volume_window:]
        volumes = [int(float(b.get('volume', 0) or 0)) for b in recent]
        return int(statistics.mean(volumes)) if volumes else 0

    def get(self, symbol: str) -> Optional[Fundamentals]:
        operator = self._load_operator_data().get(symbol.upper())
        if not operator:
            return None
        scraped = scrape_afx_company_page(symbol) or {}
        eps = scraped.get('eps') or operator.get('eps_kes', 0.0)
        dps = scraped.get('dividend_per_share') or operator.get('dividend_per_share_kes', 0.0)
        yield_pct = scraped.get('dividend_yield_pct') or operator.get('yield_ttm_pct', 0.0)
        if not yield_pct or not dps:
            return None
        payout_ratio = round(dps / eps, 4) if eps > 0 else operator.get('payout_ratio_override')
        return Fundamentals(
            symbol=symbol.upper(),
            yield_ttm_pct=float(yield_pct),
            dividend_per_share_kes=float(dps),
            eps_kes=float(eps),
            payout_ratio=payout_ratio,
            years_consecutive_paid=int(operator.get('years_consecutive_paid', 0)),
            eps_trend=operator.get('eps_trend', 'flat'),
            avg_daily_volume=self._avg_daily_volume(symbol),
            last_updated=operator.get('last_updated', datetime.now(timezone.utc).isoformat()),
        )

    def get_all(self, symbols: List[str]) -> List[Fundamentals]:
        out = []
        for s in symbols:
            f = self.get(s)
            if f:
                out.append(f)
            else:
                logger.debug(f"Sleeve: excluding {s} from ranking (no usable fundamentals)")
        return out
```

Create `config/nse_dividends.json` (ships empty — see README for why real dividend-history fields are never auto-populated):

```json
{}
```

Create `config/nse_dividends.README.md`:

```markdown
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fundamentals_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/sleeve/__init__.py src/agent/sleeve/fundamentals_store.py \
        config/nse_dividends.json config/nse_dividends.README.md \
        tests/test_fundamentals_store.py
git commit -m "feat(sleeve): fundamentals store merging scraped + operator dividend data"
```

---

### Task 4: Dividend scorer (pure deterministic scoring)

**Files:**
- Create: `src/agent/sleeve/dividend_scorer.py`
- Test: `tests/test_dividend_scorer.py`

**Interfaces:**
- Consumes: `Fundamentals` from Task 3 (`src.agent.sleeve.fundamentals_store.Fundamentals`).
- Produces: `@dataclass ScoringConfig` (fields: `yield_cap_pct, yield_weight, quality_weight, payout_ratio_min, payout_ratio_max, years_paid_target, min_avg_daily_volume, stale_days` — all with defaults).
- Produces: `@dataclass ScoredCandidate` (fields: `symbol, combined_score, yield_score, quality_score, stale`).
- Produces: `score_symbol(f: Fundamentals, cfg: ScoringConfig) -> Optional[ScoredCandidate]` and `rank_candidates(fundamentals: List[Fundamentals], cfg: ScoringConfig, top_n: int) -> List[ScoredCandidate]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dividend_scorer.py`:

```python
from src.agent.sleeve.dividend_scorer import ScoringConfig, rank_candidates, score_symbol
from src.agent.sleeve.fundamentals_store import Fundamentals


def make_fundamentals(**overrides):
    base = dict(symbol='SCOM', yield_ttm_pct=6.5, dividend_per_share_kes=2.3,
               eps_kes=2.4, payout_ratio=0.48, years_consecutive_paid=8,
               eps_trend='positive', avg_daily_volume=100_000,
               last_updated='2026-07-01')
    base.update(overrides)
    return Fundamentals(**base)


def test_liquidity_gate_excludes_illiquid_symbol():
    cfg = ScoringConfig(min_avg_daily_volume=50_000)
    f = make_fundamentals(avg_daily_volume=1_000)
    assert score_symbol(f, cfg) is None


def test_zero_yield_excluded():
    cfg = ScoringConfig()
    f = make_fundamentals(yield_ttm_pct=0.0)
    assert score_symbol(f, cfg) is None


def test_yield_above_cap_is_capped_not_boosted():
    cfg = ScoringConfig(yield_cap_pct=12.0)
    normal = score_symbol(make_fundamentals(yield_ttm_pct=10.0), cfg)
    trap = score_symbol(make_fundamentals(yield_ttm_pct=30.0), cfg)
    assert trap.yield_score == 1.0  # capped at 12%, same as any yield >= cap
    assert normal.yield_score < trap.yield_score


def test_payout_ratio_outside_band_lowers_quality_score():
    cfg = ScoringConfig(payout_ratio_min=0.30, payout_ratio_max=0.70)
    healthy = score_symbol(make_fundamentals(payout_ratio=0.5), cfg)
    unhealthy = score_symbol(make_fundamentals(payout_ratio=0.95), cfg)
    assert healthy.quality_score > unhealthy.quality_score


def test_negative_eps_trend_lowers_quality_score():
    cfg = ScoringConfig()
    positive = score_symbol(make_fundamentals(eps_trend='positive'), cfg)
    negative = score_symbol(make_fundamentals(eps_trend='negative'), cfg)
    assert positive.quality_score > negative.quality_score


def test_stale_flag_propagates():
    cfg = ScoringConfig(stale_days=1)
    stale = score_symbol(make_fundamentals(last_updated='2020-01-01'), cfg)
    assert stale.stale is True


def test_rank_candidates_orders_by_combined_score_desc():
    cfg = ScoringConfig()
    strong = make_fundamentals(symbol='SCOM', yield_ttm_pct=8.0, years_consecutive_paid=10)
    weak = make_fundamentals(symbol='EQTY', yield_ttm_pct=3.0, years_consecutive_paid=1)
    ranked = rank_candidates([weak, strong], cfg, top_n=5)
    assert [c.symbol for c in ranked] == ['SCOM', 'EQTY']


def test_rank_candidates_respects_top_n():
    cfg = ScoringConfig(min_avg_daily_volume=0)
    many = [make_fundamentals(symbol=f'SYM{i}', yield_ttm_pct=1.0 + i) for i in range(10)]
    ranked = rank_candidates(many, cfg, top_n=3)
    assert len(ranked) == 3


def test_rank_candidates_excludes_gated_symbols():
    cfg = ScoringConfig(min_avg_daily_volume=50_000)
    ok = make_fundamentals(symbol='SCOM', avg_daily_volume=100_000)
    illiquid = make_fundamentals(symbol='EQTY', avg_daily_volume=100)
    ranked = rank_candidates([ok, illiquid], cfg, top_n=5)
    assert [c.symbol for c in ranked] == ['SCOM']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dividend_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.sleeve.dividend_scorer'`.

- [ ] **Step 3: Implement the scorer**

Create `src/agent/sleeve/dividend_scorer.py`:

```python
"""Deterministic yield + quality scoring for NSE dividend sleeve candidates.

Pure function: no network calls, no LLM. Every ranking is reproducible from
a Fundamentals record and the configured weights/thresholds — this is the
one place in the sleeve where the buy decision's numbers come from.
"""
from dataclasses import dataclass
from typing import List, Optional

from src.agent.sleeve.fundamentals_store import Fundamentals


@dataclass
class ScoringConfig:
    yield_cap_pct: float = 12.0
    yield_weight: float = 0.5
    quality_weight: float = 0.5
    payout_ratio_min: float = 0.30
    payout_ratio_max: float = 0.70
    years_paid_target: int = 5
    min_avg_daily_volume: int = 50_000
    stale_days: int = 400


@dataclass
class ScoredCandidate:
    symbol: str
    combined_score: float
    yield_score: float
    quality_score: float
    stale: bool


def score_symbol(f: Fundamentals, cfg: ScoringConfig) -> Optional[ScoredCandidate]:
    """Returns None when the symbol fails a hard gate (illiquid, no/negative
    yield) rather than assigning a low score — a gated-out name never
    appears in the ranked list at all."""
    if f.avg_daily_volume < cfg.min_avg_daily_volume:
        return None
    if f.yield_ttm_pct <= 0:
        return None

    yield_capped = min(f.yield_ttm_pct, cfg.yield_cap_pct)
    yield_score = round(yield_capped / cfg.yield_cap_pct, 4)

    payout_ok = (f.payout_ratio is not None and
                cfg.payout_ratio_min <= f.payout_ratio <= cfg.payout_ratio_max)
    consistency_score = min(f.years_consecutive_paid / cfg.years_paid_target, 1.0)
    trend_ok = 1.0 if f.eps_trend != 'negative' else 0.0
    quality_score = round(
        ((1.0 if payout_ok else 0.0) + consistency_score + trend_ok) / 3, 4)

    combined = round(cfg.yield_weight * yield_score + cfg.quality_weight * quality_score, 4)
    return ScoredCandidate(
        symbol=f.symbol, combined_score=combined,
        yield_score=yield_score, quality_score=quality_score,
        stale=f.is_stale(cfg.stale_days),
    )


def rank_candidates(fundamentals: List[Fundamentals], cfg: ScoringConfig,
                    top_n: int) -> List[ScoredCandidate]:
    scored = [c for c in (score_symbol(f, cfg) for f in fundamentals) if c is not None]
    scored.sort(key=lambda c: c.combined_score, reverse=True)
    return scored[:top_n]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dividend_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/sleeve/dividend_scorer.py tests/test_dividend_scorer.py
git commit -m "feat(sleeve): deterministic yield+quality scorer for dividend candidates"
```

---

### Task 5: LLM veto layer

**Files:**
- Create: `src/agent/sleeve/llm_veto.py`
- Test: `tests/test_llm_veto.py`

**Interfaces:**
- Consumes: any object with `.enabled: bool` and `.propose_json(system_prompt: str, user_prompt: str, model_override=None) -> Optional[dict]` (matches `src.agent.llm_orchestrator.LLMOrchestrator`, already implemented — no changes needed there).
- Produces: `@dataclass VetoResult(flag: bool, reason: str, available: bool)` and `check_candidate(llm_orchestrator, symbol: str, company_name: str = '') -> VetoResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_veto.py`:

```python
from src.agent.sleeve.llm_veto import check_candidate


class FakeOrchestrator:
    def __init__(self, response, enabled=True):
        self.enabled = enabled
        self._response = response

    def propose_json(self, system_prompt, user_prompt, model_override=None):
        return self._response


def test_flag_true_passed_through():
    orch = FakeOrchestrator({'flag': True, 'reason': 'dividend cut reported'})
    result = check_candidate(orch, 'SCOM')
    assert result.flag is True
    assert result.reason == 'dividend cut reported'
    assert result.available is True


def test_flag_false_passed_through():
    orch = FakeOrchestrator({'flag': False, 'reason': ''})
    result = check_candidate(orch, 'SCOM')
    assert result.flag is False
    assert result.available is True


def test_disabled_orchestrator_marks_unavailable_never_flags():
    orch = FakeOrchestrator(None, enabled=False)
    result = check_candidate(orch, 'SCOM')
    assert result.available is False
    assert result.flag is False


def test_none_orchestrator_marks_unavailable():
    result = check_candidate(None, 'SCOM')
    assert result.available is False
    assert result.flag is False


def test_malformed_response_marks_unavailable_never_flags():
    orch = FakeOrchestrator({'unexpected': 'shape'})
    result = check_candidate(orch, 'SCOM')
    assert result.available is False
    assert result.flag is False


def test_non_dict_response_marks_unavailable():
    orch = FakeOrchestrator(None, enabled=True)
    result = check_candidate(orch, 'SCOM')
    assert result.available is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_veto.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.sleeve.llm_veto'`.

- [ ] **Step 3: Implement the veto layer**

Create `src/agent/sleeve/llm_veto.py`:

```python
"""LLM-based red-flag check for sleeve candidates. Structurally cannot pick
stocks, score them, or size positions — the deterministic scorer already
made that decision. This layer's only job is to surface a concrete, recent
reason NOT to buy, for the operator to see before approving the ticket.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a red-team reviewer for a long-term dividend investment sleeve. "
    "You do not pick stocks, score them, or size positions — another system "
    "already decided this company is a numeric candidate for accumulation. "
    "Your ONLY job is to check for a recent, concrete reason NOT to buy more: "
    "a dividend cut or suspension, a profit warning, a rights issue, delisting "
    "risk, or a governance/fraud concern. General uncertainty or lack of news "
    "is NOT a reason to flag.\n"
    "Return your response EXACTLY as a valid JSON object with this schema:\n"
    '{"flag": <true|false>, "reason": "<short explanation, empty string if not flagged>"}\n'
    "Ensure the JSON output is raw JSON without markdown codeblocks."
)


@dataclass
class VetoResult:
    flag: bool
    reason: str
    available: bool  # False when the LLM check could not be completed


def check_candidate(llm_orchestrator, symbol: str, company_name: str = '') -> VetoResult:
    if llm_orchestrator is None or not getattr(llm_orchestrator, 'enabled', False):
        return VetoResult(flag=False, reason='', available=False)

    user_prompt = f"Symbol: {symbol}\nCompany: {company_name or symbol}\n"
    result = llm_orchestrator.propose_json(_SYSTEM_PROMPT, user_prompt)
    if not isinstance(result, dict) or 'flag' not in result:
        logger.debug(f"Sleeve veto unavailable for {symbol}: no usable LLM response")
        return VetoResult(flag=False, reason='', available=False)
    return VetoResult(flag=bool(result.get('flag')),
                      reason=str(result.get('reason', '')), available=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_veto.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/sleeve/llm_veto.py tests/test_llm_veto.py
git commit -m "feat(sleeve): LLM red-flag veto layer (flag-only, never picks/sizes)"
```

---

### Task 6: Dividend ledger

**Files:**
- Create: `src/agent/sleeve/dividend_ledger.py`
- Test: `tests/test_dividend_ledger.py`

**Interfaces:**
- Produces: `DividendLedger(db_path: str = str(DATA_DIR / 'sleeve.db'))` with methods `record_dividend(symbol, amount_kes, declared_date=None, status='received', notes='') -> int`, `unswept_cash_kes() -> float`, `mark_swept() -> int`, `history(symbol=None, limit=100) -> List[Dict]`, `close()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dividend_ledger.py`:

```python
import os
import tempfile

import pytest

from src.agent.sleeve.dividend_ledger import DividendLedger


@pytest.fixture
def ledger():
    path = os.path.join(tempfile.mkdtemp(), 'sleeve_test.db')
    l = DividendLedger(path)
    yield l
    l.close()


def test_record_and_unswept_cash(ledger):
    ledger.record_dividend('SCOM', 100.0)
    ledger.record_dividend('EQTY', 50.0)
    assert ledger.unswept_cash_kes() == 150.0


def test_mark_swept_zeroes_unswept_cash(ledger):
    ledger.record_dividend('SCOM', 100.0)
    swept = ledger.mark_swept()
    assert swept == 1
    assert ledger.unswept_cash_kes() == 0.0


def test_declared_status_excluded_from_unswept_cash(ledger):
    ledger.record_dividend('SCOM', 100.0, status='declared')
    assert ledger.unswept_cash_kes() == 0.0


def test_history_filters_by_symbol(ledger):
    ledger.record_dividend('SCOM', 100.0)
    ledger.record_dividend('EQTY', 50.0)
    hist = ledger.history(symbol='SCOM')
    assert len(hist) == 1
    assert hist[0]['symbol'] == 'SCOM'


def test_history_returns_all_without_symbol_filter(ledger):
    ledger.record_dividend('SCOM', 100.0)
    ledger.record_dividend('EQTY', 50.0)
    assert len(ledger.history()) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dividend_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.sleeve.dividend_ledger'`.

- [ ] **Step 3: Implement the ledger**

Create `src/agent/sleeve/dividend_ledger.py`:

```python
"""Tracks declared/received dividends per sleeve holding and the resulting
sleeve cash available for the next accumulation cycle (manual DRIP — the
cash is just more capital for SleeveManager to allocate, no separate
reinvestment code path).
"""
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sleeve_dividends (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    declared_date TEXT,
    amount_kes    REAL NOT NULL,
    status        TEXT DEFAULT 'received',  -- 'declared' | 'received' | 'swept'
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_sleeve_dividends_symbol ON sleeve_dividends(symbol, status);
"""


class DividendLedger:
    def __init__(self, db_path: str = str(DATA_DIR / 'sleeve.db')):
        self._lock = threading.Lock()
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(f"DividendLedger initialized at {db_path}")

    def record_dividend(self, symbol: str, amount_kes: float,
                        declared_date: Optional[str] = None,
                        status: str = 'received', notes: str = '') -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sleeve_dividends (recorded_at, symbol, declared_date,"
                " amount_kes, status, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), symbol.upper(), declared_date,
                 float(amount_kes), status, notes))
            self._conn.commit()
            return cur.lastrowid

    def unswept_cash_kes(self) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(amount_kes), 0) FROM sleeve_dividends"
                " WHERE status = 'received'").fetchone()
            return float(row[0])

    def mark_swept(self) -> int:
        """Called once SleeveManager has folded unswept cash into a cycle's
        capital, so it isn't counted again next cycle."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sleeve_dividends SET status = 'swept' WHERE status = 'received'")
            self._conn.commit()
            return cur.rowcount

    def history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            if symbol:
                cur = self._conn.execute(
                    "SELECT * FROM sleeve_dividends WHERE symbol = ?"
                    " ORDER BY recorded_at DESC LIMIT ?", (symbol.upper(), limit))
            else:
                cur = self._conn.execute(
                    "SELECT * FROM sleeve_dividends ORDER BY recorded_at DESC LIMIT ?", (limit,))
            cur.row_factory = sqlite3.Row
            return [dict(r) for r in cur.fetchall()]

    def close(self):
        with self._lock:
            self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dividend_ledger.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/sleeve/dividend_ledger.py tests/test_dividend_ledger.py
git commit -m "feat(sleeve): dividend ledger tracking received cash for next cycle"
```

---

### Task 7: Sleeve manager (monthly accumulation cycle)

The only file that talks to all of Tasks 3–6 plus `NseOrderQueue`. Computes target weights from ranked candidates, sizes buy-only tickets from new capital (never touches existing holdings — no sell/rebalance in v1), runs the veto per candidate, and gates itself to once per calendar month.

**Files:**
- Create: `src/agent/sleeve/sleeve_manager.py`
- Test: `tests/test_sleeve_manager.py`
- Test: `tests/test_sleeve_integration.py`

**Interfaces:**
- Consumes: `NseOrderQueue` (`create_ticket(..., book=...)`, `positions(book=...)`) from Task 1; `FundamentalsStore.get_all` from Task 3; `ScoringConfig`/`rank_candidates` from Task 4; `check_candidate` from Task 5; `DividendLedger.unswept_cash_kes`/`mark_swept` from Task 6.
- Produces: `SleeveManager(config, nse_order_queue, fundamentals_store, dividend_ledger, llm_orchestrator=None, db_path=...)` with `run_monthly_cycle(quotes: Dict[str, float], today: Optional[date] = None) -> List[Dict[str, Any]]`, `current_holdings() -> Dict[str, Dict[str, Any]]`, `close()`. Public attributes `universe: List[str]`, `nse_capital_kes: float`, `capital_split_pct: float` (read by the API layer in Task 9).

Each result dict: `{'symbol', 'quantity', 'price', 'ticket_id', 'combined_score', 'veto_flag', 'veto_available', 'veto_reason', 'stale_data'}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sleeve_manager.py`:

```python
import os
import tempfile
from datetime import date

import pytest

from src.agent.sleeve.fundamentals_store import Fundamentals
from src.agent.sleeve.sleeve_manager import SleeveManager


class FakeQueue:
    def __init__(self):
        self.tickets = []

    def create_ticket(self, symbol, side, quantity, suggested_limit_price=None,
                      rationale='', ensemble_confidence=None, llm_reasoning='', book='trading'):
        tid = len(self.tickets) + 1
        self.tickets.append({'id': tid, 'symbol': symbol, 'side': side,
                             'quantity': quantity, 'book': book})
        return tid

    def positions(self, book=None):
        return {}


class FakeFundamentalsStore:
    def __init__(self, records):
        self._records = {r.symbol: r for r in records}

    def get_all(self, symbols):
        return [self._records[s] for s in symbols if s in self._records]


class FakeDividendLedger:
    def __init__(self, unswept=0.0):
        self._unswept = unswept
        self.swept_calls = 0

    def unswept_cash_kes(self):
        return self._unswept

    def mark_swept(self):
        self.swept_calls += 1
        return 1


class FakeOrchestrator:
    enabled = True

    def propose_json(self, system_prompt, user_prompt, model_override=None):
        return {'flag': False, 'reason': ''}


def make_config(top_n=2):
    return {
        'sleeve': {
            'enabled': True, 'nse_capital_kes': 100000, 'capital_split_pct': 0.5,
            'top_n': top_n, 'weighting_mode': 'equal', 'universe': ['SCOM', 'EQTY'],
            'scoring': {'min_avg_daily_volume': 1000},
        }
    }


def make_fundamentals():
    return [
        Fundamentals('SCOM', yield_ttm_pct=6.5, dividend_per_share_kes=2.3, eps_kes=2.4,
                    payout_ratio=0.48, years_consecutive_paid=8, eps_trend='positive',
                    avg_daily_volume=100_000, last_updated=date.today().isoformat()),
        Fundamentals('EQTY', yield_ttm_pct=5.0, dividend_per_share_kes=4.0, eps_kes=8.0,
                    payout_ratio=0.50, years_consecutive_paid=6, eps_trend='positive',
                    avg_daily_volume=80_000, last_updated=date.today().isoformat()),
    ]


@pytest.fixture
def db_path():
    return os.path.join(tempfile.mkdtemp(), 'sleeve_test.db')


def make_manager(db_path, config=None, dividend_ledger=None, orchestrator=None,
                 fundamentals=None, queue=None):
    return SleeveManager(
        config or make_config(), queue or FakeQueue(),
        FakeFundamentalsStore(fundamentals if fundamentals is not None else make_fundamentals()),
        dividend_ledger or FakeDividendLedger(), orchestrator or FakeOrchestrator(),
        db_path=db_path)


def test_generates_tickets_for_ranked_candidates(db_path):
    mgr = make_manager(db_path)
    results = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    assert {r['symbol'] for r in results} == {'SCOM', 'EQTY'}
    assert all(r['quantity'] > 0 for r in results)
    mgr.close()


def test_second_call_same_month_is_noop(db_path):
    mgr = make_manager(db_path)
    first = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 7, 1))
    second = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 7, 15))
    assert len(first) == 2
    assert second == []
    mgr.close()


def test_new_month_runs_again(db_path):
    mgr = make_manager(db_path)
    mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 7, 1))
    second = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}, today=date(2026, 8, 1))
    assert len(second) == 2
    mgr.close()


def test_missing_quote_excludes_symbol(db_path):
    mgr = make_manager(db_path)
    results = mgr.run_monthly_cycle({'SCOM': 20.0})  # no EQTY quote
    assert {r['symbol'] for r in results} == {'SCOM'}
    mgr.close()


def test_dividend_cash_increases_deployed_quantity(db_path):
    ledger = FakeDividendLedger(unswept=50000.0)
    with_div = make_manager(db_path, dividend_ledger=ledger)
    without_div = make_manager(os.path.join(tempfile.mkdtemp(), 'baseline.db'))

    with_results = with_div.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    without_results = without_div.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})

    with_scom = next(r for r in with_results if r['symbol'] == 'SCOM')['quantity']
    without_scom = next(r for r in without_results if r['symbol'] == 'SCOM')['quantity']
    assert with_scom > without_scom
    assert ledger.swept_calls == 1
    with_div.close()
    without_div.close()


def test_no_sweep_when_zero_tickets_generated(db_path):
    ledger = FakeDividendLedger(unswept=50000.0)
    mgr = make_manager(db_path, dividend_ledger=ledger, fundamentals=[])
    results = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    assert results == []
    assert ledger.swept_calls == 0
    mgr.close()


def test_veto_flag_surfaced_on_ticket_result(db_path):
    class FlaggingOrchestrator:
        enabled = True

        def propose_json(self, system_prompt, user_prompt, model_override=None):
            return {'flag': True, 'reason': 'profit warning issued'}

    mgr = make_manager(db_path, orchestrator=FlaggingOrchestrator())
    results = mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0})
    assert all(r['veto_flag'] is True for r in results)
    assert all(r['veto_reason'] == 'profit warning issued' for r in results)
    mgr.close()


def test_disabled_sleeve_returns_no_tickets(db_path):
    cfg = make_config()
    cfg['sleeve']['enabled'] = False
    mgr = make_manager(db_path, config=cfg)
    assert mgr.run_monthly_cycle({'SCOM': 20.0, 'EQTY': 40.0}) == []
    mgr.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sleeve_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.sleeve.sleeve_manager'`.

- [ ] **Step 3: Implement the manager**

Create `src/agent/sleeve/sleeve_manager.py`:

```python
"""Monthly accumulation cycle for the NSE dividend-led long-term sleeve.

Ties fundamentals, deterministic scoring, the LLM veto, and the dividend
ledger together to turn scored candidates into buy-only order tickets in
the existing NSE order-ticket queue. Never sells or trims — this sleeve
only ever deploys new capital (fixed % of NSE capital + swept dividends)
across the current top-ranked candidates.
"""
import logging
import math
import sqlite3
import threading
from datetime import date
from typing import Any, Dict, List, Optional

from src.agent.sleeve.dividend_scorer import ScoringConfig, rank_candidates
from src.agent.sleeve.llm_veto import check_candidate
from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sleeve_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _scoring_config_from_dict(d: Dict[str, Any]) -> ScoringConfig:
    return ScoringConfig(
        yield_cap_pct=d.get('yield_cap_pct', 12.0),
        yield_weight=d.get('yield_weight', 0.5),
        quality_weight=d.get('quality_weight', 0.5),
        payout_ratio_min=d.get('payout_ratio_min', 0.30),
        payout_ratio_max=d.get('payout_ratio_max', 0.70),
        years_paid_target=d.get('years_paid_target', 5),
        min_avg_daily_volume=d.get('min_avg_daily_volume', 50_000),
        stale_days=d.get('stale_days', 400),
    )


class SleeveManager:
    def __init__(self, config: Dict[str, Any], nse_order_queue, fundamentals_store,
                dividend_ledger, llm_orchestrator=None,
                db_path: str = str(DATA_DIR / 'sleeve.db')):
        sc = config.get('sleeve', {})
        self.enabled = sc.get('enabled', False)
        self.nse_capital_kes = sc.get('nse_capital_kes', 0.0)
        self.capital_split_pct = sc.get('capital_split_pct', 0.0)
        self.top_n = sc.get('top_n', 5)
        self.weighting_mode = sc.get('weighting_mode', 'equal')
        self.universe = sc.get('universe', [])
        self.scoring_cfg = _scoring_config_from_dict(sc.get('scoring', {}))

        self.nse_order_queue = nse_order_queue
        self.fundamentals_store = fundamentals_store
        self.dividend_ledger = dividend_ledger
        self.llm_orchestrator = llm_orchestrator

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _get_state(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM sleeve_state WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def _set_state(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sleeve_state (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
            self._conn.commit()

    def current_holdings(self) -> Dict[str, Dict[str, Any]]:
        """Sleeve's own holdings, isolated from the trading book by the
        `book='long_term'` tag every ticket this manager creates carries."""
        return self.nse_order_queue.positions(book='long_term')

    def _target_weights(self, ranked) -> Dict[str, float]:
        if not ranked:
            return {}
        if self.weighting_mode == 'score':
            total = sum(c.combined_score for c in ranked)
            if total <= 0:
                return {}
            return {c.symbol: c.combined_score / total for c in ranked}
        weight = 1.0 / len(ranked)
        return {c.symbol: weight for c in ranked}

    def run_monthly_cycle(self, quotes: Dict[str, float],
                          today: Optional[date] = None) -> List[Dict[str, Any]]:
        """Runs once per calendar month. `quotes` maps symbol -> current KES
        price (the caller supplies these from the NSE connector so this
        module has no direct network dependency)."""
        if not self.enabled:
            return []
        today = today or date.today()
        month_key = today.strftime('%Y-%m')
        if self._get_state('last_cycle_month') == month_key:
            return []

        candidates = self.fundamentals_store.get_all(self.universe)
        ranked = rank_candidates(candidates, self.scoring_cfg, self.top_n)
        ranked = [c for c in ranked if quotes.get(c.symbol, 0) > 0]
        weights = self._target_weights(ranked)

        new_capital_kes = (self.nse_capital_kes * self.capital_split_pct +
                           self.dividend_ledger.unswept_cash_kes())

        results: List[Dict[str, Any]] = []
        for candidate in ranked:
            symbol = candidate.symbol
            price = quotes[symbol]
            allocation_kes = new_capital_kes * weights.get(symbol, 0)
            quantity = int(math.floor(allocation_kes / price))
            if quantity <= 0:
                continue

            veto = check_candidate(self.llm_orchestrator, symbol)
            tags = []
            if candidate.stale:
                tags.append('STALE_DATA')
            if veto.flag:
                tags.append('VETO_FLAG')
            if not veto.available:
                tags.append('VETO_UNAVAILABLE')
            rationale = (f"Sleeve accumulation: yield_score={candidate.yield_score}, "
                        f"quality_score={candidate.quality_score}, "
                        f"combined={candidate.combined_score}")
            if tags:
                rationale += " | " + ", ".join(tags)

            ticket_id = self.nse_order_queue.create_ticket(
                symbol, 'buy', quantity, suggested_limit_price=price,
                rationale=rationale, ensemble_confidence=candidate.combined_score,
                llm_reasoning=veto.reason, book='long_term')

            results.append({
                'symbol': symbol, 'quantity': quantity, 'price': price,
                'ticket_id': ticket_id, 'combined_score': candidate.combined_score,
                'veto_flag': veto.flag, 'veto_available': veto.available,
                'veto_reason': veto.reason, 'stale_data': candidate.stale,
            })

        # Only mark dividend cash "swept" if it was actually deployed this
        # cycle — otherwise a zero-ticket month would silently lose that
        # cash instead of rolling it into next month's capital.
        if results:
            self.dividend_ledger.mark_swept()
        self._set_state('last_cycle_month', month_key)
        return results

    def close(self):
        with self._lock:
            self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sleeve_manager.py -v`
Expected: PASS

- [ ] **Step 5: Write the end-to-end integration test against the real `NseOrderQueue`**

Create `tests/test_sleeve_integration.py`:

```python
"""End-to-end: a full monthly cycle against a fixture universe, asserting
the right tickets land in the REAL NseOrderQueue with correct tags, sizes,
and isolation from a trading-book ticket for the same symbol."""
import os
import tempfile
from datetime import date

import pytest

from src.agent.nse_order_queue import NseOrderQueue
from src.agent.sleeve.fundamentals_store import Fundamentals
from src.agent.sleeve.sleeve_manager import SleeveManager


class FakeFundamentalsStore:
    def __init__(self, records):
        self._records = {r.symbol: r for r in records}

    def get_all(self, symbols):
        return [self._records[s] for s in symbols if s in self._records]


class FakeDividendLedger:
    def unswept_cash_kes(self):
        return 0.0

    def mark_swept(self):
        return 0


class FakeOrchestrator:
    enabled = True

    def propose_json(self, system_prompt, user_prompt, model_override=None):
        return {'flag': False, 'reason': ''}


@pytest.fixture
def queue():
    path = os.path.join(tempfile.mkdtemp(), 'nse_test.db')
    q = NseOrderQueue(path)
    yield q
    q.close()


def test_monthly_cycle_lands_tagged_tickets_in_real_queue(queue):
    records = [
        Fundamentals('SCOM', yield_ttm_pct=6.5, dividend_per_share_kes=2.3, eps_kes=2.4,
                    payout_ratio=0.48, years_consecutive_paid=8, eps_trend='positive',
                    avg_daily_volume=100_000, last_updated=date.today().isoformat()),
    ]
    config = {
        'sleeve': {
            'enabled': True, 'nse_capital_kes': 100000, 'capital_split_pct': 0.5,
            'top_n': 5, 'weighting_mode': 'equal', 'universe': ['SCOM'],
            'scoring': {'min_avg_daily_volume': 1000},
        }
    }
    mgr = SleeveManager(config, queue, FakeFundamentalsStore(records),
                        FakeDividendLedger(), FakeOrchestrator(),
                        db_path=os.path.join(tempfile.mkdtemp(), 'sleeve.db'))

    results = mgr.run_monthly_cycle({'SCOM': 20.0})
    assert len(results) == 1

    pending = queue.get_pending()
    sleeve_tickets = [t for t in pending if t['book'] == 'long_term']
    assert len(sleeve_tickets) == 1
    assert sleeve_tickets[0]['symbol'] == 'SCOM'
    assert sleeve_tickets[0]['quantity'] > 0

    # A trading-book ticket for the same symbol/side is a separate ticket,
    # not deduped against the sleeve one (book-scoped duplicate check).
    trading_id = queue.create_ticket('SCOM', 'buy', 50, book='trading')
    assert trading_id is not None
    mgr.close()
```

- [ ] **Step 6: Run the integration test to verify it passes**

Run: `pytest tests/test_sleeve_integration.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent/sleeve/sleeve_manager.py tests/test_sleeve_manager.py \
        tests/test_sleeve_integration.py
git commit -m "feat(sleeve): monthly accumulation cycle orchestrator"
```

---

### Task 8: Wire the sleeve into the main trading loop + config

Follows the exact pattern `_evaluate_nse_symbols()` already uses for the NSE trading path: a rate-limited (here, month-gated inside `SleeveManager` itself) method called from the main loop, guarded by `try/except` so a sleeve error never takes down the trading loop.

**Files:**
- Modify: `config/config.json`
- Modify: `src/agent/main.py`

**Interfaces:**
- Produces: `self.components['fundamentals_store']`, `self.components['dividend_ledger']`, `self.components['sleeve_manager']` on the `TradingAgent` instance.
- Produces: `TradingAgent._run_sleeve_cycle(self) -> None`.

- [ ] **Step 1: Add the `sleeve` config block**

In `config/config.json`, insert a new top-level key right after the existing `"cash_policy"` block (currently lines 79–86):

```json
  "cash_policy": {
    "enabled": true,
    "target_deployment": 0.80,
    "min_cash_buffer": 0.10,
    "base_risk_per_trade": 0.02,
    "max_risk_per_trade": 0.05,
    "min_confidence_to_boost": 0.55
  },
  "sleeve": {
    "enabled": false,
    "nse_capital_kes": 0,
    "capital_split_pct": 0.5,
    "top_n": 5,
    "weighting_mode": "equal",
    "universe": ["SCOM", "EQTY", "KCB", "COOP", "SCBK", "ABSA", "BAT", "EABL", "NCBA"],
    "scoring": {
      "yield_cap_pct": 12.0,
      "yield_weight": 0.5,
      "quality_weight": 0.5,
      "payout_ratio_min": 0.30,
      "payout_ratio_max": 0.70,
      "years_paid_target": 5,
      "min_avg_daily_volume": 50000,
      "stale_days": 400
    }
  },
```

`enabled: false` and `nse_capital_kes: 0` by default — the operator must explicitly opt in and set real capital before the sleeve creates any ticket (matches `SleeveManager.run_monthly_cycle`'s `if not self.enabled: return []` guard).

- [ ] **Step 2: Wire the components in `main.py`**

In `src/agent/main.py`, right after the existing NSE order-ticket queue setup (currently lines 226–231):

```python
            # NSE order-ticket queue: NSE has no broker API, so NSE trade
            # decisions become tickets a human keys into the AIB-AXYS portal.
            from src.agent.nse_order_queue import NseOrderQueue
            self.components['nse_order_queue'] = NseOrderQueue()
            self._last_nse_eval = 0.0
            self._nse_last_price = {}  # symbol -> last-evaluated price (freshness guard)
```

add:

```python
            # Long-term dividend sleeve: separate NSE accumulation book,
            # isolated from trading positions via the `book` tag on tickets.
            from src.agent.sleeve.fundamentals_store import FundamentalsStore
            from src.agent.sleeve.dividend_ledger import DividendLedger
            from src.agent.sleeve.sleeve_manager import SleeveManager
            self.components['fundamentals_store'] = FundamentalsStore()
            self.components['dividend_ledger'] = DividendLedger()
            self.components['sleeve_manager'] = SleeveManager(
                self.config,
                self.components['nse_order_queue'],
                self.components['fundamentals_store'],
                self.components['dividend_ledger'],
                self.components['llm_orchestrator'],
            )
```

- [ ] **Step 3: Add cleanup on shutdown**

Find the existing shutdown block that closes `nse_order_queue` (around line 327):

```python
            if 'nse_order_queue' in self.components:
                self.components['nse_order_queue'].close()
```

add immediately after:

```python
            if 'dividend_ledger' in self.components:
                self.components['dividend_ledger'].close()
            if 'sleeve_manager' in self.components:
                self.components['sleeve_manager'].close()
```

- [ ] **Step 4: Add the loop hook**

Find the existing NSE evaluation call site (currently lines 969–975):

```python
                # NSE Kenya decision pass — separate cadence from the US loop
                # (NSE prices only refresh on the ~30-min scrape). Internally
                # rate-limited and guarded by market hours + price freshness.
                try:
                    self._evaluate_nse_symbols()
                except Exception as e:
                    logger.error(f"NSE evaluation error: {e}")
```

add immediately after:

```python
                # Long-term dividend sleeve — separate cadence again (at most
                # once per calendar month; SleeveManager gates itself).
                try:
                    self._run_sleeve_cycle()
                except Exception as e:
                    logger.error(f"Sleeve cycle error: {e}")
```

- [ ] **Step 5: Add the `_run_sleeve_cycle` method**

Immediately after the existing `_evaluate_nse_symbols` method definition (starts at line 987 today — add this new method right after it ends), add:

```python
    def _run_sleeve_cycle(self):
        """Monthly dividend-sleeve accumulation pass. Pulls current NSE
        quotes for the sleeve's configured universe and hands them to
        SleeveManager, which gates itself to once per calendar month."""
        sleeve = self.components.get('sleeve_manager')
        dm = self.components.get('data_manager')
        nse = dm.connectors.get('nse') if dm and hasattr(dm, 'connectors') else None
        if not sleeve or not nse:
            return

        quotes = {}
        for symbol in sleeve.universe:
            quote = nse.get_quote(symbol)
            if quote and not quote.get('_stale') and quote.get('price_kes'):
                quotes[symbol] = float(quote['price_kes'])

        results = sleeve.run_monthly_cycle(quotes)
        if results:
            logger.info(f"Sleeve cycle generated {len(results)} accumulation "
                       f"ticket(s): {[r['symbol'] for r in results]}")
```

- [ ] **Step 6: Verify the module imports cleanly**

Run: `python -c "import ast; ast.parse(open('src/agent/main.py').read())"`
Expected: no output (parses without a `SyntaxError`).

Run: `pytest tests/ -v -k "sleeve or nse_order_queue or nse_company_scraper"`
Expected: PASS (re-confirms every sleeve-related test still passes after touching `main.py` and `config.json`; `main.py` itself has no direct unit test — it's exercised indirectly via the component tests plus manual verification in Task 10).

- [ ] **Step 7: Commit**

```bash
git add config/config.json src/agent/main.py
git commit -m "feat(sleeve): wire dividend sleeve into main trading loop"
```

---

### Task 9: Dashboard API endpoint

**Files:**
- Create: `src/api/sleeve_view.py`
- Modify: `src/api/api_server.py`
- Test: `tests/test_sleeve_view.py`

**Interfaces:**
- Produces: `build_sleeve_view(holdings: Dict[str, Dict], pending_tickets: List[Dict], dividend_history: List[Dict], nse_capital_kes: float, capital_split_pct: float) -> Dict[str, Any]`.
- Produces: `GET /api/sleeve` route returning that payload as JSON.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sleeve_view.py`:

```python
from src.api.sleeve_view import build_sleeve_view


def test_filters_to_long_term_book():
    tickets = [
        {'symbol': 'SCOM', 'quantity': 100, 'book': 'trading', 'rationale': '', 'llm_reasoning': ''},
        {'symbol': 'EQTY', 'quantity': 50, 'book': 'long_term',
         'rationale': 'Sleeve accumulation: combined=0.7', 'llm_reasoning': ''},
    ]
    view = build_sleeve_view({}, tickets, [], nse_capital_kes=100000, capital_split_pct=0.5)
    assert len(view['pending_tickets']) == 1
    assert view['pending_tickets'][0]['symbol'] == 'EQTY'


def test_veto_flag_detected_from_rationale():
    tickets = [{'symbol': 'SCOM', 'quantity': 10, 'book': 'long_term',
               'rationale': 'Sleeve accumulation: combined=0.7 | VETO_FLAG',
               'llm_reasoning': 'profit warning'}]
    view = build_sleeve_view({}, tickets, [], nse_capital_kes=100000, capital_split_pct=0.5)
    assert view['pending_tickets'][0]['veto_flagged'] is True
    assert view['pending_tickets'][0]['llm_reasoning'] == 'profit warning'


def test_veto_unavailable_detected_from_rationale():
    tickets = [{'symbol': 'SCOM', 'quantity': 10, 'book': 'long_term',
               'rationale': 'Sleeve accumulation: combined=0.7 | VETO_UNAVAILABLE',
               'llm_reasoning': ''}]
    view = build_sleeve_view({}, tickets, [], nse_capital_kes=100000, capital_split_pct=0.5)
    assert view['pending_tickets'][0]['veto_unavailable'] is True
    assert view['pending_tickets'][0]['veto_flagged'] is False


def test_capital_target_computed():
    view = build_sleeve_view({}, [], [], nse_capital_kes=200000, capital_split_pct=0.4)
    assert view['capital_target_kes'] == 80000.0


def test_cumulative_dividends_excludes_declared_only():
    history = [
        {'amount_kes': 100.0, 'status': 'received'},
        {'amount_kes': 50.0, 'status': 'swept'},
        {'amount_kes': 30.0, 'status': 'declared'},
    ]
    view = build_sleeve_view({}, [], history, nse_capital_kes=0, capital_split_pct=0)
    assert view['cumulative_dividends_kes'] == 150.0


def test_holdings_passed_through():
    holdings = {'SCOM': {'quantity': 200, 'avg_entry_price_kes': 15.0}}
    view = build_sleeve_view(holdings, [], [], nse_capital_kes=0, capital_split_pct=0)
    assert view['holdings'] == [{'symbol': 'SCOM', 'quantity': 200, 'avg_entry_price_kes': 15.0}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sleeve_view.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.sleeve_view'`.

- [ ] **Step 3: Implement the view builder**

Create `src/api/sleeve_view.py`:

```python
"""Long-Term Sleeve dashboard view: holdings, pending accumulation tickets
(with veto flags surfaced), and cumulative dividend income."""
from typing import Any, Dict, List


def build_sleeve_view(holdings: Dict[str, Dict[str, Any]], pending_tickets: List[Dict[str, Any]],
                      dividend_history: List[Dict[str, Any]], nse_capital_kes: float,
                      capital_split_pct: float) -> Dict[str, Any]:
    sleeve_tickets = [t for t in pending_tickets if t.get('book') == 'long_term']
    total_received = sum(d['amount_kes'] for d in dividend_history
                         if d.get('status') in ('received', 'swept'))
    return {
        'capital_target_kes': round(nse_capital_kes * capital_split_pct, 2),
        'holdings': [{'symbol': symbol, **data} for symbol, data in holdings.items()],
        'pending_tickets': [
            {
                'symbol': t['symbol'], 'quantity': t['quantity'],
                'suggested_limit_price': t.get('suggested_limit_price'),
                'rationale': t.get('rationale'),
                'veto_flagged': 'VETO_FLAG' in (t.get('rationale') or ''),
                'veto_unavailable': 'VETO_UNAVAILABLE' in (t.get('rationale') or ''),
                'llm_reasoning': t.get('llm_reasoning'),
            }
            for t in sleeve_tickets
        ],
        'cumulative_dividends_kes': round(total_received, 2),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sleeve_view.py -v`
Expected: PASS

- [ ] **Step 5: Add the Flask route**

In `src/api/api_server.py`, immediately after the existing `/api/agent-focus` route block (ends at line 692 with `'cash': {'cash': 0, 'equity': 0, 'deployed_pct': 0.0}})`), add:

```python
        @self.app.route('/api/sleeve', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_sleeve_view():
            """Long-term dividend sleeve: holdings, pending accumulation
            tickets (with veto flags), and cumulative dividend income."""
            def produce():
                from src.api.sleeve_view import build_sleeve_view
                queue = self.trading_agent.components.get('nse_order_queue')
                ledger = self.trading_agent.components.get('dividend_ledger')
                sleeve = self.trading_agent.components.get('sleeve_manager')
                if not queue or not sleeve:
                    return {'holdings': [], 'pending_tickets': [],
                           'cumulative_dividends_kes': 0, 'capital_target_kes': 0}
                holdings = queue.positions(book='long_term')
                pending = queue.get_pending()
                history = ledger.history() if ledger else []
                return build_sleeve_view(holdings, pending, history,
                                         sleeve.nse_capital_kes, sleeve.capital_split_pct)
            try:
                return jsonify(self._cached('sleeve_view', 30, produce))
            except Exception as e:
                logger.error(f"Error building sleeve view: {e}")
                return jsonify({'holdings': [], 'pending_tickets': [],
                               'cumulative_dividends_kes': 0, 'capital_target_kes': 0})
```

- [ ] **Step 6: Verify the module imports cleanly**

Run: `python -c "import ast; ast.parse(open('src/api/api_server.py').read())"`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add src/api/sleeve_view.py src/api/api_server.py tests/test_sleeve_view.py
git commit -m "feat(sleeve): dashboard API endpoint for the long-term sleeve"
```

---

### Task 10: Frontend panel

Mirrors the existing `AgentFocus` component/mount pattern exactly (same polling approach, same `theme`/`getApiBase` helpers).

**Files:**
- Create: `frontend/components/SleeveDashboard.js`
- Modify: `frontend/components/AdvancedDashboard.js`

**Interfaces:**
- Consumes: `GET /api/sleeve` (Task 9's payload shape).
- Produces: `SleeveDashboard` default-exported React component, no props required.

- [ ] **Step 1: Create the component**

Create `frontend/components/SleeveDashboard.js`:

```jsx
import { useState, useEffect } from 'react';
import { theme } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const SleeveDashboard = () => {
  const [view, setView] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/sleeve`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setView(json);
      } catch (e) {}
    };
    load();
    const id = setInterval(load, 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!view) return null;
  const row = { display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '13px' };
  const empty = (msg) => <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>{msg}</div>;

  return (
    <div style={{ ...theme.glass, padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
        <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 800, letterSpacing: '1px', color: '#fff' }}>LONG-TERM SLEEVE — NSE DIVIDENDS</h3>
        <span style={{ fontSize: '12px', color: theme.colors.textMuted }}>
          Target: <span style={{ color: theme.colors.primary, fontWeight: 800 }}>KES {(view.capital_target_kes ?? 0).toLocaleString()}</span>
          {' · '}Dividends received: <span style={{ color: theme.colors.primary, fontWeight: 800 }}>KES {(view.cumulative_dividends_kes ?? 0).toLocaleString()}</span>
        </span>
      </div>
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '200px' }}>
          <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '1px', color: theme.colors.primary, marginBottom: '10px' }}>
            HOLDINGS ({view.holdings.length})
          </div>
          {view.holdings.length === 0 && empty('No sleeve holdings yet')}
          {view.holdings.map(h => (
            <div key={h.symbol} style={row}>
              <span style={{ fontWeight: 700 }}>{h.symbol}</span>
              <span>{h.quantity} @ avg {h.avg_entry_price_kes}</span>
            </div>
          ))}
        </div>
        <div style={{ flex: 1, minWidth: '200px' }}>
          <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '1px', color: theme.colors.warning, marginBottom: '10px' }}>
            PENDING ACCUMULATION ({view.pending_tickets.length})
          </div>
          {view.pending_tickets.length === 0 && empty('Nothing pending this cycle')}
          {view.pending_tickets.map(t => (
            <div key={t.symbol} style={row}>
              <span style={{ fontWeight: 700 }}>
                {t.symbol}
                {t.veto_flagged && (
                  <span style={{ color: theme.colors.danger, marginLeft: '6px', fontSize: '11px' }} title={t.llm_reasoning}>
                    ⚑ FLAGGED
                  </span>
                )}
              </span>
              <span style={{ fontSize: '11px' }}>{t.quantity} @ {t.suggested_limit_price}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SleeveDashboard;
```

- [ ] **Step 2: Mount it in the dashboard**

In `frontend/components/AdvancedDashboard.js`, the import block (lines 12–14 today):

```javascript
import AgentActivity from './AgentActivity';
import AgentFocus from './AgentFocus';
import MarketClock from './MarketClock';
```

becomes:

```javascript
import AgentActivity from './AgentActivity';
import AgentFocus from './AgentFocus';
import SleeveDashboard from './SleeveDashboard';
import MarketClock from './MarketClock';
```

And the mount point (line 646 today):

```javascript
          <AgentFocus onDrill={setDrilldownSymbol} />
          <AgentActivity activities={data.agentActivity} />
```

becomes:

```javascript
          <AgentFocus onDrill={setDrilldownSymbol} />
          <SleeveDashboard />
          <AgentActivity activities={data.agentActivity} />
```

- [ ] **Step 3: Verify the frontend builds**

Run: `cd frontend && npm run build`
Expected: build succeeds with no new errors (pre-existing warnings unrelated to this change are fine).

- [ ] **Step 4: Manual smoke check**

Run: `cd frontend && npm run dev`, open the dashboard in a browser, confirm the "LONG-TERM SLEEVE — NSE DIVIDENDS" panel renders (showing "No sleeve holdings yet" / "Nothing pending this cycle" against a backend with `sleeve.enabled: false` or no data yet is the expected empty state — it should not error or blank the page).

- [ ] **Step 5: Commit**

```bash
git add frontend/components/SleeveDashboard.js frontend/components/AdvancedDashboard.js
git commit -m "feat(sleeve): dashboard panel for the long-term sleeve"
```

---

## Post-Implementation Notes

- The sleeve ships **disabled** (`config/config.json`'s `sleeve.enabled: false`, `nse_capital_kes: 0`) and with an **empty** `config/nse_dividends.json` — no ticket can be generated until the operator explicitly configures capital, flips `enabled: true`, and populates at least one symbol's dividend history. This is intentional: a real-money accumulation sleeve should never activate itself.
- Before first real use, the operator should: (1) populate `config/nse_dividends.json` for the symbols in `sleeve.universe` (see `config/nse_dividends.README.md`), (2) set `nse_capital_kes` to the actual KES capital allocated to NSE activity, (3) set `capital_split_pct`, (4) flip `enabled: true`.
- `dividend_ledger.record_dividend(...)` has no API/UI entry point in this plan — dividends are recorded by calling it directly (e.g. from a script or REPL) until/unless a dedicated operator UI is requested. This matches the design's non-goal of automated dividend-payment detection.
