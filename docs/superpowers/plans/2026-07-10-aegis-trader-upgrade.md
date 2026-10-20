# AEGIS Trader Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five QA-confirmed bugs, correct NSE Kenya session hours, make every symbol click through to the drill-down, surface what the agent is holding/reviewing/trading in real time, deploy idle cash under a portfolio-level policy, and make the agent progressively smarter via outcome feedback — benchmarked against TradingView's UX patterns.

**Architecture:** Backend is Flask (waitress) in `src/api/api_server.py` fronting a `TradingAgent` (`src/agent/main.py`) that runs a 60s decision loop over US (Alpaca paper) + NSE Kenya symbols, journaling every decision to SQLite. Frontend is a Next.js single-page dashboard (`frontend/components/AdvancedDashboard.js`) with inline styles and recharts. All new logic goes in pure, unit-testable modules; API routes stay thin; frontend changes are verified via the dev preview (no JS test harness exists).

**Tech Stack:** Python 3 + Flask + sqlite3 + pytest; React/Next.js + recharts; Gemini free tier (existing `_call_gemini` path) for LLM validation.

## Global Constraints

- PAPER TRADING ONLY — never touch the paper-only guard, HALT logic, or auth (`src/api/auth.py`).
- Every new API route gets `@require_rate_limit` and `@token_required` decorators, in that order, matching existing routes.
- New backend logic lives in pure functions/classes with pytest coverage; API handlers only wire them up.
- Frontend: match existing inline-style idiom (no CSS files, no new UI libraries). Colors come from `theme.colors` in `DashboardStyles.js`.
- Initial capital baseline stays $100,000 (`initial_cash: 100000` in main.py).
- `max_position_size` risk limit (0.05) is a hard cap — no task may exceed it.
- Run tests with: `python -m pytest tests/<file> -v` from repo root (Windows).
- Commit after every task with the message given in the task.

---

## Phase 0 — QA bug fixes (each independently shippable)

### Task 1: Agent Activity schema fix

The Dashboard's Agent Activity panel is blank because `/api/agent-activity` returns `{timestamp, component, message}` while `AgentActivity.js` renders `{id, time, type, symbol, quantity, price, reason}`. Fix by emitting a superset record from a pure formatter, and make the component tolerate missing fields.

**Files:**
- Create: `src/api/activity_format.py`
- Create: `tests/test_activity_format.py`
- Modify: `src/api/api_server.py:636-652` (the `produce()` body of `get_agent_activity`)
- Modify: `frontend/components/AgentActivity.js:32-48`

**Interfaces:**
- Produces: `format_agent_activity(decisions: list[dict]) -> list[dict]` — each output dict has keys `id, timestamp, time, component, message, type, symbol, quantity, price, reason`. Task 10 reuses the same decision-journal records.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_activity_format.py
from src.api.activity_format import format_agent_activity


def test_executed_decision_maps_to_full_activity_record():
    decisions = [{
        'ts': '2026-07-10T09:31:05', 'symbol': 'SCOM', 'action': 'buy',
        'price': 34.2, 'quantity': 100, 'executed': True, 'skip_reason': None,
    }]
    out = format_agent_activity(decisions)
    assert len(out) == 1
    rec = out[0]
    assert rec['id'] == '2026-07-10T09:31:05-SCOM'
    assert rec['time'] == '09:31:05'
    assert rec['type'] == 'BUY'
    assert rec['symbol'] == 'SCOM'
    assert rec['price'] == 34.2
    assert rec['quantity'] == 100
    assert rec['reason'] == 'Executed'
    assert rec['message'] == 'BUY executed at 34.2'
    # legacy keys still present for the System-tab Agent Log
    assert rec['timestamp'] == '2026-07-10T09:31:05'
    assert rec['component'] == 'SCOM'


def test_blocked_decision_uses_skip_reason():
    decisions = [{'ts': '2026-07-10T10:00:00', 'symbol': 'EQTY', 'action': 'buy',
                  'price': None, 'executed': False, 'skip_reason': 'risk_limit'}]
    rec = format_agent_activity(decisions)[0]
    assert rec['type'] == 'BUY'
    assert rec['reason'] == 'Blocked: risk_limit'
    assert rec['message'] == 'BUY blocked: risk_limit'


def test_missing_fields_do_not_crash():
    rec = format_agent_activity([{}])[0]
    assert rec['type'] == 'HOLD'
    assert rec['time'] == ''
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_activity_format.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.activity_format'`

- [ ] **Step 3: Write the implementation**

```python
# src/api/activity_format.py
"""Shapes decision-journal records for the dashboard activity feeds.

One record serves two consumers: the Dashboard's AgentActivity panel
(type/symbol/price/reason) and the System tab's Agent Log
(timestamp/component/message).
"""
from typing import Any, Dict, List


def format_agent_activity(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in decisions:
        ts = d.get('ts') or ''
        action = (d.get('action') or 'hold').upper()
        if d.get('executed'):
            message = f"{action} executed at {d.get('price')}"
            reason = 'Executed'
        elif d.get('skip_reason'):
            message = f"{action} blocked: {d.get('skip_reason')}"
            reason = f"Blocked: {d.get('skip_reason')}"
        else:
            message = action
            reason = 'Evaluated'
        out.append({
            'id': f"{ts}-{d.get('symbol')}",
            'timestamp': ts,
            'time': ts[11:19] if len(ts) >= 19 else '',
            'component': d.get('symbol'),
            'message': message,
            'type': action,
            'symbol': d.get('symbol'),
            'quantity': d.get('quantity'),
            'price': d.get('price'),
            'reason': reason,
        })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_activity_format.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Wire the API route to the formatter**

In `src/api/api_server.py`, replace the body of `produce()` inside `get_agent_activity` (currently lines 636–652, the `for d in decisions:` loop and `activity` list) with:

```python
            def produce():
                from src.api.activity_format import format_agent_activity
                dj = getattr(self.trading_agent, 'decision_journal', None)
                decisions = dj.recent(None, limit=50) if dj else []
                return format_agent_activity(decisions)
```

- [ ] **Step 6: Make AgentActivity.js tolerant of partial data**

In `frontend/components/AgentActivity.js`, replace the `activities.map(...)` block (lines 32–48) with:

```jsx
        {activities.length === 0 && (
          <div style={{ color: currentTheme.text, opacity: 0.5, fontSize: '13px', padding: '12px 0' }}>
            No agent decisions yet this session.
          </div>
        )}
        {activities.map((activity, i) => (
          <div key={activity.id || i} style={{
            display: 'flex',
            alignItems: 'center',
            padding: '12px 0',
            borderBottom: `1px solid ${currentTheme.border}`
          }}>
            <div style={{ width: '100px', color: currentTheme.text, fontSize: '14px' }}>{activity.time || ''}</div>
            <div style={{ width: '80px' }}>
              <span style={getTypeStyle(activity.type)}>{activity.type || '—'}</span>
            </div>
            <div style={{ flex: 1, color: currentTheme.text, fontSize: '14px', fontWeight: '500' }}>
              {activity.symbol ? `${activity.symbol}${activity.price ? ` @ ${activity.price}` : ''}` : ''}
            </div>
            <div style={{ flex: 2, color: currentTheme.text, fontSize: '14px' }}>{activity.reason || activity.message || ''}</div>
          </div>
        ))}
```

- [ ] **Step 7: Verify in the browser**

Start the dev preview, open the Dashboard tab, confirm the Agent Activity panel shows rows with time / BUY–SELL–HOLD chip / symbol / reason and the React key warning is gone from the console.

- [ ] **Step 8: Commit**

```bash
git add src/api/activity_format.py tests/test_activity_format.py src/api/api_server.py frontend/components/AgentActivity.js
git commit -m "fix(dashboard): agent activity feed renders decision records (schema mismatch)"
```

---

### Task 2: Fetch pipeline correctness — parallel loads, loading state, no error-object clobbering

`fetchData` fetches 12 endpoints **sequentially** and then trusts every response: a failed fetch pushes `{error}` which is truthy and overwrites good cached state, and `setIsConnected(true)` runs even when everything failed. This produces both the "$0 / OFFLINE" first-load flash and the Dashboard-vs-System-tab contradiction.

**Files:**
- Modify: `frontend/components/AdvancedDashboard.js:412-453` (`fetchData`), initial state block around line 345, and the overview header around line 484.

**Interfaces:**
- Produces: `data.loading` boolean consumed by the overview render; `isConnected` now means "the /api/status fetch succeeded on the last sweep".

- [ ] **Step 1: Add a loading flag to initial state**

In the `useState` initial object (line ~345), add `loading: true` as the first key:

```js
  const [data, setData] = useState({
    loading: true,
    status: { components: {} },
    ...
```

- [ ] **Step 2: Rewrite fetchData with Promise.all and response guards**

Replace the whole `fetchData` function (lines 412–453) with:

```js
  const fetchData = async () => {
    const endpoints = [
      'status', 'performance', 'positions', 'alerts', 'news-feed',
      'risk-metrics', 'model-performance', 'strategy-performance',
      'market-heatmap', 'system-health', 'agent-activity', 'portfolio-allocation'
    ];

    const token = localStorage.getItem('trading_token');
    if (!token) { onLogout(); return; }

    let unauthorized = false;
    const newResponses = await Promise.all(endpoints.map(async (endpoint) => {
      try {
        const res = await fetch(`${getApiBase()}/api/${endpoint}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.status === 401) { unauthorized = true; return null; }
        if (!res.ok) return null;
        return await res.json();
      } catch (err) {
        return null;
      }
    }));
    if (unauthorized) { onLogout(); return; }

    // A response only replaces cached state when it parsed and isn't an error envelope.
    const ok = (r) => r != null && !(typeof r === 'object' && !Array.isArray(r) && r.error);

    setData(prev => ({
      loading: false,
      status: ok(newResponses[0]) ? newResponses[0] : prev.status,
      performance: ok(newResponses[1]) ? newResponses[1] : prev.performance,
      positions: Array.isArray(newResponses[2]) ? newResponses[2] : prev.positions,
      alerts: Array.isArray(newResponses[3]) ? newResponses[3] : prev.alerts,
      news: Array.isArray(newResponses[4]) ? newResponses[4] : prev.news,
      riskMetrics: ok(newResponses[5]) ? newResponses[5] : prev.riskMetrics,
      modelPerf: ok(newResponses[6]) ? newResponses[6] : prev.modelPerf,
      strategies: ok(newResponses[7])
        ? Object.entries(newResponses[7])
            .filter(([, stats]) => stats && typeof stats === 'object' && 'realized_pnl' in stats)
            .map(([name, stats]) => ({ name, ...stats }))
        : prev.strategies,
      heatmap: Array.isArray(newResponses[8]) ? newResponses[8] : prev.heatmap,
      systemHealth: ok(newResponses[9]) ? newResponses[9] : prev.systemHealth,
      agentActivity: Array.isArray(newResponses[10]) ? newResponses[10] : prev.agentActivity,
      allocation: Array.isArray(newResponses[11]) ? newResponses[11] : prev.allocation
    }));
    setIsConnected(ok(newResponses[0]));
  };
```

- [ ] **Step 3: Show a loading state instead of "$0 / OFFLINE"**

In `renderOverview` (line ~478), add as the first statement:

```js
  const renderOverview = () => {
    if (data.loading) {
      return (
        <div style={{ padding: '80px', textAlign: 'center', color: theme.colors.textMuted }}>
          <div style={{ fontSize: '14px', letterSpacing: '2px' }}>ESTABLISHING SECURE LINK…</div>
          <div style={{ fontSize: '11px', marginTop: '8px' }}>Loading portfolio, risk and market state</div>
        </div>
      );
    }
    return (
    <div style={{ display: 'grid', gap: '30px' }}>
    ...
```

(and close the wrapper: the existing JSX return becomes the `return (...)` after the guard, with a closing `); }` at the end of `renderOverview`).

Also change the System Health card (line ~484) so its count can't contradict the label:

```jsx
        <HUDCard title="System Health" value={isConnected ? 'OPTIMAL' : 'DEGRADED'} subValue={isConnected ? `${Object.keys(data.status.components || {}).length} Services Active` : 'Reconnecting…'} icon={Activity} color={isConnected ? theme.colors.accent : theme.colors.warning} />
```

- [ ] **Step 4: Verify in the browser**

Reload the app: the dashboard must show the loading placeholder (never "$0"), then populate in one paint. Stop the backend briefly mid-session: the card should flip to DEGRADED / Reconnecting… while cached numbers stay on screen, and recover on restart.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/AdvancedDashboard.js
git commit -m "fix(dashboard): parallel data fetch, loading state, and truthful connection status"
```

---

### Task 3: Anchor SQLite paths to the project root

`EscalationManager` (and any other journal using `os.path.join('data', ...)`) resolves relative to the process CWD — a second process started from another directory writes to a *different* database, which is the likeliest cause of the Research tab's "disappearing" uploads/watchlist.

**Files:**
- Create: `src/utils/paths.py`
- Create: `tests/test_paths.py`
- Modify: `src/agent/escalation_manager.py:74`
- Modify: every other module matching `os.path.join('data'` (sweep in Step 5)

**Interfaces:**
- Produces: `from src.utils.paths import DATA_DIR` — an absolute `pathlib.Path` to `<repo>/data`, created on import.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
import os
from pathlib import Path


def test_data_dir_is_absolute_and_inside_repo():
    from src.utils.paths import DATA_DIR, PROJECT_ROOT
    assert DATA_DIR.is_absolute()
    assert DATA_DIR == PROJECT_ROOT / 'data'
    assert (PROJECT_ROOT / 'src').is_dir()  # sanity: root really is the repo


def test_data_dir_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    import src.utils.paths as p
    importlib.reload(p)
    assert str(tmp_path) not in str(p.DATA_DIR)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.paths'`

- [ ] **Step 3: Write the implementation**

```python
# src/utils/paths.py
"""Absolute filesystem anchors. Import DATA_DIR instead of joining 'data'
relatively — relative paths silently fork state when a process starts from
a different working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paths.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Migrate EscalationManager and sweep the rest**

In `src/agent/escalation_manager.py:74` change the signature:

```python
from src.utils.paths import DATA_DIR

class EscalationManager:
    def __init__(self, db_path: str = str(DATA_DIR / 'escalations.db')):
```

Then sweep: `grep -rn "os.path.join('data'" src/` and `grep -rn "\"data/" src/` — for each hit (decision journal, order journal, swarm_state_store, portfolio history, etc.) replace the relative default with `str(DATA_DIR / '<same filename>')`. Do not change call sites that already pass an explicit path.

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/utils/paths.py tests/test_paths.py src/agent/escalation_manager.py src/
git commit -m "fix(persistence): anchor all sqlite paths to repo root (CWD-independent)"
```

---

### Task 4: Analytics 1D x-axis shows time, not a repeated date

**Files:**
- Modify: `frontend/components/AdvancedAnalytics.js:34-44` and the XAxis at line 112.

- [ ] **Step 1: Format tick labels by range**

Replace the `forEach` block (lines 34–44) with:

```js
  const fmtTick = (ts) => {
    const d = new Date(ts);
    return timeRange === '1d'
      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  filteredChart.forEach(point => {
    const val = point.portfolio_value || point.equity || 0;
    if (val > peak) peak = val;
    const drawdownPct = peak > 0 ? ((val - peak) / peak) * 100 : 0;
    drawdownAnalysis.push({
      date: fmtTick(point.timestamp),
      equity: val,
      drawdown: parseFloat(drawdownPct.toFixed(2)),
      underwater: parseFloat((drawdownPct * 1.2).toFixed(2))
    });
  });
```

- [ ] **Step 2: Thin the ticks**

Change the XAxis (line ~112) to avoid label pile-up:

```jsx
                <XAxis dataKey="date" stroke={theme.colors.textMuted} fontSize={10} minTickGap={40} />
```

- [ ] **Step 3: Verify in the browser**

Analytics tab → 1D shows distinct times (e.g. 09:40, 10:20…); 7D/30D show `Jul 4`, `Jul 7`-style dates.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/AdvancedAnalytics.js
git commit -m "fix(analytics): 1D chart axis shows intraday times instead of repeated date"
```

---

### Task 5: Friendly symbol-not-found message

**Files:**
- Modify: `frontend/components/SymbolDrilldown.js:54` and `:102`

- [ ] **Step 1: Map the 404 to human copy**

Line 54 becomes:

```js
        if (!res.ok) { setError(res.status === 404 ? 'not_found' : `HTTP ${res.status}`); return; }
```

Line 102 becomes:

```jsx
        {error && (
          <div style={{ color: theme.colors.danger, padding: '20px' }}>
            {error === 'not_found'
              ? `"${symbol}" isn't a tracked symbol. Check the ticker — e.g. SCOM, EQTY (NSE) or AAPL, NVDA (US).`
              : `Failed to load: ${error}`}
          </div>
        )}
```

- [ ] **Step 2: Verify in the browser**

Search a junk ticker (`ZZZZ`) in the Drill symbol box → the friendly message appears; a valid ticker still loads.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/SymbolDrilldown.js
git commit -m "fix(drilldown): friendly message for unknown symbols"
```

---

## Phase 1 — NSE Kenya session hours

The code treats NSE as open 09:00–15:00 EAT (`nse_connector.py:19-20`, `nse_scraper.py:531`). The real Nairobi Securities Exchange equities session is: **pre-open 09:00–09:30, continuous trading 09:30–15:00, closing auction/price run to ~15:30, Mon–Fri**. The agent must not submit orders during pre-open, and the UI should show the session phase.

### Task 6: Minute-granular NSE session phases (backend)

**Files:**
- Modify: `src/connectors/nse_connector.py:15-21, 95-112`
- Modify: `src/connectors/nse_scraper.py:528-533`
- Create: `tests/test_nse_hours.py`

**Interfaces:**
- Produces: `market_phase(now: datetime | None = None) -> str` returning `'preopen' | 'open' | 'closed'`; `is_market_open()` true only during `'open'`; `get_status()` gains `market_phase` and `next_transition` (ISO time string).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nse_hours.py
from datetime import datetime
from src.connectors.nse_connector import NSEConnector, EAT_OFFSET


def _at(h, m, weekday_date='2026-07-10'):  # 2026-07-10 is a Friday
    return datetime.fromisoformat(f'{weekday_date}T{h:02d}:{m:02d}:00').replace(tzinfo=EAT_OFFSET)


def test_preopen_is_not_tradeable():
    nse = NSEConnector.__new__(NSEConnector)  # skip network/db init
    assert nse.market_phase(_at(9, 15)) == 'preopen'
    assert nse.is_market_open(_at(9, 15)) is False


def test_continuous_session_bounds():
    nse = NSEConnector.__new__(NSEConnector)
    assert nse.market_phase(_at(9, 30)) == 'open'
    assert nse.market_phase(_at(14, 59)) == 'open'
    assert nse.market_phase(_at(15, 0)) == 'closed'


def test_weekend_closed():
    nse = NSEConnector.__new__(NSEConnector)
    sat = datetime.fromisoformat('2026-07-11T10:00:00').replace(tzinfo=EAT_OFFSET)
    assert nse.market_phase(sat) == 'closed'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nse_hours.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'market_phase'`

- [ ] **Step 3: Implement session phases**

In `src/connectors/nse_connector.py` replace lines 18–20 with:

```python
# NSE equities session (Nairobi time): pre-open auction 09:00-09:30,
# continuous trading 09:30-15:00. Orders may only be submitted while 'open'.
from datetime import time as dtime
NSE_PREOPEN_START = dtime(9, 0)
NSE_OPEN = dtime(9, 30)
NSE_CLOSE = dtime(15, 0)
```

Replace `is_market_open` (lines 108–112) with:

```python
    def market_phase(self, now=None) -> str:
        now = now or datetime.now(EAT_OFFSET)
        if now.weekday() > 4:
            return 'closed'
        t = now.timetz().replace(tzinfo=None)
        if NSE_PREOPEN_START <= t < NSE_OPEN:
            return 'preopen'
        if NSE_OPEN <= t < NSE_CLOSE:
            return 'open'
        return 'closed'

    def is_market_open(self, now=None) -> bool:
        return self.market_phase(now) == 'open'
```

In `get_status()` (line ~100) add after `"market_open": self.is_market_open(),`:

```python
            "market_phase": self.market_phase(),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_nse_hours.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Align the scraper window**

In `src/connectors/nse_scraper.py:529-532`, the scraper may keep polling from 09:00 (pre-open prices are useful) — change the magic numbers to the named constants so the two files can't drift:

```python
                from src.connectors.nse_connector import NSE_PREOPEN_START, NSE_CLOSE
                is_market_hours = (
                    now.weekday() < 5
                    and NSE_PREOPEN_START <= now.timetz().replace(tzinfo=None) < NSE_CLOSE
                )
```

- [ ] **Step 6: Run the full suite and commit**

Run: `python -m pytest tests/ -v` → green.

```bash
git add src/connectors/nse_connector.py src/connectors/nse_scraper.py tests/test_nse_hours.py
git commit -m "fix(nse): correct session hours - preopen 09:00-09:30, trading 09:30-15:00 EAT"
```

---

### Task 7: NSE session clock in the UI

**Files:**
- Modify: `frontend/components/AdvancedDashboard.js:621` (NSE Status HUDCard)

**Interfaces:**
- Consumes: `nseData.status.market_phase` from `/api/nse-market` (Task 6). Note the API nests connector status under `status`.

- [ ] **Step 1: Compute phase + countdown client-side (EAT = UTC+3 fixed, no DST)**

Add above `renderKenyaNSE`:

```js
  const nseSession = () => {
    const now = new Date();
    const eatMin = ((now.getUTCHours() + 3) % 24) * 60 + now.getUTCMinutes();
    const day = (now.getUTCDay() + (now.getUTCHours() + 3 >= 24 ? 1 : 0)) % 7;
    const OPEN = 9 * 60 + 30, CLOSE = 15 * 60, PRE = 9 * 60;
    if (day === 0 || day === 6) return { label: 'CLOSED · WEEKEND', color: theme.colors.textMuted };
    if (eatMin < PRE) return { label: `PRE-OPEN IN ${Math.floor((PRE - eatMin) / 60)}h ${(PRE - eatMin) % 60}m`, color: theme.colors.textMuted };
    if (eatMin < OPEN) return { label: `PRE-OPEN · TRADING IN ${OPEN - eatMin}m`, color: theme.colors.warning };
    if (eatMin < CLOSE) return { label: `OPEN · CLOSES IN ${Math.floor((CLOSE - eatMin) / 60)}h ${(CLOSE - eatMin) % 60}m`, color: theme.colors.primary };
    return { label: 'CLOSED', color: theme.colors.warning };
  };
```

- [ ] **Step 2: Replace the NSE Status card (line ~621)**

```jsx
        {(() => { const s = nseSession(); return (
          <HUDCard title="NSE Session" value={s.label.split(' · ')[0]} subValue={s.label.includes('·') ? s.label.split(' · ')[1] : `${nseData.quotes?.length || 0} Symbols`} icon={Activity} color={s.color} />
        ); })()}
```

- [ ] **Step 3: Verify in the browser**

NSE Kenya tab shows the phase matching Nairobi time right now (e.g. OPEN with a countdown to 15:00 EAT during the session).

- [ ] **Step 4: Commit**

```bash
git add frontend/components/AdvancedDashboard.js
git commit -m "feat(nse): session-phase clock with open/close countdown"
```

---

## Phase 2 — Universal symbol click-through

`/api/symbol/<sym>` already accepts both US and NSE symbols (validated at `api_server.py:321-326`), and the modal exists. Only the wiring is missing: NSE table rows and mover cards have no `onClick`.

### Task 8: Make every rendered symbol clickable

**Files:**
- Modify: `frontend/components/AdvancedDashboard.js:640-653` (NSE table rows), `:622-623` (mover cards), and the heatmap tiles in the Market view (search for the `data.heatmap` render).

- [ ] **Step 1: NSE Market Watch rows**

Replace the `<tr key={i} ...>` opening tag at line 643 with:

```jsx
                    <tr key={q.symbol || i}
                        onClick={() => setDrilldownSymbol(q.symbol)}
                        title={`Open ${q.symbol} performance drill-down`}
                        style={{ borderBottom: `1px solid ${theme.colors.border}`, background: pos ? `${theme.colors.primary}15` : 'transparent', cursor: 'pointer' }}
                        onMouseEnter={(e) => e.currentTarget.style.background = `${theme.colors.primary}25`}
                        onMouseLeave={(e) => e.currentTarget.style.background = pos ? `${theme.colors.primary}15` : 'transparent'}>
```

And append a chevron affordance to the symbol cell (line 644):

```jsx
                      <td style={{ padding: '12px', fontWeight: '700' }}>{q.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span></td>
```

- [ ] **Step 2: Top gainer / loser cards (lines 622–623)**

Wrap each HUDCard in a clickable div:

```jsx
        <div onClick={() => nseData.movers?.gainers?.[0]?.symbol && setDrilldownSymbol(nseData.movers.gainers[0].symbol)} style={{ cursor: 'pointer' }}>
          <HUDCard title="Top NSE Gainer" value={nseData.movers?.gainers?.[0]?.symbol || '—'} subValue={`+${nseData.movers?.gainers?.[0]?.change_pct?.toFixed(2) || 0}%`} icon={TrendingUp} color={theme.colors.primary} />
        </div>
        <div onClick={() => nseData.movers?.losers?.[0]?.symbol && setDrilldownSymbol(nseData.movers.losers[0].symbol)} style={{ cursor: 'pointer' }}>
          <HUDCard title="Top NSE Loser" value={nseData.movers?.losers?.[0]?.symbol || '—'} subValue={`${nseData.movers?.losers?.[0]?.change_pct?.toFixed(2) || 0}%`} icon={TrendingDown} color={theme.colors.danger} />
        </div>
```

- [ ] **Step 3: Sweep remaining symbol renders**

Grep `frontend/components/AdvancedDashboard.js` for `.symbol` renders without an `onClick` in their row/tile (US heatmap tiles already open the *sector* drill-down — keep that; add symbol click-through only where an individual ticker is shown, e.g. watchlist rows in ResearchView: `onClick={() => setDrilldownSymbol(item.symbol)}` on the row, but `e.stopPropagation()` on the PAUSE/RESUME/REMOVE buttons — the buttons at lines 323–327 already need `(e) => { e.stopPropagation(); handleWatchlistAction(...) }`).

Note: `ResearchView` doesn't receive `setDrilldownSymbol` — pass it as a prop at line 833: `component: () => <ResearchView activeTab={activeTab} fetchData={fetchData} onDrill={setDrilldownSymbol} />` and accept `onDrill` in the `ResearchView` signature at line 57.

- [ ] **Step 4: Verify in the browser**

Click SCOM in NSE Market Watch → drill-down modal opens with price/decision tape. Click a watchlist row → same. PAUSE button still pauses without opening the modal.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/AdvancedDashboard.js
git commit -m "feat(ux): every symbol click-through to drill-down (NSE table, movers, watchlist)"
```

---

## Phase 3 — Live transparency: what is the agent holding / reviewing / trading?

TradingView's best pattern here is the *watchlist with live status columns*. We already journal every per-cycle decision — expose it as a three-lane "Agent Operations" board.

### Task 9: `/api/agent-focus` endpoint

**Files:**
- Create: `src/api/agent_focus.py`
- Create: `tests/test_agent_focus.py`
- Modify: `src/api/api_server.py` (new route after `get_agent_activity`)

**Interfaces:**
- Produces: `build_agent_focus(positions, decisions, cash, equity) -> dict` with keys `holding` (list of `{symbol, quantity, unrealized_pl_pct}`), `reviewing` (list of `{symbol, action, reason, ts}` from the latest cycle's non-executed decisions), `traded` (list of `{symbol, action, price, ts}` executed today), `cash` (`{cash, equity, deployed_pct}`). Route: `GET /api/agent-focus`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_focus.py
from src.api.agent_focus import build_agent_focus


def test_lanes_partition_decisions():
    positions = [{'symbol': 'SCOM', 'quantity': 100, 'unrealized_pl_pct': 2.1}]
    decisions = [
        {'ts': '2026-07-10T09:40:00', 'symbol': 'EQTY', 'action': 'hold',
         'executed': False, 'skip_reason': 'low_confidence'},
        {'ts': '2026-07-10T09:40:01', 'symbol': 'KCB', 'action': 'buy',
         'executed': True, 'price': 41.5},
    ]
    focus = build_agent_focus(positions, decisions, cash=60000, equity=100000)
    assert focus['holding'] == [{'symbol': 'SCOM', 'quantity': 100, 'unrealized_pl_pct': 2.1}]
    assert focus['reviewing'][0]['symbol'] == 'EQTY'
    assert focus['reviewing'][0]['reason'] == 'low_confidence'
    assert focus['traded'][0]['symbol'] == 'KCB'
    assert focus['cash']['deployed_pct'] == 40.0


def test_empty_inputs():
    focus = build_agent_focus([], [], cash=0, equity=0)
    assert focus == {'holding': [], 'reviewing': [], 'traded': [],
                     'cash': {'cash': 0, 'equity': 0, 'deployed_pct': 0.0}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_focus.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

```python
# src/api/agent_focus.py
"""Three-lane live view of agent behaviour: holding / reviewing / traded."""
from typing import Any, Dict, List


def build_agent_focus(positions: List[Dict[str, Any]], decisions: List[Dict[str, Any]],
                      cash: float, equity: float) -> Dict[str, Any]:
    holding = [{'symbol': p.get('symbol'), 'quantity': p.get('quantity'),
                'unrealized_pl_pct': p.get('unrealized_pl_pct')} for p in positions]
    reviewing, traded, seen = [], [], set()
    for d in decisions:  # journal returns newest first; keep first per symbol
        sym = d.get('symbol')
        if sym in seen:
            continue
        seen.add(sym)
        if d.get('executed'):
            traded.append({'symbol': sym, 'action': (d.get('action') or '').upper(),
                           'price': d.get('price'), 'ts': d.get('ts')})
        else:
            reviewing.append({'symbol': sym, 'action': (d.get('action') or 'hold').upper(),
                              'reason': d.get('skip_reason') or 'signal below threshold',
                              'ts': d.get('ts')})
    deployed_pct = round((1 - cash / equity) * 100, 1) if equity else 0.0
    return {'holding': holding, 'reviewing': reviewing, 'traded': traded,
            'cash': {'cash': cash, 'equity': equity, 'deployed_pct': deployed_pct}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_focus.py -v` → 2 PASSED

- [ ] **Step 5: Add the route**

In `src/api/api_server.py`, after the `get_agent_activity` route:

```python
        @self.app.route('/api/agent-focus', methods=['GET'])
        @require_rate_limit
        @token_required
        def get_agent_focus():
            """Live board: holdings, symbols under review, today's executions."""
            def produce():
                from src.api.agent_focus import build_agent_focus
                dj = getattr(self.trading_agent, 'decision_journal', None)
                decisions = dj.recent(None, limit=100) if dj else []
                positions, cash, equity = [], 0.0, 0.0
                broker = self.trading_agent.components.get('broker')
                try:
                    acct = broker.get_account_info() if broker else None
                    cash = float(acct.cash) if acct else 0.0
                    equity = float(acct.equity) if acct else 0.0
                except Exception:
                    pass
                try:
                    positions = self.trading_agent.get_positions_snapshot()
                except Exception:
                    pass
                return build_agent_focus(positions, decisions, cash, equity)
            try:
                return jsonify(self._cached('agent_focus', 15, produce))
            except Exception as e:
                logger.error(f"Error building agent focus: {e}")
                return jsonify({'holding': [], 'reviewing': [], 'traded': [], 'cash': {}})
```

If `get_positions_snapshot` doesn't exist on the agent, use the same positions source the existing `/api/positions` route uses (read that route and copy its accessor — keep the two consistent).

- [ ] **Step 6: Commit**

```bash
git add src/api/agent_focus.py tests/test_agent_focus.py src/api/api_server.py
git commit -m "feat(api): /api/agent-focus - holding/reviewing/traded live board"
```

---

### Task 10: "Agent Operations" board on the Dashboard

**Files:**
- Create: `frontend/components/AgentFocus.js`
- Modify: `frontend/components/AdvancedDashboard.js` (render below the Performance Curve, next to `<AgentActivity …/>` at line ~517)

**Interfaces:**
- Consumes: `GET /api/agent-focus` (Task 9), `getApiBase()` from `frontend/utils` (import the same way `SymbolDrilldown.js` does), `theme` from `./DashboardStyles`, `onDrill(symbol)` prop opening the drill-down.

- [ ] **Step 1: Build the component**

```jsx
// frontend/components/AgentFocus.js
import { useState, useEffect } from 'react';
import { theme } from './DashboardStyles';
import { getApiBase } from '../utils/api';

const Lane = ({ title, color, children }) => (
  <div style={{ flex: 1, minWidth: '180px' }}>
    <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '1px', color, marginBottom: '10px' }}>{title}</div>
    {children}
  </div>
);

const AgentFocus = ({ onDrill }) => {
  const [focus, setFocus] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/agent-focus`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setFocus(json);
      } catch (e) {}
    };
    load();
    const id = setInterval(load, 20000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!focus) return null;
  const row = { display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '13px', cursor: 'pointer' };
  const empty = (msg) => <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>{msg}</div>;

  return (
    <div style={{ ...theme.glass, padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 800, letterSpacing: '1px', color: '#fff' }}>AGENT OPERATIONS — LIVE</h3>
        <span style={{ fontSize: '12px', color: theme.colors.textMuted }}>
          Capital deployed: <span style={{ color: theme.colors.primary, fontWeight: 800 }}>{focus.cash?.deployed_pct ?? 0}%</span>
        </span>
      </div>
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        <Lane title={`HOLDING (${focus.holding.length})`} color={theme.colors.primary}>
          {focus.holding.length === 0 && empty('No open positions')}
          {focus.holding.map(h => (
            <div key={h.symbol} style={row} onClick={() => onDrill(h.symbol)}>
              <span style={{ fontWeight: 700 }}>{h.symbol}</span>
              <span style={{ color: (h.unrealized_pl_pct ?? 0) >= 0 ? theme.colors.primary : theme.colors.danger }}>
                {h.quantity} · {(h.unrealized_pl_pct ?? 0) >= 0 ? '+' : ''}{(h.unrealized_pl_pct ?? 0).toFixed(2)}%
              </span>
            </div>
          ))}
        </Lane>
        <Lane title={`REVIEWING (${focus.reviewing.length})`} color={theme.colors.warning}>
          {focus.reviewing.length === 0 && empty('Nothing under review this cycle')}
          {focus.reviewing.slice(0, 8).map(r => (
            <div key={r.symbol} style={row} onClick={() => onDrill(r.symbol)}>
              <span style={{ fontWeight: 700 }}>{r.symbol}</span>
              <span style={{ color: theme.colors.textMuted, fontSize: '11px' }}>{r.action} · {r.reason}</span>
            </div>
          ))}
        </Lane>
        <Lane title={`TRADED (${focus.traded.length})`} color={theme.colors.accent}>
          {focus.traded.length === 0 && empty('No executions yet today')}
          {focus.traded.slice(0, 8).map(t => (
            <div key={`${t.symbol}-${t.ts}`} style={row} onClick={() => onDrill(t.symbol)}>
              <span style={{ fontWeight: 700 }}>{t.symbol}</span>
              <span style={{ fontSize: '11px' }}>{t.action} @ {t.price}</span>
            </div>
          ))}
        </Lane>
      </div>
    </div>
  );
};

export default AgentFocus;
```

(Adjust the `getApiBase` import path to match how `AdvancedDashboard.js` imports it — copy that exact import line.)

- [ ] **Step 2: Mount it on the overview**

In `AdvancedDashboard.js`: `import AgentFocus from './AgentFocus';` (next to the AgentActivity import at line 12), then directly above `<AgentActivity activities={data.agentActivity} />` (line ~517) insert:

```jsx
          <AgentFocus onDrill={setDrilldownSymbol} />
```

- [ ] **Step 3: Verify in the browser**

Dashboard shows the three-lane board; every symbol in it opens the drill-down; "Capital deployed" percentage matches Positions vs cash.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/AgentFocus.js frontend/components/AdvancedDashboard.js
git commit -m "feat(dashboard): live Agent Operations board (holding/reviewing/traded)"
```

---

## Phase 4 — Idle-cash deployment policy

Root cause of cash drag: `src/agent/main.py:1038` hardcodes `max_risk_per_trade = 0.02`, so no position can exceed 2% of equity regardless of conviction — with ~29 symbols and mostly holds, ~90% of the $100k idles. The professional fix is a **portfolio-level deployment controller**, not a bigger per-trade dial: scale conviction-ranked signals up when deployment is chronically below target, always respecting the 5% per-position hard cap and existing VaR/drawdown limits.

### Task 11: `CashDeploymentPolicy`

**Files:**
- Create: `src/agent/cash_policy.py`
- Create: `tests/test_cash_policy.py`
- Modify: `config/config.json` (new top-level `cash_policy` block)
- Modify: `src/agent/main.py:1038-1041`

**Interfaces:**
- Produces: `CashDeploymentPolicy(config).risk_cap(deployed_pct: float, confidence: float) -> float` — returns the per-trade fraction-of-equity cap replacing the hardcoded `0.02`.

- [ ] **Step 1: Add config**

In `config/config.json`, add at the top level (sibling of `risk_management`):

```json
  "cash_policy": {
    "enabled": true,
    "target_deployment": 0.80,
    "min_cash_buffer": 0.10,
    "base_risk_per_trade": 0.02,
    "max_risk_per_trade": 0.05,
    "min_confidence_to_boost": 0.55
  },
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_cash_policy.py
from src.agent.cash_policy import CashDeploymentPolicy

CFG = {'cash_policy': {
    'enabled': True, 'target_deployment': 0.80, 'min_cash_buffer': 0.10,
    'base_risk_per_trade': 0.02, 'max_risk_per_trade': 0.05,
    'min_confidence_to_boost': 0.55,
}}


def test_base_cap_when_fully_deployed():
    p = CashDeploymentPolicy(CFG)
    assert p.risk_cap(deployed_pct=0.85, confidence=0.9) == 0.02


def test_boost_scales_with_gap_and_conviction():
    p = CashDeploymentPolicy(CFG)
    # 10% deployed, high conviction -> boosted toward the 5% ceiling
    cap = p.risk_cap(deployed_pct=0.10, confidence=0.9)
    assert 0.02 < cap <= 0.05
    # same gap, weak conviction -> no boost
    assert p.risk_cap(deployed_pct=0.10, confidence=0.4) == 0.02


def test_never_exceeds_hard_ceiling():
    p = CashDeploymentPolicy(CFG)
    assert p.risk_cap(deployed_pct=0.0, confidence=1.0) <= 0.05


def test_disabled_policy_returns_base():
    cfg = {'cash_policy': dict(CFG['cash_policy'], enabled=False)}
    assert CashDeploymentPolicy(cfg).risk_cap(0.0, 1.0) == 0.02
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_cash_policy.py -v`
Expected: FAIL (module missing)

- [ ] **Step 4: Implement**

```python
# src/agent/cash_policy.py
"""Portfolio-level cash deployment controller.

The per-trade risk cap floats between base (2%) and ceiling (5%) in
proportion to (a) how far current deployment sits below target and
(b) signal conviction. Weak signals never get boosted, so idle cash is
deployed into the *best* ideas, not sprayed across the book.
"""
from typing import Any, Dict


class CashDeploymentPolicy:
    def __init__(self, config: Dict[str, Any]):
        cp = config.get('cash_policy', {})
        self.enabled = cp.get('enabled', False)
        self.target = cp.get('target_deployment', 0.80)
        self.base = cp.get('base_risk_per_trade', 0.02)
        self.ceiling = cp.get('max_risk_per_trade', 0.05)
        self.min_conf = cp.get('min_confidence_to_boost', 0.55)

    def risk_cap(self, deployed_pct: float, confidence: float) -> float:
        if not self.enabled or confidence < self.min_conf:
            return self.base
        gap = max(0.0, self.target - deployed_pct)
        if gap <= 0:
            return self.base
        # gap=target (empty book) with confidence=1.0 reaches the ceiling.
        boost = (self.ceiling - self.base) * (gap / self.target) * confidence
        return round(min(self.ceiling, self.base + boost), 4)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_cash_policy.py -v` → 4 PASSED

- [ ] **Step 6: Integrate into the execution path**

In `src/agent/main.py`, replace lines 1037–1041:

```python
                        portfolio_value = account_info.equity
                        cash = float(getattr(account_info, 'cash', 0) or 0)
                        deployed_pct = 1 - (cash / portfolio_value) if portfolio_value else 0.0
                        if not hasattr(self, 'cash_policy'):
                            from src.agent.cash_policy import CashDeploymentPolicy
                            self.cash_policy = CashDeploymentPolicy(self.config)
                        max_risk_per_trade = self.cash_policy.risk_cap(
                            deployed_pct, signal_data.get('confidence', 0.0))

                        # Position value based on signal (confidence * position_size)
                        target_pos_value = portfolio_value * min(position_size, max_risk_per_trade)
```

Note: `position_size` still bounds the result, and the risk manager's `max_position_size: 0.05` check at `main.py:904-907` remains the hard backstop.

- [ ] **Step 7: Full suite + commit**

Run: `python -m pytest tests/ -v` → green.

```bash
git add src/agent/cash_policy.py tests/test_cash_policy.py src/agent/main.py config/config.json
git commit -m "feat(sizing): conviction-scaled cash deployment policy replaces hardcoded 2% cap"
```

---

### Task 12: Surface deployment on the dashboard

**Files:**
- Modify: `src/api/api_server.py` (extend the `/api/agent-focus` produce with `policy` fields), `frontend/components/AgentFocus.js`

- [ ] **Step 1: Extend the focus payload**

In the Task 9 route's `produce()`, after computing `cash`/`equity`, add:

```python
                payload = build_agent_focus(positions, decisions, cash, equity)
                cp = self.config.get('cash_policy', {})
                payload['cash']['target_deployment_pct'] = round(cp.get('target_deployment', 0.8) * 100, 1)
                payload['cash']['policy_enabled'] = bool(cp.get('enabled'))
                return payload
```

- [ ] **Step 2: Show target vs actual in AgentFocus header**

Replace the "Capital deployed" span in `AgentFocus.js` with:

```jsx
        <span style={{ fontSize: '12px', color: theme.colors.textMuted }}>
          Deployed: <span style={{ color: theme.colors.primary, fontWeight: 800 }}>{focus.cash?.deployed_pct ?? 0}%</span>
          {focus.cash?.policy_enabled && <> / target {focus.cash?.target_deployment_pct}%</>}
        </span>
```

- [ ] **Step 3: Verify in the browser, then commit**

```bash
git add src/api/api_server.py frontend/components/AgentFocus.js
git commit -m "feat(dashboard): show capital deployed vs policy target"
```

---

## Phase 5 — Progressive intelligence

Three mechanisms, in dependency order: (a) free-tier LLM with a verdict cache so quota is never the bottleneck, (b) outcome scoring that feeds the agent's own hit-rate back into its prompts, (c) operator-approved universe expansion so the agent can propose new stocks.

### Task 13: Gemini free tier + LLM verdict cache

**Files:**
- Modify: `src/agent/llm_orchestrator.py`
- Modify: `config/config.json` (`"primary_llm_provider": "gemini"` inside a new `llm` block or top level to match `config.get("primary_llm_provider", ...)`)
- Create: `tests/test_llm_cache.py`

**Interfaces:**
- Produces: `validate_trade(...)` behaviour unchanged externally, but consecutive calls for the same `(symbol, action, confidence-bucket)` within `cache_ttl` seconds return the cached verdict without an HTTP call.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_cache.py
from src.agent.llm_orchestrator import LLMOrchestrator


def _orch(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    return LLMOrchestrator({'primary_llm_provider': 'gemini', 'llm_cache_ttl': 900})


def test_second_identical_call_is_served_from_cache(monkeypatch):
    orch = _orch(monkeypatch)
    calls = {'n': 0}

    def fake_gemini(sys_p, usr_p, fb, model_override=None):
        calls['n'] += 1
        return {'action': 'buy', 'confidence': 0.7, 'position_size': 0.03, 'reasoning': 'ok'}

    monkeypatch.setattr(orch, '_call_gemini', fake_gemini)
    sig = {'action': 'buy', 'confidence': 0.71}
    r1 = orch.validate_trade('SCOM', sig, {'close': 34.0})
    r2 = orch.validate_trade('SCOM', sig, {'close': 34.1})
    assert calls['n'] == 1
    assert r1['action'] == r2['action'] == 'buy'


def test_different_action_bypasses_cache(monkeypatch):
    orch = _orch(monkeypatch)
    calls = {'n': 0}
    monkeypatch.setattr(orch, '_call_gemini',
                        lambda *a, **k: calls.__setitem__('n', calls['n'] + 1) or {'action': 'hold', 'confidence': 0.5, 'position_size': 0, 'reasoning': 'x'})
    orch.validate_trade('SCOM', {'action': 'buy', 'confidence': 0.7}, {'close': 34.0})
    orch.validate_trade('SCOM', {'action': 'sell', 'confidence': 0.7}, {'close': 34.0})
    assert calls['n'] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_cache.py -v`
Expected: FAIL — `calls['n'] == 2` in the first test (no cache yet).

- [ ] **Step 3: Implement the cache and provider default**

In `src/agent/llm_orchestrator.py`:

`__init__` additions (after line 25):

```python
        self.cache_ttl = config.get('llm_cache_ttl', 900)  # 15 min default
        self._verdict_cache: Dict[str, tuple] = {}  # key -> (expires_at, verdict)
```

Add `import time` at the top. In `validate_trade`, after the weak-hold early return (line 36), insert:

```python
        conf_bucket = round(strategy_signal.get('confidence', 0) * 10)  # 0.71/0.73 share a verdict
        cache_key = f"{symbol}:{action}:{conf_bucket}"
        cached = self._verdict_cache.get(cache_key)
        if cached and cached[0] > time.time():
            return dict(cached[1])
```

And wrap every successful provider return so it lands in the cache. Replace the provider block (lines 85–98) with:

```python
        def _remember(verdict):
            if isinstance(verdict, dict):
                self._verdict_cache[cache_key] = (time.time() + self.cache_ttl, dict(verdict))
            return verdict

        try:
            if self.primary_provider == "openrouter" and self.openrouter_api_key:
                try:
                    return _remember(self._call_openrouter(system_prompt, user_prompt, strategy_signal, model_override=model))
                except Exception as openrouter_err:
                    logger.warning(f"OpenRouter validation failed ({openrouter_err}). Falling back to Gemini...")

            if self.gemini_api_key:
                return _remember(self._call_gemini(system_prompt, user_prompt, strategy_signal, model_override=model))

            return strategy_signal
        except Exception as e:
            logger.error(f"LLM validation failed: {str(e)}. Falling back to base signal.")
            return strategy_signal
```

Update the Gemini default model (line 163): `model = model_override or "gemini-2.5-flash-lite"` (keep the `"/" in model` fallback mapping, pointing at the same default).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_cache.py -v` → 2 PASSED

- [ ] **Step 5: Flip config to Gemini and document the env var**

In `config/config.json` add `"primary_llm_provider": "gemini",` at the top level (the code reads `config.get("primary_llm_provider", "openrouter")`). In `.env.example`, move `GEMINI_API_KEY` above `OPENROUTER_API_KEY` and comment: `# Free tier: ~10 RPM / 1,500 req-day on gemini-2.5-flash-lite — get a key at ai.google.dev`.

- [ ] **Step 6: Full suite + commit**

```bash
git add src/agent/llm_orchestrator.py tests/test_llm_cache.py config/config.json .env.example
git commit -m "feat(llm): Gemini free tier as primary provider + 15-min verdict cache"
```

---

### Task 14: Outcome scoreboard — the agent learns its own hit rate

**Files:**
- Create: `src/agent/verdict_scoreboard.py`
- Create: `tests/test_verdict_scoreboard.py`
- Modify: `src/agent/llm_orchestrator.py` (`validate_trade` prompt), `src/agent/main.py` (pass the scoreboard summary in)

**Interfaces:**
- Produces: `score_decisions(decisions: list[dict], price_lookup: Callable[[str], float | None]) -> dict` returning `{symbol: {'evaluated': int, 'hits': int, 'hit_rate': float}}`; `summary_line(scores: dict, symbol: str) -> str` — one sentence for the LLM prompt.
- A decision counts as a *hit* when: executed buy and current price > decision price, executed sell and current price < decision price. Non-executed decisions are ignored.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verdict_scoreboard.py
from src.agent.verdict_scoreboard import score_decisions, summary_line


def test_buy_that_appreciated_is_a_hit():
    decisions = [
        {'symbol': 'SCOM', 'action': 'buy', 'executed': True, 'price': 30.0},
        {'symbol': 'SCOM', 'action': 'buy', 'executed': True, 'price': 40.0},
        {'symbol': 'SCOM', 'action': 'hold', 'executed': False, 'price': 31.0},
    ]
    scores = score_decisions(decisions, lambda s: 34.0)
    assert scores['SCOM']['evaluated'] == 2      # holds/skips not scored
    assert scores['SCOM']['hits'] == 1           # 30->34 hit, 40->34 miss
    assert scores['SCOM']['hit_rate'] == 0.5


def test_summary_line_mentions_rate():
    line = summary_line({'SCOM': {'evaluated': 10, 'hits': 6, 'hit_rate': 0.6}}, 'SCOM')
    assert '60' in line and 'SCOM' in line


def test_unknown_symbol_gives_neutral_line():
    assert 'No scored history' in summary_line({}, 'EQTY')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verdict_scoreboard.py -v` → FAIL (module missing)

- [ ] **Step 3: Implement**

```python
# src/agent/verdict_scoreboard.py
"""Scores past executed decisions against current prices so the LLM
prompt can carry the agent's own track record ('you were right 60% of
the time on SCOM'). Pure functions — journal in, summary out."""
from typing import Any, Callable, Dict, List, Optional


def score_decisions(decisions: List[Dict[str, Any]],
                    price_lookup: Callable[[str], Optional[float]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    prices: Dict[str, Optional[float]] = {}
    for d in decisions:
        if not d.get('executed') or not d.get('price'):
            continue
        action = (d.get('action') or '').lower()
        if action not in ('buy', 'sell'):
            continue
        sym = d.get('symbol')
        if sym not in prices:
            try:
                prices[sym] = price_lookup(sym)
            except Exception:
                prices[sym] = None
        now = prices[sym]
        if not now:
            continue
        rec = out.setdefault(sym, {'evaluated': 0, 'hits': 0, 'hit_rate': 0.0})
        rec['evaluated'] += 1
        if (action == 'buy' and now > d['price']) or (action == 'sell' and now < d['price']):
            rec['hits'] += 1
    for rec in out.values():
        rec['hit_rate'] = round(rec['hits'] / rec['evaluated'], 3) if rec['evaluated'] else 0.0
    return out


def summary_line(scores: Dict[str, Dict[str, Any]], symbol: str) -> str:
    rec = scores.get(symbol)
    if not rec or not rec['evaluated']:
        return f"No scored history for {symbol} yet — treat the ensemble signal on its merits."
    pct = round(rec['hit_rate'] * 100)
    return (f"Track record on {symbol}: {rec['hits']}/{rec['evaluated']} executed calls "
            f"({pct}%) moved in the traded direction. Weigh your confidence accordingly.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_verdict_scoreboard.py -v` → 3 PASSED

- [ ] **Step 5: Feed the line into the LLM prompt**

`validate_trade` in `llm_orchestrator.py` gains a keyword arg `track_record: str = None`; append to `user_prompt` before the final instruction:

```python
        if track_record:
            user_prompt += f"Agent Track Record: {track_record}\n"
```

In `src/agent/main.py`, at the `validate_trade` call site (line ~769), compute once per cycle (not per symbol):

```python
                                    if not hasattr(self, '_verdict_scores_at') or time.time() - self._verdict_scores_at > 600:
                                        from src.agent.verdict_scoreboard import score_decisions
                                        from src.utils.real_price_feed import price_feed
                                        journal = getattr(self, 'decision_journal', None)
                                        self._verdict_scores = score_decisions(
                                            journal.recent(None, limit=500) if journal else [],
                                            price_feed.get_price)
                                        self._verdict_scores_at = time.time()
                                    from src.agent.verdict_scoreboard import summary_line
                                    validated_signal = self.components['llm_orchestrator'].validate_trade(
                                        symbol, symbol_signals, market_data, news_data,
                                        research_context=research_ctx, sector_outlook=sector_ctx,
                                        track_record=summary_line(self._verdict_scores, symbol))
```

(Match the existing call's actual argument names at line 769 — read them before editing and keep every existing argument.)

- [ ] **Step 6: Full suite + commit**

```bash
git add src/agent/verdict_scoreboard.py tests/test_verdict_scoreboard.py src/agent/llm_orchestrator.py src/agent/main.py
git commit -m "feat(intelligence): agent's own hit-rate feeds back into LLM validation prompts"
```

---

### Task 15: Agent-proposed universe expansion (operator-approved)

The agent can already ingest research PDFs → auto-watchlist. Extend discovery: strong NSE movers and news-mentioned symbols the agent does *not* track become **escalations** the operator approves in the existing Research tab UI; approval adds the symbol to the runtime universe and the persistent watchlist. No auto-trading of unapproved names.

**Files:**
- Create: `src/agent/universe_scout.py`
- Create: `tests/test_universe_scout.py`
- Modify: `src/agent/main.py` (call scout once per assessment interval)
- Modify: `src/api/api_server.py` (the existing escalation-approve handler gains the `add_symbol` action; read `handleResolveEscalation`'s backend route first and extend it)

**Interfaces:**
- Produces: `propose_candidates(tracked: set[str], movers: list[dict], news_texts: list[str], threshold_pct: float = 3.0) -> list[dict]` — each `{symbol, reason}`; the escalation record uses `action='add_symbol'`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_universe_scout.py
from src.agent.universe_scout import propose_candidates


def test_untracked_big_mover_is_proposed():
    movers = [{'symbol': 'BAMB', 'change_pct': 4.2}, {'symbol': 'SCOM', 'change_pct': 5.0}]
    out = propose_candidates(tracked={'SCOM'}, movers=movers, news_texts=[])
    assert out == [{'symbol': 'BAMB', 'reason': 'Moved +4.2% today while untracked'}]


def test_small_moves_ignored():
    assert propose_candidates({'SCOM'}, [{'symbol': 'BAMB', 'change_pct': 1.0}], []) == []


def test_news_mention_of_untracked_symbol():
    out = propose_candidates({'SCOM'}, [], ['BAMB Cement announces record dividend'],
                             known_universe={'BAMB': 'BAMB Cement'})
    assert out and out[0]['symbol'] == 'BAMB'
```

- [ ] **Step 2: Run to verify it fails, then implement**

```python
# src/agent/universe_scout.py
"""Proposes new symbols for the trading universe. Proposals become
operator escalations — the agent never trades an unapproved name."""
from typing import Any, Dict, List, Optional, Set


def propose_candidates(tracked: Set[str], movers: List[Dict[str, Any]],
                       news_texts: List[str], threshold_pct: float = 3.0,
                       known_universe: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    seen, out = set(), []
    for m in movers:
        sym = (m.get('symbol') or '').upper()
        pct = m.get('change_pct') or 0
        if sym and sym not in tracked and sym not in seen and abs(pct) >= threshold_pct:
            seen.add(sym)
            out.append({'symbol': sym, 'reason': f"Moved {'+' if pct >= 0 else ''}{pct}% today while untracked"})
    for sym, name in (known_universe or {}).items():
        if sym in tracked or sym in seen:
            continue
        for text in news_texts:
            if sym in text or (name and name.lower() in text.lower()):
                seen.add(sym)
                out.append({'symbol': sym, 'reason': f"In the news: {text[:80]}"})
                break
    return out
```

Run: `python -m pytest tests/test_universe_scout.py -v` → 3 PASSED

- [ ] **Step 3: Wire into the agent loop**

In `src/agent/main.py`, inside the periodic self-assessment block (search for `assessment_interval`), add:

```python
                try:
                    from src.agent.universe_scout import propose_candidates
                    nse = self.components.get('data_manager')
                    em = self.components.get('escalation_manager')
                    if em and nse:
                        tracked = set(self.config.get('data_manager', {}).get('nse_symbols', []))
                        movers = []
                        nse_conn = getattr(nse, 'connectors', {}).get('nse')
                        if nse_conn:
                            quotes = nse_conn.get_all_quotes()
                            movers = [{'symbol': q.get('symbol'), 'change_pct': q.get('change_pct')} for q in quotes]
                        for cand in propose_candidates(tracked, movers, [])[:3]:
                            em.create_escalation(None, cand['symbol'], 'add_symbol', cand['reason'], 'low')
                except Exception as e:
                    logger.warning(f"Universe scout skipped: {e}")
```

(NSE quotes are all currently tracked, so this fires only when the scraper starts returning symbols outside `nse_symbols` — that's intentional future-proofing; the news path activates when `regional_news` texts are passed instead of `[]`. Wire `news_texts` from the same source the sentiment analyst reads.)

- [ ] **Step 4: Handle approval**

Find the escalation-resolve route in `api_server.py` (the one `handleResolveEscalation` posts to). In its approval branch, add:

```python
                if esc_action == 'add_symbol' and resolution == 'approved':
                    sym = escalation_symbol.upper()
                    nse_list = self.trading_agent.config.setdefault('data_manager', {}).setdefault('nse_symbols', [])
                    if sym not in nse_list:
                        nse_list.append(sym)
                    em.add_to_watchlist(sym, market='kenyan', source='universe_scout')
```

(Use the route's actual local variable names — read the handler before editing.)

- [ ] **Step 5: Full suite + commit**

```bash
git add src/agent/universe_scout.py tests/test_universe_scout.py src/agent/main.py src/api/api_server.py
git commit -m "feat(universe): agent proposes new symbols via operator-approved escalations"
```

---

## Phase 6 — TradingView-benchmark quick win

### Task 16: Sortable, filterable NSE Market Watch

TradingView's watchlist lives on sortable columns + advanced view. Cheapest high-value equivalent for our 18-symbol table: client-side sort on every column and an All / Held / Movers filter.

**Files:**
- Modify: `frontend/components/AdvancedDashboard.js` (`renderKenyaNSE`, lines 614–663)

- [ ] **Step 1: Add sort/filter state and derivation**

At the top of the dashboard component (near line 362):

```js
  const [nseSort, setNseSort] = useState({ key: 'symbol', dir: 1 });
  const [nseFilter, setNseFilter] = useState('all'); // all | held | movers
```

Inside `renderKenyaNSE`, after `positionsBySymbol`:

```js
    const sorted = [...(nseData.quotes || [])]
      .filter(q => nseFilter === 'all' || (nseFilter === 'held' ? positionsBySymbol[q.symbol] : Math.abs(q.change_pct || 0) >= 1))
      .sort((a, b) => {
        const va = a[nseSort.key] ?? '', vb = b[nseSort.key] ?? '';
        return (typeof va === 'number' ? va - vb : String(va).localeCompare(String(vb))) * nseSort.dir;
      });
    const sortBtn = (key, label) => (
      <th key={key} onClick={() => setNseSort(s => ({ key, dir: s.key === key ? -s.dir : 1 }))}
          style={{ padding: '12px', cursor: 'pointer', userSelect: 'none' }}>
        {label}{nseSort.key === key ? (nseSort.dir === 1 ? ' ▲' : ' ▼') : ''}
      </th>
    );
```

- [ ] **Step 2: Replace the header row and body source**

Header `<tr>` (lines 634–636) becomes:

```jsx
              <tr style={{ textAlign: 'left', color: theme.colors.textMuted, fontSize: '12px', borderBottom: `1px solid ${theme.colors.border}` }}>
                {sortBtn('symbol', 'SYMBOL')}{sortBtn('price_kes', 'PRICE (KES)')}{sortBtn('change_pct', 'CHANGE')}{sortBtn('volume', 'VOLUME')}<th style={{ padding: '12px' }}>YOUR POSITION</th>
              </tr>
```

Change `nseData.quotes.map(...)` to `sorted.map(...)` (both the `.length` check and the map). Above the table, add the filter pills:

```jsx
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
          {[['all', 'ALL'], ['held', 'HELD'], ['movers', 'MOVERS ±1%']].map(([id, label]) => (
            <button key={id} onClick={() => setNseFilter(id)} style={{
              backgroundColor: nseFilter === id ? theme.colors.primary : 'rgba(255,255,255,0.05)',
              color: nseFilter === id ? '#000' : theme.colors.textSecondary,
              border: 'none', padding: '5px 12px', borderRadius: '6px', fontSize: '11px', fontWeight: 800, cursor: 'pointer'
            }}>{label}</button>
          ))}
        </div>
```

- [ ] **Step 3: Verify in the browser**

Sort by CHANGE descending puts the top gainer first; HELD shows only highlighted rows; row click-through (Task 8) still works.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/AdvancedDashboard.js
git commit -m "feat(nse): sortable columns and All/Held/Movers filter on market watch"
```

---

## Appendix A — TradingView benchmark backlog (not in this plan's tasks)

| TradingView feature | AEGIS equivalent | Priority | Notes |
|---|---|---|---|
| Price alerts (multi-condition) | `/api/alerts` exists but is agent-generated only; add operator-set price alerts per symbol | P1 next plan | Needs an alerts table + a check in the 60s loop + a bell UI |
| Symbol page (full page, not modal) | Promote `SymbolDrilldown` to a routed page with the decision tape, target price, news | P2 | Next.js route `pages/symbol/[sym].js` |
| Mini sparklines in watchlist rows | 7-day close sparkline per NSE row | P2 | Requires a `/api/nse-history/<sym>` endpoint over `nse_daily_prices` |
| Screener | Filter pills (Task 16) are the seed; add RSI/volume filters server-side | P3 | |
| Multi-timeframe charting | Out of scope — link out to TradingView chart for the symbol | P3 | One `<a href>` per drill-down header, zero maintenance |
| News per symbol | `regional_news` exists; join news to drill-down by symbol mention | P2 | |

## Appendix B — deliberately not doing

- **"Training" ML models with idle cash**: idle cash is a *portfolio allocation* problem (fixed by Task 11), not a compute-budget one. The learning loop (Task 14) uses the decision journal, which costs nothing.
- **Auto-trading agent-discovered symbols without approval**: universe expansion always goes through the operator escalation gate (Task 15). On a shared paper portfolio, silent universe drift would make the P&L unattributable.
- **Raising `max_position_size` above 5%**: concentration cap stays; deployment comes from breadth and conviction-scaling, not bigger single bets.
- **NSE holidays calendar**: weekday+time check only for now; holiday closures just mean a quiet scraper day. Add `exchange_calendars` later if stale-price trades appear.
