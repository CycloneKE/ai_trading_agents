# Regime-Routed Signals & Dynamic Universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the ensemble averaging two structurally opposite strategies into indecision — route to exactly one strategy per market regime — and make the agent's universe additions and removals actually reach the trading loop.

**Architecture:** A stateless regime classifier (Kaufman efficiency ratio with a dead band and hysteresis) selects one strategy per symbol per bar. It plugs in as a sixth `ensemble_method` in `StrategyManager`, so the existing methods stay intact and the backtest harness can A/B them by changing one config string. A portfolio net-exposure cap in `ExposureGuard` delivers the low-SPY-correlation objective. A new `universe_manager` proposes additions and removals as operator escalations and fixes the three breaks in the approval path.

**Tech Stack:** Python 3.13, pytest, SQLite (via `src/utils/sqlite_util`), existing `StrategyManager` / `ExposureGuard` / `EscalationManager`. No new third-party dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-02-regime-routed-signals-design.md`
- TDD is mandatory. Write the failing test, **run it and watch it fail**, then implement. A test that passes on first run is testing existing behaviour — fix the test.
- Run tests with `.venv/Scripts/python.exe -m pytest` from the repo root.
- The full suite must stay green: currently **383 passed, 2 skipped**.
- No new deprecation warnings. `datetime.utcnow()` is banned by `tests/test_no_deprecated_datetime.py`; use `src.utils.timeutil.utc_now`.
- Existing ensemble methods (`weighted_average`, `majority_vote`, `confidence_weighted`, `performance_weighted`, `adaptive_confidence`) must behave **exactly** as before. They stay in the codebase, unused by default, so the A/B stays runnable.
- **Exits are never blocked** by any gate, under any circumstance. This applies to every cap added in this plan.
- Universe additions and removals **always** produce an operator escalation. Nothing auto-applies.
- Acceptance gate (Task 6) — out-of-sample must show Sharpe > 0.5, |SPY correlation| < 0.3, max drawdown < 15%, > 100 trades, and better Sharpe than the current ensemble. If it misses, **stop and report**; do not proceed to live config changes.
- Commit after every task.

## File Structure

| File | Responsibility |
|---|---|
| `src/agent/regime.py` (create) | Pure regime classification. No state, no I/O. |
| `tests/test_regime.py` (create) | Classifier unit tests. |
| `src/agent/strategy_manager.py` (modify) | Add `_regime_routed_ensemble`, price buffer, regime state. |
| `tests/test_regime_routing.py` (create) | Routing behaviour through the real StrategyManager. |
| `src/agent/exposure.py` (modify) | Portfolio net-exposure cap. |
| `tests/test_exposure.py` (modify) | Net-exposure tests alongside existing per-symbol tests. |
| `src/agent/main.py` (modify) | Pass net exposure into the gate; read effective universe. |
| `scripts/calibrate_regime.py` (create) | Grid search on the first half of history. |
| `src/agent/universe_manager.py` (create) | Addition/removal proposals with screens. |
| `tests/test_universe_manager.py` (create) | Screen and trigger tests. |
| `src/api/api_server.py` (modify) | Approval routes by market, persists via overlay. |

---

### Task 1: Regime classifier

**Files:**
- Create: `src/agent/regime.py`
- Test: `tests/test_regime.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `efficiency_ratio(closes: list[float], window: int) -> float` — returns 0.0 when history is short or the series is flat.
  - `classify(closes: list[float], previous: Optional[str], cfg: dict) -> str` — returns `'trending'`, `'ranging'` or `'undefined'`.
  - `conviction(closes: list[float], regime: str, cfg: dict) -> float` — returns a float in [0.7, 1.0].
  - Constants `TRENDING = 'trending'`, `RANGING = 'ranging'`, `UNDEFINED = 'undefined'`.
  - `cfg` keys read: `regime.window` (default 20), `regime.trend_threshold` (default 0.35), `regime.range_threshold` (default 0.20).

- [ ] **Step 1: Write the failing test**

Create `tests/test_regime.py`:

```python
"""Regime classification.

The ensemble cancelled a live signal on 62.7% of bars because momentum and
mean_reversion measure the same thing with opposite sign. Routing to one
strategy per regime replaces that averaging, so this classifier decides
which strategy runs.
"""

import pytest

from src.agent.regime import (
    RANGING, TRENDING, UNDEFINED, classify, conviction, efficiency_ratio,
)

CFG = {'regime': {'window': 10, 'trend_threshold': 0.35, 'range_threshold': 0.20}}

STRAIGHT_UP = [100.0 + i for i in range(21)]
ZIGZAG = [100.0 + (2 if i % 2 else 0) for i in range(21)]
FLAT = [100.0] * 21


def test_monotonic_series_has_efficiency_ratio_of_one():
    assert efficiency_ratio(STRAIGHT_UP, 10) == pytest.approx(1.0)


def test_zigzag_series_has_efficiency_ratio_near_zero():
    assert efficiency_ratio(ZIGZAG, 10) < 0.15


def test_flat_series_returns_zero_not_a_division_error():
    assert efficiency_ratio(FLAT, 10) == 0.0


def test_short_history_returns_zero():
    assert efficiency_ratio([100.0, 101.0], 10) == 0.0


def test_efficiency_ratio_is_direction_agnostic():
    down = [100.0 - i for i in range(21)]
    assert efficiency_ratio(down, 10) == pytest.approx(1.0)


def test_strong_trend_classifies_as_trending():
    assert classify(STRAIGHT_UP, None, CFG) == TRENDING


def test_chop_classifies_as_ranging():
    assert classify(ZIGZAG, None, CFG) == RANGING


def test_flat_series_is_undefined_not_ranging():
    """A flat series has no measurable character; trading it on a
    mean-reversion signal would be trading noise."""
    assert classify(FLAT, None, CFG) == UNDEFINED


def test_short_history_is_undefined():
    assert classify([100.0, 101.0], None, CFG) == UNDEFINED


def test_dead_band_holds_the_previous_regime():
    """Hysteresis: without it the router flips at the boundary every other
    bar and pays costs for the privilege."""
    band = [100.0, 101.0, 100.5, 101.5, 101.0, 102.0, 101.5, 102.5, 102.0, 103.0, 102.5]
    er = efficiency_ratio(band, 10)
    assert 0.20 < er < 0.35, f'fixture must sit in the dead band, got {er}'
    assert classify(band, TRENDING, CFG) == TRENDING
    assert classify(band, RANGING, CFG) == RANGING


def test_dead_band_with_no_previous_regime_is_undefined():
    band = [100.0, 101.0, 100.5, 101.5, 101.0, 102.0, 101.5, 102.5, 102.0, 103.0, 102.5]
    assert classify(band, None, CFG) == UNDEFINED


def test_crossing_the_opposite_threshold_does_switch():
    assert classify(ZIGZAG, TRENDING, CFG) == RANGING
    assert classify(STRAIGHT_UP, RANGING, CFG) == TRENDING


def test_conviction_is_full_at_an_extreme_regime():
    assert conviction(STRAIGHT_UP, TRENDING, CFG) == pytest.approx(1.0)


def test_conviction_is_floored_at_a_marginal_regime():
    marginal = [100.0] * 11
    assert conviction(marginal, UNDEFINED, CFG) == pytest.approx(0.7)


def test_conviction_stays_within_bounds():
    for series in (STRAIGHT_UP, ZIGZAG, FLAT):
        for regime in (TRENDING, RANGING, UNDEFINED):
            assert 0.7 <= conviction(series, regime, CFG) <= 1.0


def test_defaults_apply_when_config_omits_the_regime_section():
    assert classify(STRAIGHT_UP, None, {}) == TRENDING
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime.py -q`
Expected: `ModuleNotFoundError: No module named 'src.agent.regime'`

- [ ] **Step 3: Write the implementation**

Create `src/agent/regime.py`:

```python
"""Market regime classification for strategy routing.

Momentum and mean-reversion measure the same thing with opposite sign. A
diagnostic over 13,145 symbol-bars found they contradict each other on 44.3%
of bars and agree on 0.6%, so averaging them cancelled a live signal 62.7% of
the time and pinned surviving confidence at ~0.5.

The fix is to run exactly one of them, chosen by regime. This module decides
which, using Kaufman's efficiency ratio: the net distance travelled divided
by the total distance walked. High means price went somewhere directly
(trending); low means it chopped (ranging).

Chosen over ADX or the Hurst exponent because it needs no library, has one
parameter, and measures the routing criterion directly rather than by proxy.
"""

from typing import Any, Dict, List, Optional

TRENDING = 'trending'
RANGING = 'ranging'
UNDEFINED = 'undefined'

DEFAULT_WINDOW = 20
DEFAULT_TREND_THRESHOLD = 0.35
DEFAULT_RANGE_THRESHOLD = 0.20

CONVICTION_FLOOR = 0.7


def _cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    section = (cfg or {}).get('regime') or {}
    return {
        'window': int(section.get('window', DEFAULT_WINDOW)),
        'trend': float(section.get('trend_threshold', DEFAULT_TREND_THRESHOLD)),
        'range': float(section.get('range_threshold', DEFAULT_RANGE_THRESHOLD)),
    }


def efficiency_ratio(closes: List[float], window: int) -> float:
    """Net move over `window` bars divided by the total path walked.

    Returns 0.0 for insufficient history or a flat series — never raises,
    because this runs inside the per-bar signal path.
    """
    if not closes or window < 1 or len(closes) < window + 1:
        return 0.0
    segment = closes[-(window + 1):]
    net = abs(segment[-1] - segment[0])
    path = sum(abs(segment[i] - segment[i - 1]) for i in range(1, len(segment)))
    if path <= 0:
        return 0.0
    return min(1.0, net / path)


def classify(closes: List[float], previous: Optional[str],
             cfg: Optional[Dict[str, Any]] = None) -> str:
    """Current regime, with a dead band and hysteresis.

    Between the two thresholds the prior regime is held. Without that, a
    series hovering near a boundary would flip the router every other bar and
    pay the round-trip cost each time.
    """
    c = _cfg(cfg)
    if len(closes or []) < c['window'] + 1:
        return UNDEFINED
    er = efficiency_ratio(closes, c['window'])
    if er <= 0.0:
        return UNDEFINED          # flat series has no character to trade
    if er >= c['trend']:
        return TRENDING
    if er <= c['range']:
        return RANGING
    return previous if previous in (TRENDING, RANGING) else UNDEFINED


def conviction(closes: List[float], regime: str,
               cfg: Optional[Dict[str, Any]] = None) -> float:
    """How firmly the series sits inside its regime, in [0.7, 1.0].

    Scales position size so a marginal regime trades smaller than an obvious
    one. Never returns 0 — a routed signal that passed classification should
    still trade, just less.
    """
    c = _cfg(cfg)
    er = efficiency_ratio(closes, c['window'])
    if regime == TRENDING:
        span = max(1e-9, 1.0 - c['trend'])
        depth = (er - c['trend']) / span
    elif regime == RANGING:
        span = max(1e-9, c['range'])
        depth = (c['range'] - er) / span
    else:
        depth = 0.0
    depth = max(0.0, min(1.0, depth))
    return CONVICTION_FLOOR + (1.0 - CONVICTION_FLOOR) * depth
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime.py -q`
Expected: `16 passed`

If `test_dead_band_holds_the_previous_regime` fails on its fixture assertion, adjust the `band` fixture values until `efficiency_ratio(band, 10)` lands strictly between 0.20 and 0.35 — the assertion exists to catch exactly that.

- [ ] **Step 5: Commit**

```bash
git add src/agent/regime.py tests/test_regime.py
git commit -m "feat(regime): efficiency-ratio classifier with dead band and hysteresis"
```

---

### Task 2: Regime-routed ensemble method

**Files:**
- Modify: `src/agent/strategy_manager.py`
- Test: `tests/test_regime_routing.py`

**Interfaces:**
- Consumes: `classify`, `conviction`, `TRENDING`, `RANGING`, `UNDEFINED` from Task 1.
- Produces: `StrategyManager._regime_routed_ensemble(strategy_signals: Dict[str, Dict], symbol: str, data: Dict) -> Dict`, reachable by setting `ensemble_method: "regime_routed"`. Returned dict includes keys `symbol`, `action`, `confidence`, `position_size`, `regime`, `routed_strategy`, `ensemble_method`, `timestamp`.
- `StrategyManager` gains `self._regime_history: Dict[str, List[float]]` and `self._regime_state: Dict[str, str]`.

Routing table, from the spec:

| Regime | Strategy executed | RSI |
|---|---|---|
| `trending` | momentum only | ignored |
| `ranging` | mean_reversion only | must agree, else hold |
| `undefined` | none | — |

- [ ] **Step 1: Write the failing test**

Create `tests/test_regime_routing.py`:

```python
"""Routing replaces averaging.

The regression this exists to fix: `_weighted_average_ensemble` divides the
winning action's score by total weight, so a 2-vs-1 disagreement lands at
~0.5 by construction. Since position_size = confidence * max_position_size,
every surviving trade was sized at ~2.6% regardless of conviction.
"""

import pytest

from src.agent.regime import RANGING, TRENDING, UNDEFINED
from src.agent.strategy_manager import StrategyManager

CONFIG = {
    'ensemble_method': 'regime_routed',
    'regime': {'window': 10, 'trend_threshold': 0.35, 'range_threshold': 0.20},
    'strategies': {
        'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0,
                     'lookback_period': 20, 'max_position_size': 0.05},
        'mean_reversion': {'type': 'technical', 'enabled': True, 'weight': 1.0,
                           'lookback_period': 20, 'max_position_size': 0.05},
        'rsi_strategy': {'type': 'technical', 'enabled': True, 'weight': 0.8,
                         'lookback_period': 20, 'max_position_size': 0.05},
    },
}


def feed(mgr, symbol, closes):
    """Push a price series through generate_signals, returning the last result."""
    out = {}
    for price in closes:
        out = mgr.generate_signals({'symbol': symbol, 'price': price, 'close': price})
    return out


def signals(momentum, mean_reversion, rsi):
    return {
        'momentum': {'action': momentum, 'confidence': 0.8, 'position_size': 0.04},
        'mean_reversion': {'action': mean_reversion, 'confidence': 0.9, 'position_size': 0.045},
        'rsi_strategy': {'action': rsi, 'confidence': 0.7, 'position_size': 0.035},
    }


def test_trending_regime_routes_to_momentum():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert out['regime'] == TRENDING
    assert out['routed_strategy'] == 'momentum'
    assert out['action'] == 'buy'


def test_ranging_regime_routes_to_mean_reversion():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + (2 if i % 2 else 0) for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert out['regime'] == RANGING
    assert out['routed_strategy'] == 'mean_reversion'
    assert out['action'] == 'sell'


def test_ranging_regime_requires_rsi_agreement():
    """RSI holds 88.8% of the time, so as a voter it is inert. As a
    confirmation gate in the one regime where it is meaningful, it earns
    its place."""
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + (2 if i % 2 else 0) for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'buy'), 'X', {'symbol': 'X'})
    assert out['action'] == 'hold'


def test_undefined_regime_produces_hold():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0] * 21
    out = mgr._regime_routed_ensemble(signals('buy', 'buy', 'buy'), 'X', {'symbol': 'X'})
    assert out['regime'] == UNDEFINED
    assert out['action'] == 'hold'


def test_confidence_is_no_longer_pinned_near_half():
    """THE regression. Under weighted-average this returned ~0.5 whenever the
    strategies disagreed; the diagnostic measured a median of 0.517 with a
    maximum of 0.629 across 306 live signals."""
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    out = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert out['confidence'] > 0.7


def test_position_size_scales_with_confidence():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    strong = mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    weak_signals = signals('buy', 'sell', 'sell')
    weak_signals['momentum'] = {'action': 'buy', 'confidence': 0.5, 'position_size': 0.01}
    weak = mgr._regime_routed_ensemble(weak_signals, 'X', {'symbol': 'X'})
    assert strong['position_size'] > weak['position_size']


def test_a_missing_routed_strategy_degrades_to_hold():
    """A config without a momentum strategy must not raise in the signal path."""
    cfg = {k: v for k, v in CONFIG.items()}
    cfg['strategies'] = {'mean_reversion': CONFIG['strategies']['mean_reversion']}
    mgr = StrategyManager(cfg)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    out = mgr._regime_routed_ensemble({'mean_reversion': {'action': 'sell', 'confidence': 0.9}},
                                      'X', {'symbol': 'X'})
    assert out['action'] == 'hold'


def test_generate_signals_builds_its_own_price_history():
    mgr = StrategyManager(CONFIG)
    feed(mgr, 'X', [100.0 + i for i in range(30)])
    assert len(mgr._regime_history['X']) >= 21


def test_price_history_is_bounded():
    """An unbounded buffer leaks in a process that runs for weeks."""
    mgr = StrategyManager(CONFIG)
    feed(mgr, 'X', [100.0 + i for i in range(1200)])
    assert len(mgr._regime_history['X']) <= 500


def test_regime_state_persists_between_calls_for_hysteresis():
    mgr = StrategyManager(CONFIG)
    mgr._regime_history['X'] = [100.0 + i for i in range(21)]
    mgr._regime_routed_ensemble(signals('buy', 'sell', 'sell'), 'X', {'symbol': 'X'})
    assert mgr._regime_state['X'] == TRENDING


def test_other_ensemble_methods_are_unchanged():
    """The A/B depends on the old path behaving exactly as before."""
    cfg = dict(CONFIG, ensemble_method='weighted_average')
    mgr = StrategyManager(cfg)
    out = mgr._combine_signals(signals('buy', 'sell', 'sell'), {'symbol': 'X'})
    assert out['ensemble_method'] == 'weighted_average'
    assert 'regime' not in out


def test_warm_start_seeds_the_regime_buffer():
    mgr = StrategyManager(CONFIG)
    mgr.warm_start({'X': [100.0 + i for i in range(60)]})
    assert len(mgr._regime_history.get('X', [])) >= 21
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_routing.py -q`
Expected: FAIL with `AttributeError: 'StrategyManager' object has no attribute '_regime_history'`

- [ ] **Step 3: Write the implementation**

In `src/agent/strategy_manager.py`, add the import next to the existing strategy imports (after the `from src.utils.timeutil import utc_now` line):

```python
from src.agent.regime import RANGING, TRENDING, UNDEFINED, classify, conviction
```

In `StrategyManager.__init__`, alongside `self.strategy_weights = {}`, add:

```python
        # Price buffer for regime classification. Kept here rather than read
        # out of a strategy's internals so the manager does not depend on any
        # particular strategy's private state.
        self._regime_history: Dict[str, List[float]] = {}
        self._regime_state: Dict[str, str] = {}
        self._regime_history_max = 500
```

In `generate_signals`, immediately after the `if not self.strategies:` guard block and before `strategy_signals = self.collect_strategy_signals(data)`, add:

```python
            symbol_key = data.get('symbol', 'UNKNOWN')
            price = data.get('price') or data.get('close')
            if price:
                buf = self._regime_history.setdefault(symbol_key, [])
                buf.append(float(price))
                if len(buf) > self._regime_history_max:
                    del buf[:-self._regime_history_max]
```

In `_combine_signals`, add the branch alongside the existing method checks, immediately before the `else:` that logs an unknown method:

```python
            elif self.ensemble_method == 'regime_routed':
                return self._regime_routed_ensemble(strategy_signals, symbol, data)
```

Add the method itself, next to `_adaptive_confidence_ensemble`:

```python
    def _regime_routed_ensemble(self, strategy_signals: Dict[str, Dict[str, Any]],
                                symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Route to exactly ONE strategy based on the current regime.

        Nothing is averaged, so nothing cancels. Momentum and mean-reversion
        measure the same thing with opposite sign — blending them produced a
        hold on 62.7% of bars and pinned confidence at ~0.5, which capped
        every position at ~2.6% no matter how strong the signal.
        """
        closes = self._regime_history.get(symbol, [])
        previous = self._regime_state.get(symbol)
        regime = classify(closes, previous, self.config)
        self._regime_state[symbol] = regime

        def held(reason: str) -> Dict[str, Any]:
            return {
                'symbol': symbol, 'action': 'hold', 'confidence': 0.0,
                'position_size': 0.0, 'regime': regime, 'routed_strategy': None,
                'skip_reason': reason, 'ensemble_method': 'regime_routed',
                'timestamp': utc_now().isoformat(),
            }

        if regime == UNDEFINED:
            return held('regime_undefined')

        routed_name = 'momentum' if regime == TRENDING else 'mean_reversion'
        routed = strategy_signals.get(routed_name)
        if not routed:
            logger.warning(f"regime_routed: no '{routed_name}' strategy for {symbol}")
            return held('routed_strategy_missing')

        action = str(routed.get('action', 'hold')).lower()
        if action not in ('buy', 'sell'):
            return held('routed_strategy_held')

        # RSI confirms mean-reversion entries only. It holds 88.8% of bars, so
        # as a voter it is inert; as a gate in the regime where it is
        # meaningful it filters rather than dilutes.
        if regime == RANGING:
            rsi = strategy_signals.get('rsi_strategy') or {}
            if str(rsi.get('action', 'hold')).lower() != action:
                return held('rsi_disagreed')

        factor = conviction(closes, regime, self.config)
        confidence = max(0.0, min(1.0, float(routed.get('confidence', 0.0)) * factor))
        position_size = max(0.0, min(1.0, float(routed.get('position_size', 0.0)) * factor))

        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'regime': regime,
            'routed_strategy': routed_name,
            'regime_conviction': factor,
            'strategy_signals': strategy_signals,
            'ensemble_method': 'regime_routed',
            'timestamp': utc_now().isoformat(),
        }
```

In `warm_start`, after the existing per-strategy seeding loop and before `return seeded`, add:

```python
        # Seed the regime buffer too, or the router reports 'undefined' for
        # `window` live cycles after every restart.
        for sym, bars in (bars_by_symbol or {}).items():
            closes = [float(b) for b in bars if b]
            if closes:
                self._regime_history[sym] = closes[-self._regime_history_max:]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regime_routing.py -q`
Expected: `12 passed`

Then the full suite: `.venv/Scripts/python.exe -m pytest -q`
Expected: `411 passed, 2 skipped` (383 baseline + 16 from Task 1 + 12 here)

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategy_manager.py tests/test_regime_routing.py
git commit -m "feat(signals): regime_routed ensemble routes to one strategy per regime"
```

---

### Task 3: Portfolio net-exposure cap

**Files:**
- Modify: `src/agent/exposure.py`
- Test: `tests/test_exposure.py` (append)

**Interfaces:**
- Consumes: existing `ExposureGuard.permitted_notional`.
- Produces: `permitted_notional(..., net_notional: Optional[float] = None)` — a new trailing keyword argument. `net_notional` is the **signed** sum of all position values (long positive, short negative). Omitting it skips the check, preserving every existing caller.
- `ExposureGuard.max_net_exposure` read from `risk_limits.max_net_exposure`, default `0.30`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_exposure.py`:

```python
# ---------------------------------------------------------------------------
# Portfolio net exposure
#
# The objective is absolute return uncorrelated to the market. A per-symbol
# cap does not deliver that: eleven separate 5% longs is a 55% long book that
# tracks SPY. This bounds the directional bet across the whole portfolio.
# ---------------------------------------------------------------------------

def test_add_within_the_net_cap_is_allowed(guard):
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=10_000.0)
    assert got == 2_000.0


def test_add_beyond_the_net_cap_is_trimmed(guard):
    """Net long $29k of a $30k cap leaves $1k of headroom."""
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 4_000.0,
                                   net_notional=29_000.0)
    assert got == pytest.approx(1_000.0)


def test_add_at_the_net_cap_is_refused(guard):
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=30_000.0)
    assert got == 0.0


def test_short_side_net_cap_is_symmetric(guard):
    got = guard.permitted_notional('TSLA', 'sell', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=-30_000.0)
    assert got == 0.0


def test_an_order_that_reduces_net_exposure_always_passes(guard):
    """Net long $50k (over cap) and selling: this brings the book back toward
    neutral and must never be blocked."""
    got = guard.permitted_notional('TSLA', 'sell', 200.0, PRICE, EQUITY, 5_000.0,
                                   net_notional=50_000.0)
    assert got == 5_000.0


def test_exits_are_never_blocked_by_the_net_cap(guard):
    held = 40_000.0 / PRICE
    got = guard.permitted_notional('TSLA', 'sell', held, PRICE, EQUITY, 40_000.0,
                                   net_notional=60_000.0)
    assert got == 40_000.0


def test_omitting_net_notional_skips_the_check(guard):
    """Back-compat: every existing caller passes six arguments."""
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0)
    assert got == 2_000.0


def test_the_tighter_of_the_two_caps_wins(guard):
    """Per-symbol headroom is $5,000; net headroom is $500."""
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 4_000.0,
                                   net_notional=29_500.0)
    assert got == pytest.approx(500.0)


def test_net_dust_headroom_is_refused(guard):
    got = guard.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                   net_notional=29_999.0)
    assert got == 0.0


def test_net_cap_is_configurable():
    g, path = make_guard(risk_limits={'max_net_exposure': 0.10})
    try:
        assert g.permitted_notional('TSLA', 'buy', 0.0, PRICE, EQUITY, 2_000.0,
                                    net_notional=10_000.0) == 0.0
    finally:
        g.close()
        os.unlink(path)


def test_net_cap_default_is_thirty_percent(guard):
    assert guard.max_net_exposure == pytest.approx(0.30)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_exposure.py -q -k net`
Expected: FAIL with `TypeError: permitted_notional() got an unexpected keyword argument 'net_notional'`

- [ ] **Step 3: Write the implementation**

In `src/agent/exposure.py`, add the constant next to `DEFAULT_MIN_NOTIONAL`:

```python
DEFAULT_MAX_NET_EXPOSURE = 0.30     # |long - short| / equity
```

In `ExposureGuard.__init__`, after `self.cooldown_seconds`:

```python
        # Bounds the directional bet across the whole book. The per-symbol cap
        # alone does not: eleven separate 5% longs is a 55% long book that
        # simply tracks the index.
        self.max_net_exposure = float(
            limits.get('max_net_exposure', DEFAULT_MAX_NET_EXPOSURE))
```

Change the signature of `permitted_notional`:

```python
    def permitted_notional(self, symbol: str, side: str,
                           existing_qty: Optional[float],
                           price: Optional[float],
                           equity: Optional[float],
                           target_value: Optional[float],
                           net_notional: Optional[float] = None) -> float:
```

Then, immediately before the final `return allowed` (after the existing
`if allowed < target_value:` logging block), insert:

```python
        allowed = self._apply_net_cap(symbol, side, allowed, equity, net_notional)
        return allowed
```

and delete the old bare `return allowed`. Add the helper below `_cap`:

```python
    def _apply_net_cap(self, symbol: str, side: str, allowed: float,
                       equity: Optional[float],
                       net_notional: Optional[float]) -> float:
        """Trim an ADD so the book's net directional exposure stays bounded.

        `net_notional` is signed: long positive, short negative. Passing None
        skips the check entirely, so existing six-argument callers are
        unaffected. An order that moves the book toward neutral always
        passes — capping a de-risking trade would be perverse.
        """
        if net_notional is None or allowed <= 0:
            return allowed
        cap = self._cap_net(equity)
        if cap is None:
            return allowed

        signed = allowed if side == 'buy' else -allowed
        prospective = net_notional + signed
        if abs(prospective) <= abs(net_notional) or abs(prospective) <= cap:
            return allowed

        # Trim to whatever keeps |net| exactly at the cap.
        headroom = cap - abs(net_notional)
        if headroom < self.min_notional:
            logger.warning(
                f"Exposure gate: refusing {side} {symbol} — net exposure "
                f"${abs(net_notional):,.0f} against a ${cap:,.0f} cap")
            return 0.0
        logger.info(
            f"Exposure gate: trimming {side} {symbol} to ${headroom:,.0f} "
            f"on net-exposure cap")
        return headroom

    def _cap_net(self, equity: Optional[float]) -> Optional[float]:
        if not equity or equity <= 0:
            return None
        return float(equity) * self.max_net_exposure
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_exposure.py -q`
Expected: `38 passed` (27 existing + 11 new)

- [ ] **Step 5: Commit**

```bash
git add src/agent/exposure.py tests/test_exposure.py
git commit -m "feat(risk): portfolio net-exposure cap for market-neutral objective"
```

---

### Task 4: Wire net exposure into the trading loop

**Files:**
- Modify: `src/agent/main.py` (the `_execute_trades` exposure-gate block added in commit `5f2e74d`)
- Test: `tests/test_execute_trades_exposure.py` (append)

**Interfaces:**
- Consumes: `permitted_notional(..., net_notional=...)` from Task 3.
- Produces: no new public interface.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_execute_trades_exposure.py`:

```python
def test_net_exposure_is_passed_to_the_gate(rig):
    """The cap is worthless if the loop never tells the gate what the book
    currently looks like."""
    agent, broker = rig
    seen = {}
    real = agent.exposure_guard.permitted_notional

    def spy(*args, **kwargs):
        seen['net'] = kwargs.get('net_notional')
        return real(*args, **kwargs)

    agent.exposure_guard.permitted_notional = spy
    run_cycles(agent, buy_signal(), 1)
    assert seen.get('net') is not None


def test_net_exposure_reflects_existing_positions(rig):
    agent, broker = rig
    broker.qty = 100.0                      # 100 sh * $310 = $31,000 long
    seen = {}
    real = agent.exposure_guard.permitted_notional

    def spy(*args, **kwargs):
        seen['net'] = kwargs.get('net_notional')
        return real(*args, **kwargs)

    agent.exposure_guard.permitted_notional = spy
    run_cycles(agent, buy_signal(), 1)
    assert seen['net'] == pytest.approx(31_000.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_execute_trades_exposure.py -q -k net`
Expected: FAIL — `assert None is not None`

- [ ] **Step 3: Write the implementation**

In `src/agent/main.py`, in `_execute_trades`, replace the `permitted = self.exposure_guard.permitted_notional(...)` call with:

```python
                        # Signed net across the whole book, so the gate can
                        # bound the directional bet, not just per-symbol size.
                        net_notional = 0.0
                        try:
                            for p in (broker.get_positions() or []):
                                px = getattr(p, 'current_price', None) or price
                                net_notional += float(p.quantity) * float(px)
                        except Exception as e:
                            logger.debug(f"Net exposure lookup failed: {e}")

                        permitted = self.exposure_guard.permitted_notional(
                            symbol, action, existing_qty, price,
                            portfolio_value, target_pos_value,
                            net_notional=net_notional)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_execute_trades_exposure.py -q`
Expected: `9 passed` (7 existing + 2 new)

Then: `.venv/Scripts/python.exe -m pytest -q`
Expected: `424 passed, 2 skipped`

- [ ] **Step 5: Commit**

```bash
git add src/agent/main.py tests/test_execute_trades_exposure.py
git commit -m "feat(risk): feed portfolio net exposure into the entry gate"
```

---

### Task 5: Calibrate regime thresholds on the first half

**Files:**
- Create: `scripts/calibrate_regime.py`

**Interfaces:**
- Consumes: `run_backtest`, `sharpe_ratio` from `src/agent/strategy_backtest.py`; `regime_routed` from Task 2; the net cap from Task 3.
- Produces: `data/regime_calibration.json` containing the winning `{window, trend_threshold, range_threshold}` and the full grid for inspection.

**This task must not look at the second half of history.** The out-of-sample window is spent once, in Task 6.

- [ ] **Step 1: Write the calibration script**

Create `scripts/calibrate_regime.py`:

```python
"""Grid-search regime parameters on the FIRST HALF of history only.

The second half is the out-of-sample window and is evaluated exactly once, in
scripts/run_strategy_backtest.py, after these parameters are frozen. Looking
at it here would contaminate the only honest number this project has.

The grid is deliberately coarse: a fine search over four parameters is easy
to overfit, and the difference between 0.35 and 0.36 is noise at this sample
size.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.strategy_backtest import run_backtest  # noqa: E402

HISTORY = 'data/backtest_history.json'
CONFIG = 'config/config.json'
OUT = 'data/regime_calibration.json'

WINDOWS = [10, 20, 30, 40]
TREND = [0.30, 0.35, 0.40, 0.45]
RANGE = [0.10, 0.15, 0.20, 0.25]


def main():
    history = json.load(open(HISTORY, encoding='utf-8'))
    config = json.load(open(CONFIG, encoding='utf-8'))
    history = {s: c for s, c in history.items() if c and len(c) > 60}

    n = min(len(c) for c in history.values())
    first_half = {s: c[: n // 2] for s, c in history.items()}
    print(f'calibrating on bars 0..{n // 2} of {n} — second half NOT touched')

    results = []
    for window in WINDOWS:
        for trend in TREND:
            for rng in RANGE:
                if rng >= trend:
                    continue          # dead band must be non-empty
                cfg = json.loads(json.dumps(config))
                cfg['ensemble_method'] = 'regime_routed'
                cfg['regime'] = {'window': window, 'trend_threshold': trend,
                                 'range_threshold': rng}
                res = run_backtest(first_half, cfg, initial_capital=100_000.0)
                row = {'window': window, 'trend_threshold': trend,
                       'range_threshold': rng, 'sharpe': round(res.sharpe, 3),
                       'total_return': round(res.total_return, 4),
                       'trades': len(res.trades),
                       'max_drawdown': round(res.max_drawdown, 4)}
                results.append(row)
                print(f"  w={window:2d} t={trend:.2f} r={rng:.2f} -> "
                      f"sharpe {row['sharpe']:6.2f}  trades {row['trades']:4d}")

    # Require enough trades that the Sharpe means something. A 3-trade
    # parameter set with a great ratio is noise, not a strategy.
    viable = [r for r in results if r['trades'] >= 50]
    if not viable:
        print('\nNo parameter set produced 50+ trades on the first half.')
        print('Report this rather than lowering the bar.')
        json.dump({'grid': results, 'best': None}, open(OUT, 'w'), indent=2)
        return 1

    best = max(viable, key=lambda r: r['sharpe'])
    print(f"\nBEST (in-sample): window={best['window']} "
          f"trend={best['trend_threshold']} range={best['range_threshold']} "
          f"sharpe={best['sharpe']} trades={best['trades']}")
    json.dump({'grid': results, 'best': best}, open(OUT, 'w'), indent=2)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Run the calibration**

Run: `.venv/Scripts/python.exe scripts/calibrate_regime.py`
Expected: a grid of ~48 rows, then a `BEST (in-sample)` line and `data/regime_calibration.json`.

If every row shows fewer than 50 trades, **stop and report**. That means the router is even more selective than the current ensemble, and the design needs revisiting before the out-of-sample window is spent.

- [ ] **Step 3: Write the winning parameters into config**

Add to `config/config.json` at the top level, using the values from `BEST`:

```json
  "regime": {
    "_comment": "Fitted on the first half of data/backtest_history.json by scripts/calibrate_regime.py. See docs/superpowers/specs/2026-08-02-regime-routed-signals-design.md section 7.",
    "window": 20,
    "trend_threshold": 0.35,
    "range_threshold": 0.20
  },
```

Replace the three numbers with the actual `BEST` values. Do **not** change `ensemble_method` yet — that happens only if Task 6 passes.

- [ ] **Step 4: Verify config still loads**

Run:
```bash
.venv/Scripts/python.exe -c "import json; from src.utils.config_validator import validate_config; print(validate_config(json.load(open('config/config.json'))) or 'ok')"
```
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_regime.py config/config.json data/regime_calibration.json
git commit -m "feat(regime): calibrate thresholds on the in-sample half"
```

---

### Task 6: Out-of-sample validation and the acceptance gate

**Files:**
- Modify: `scripts/run_strategy_backtest.py`

**Interfaces:**
- Consumes: frozen parameters from Task 5.
- Produces: an A/B comparison in `data/strategy_backtest_results.json` with a `correlation_to_spy` field and an explicit pass/fail verdict.

- [ ] **Step 1: Add correlation and the A/B to the runner**

In `scripts/run_strategy_backtest.py`, add above `summarise`:

```python
def correlation(a, b):
    """Pearson correlation of two equal-length return series."""
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def daily_returns(values):
    return [(values[i] - values[i - 1]) / values[i - 1]
            for i in range(1, len(values)) if values[i - 1]]


ACCEPTANCE = {
    'min_sharpe': 0.5, 'max_abs_correlation': 0.3,
    'max_drawdown': 0.15, 'min_trades': 100,
}


def verdict(summary, baseline_sharpe):
    """Every criterion from the spec, reported individually."""
    checks = {
        'sharpe > 0.5': summary['sharpe'] > ACCEPTANCE['min_sharpe'],
        '|SPY correlation| < 0.3':
            abs(summary['correlation_to_spy']) < ACCEPTANCE['max_abs_correlation'],
        'max drawdown < 15%': summary['max_drawdown'] < ACCEPTANCE['max_drawdown'],
        'trades > 100': summary['trades'] > ACCEPTANCE['min_trades'],
        'beats baseline sharpe': summary['sharpe'] > baseline_sharpe,
    }
    return checks, all(checks.values())
```

In `summarise`, add the correlation field by extending the signature to
`summarise(name, res, benchmark, spy_returns)` and adding to the returned dict:

```python
        'correlation_to_spy': round(
            correlation(daily_returns([p['equity'] for p in res.equity_curve]),
                        spy_returns), 3),
```

Update the three `summarise(...)` call sites to pass
`daily_returns(spy)` and `daily_returns(spy[sl])` respectively.

- [ ] **Step 2: Add the A/B and gate to `main()`**

Replace the body of `main()` after the walk-forward loop with:

```python
    for s in results:
        show(s)

    # A/B on the out-of-sample half, with parameters frozen by Task 5.
    n = min(len(c) for c in history.values())
    oos = {s: c[n // 2:] for s, c in history.items()}
    oos_spy = daily_returns(spy[n // 2:])

    baseline_cfg = json.loads(json.dumps(config))
    baseline_cfg['ensemble_method'] = 'adaptive_confidence'
    baseline = summarise('OOS baseline (current ensemble)',
                         run_backtest(oos, baseline_cfg, initial_capital=CAPITAL),
                         buy_and_hold_return(spy[n // 2:]), oos_spy)

    routed_cfg = json.loads(json.dumps(config))
    routed_cfg['ensemble_method'] = 'regime_routed'
    routed = summarise('OOS regime_routed',
                       run_backtest(oos, routed_cfg, initial_capital=CAPITAL),
                       buy_and_hold_return(spy[n // 2:]), oos_spy)

    show(baseline)
    show(routed)

    checks, passed = verdict(routed, baseline['sharpe'])
    print('\n--- ACCEPTANCE GATE (out-of-sample) ---')
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  VERDICT: {'PASS — cleared for live config' if passed else 'FAIL — do not ship'}")

    results.extend([baseline, routed])
    json.dump({'results': results, 'acceptance': checks, 'passed': passed},
              open(OUT, 'w'), indent=2)
    print(f"\nwrote {OUT}")
    return 0 if passed else 1
```

- [ ] **Step 3: Run the validation**

Run: `.venv/Scripts/python.exe scripts/run_strategy_backtest.py`
Expected: the gate block, then `VERDICT: PASS` or `VERDICT: FAIL`.

- [ ] **Step 4: Act on the verdict**

**If PASS** — set `"ensemble_method": "regime_routed"` in `config/config.json` and continue to Task 7.

**If FAIL** — do **not** change `ensemble_method`. Stop, and report which criteria failed and by how much. The out-of-sample window is now spent; re-tuning against it would contaminate the result. Per spec section 7, the honest next step is fetching a genuinely fresh window, not adjusting thresholds until the number looks better.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_strategy_backtest.py config/config.json data/strategy_backtest_results.json
git commit -m "test(regime): out-of-sample A/B against the acceptance gate"
```

---

### Task 7: Universe manager — screened additions

**Files:**
- Create: `src/agent/universe_manager.py`
- Test: `tests/test_universe_manager.py`

**Interfaces:**
- Consumes: `src/agent/universe_scout.propose_candidates`.
- Produces:
  - `correlation(a: list[float], b: list[float]) -> float`
  - `propose_additions(tracked: set, candidates: list[dict], history: dict[str, list[float]], cfg: dict) -> list[dict]` — each returned dict has keys `symbol`, `reason`, `market`.
  - `cfg` keys read: `universe.min_avg_dollar_volume` (default 1_000_000), `universe.min_price` (default 5.0), `universe.max_correlation` (default 0.9).
  - Candidate dicts carry `symbol`, `reason`, optional `market` (default `'us'`), `avg_dollar_volume`, `price`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_universe_manager.py`:

```python
"""Universe additions must be screened.

Today `universe_scout` proposes anything that moved 3%. The traded universe
holds SPY, QQQ and DIA — three near-identical instruments — which leaves
roughly seven independent names out of eleven. For an uncorrelated-return
objective that is the wrong shape, so a candidate that duplicates existing
exposure is rejected rather than added.
"""

import pytest

from src.agent.universe_manager import correlation, propose_additions

CFG = {'universe': {'min_avg_dollar_volume': 1_000_000, 'min_price': 5.0,
                    'max_correlation': 0.9}}

UP = [100.0 + i for i in range(60)]
ALSO_UP = [50.0 + i * 0.5 for i in range(60)]        # perfectly correlated
DOWN = [100.0 - i for i in range(60)]                # perfectly anti-correlated
NOISE = [100.0 + (7 * i % 11) for i in range(60)]


def candidate(symbol, **over):
    base = {'symbol': symbol, 'reason': 'moved 4% today', 'market': 'us',
            'avg_dollar_volume': 5_000_000, 'price': 50.0}
    base.update(over)
    return base


def test_a_clean_candidate_is_proposed():
    out = propose_additions({'AAPL'}, [candidate('NFLX')],
                            {'AAPL': UP, 'NFLX': NOISE}, CFG)
    assert [c['symbol'] for c in out] == ['NFLX']


def test_an_already_tracked_symbol_is_skipped():
    out = propose_additions({'NFLX'}, [candidate('NFLX')],
                            {'NFLX': NOISE}, CFG)
    assert out == []


def test_an_illiquid_candidate_is_rejected():
    """You cannot exit what does not trade."""
    out = propose_additions({'AAPL'}, [candidate('THIN', avg_dollar_volume=10_000)],
                            {'AAPL': UP, 'THIN': NOISE}, CFG)
    assert out == []


def test_a_sub_minimum_price_candidate_is_rejected():
    out = propose_additions({'AAPL'}, [candidate('PENNY', price=1.2)],
                            {'AAPL': UP, 'PENNY': NOISE}, CFG)
    assert out == []


def test_a_correlated_candidate_is_rejected():
    """The SPY/QQQ/DIA problem: adding a fourth index tracker buys no
    diversification but consumes a universe slot."""
    out = propose_additions({'SPY'}, [candidate('QQQ')],
                            {'SPY': UP, 'QQQ': ALSO_UP}, CFG)
    assert out == []


def test_an_anticorrelated_candidate_is_also_rejected():
    """|rho| > 0.9 either way is redundant exposure — a perfect inverse is
    the same bet with the sign flipped."""
    out = propose_additions({'SPY'}, [candidate('SH')],
                            {'SPY': UP, 'SH': DOWN}, CFG)
    assert out == []


def test_a_candidate_with_no_history_is_rejected():
    """Unscreenable is not the same as clean."""
    out = propose_additions({'AAPL'}, [candidate('NEW')], {'AAPL': UP}, CFG)
    assert out == []


def test_the_rejection_reason_is_not_silently_swallowed(caplog):
    import logging
    with caplog.at_level(logging.INFO):
        propose_additions({'SPY'}, [candidate('QQQ')], {'SPY': UP, 'QQQ': ALSO_UP}, CFG)
    assert 'QQQ' in caplog.text


def test_market_is_carried_through():
    out = propose_additions(set(), [candidate('SCOM', market='kenyan')],
                            {'SCOM': NOISE}, CFG)
    assert out[0]['market'] == 'kenyan'


def test_correlation_of_identical_series_is_one():
    assert correlation(UP, UP) == pytest.approx(1.0)


def test_correlation_of_inverse_series_is_minus_one():
    assert correlation(UP, DOWN) == pytest.approx(-1.0)


def test_correlation_of_a_flat_series_is_zero_not_nan():
    assert correlation(UP, [100.0] * 60) == 0.0


def test_correlation_of_short_series_is_zero():
    assert correlation([1.0, 2.0], [1.0, 2.0]) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_universe_manager.py -q`
Expected: `ModuleNotFoundError: No module named 'src.agent.universe_manager'`

- [ ] **Step 3: Write the implementation**

Create `src/agent/universe_manager.py`:

```python
"""What should the agent be trading?

`universe_scout` proposes anything that moved 3% today, with no screening for
whether the name is tradeable or whether it duplicates exposure the book
already has. With SPY, QQQ and DIA all tracked, eleven symbols carry roughly
seven independent bets — the wrong shape for an uncorrelated-return mandate.

Everything here produces PROPOSALS. Nothing auto-applies; every change goes
to the operator as an escalation.
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_MIN_DOLLAR_VOLUME = 1_000_000.0
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MAX_CORRELATION = 0.9

MIN_CORRELATION_BARS = 30


def _returns(closes: List[float]) -> List[float]:
    return [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1]]


def correlation(a: List[float], b: List[float]) -> float:
    """Pearson correlation of two price series' daily returns.

    Returns 0.0 for short or zero-variance input rather than nan — a nan
    would silently compare False against every threshold and wave the
    candidate through.
    """
    ra, rb = _returns(a or []), _returns(b or [])
    n = min(len(ra), len(rb))
    if n < MIN_CORRELATION_BARS:
        return 0.0
    ra, rb = ra[-n:], rb[-n:]
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / (va * vb)


def propose_additions(tracked: Set[str], candidates: List[Dict[str, Any]],
                      history: Dict[str, List[float]],
                      cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Screen raw scout candidates down to ones worth an operator's attention."""
    section = (cfg or {}).get('universe') or {}
    min_volume = float(section.get('min_avg_dollar_volume', DEFAULT_MIN_DOLLAR_VOLUME))
    min_price = float(section.get('min_price', DEFAULT_MIN_PRICE))
    max_corr = float(section.get('max_correlation', DEFAULT_MAX_CORRELATION))

    out = []
    for cand in candidates or []:
        symbol = (cand.get('symbol') or '').upper()
        if not symbol or symbol in tracked:
            continue

        if float(cand.get('avg_dollar_volume', 0) or 0) < min_volume:
            logger.info(f"Universe: rejecting {symbol} — illiquid "
                        f"(${cand.get('avg_dollar_volume', 0):,.0f}/day)")
            continue

        if float(cand.get('price', 0) or 0) < min_price:
            logger.info(f"Universe: rejecting {symbol} — price "
                        f"${cand.get('price', 0)} below ${min_price}")
            continue

        closes = history.get(symbol)
        if not closes or len(_returns(closes)) < MIN_CORRELATION_BARS:
            logger.info(f"Universe: rejecting {symbol} — not enough history to screen")
            continue

        worst = None
        for held in tracked:
            other = history.get(held)
            if not other:
                continue
            rho = correlation(closes, other)
            if worst is None or abs(rho) > abs(worst[1]):
                worst = (held, rho)
        if worst and abs(worst[1]) > max_corr:
            logger.info(f"Universe: rejecting {symbol} — {worst[1]:+.2f} correlated "
                        f"with {worst[0]}, duplicates existing exposure")
            continue

        out.append({'symbol': symbol,
                    'reason': cand.get('reason', 'scout candidate'),
                    'market': cand.get('market', 'us')})
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_universe_manager.py -q`
Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agent/universe_manager.py tests/test_universe_manager.py
git commit -m "feat(universe): screen additions for liquidity, price and correlation"
```

---

### Task 8: Universe manager — removals

**Files:**
- Modify: `src/agent/universe_manager.py`
- Test: `tests/test_universe_manager.py` (append)

**Interfaces:**
- Consumes: Task 7's module.
- Produces: `propose_removals(tracked: set, stats: dict[str, dict], cfg: dict) -> list[dict]` — each returned dict has keys `symbol`, `reason`.
  `stats[symbol]` keys: `consecutive_losses` (int), `undefined_regime_days` (int), `avg_dollar_volume` (float), `attribution` (float).
  `cfg` keys read: `universe.max_consecutive_losses` (default 4), `universe.max_undefined_days` (default 30), plus `min_avg_dollar_volume` from Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_universe_manager.py`:

```python
# ---------------------------------------------------------------------------
# Removals
#
# Nothing proposes removals today, so a dead or persistently losing symbol
# stays in the universe forever, consuming API quota and LLM calls.
# ---------------------------------------------------------------------------

from src.agent.universe_manager import propose_removals

RCFG = {'universe': {'min_avg_dollar_volume': 1_000_000,
                     'max_consecutive_losses': 4, 'max_undefined_days': 30}}


def stat(**over):
    base = {'consecutive_losses': 0, 'undefined_regime_days': 0,
            'avg_dollar_volume': 5_000_000, 'attribution': 100.0}
    base.update(over)
    return base


def test_a_healthy_symbol_is_kept():
    assert propose_removals({'AAPL'}, {'AAPL': stat()}, RCFG) == []


def test_consecutive_losses_trigger_a_removal():
    out = propose_removals({'TSLA'}, {'TSLA': stat(consecutive_losses=5)}, RCFG)
    assert [r['symbol'] for r in out] == ['TSLA']
    assert 'consecutive' in out[0]['reason'].lower()


def test_a_symbol_stuck_in_undefined_regime_is_removed():
    """The router can never trade it, so it is pure overhead."""
    out = propose_removals({'XLE'}, {'XLE': stat(undefined_regime_days=45)}, RCFG)
    assert [r['symbol'] for r in out] == ['XLE']
    assert 'regime' in out[0]['reason'].lower()


def test_a_symbol_gone_illiquid_is_removed():
    out = propose_removals({'THIN'}, {'THIN': stat(avg_dollar_volume=5_000)}, RCFG)
    assert [r['symbol'] for r in out] == ['THIN']


def test_persistently_negative_attribution_triggers_a_removal():
    out = propose_removals({'XLV'}, {'XLV': stat(attribution=-500.0)}, RCFG)
    assert [r['symbol'] for r in out] == ['XLV']


def test_a_symbol_at_the_threshold_is_kept():
    """Boundary: 4 losses is the limit, not a breach."""
    assert propose_removals({'TSLA'}, {'TSLA': stat(consecutive_losses=4)}, RCFG) == []


def test_a_symbol_with_no_stats_is_kept():
    """Absence of evidence is not evidence of failure."""
    assert propose_removals({'NEW'}, {}, RCFG) == []


def test_only_one_proposal_per_symbol():
    out = propose_removals({'BAD'}, {'BAD': stat(consecutive_losses=9,
                                                 attribution=-900.0,
                                                 avg_dollar_volume=1.0)}, RCFG)
    assert len(out) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_universe_manager.py -q -k remov`
Expected: `ImportError: cannot import name 'propose_removals'`

- [ ] **Step 3: Write the implementation**

Append to `src/agent/universe_manager.py`:

```python
DEFAULT_MAX_CONSECUTIVE_LOSSES = 4
DEFAULT_MAX_UNDEFINED_DAYS = 30


def propose_removals(tracked: Set[str], stats: Dict[str, Dict[str, Any]],
                     cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Symbols that have earned their way out of the universe.

    One proposal per symbol, first matching reason wins — an operator does
    not need four escalations telling them the same name is bad.

    A symbol with no stats is KEPT. Absence of evidence is not evidence of
    failure, and silently dropping unmeasured names would shrink the universe
    by accident.
    """
    section = (cfg or {}).get('universe') or {}
    min_volume = float(section.get('min_avg_dollar_volume', DEFAULT_MIN_DOLLAR_VOLUME))
    max_losses = int(section.get('max_consecutive_losses', DEFAULT_MAX_CONSECUTIVE_LOSSES))
    max_undefined = int(section.get('max_undefined_days', DEFAULT_MAX_UNDEFINED_DAYS))

    out = []
    for symbol in sorted(tracked or set()):
        s = stats.get(symbol)
        if not s:
            continue

        losses = int(s.get('consecutive_losses', 0) or 0)
        undefined_days = int(s.get('undefined_regime_days', 0) or 0)
        volume = float(s.get('avg_dollar_volume', 0) or 0)
        attribution = float(s.get('attribution', 0) or 0)

        if losses > max_losses:
            reason = f'{losses} consecutive losing round trips (limit {max_losses})'
        elif undefined_days > max_undefined:
            reason = (f'regime undefined for {undefined_days} days — '
                      f'the router can never trade it')
        elif volume < min_volume:
            reason = f'average dollar volume ${volume:,.0f} below ${min_volume:,.0f}'
        elif attribution < 0:
            reason = f'negative attribution ${attribution:,.0f} over the review window'
        else:
            continue

        logger.info(f"Universe: proposing removal of {symbol} — {reason}")
        out.append({'symbol': symbol, 'reason': reason})
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_universe_manager.py -q`
Expected: `21 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agent/universe_manager.py tests/test_universe_manager.py
git commit -m "feat(universe): propose removals for dead, illiquid and losing symbols"
```

---

### Task 9: Make approval reach the trading loop

**Files:**
- Modify: `src/api/api_server.py` (the `resolve_escalation` handler)
- Modify: `src/agent/main.py` (universe read in the trading loop)
- Test: `tests/test_universe_approval.py` (create)

**Interfaces:**
- Consumes: `apply_overlay` and `_persist_overlay` semantics from `src/agent/self_assessment.py`.
- Produces: `src/agent/universe_manager.effective_universe(config: dict, watchlist: list[dict], market: str) -> list[str]` and `persist_universe(config: dict, symbol: str, market: str, overlay_path: str) -> None`.

Three breaks being fixed, all verified in the codebase:
1. `resolve_escalation` appends only to `nse_symbols` — US symbols have no path.
2. That write is in-memory; a restart reverts it.
3. `main.py:633` reads `symbols` from config while `get_active_watchlist()` feeds only LLM prompts — so an approved symbol never trades.

- [ ] **Step 1: Write the failing test**

Create `tests/test_universe_approval.py`:

```python
"""An approved symbol must actually get traded.

Verified against the code before this change: resolve_escalation appends to
config['data_manager']['nse_symbols'] only, in memory, and the trading loop
reads config['data_manager']['symbols'] at main.py:633 while
get_active_watchlist() is used solely to build LLM research context. So
operator approval had no effect on what the agent traded.
"""

import json
import os
import tempfile

import pytest

from src.agent.universe_manager import effective_universe, persist_universe


def base_config():
    return {'data_manager': {'symbols': ['AAPL', 'MSFT'],
                             'nse_symbols': ['SCOM', 'EQTY']}}


def test_effective_universe_includes_config_symbols():
    cfg = base_config()
    assert set(effective_universe(cfg, [], 'us')) == {'AAPL', 'MSFT'}


def test_an_approved_watchlist_symbol_joins_the_universe():
    """THE regression: this was written to the watchlist and never read."""
    cfg = base_config()
    watchlist = [{'symbol': 'NVDA', 'market': 'us', 'status': 'active'}]
    assert 'NVDA' in effective_universe(cfg, watchlist, 'us')


def test_a_removed_watchlist_symbol_leaves_the_universe():
    cfg = base_config()
    watchlist = [{'symbol': 'AAPL', 'market': 'us', 'status': 'removed'}]
    assert 'AAPL' not in effective_universe(cfg, watchlist, 'us')


def test_a_paused_watchlist_symbol_leaves_the_universe():
    cfg = base_config()
    watchlist = [{'symbol': 'AAPL', 'market': 'us', 'status': 'paused'}]
    assert 'AAPL' not in effective_universe(cfg, watchlist, 'us')


def test_markets_do_not_leak_into_each_other():
    cfg = base_config()
    watchlist = [{'symbol': 'NVDA', 'market': 'us', 'status': 'active'}]
    assert 'NVDA' not in effective_universe(cfg, watchlist, 'kenyan')
    assert set(effective_universe(cfg, watchlist, 'kenyan')) == {'SCOM', 'EQTY'}


def test_the_universe_has_no_duplicates():
    cfg = base_config()
    watchlist = [{'symbol': 'AAPL', 'market': 'us', 'status': 'active'}]
    universe = effective_universe(cfg, watchlist, 'us')
    assert len(universe) == len(set(universe))


def test_a_us_symbol_persists_to_the_us_list():
    """US symbols previously had no approval path at all."""
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'NVDA', 'us', overlay)
        assert 'NVDA' in cfg['data_manager']['symbols']
        saved = json.loads(open(overlay, encoding='utf-8').read())
        assert 'NVDA' in saved['data_manager.symbols']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_a_kenyan_symbol_persists_to_the_nse_list():
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'KCB', 'kenyan', overlay)
        assert 'KCB' in cfg['data_manager']['nse_symbols']
        assert 'KCB' not in cfg['data_manager']['symbols']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_an_approval_survives_a_restart():
    """The in-memory write reverted on every restart."""
    from src.agent.self_assessment import apply_overlay
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'NVDA', 'us', overlay)

        fresh = base_config()
        apply_overlay(fresh, overlay)
        assert 'NVDA' in fresh['data_manager']['symbols']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_persisting_the_same_symbol_twice_is_idempotent():
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'NVDA', 'us', overlay)
        persist_universe(cfg, 'NVDA', 'us', overlay)
        assert cfg['data_manager']['symbols'].count('NVDA') == 1
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_universe_approval.py -q`
Expected: `ImportError: cannot import name 'effective_universe'`

- [ ] **Step 3: Write the implementation**

Append to `src/agent/universe_manager.py`:

```python
MARKET_CONFIG_KEY = {'us': 'symbols', 'kenyan': 'nse_symbols'}


def effective_universe(config: Dict[str, Any], watchlist: List[Dict[str, Any]],
                       market: str = 'us') -> List[str]:
    """What the trading loop should actually iterate over.

    config symbols + operator-approved additions - paused/removed names.
    Previously the loop read config alone, so an approved symbol was written
    to the watchlist and never traded.
    """
    key = MARKET_CONFIG_KEY.get(market, 'symbols')
    universe = list((config.get('data_manager') or {}).get(key) or [])

    for row in watchlist or []:
        symbol = (row.get('symbol') or '').upper()
        if not symbol or row.get('market', 'us') != market:
            continue
        status = row.get('status', 'active')
        if status == 'active':
            if symbol not in universe:
                universe.append(symbol)
        elif symbol in universe:
            universe.remove(symbol)

    seen, out = set(), []
    for symbol in universe:
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def persist_universe(config: Dict[str, Any], symbol: str, market: str,
                     overlay_path: Optional[str] = None) -> None:
    """Add an approved symbol to the right list AND to the on-disk overlay.

    Without the overlay the approval is in-memory only and the next restart
    silently reverts it.
    """
    from src.agent.self_assessment import DEFAULT_OVERLAY_PATH, _write_overlay_key

    symbol = (symbol or '').upper()
    if not symbol:
        return
    key = MARKET_CONFIG_KEY.get(market, 'symbols')
    target = config.setdefault('data_manager', {}).setdefault(key, [])
    if symbol not in target:
        target.append(symbol)
    _write_overlay_key(f'data_manager.{key}', list(target),
                       overlay_path or DEFAULT_OVERLAY_PATH)
    logger.warning(f"Universe: {symbol} added to {key} and persisted")
```

In `src/agent/self_assessment.py`, extract the overlay write so it can be
reused. Replace the body of `_persist_overlay` with a delegation and add the
module-level function above `apply_overlay`:

```python
def _write_overlay_key(dotted_key: str, value: Any, path: str) -> None:
    """Merge one key into the overlay and rewrite it atomically."""
    try:
        saved = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    saved = loaded
            except (OSError, ValueError):
                logger.warning("Overwriting unreadable config overlay")
        saved[dotted_key] = value
        saved['_updated_at'] = utc_now_iso()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = f'{path}.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(saved, fh, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.error(f"Failed to persist config overlay: {e}")
```

and make the method delegate:

```python
    def _persist_overlay(self, dotted_key: str, value: Any) -> None:
        """Record the change so `apply_overlay` can replay it after a restart."""
        if not self.overlay_path:
            return
        _write_overlay_key(dotted_key, value, self.overlay_path)
```

In `src/api/api_server.py`, replace the `add_symbol` approval branch:

```python
            if status == 'approved' and escalation and escalation.get('action') == 'add_symbol':
                # Route by market and persist, so the approval survives a
                # restart and reaches the market it belongs to. US symbols
                # previously had no path here at all.
                from src.agent.universe_manager import persist_universe
                symbol = (escalation.get('symbol') or '').upper()
                market = escalation.get('market') or 'kenyan'
                persist_universe(self.trading_agent.config, symbol, market)
                em.add_to_watchlist(symbol, market=market, source='universe_scout')
```

In `src/agent/main.py`, replace line 633's universe read:

```python
        # config + operator-approved additions - paused/removed. Reading config
        # alone meant an approved symbol was written to the watchlist and
        # never actually traded.
        from src.agent.universe_manager import effective_universe
        em = self.components.get('escalation_manager')
        watchlist = em.get_active_watchlist() if em else []
        symbols = effective_universe(self.config, watchlist, 'us')
        if not symbols:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_universe_approval.py -q`
Expected: `10 passed`

Then the full suite: `.venv/Scripts/python.exe -m pytest -q`
Expected: `455 passed, 2 skipped`

- [ ] **Step 5: Commit**

```bash
git add src/agent/universe_manager.py src/agent/self_assessment.py src/api/api_server.py src/agent/main.py tests/test_universe_approval.py
git commit -m "fix(universe): approved symbols now reach the trading loop and survive restart"
```

---

## Completion checklist

- [ ] All nine tasks committed
- [ ] Full suite green, no new deprecation warnings
- [ ] `data/regime_calibration.json` records the in-sample grid
- [ ] `data/strategy_backtest_results.json` records the out-of-sample A/B and verdict
- [ ] `config/config.json` has `ensemble_method: "regime_routed"` **only if** the gate passed
- [ ] If the gate failed: report which criteria missed and by how much. Do not re-tune against the spent out-of-sample window.
