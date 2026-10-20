# Regime-Routed Signals & Dynamic Universe — Design

**Date:** 2026-08-02
**Status:** Approved for planning
**Objective:** absolute return, uncorrelated to the market. Judged on Sharpe
and SPY correlation, not on beating buy-and-hold.

## 1. Why

`scripts/run_strategy_backtest.py` over 1,255 daily bars × 11 symbols, costs
on (0.10% commission + 0.05% slippage):

| Window | Strategy | SPY | Sharpe | Trades |
|---|---|---|---|---|
| Full period | +4.16% | +70.71% | −0.66 | 73 |
| First half | −0.00% | +12.27% | −2.33 | 23 |
| Second half (OOS) | +4.43% | +52.18% | −0.10 | 44 |

Max drawdown was 1.85%. The agent is not losing aggressively — it is sitting
in cash. Instrumenting the real `StrategyManager` over the same history
(13,145 symbol-bars) shows exactly why:

```
PER-STRATEGY ACTION MIX
  momentum         buy=30.5%  sell=21.5%  hold=48.0%
  mean_reversion   buy=23.6%  sell=33.5%  hold=42.9%
  rsi_strategy     buy= 2.5%  sell= 8.8%  hold=88.8%

AGREEMENT
  contradict     44.3%
  all hold       34.9%
  lone voice     20.2%
  agree (2+)      0.6%

WHERE THE SIGNAL DIES
  ensemble cancelled a live signal   62.7%
  no strategy had a view             34.9%
  survived to execution               2.3%

CONFIDENCE OF SURVIVORS (n=306)
  median 0.517   p75 0.534   max 0.629
```

Three root causes:

1. **Momentum and mean-reversion are structural opposites.** They measure the
   same thing with opposite sign, so `_weighted_average_ensemble` cancels a
   live signal 62.7% of the time. The ensemble manufactures indecision.
2. **Confidence cannot escape ~0.5.** The winning action's score is divided by
   total weight, so a 2-vs-1 disagreement lands at ≈0.5 by construction. Since
   `position_size = confidence × max_position_size`, every surviving trade is
   sized at ~2.6%. The strategy cannot express conviction.
3. **It sells 5× more than it buys** (2.0% vs 0.4%) across a period when SPY
   rose 71% — a persistent short bias pointed the wrong way.

## 2. Acceptance gate

This ships only if the **out-of-sample** half shows all of:

- Sharpe > 0.5 (currently −0.10)
- |correlation to SPY daily returns| < 0.3
- max drawdown < 15%
- more than 100 trades (below this it is luck, not evidence)
- strictly better Sharpe than the current ensemble on the same window

If any fails, it does not go live. This gate is the discipline that was
missing when the current strategy was deployed.

## 3. Part 1 — Regime classifier

**New module: `src/agent/regime.py`.** Stateless, no dependencies beyond the
standard library.

```python
efficiency_ratio(closes: list[float], window: int) -> float
classify(closes: list[float], previous: str | None, cfg: dict) -> str
```

**Kaufman efficiency ratio:**

```
ER = |close[t] − close[t−N]| / Σ|close[i] − close[i−1]|   over the same N bars
```

A single number in [0, 1]. High means price travelled somewhere directly
(trending); low means it chopped (ranging). Chosen over ADX/Hurst because it
needs no library, has exactly one parameter, and measures the routing
criterion directly rather than by proxy.

**Dead band and hysteresis:**

- `ER ≥ trend_threshold` → `trending`
- `ER ≤ range_threshold` → `ranging`
- otherwise → `undefined`, and no trade

Once in a regime, switching requires crossing the *opposite* threshold
(`previous` carries the prior state). Without hysteresis the router flips at
the boundary every other bar and pays costs for the privilege.

Insufficient history (fewer than `window + 1` bars) returns `undefined`.
A zero-denominator series (flat price) returns `undefined`, never a
divide-by-zero.

## 4. Part 2 — Routing

**`ensemble_method: "regime_routed"`** — a sixth method in `StrategyManager`,
selected by config. Existing methods are untouched, so the harness can A/B the
two by changing one string.

| Regime | Strategy executed | RSI's role |
|---|---|---|
| `trending` | momentum only | ignored |
| `ranging` | mean_reversion only | confirmation gate — must agree, else no trade |
| `undefined` | none | — |

Exactly one strategy is consulted. Nothing is averaged, so nothing cancels.

RSI moves from voter to filter. At 88.8% hold it is inert as a voter; as a
confirmation gate in the one regime where it is meaningful, it earns its place.

**Confidence** passes through from the routed strategy unmodified — so it can
span 0.5–1.0 as `TechnicalStrategy._scale` already intends — then is scaled by
a regime-conviction factor in [0.7, 1.0] derived from how far ER sits past its
threshold. A marginal regime sizes smaller. This is what lets sizing vary
instead of pinning at 2.6%.

## 5. Part 3 — Net exposure cap

The router is still directional; on its own it will not deliver low SPY
correlation. `ExposureGuard` gains a portfolio-level check alongside the
existing per-symbol cap:

```
|Σ long notional − Σ short notional| / equity ≤ max_net_exposure   (default 0.30)
```

Semantics match the per-symbol cap and must not regress it:

- Applies to **adds only**. An order that *reduces* net exposure always passes.
- Exits are never blocked, under any circumstance.
- Unusable equity blocks adds rather than sizing off a default.
- An add is trimmed to the remaining net-exposure headroom, and refused
  outright when that headroom falls below `min_notional`.

Both caps apply; an order must satisfy the tighter of the two.

## 6. Part 4 — Universe manager

**New module: `src/agent/universe_manager.py`** — the single answer to "what
should I be trading?"

### Current gaps

- `resolve_escalation` appends approved symbols to
  `config['data_manager']['nse_symbols']` only. **US symbols have no path.**
- That write is in-memory; a restart reverts it.
- The trading loop reads `symbols` from config (`main.py:633`).
  `get_active_watchlist()` is only ever read to build LLM research context, so
  **an approved symbol never reaches the trading loop.**
- No removal scout exists at all.

### Additions

Extends `universe_scout.propose_candidates` beyond NSE, adding screens it
lacks:

- **liquidity** — minimum average dollar volume; you cannot exit what does not trade
- **price floor** — avoid sub-$5 noise
- **correlation** — reject any candidate >0.9 correlated with a tracked symbol.
  Serves the uncorrelated objective directly and fixes the SPY/QQQ/DIA
  near-duplication that leaves 11 symbols with roughly 7 independent names.

### Removals

Data-driven, from journals already in place:

- N consecutive losing round trips (order journal)
- regime `undefined` for X consecutive days — the router can never trade it
- average dollar volume fallen below the liquidity floor
- persistently negative attribution over the review window
  (`strategy_attribution.py`)

### Invariants

- Both additions and removals produce **operator escalations**. Nothing
  auto-applies. This preserves the existing approval invariant, which is
  correct and is not being softened.
- Approval routes to `symbols` or `nse_symbols` by market.
- Approval persists through `config_overlay.json` (built during the risk work)
  so it survives restart.
- The trading loop reads its effective universe from the manager:
  config + approved − removed.

Runs daily alongside self-assessment, not per cycle.

## 7. Calibration protocol

`window`, `trend_threshold` and `range_threshold` are fitted on the **first
half** of history (bars 0 → n/2) and validated on the **untouched second
half**. The out-of-sample number is the one reported and the one the
acceptance gate reads.

Constraints that keep this honest:

- The OOS half is evaluated **once**, after parameters are frozen. Re-tuning
  after seeing OOS results contaminates it; if that becomes necessary, the
  spec requires fetching a genuinely fresh window instead.
- The fitted parameters are written to config and committed, so the exact
  values behind any reported number are recoverable.
- Fitting searches a coarse grid (window ∈ {10, 20, 30, 40}, thresholds on a
  0.05 lattice) rather than a fine one. A coarse grid over 4 parameters is
  harder to overfit than a fine search, and the difference between 0.35 and
  0.36 is noise at this sample size.

## 8. Testing

Every component is TDD'd, and the failing test is watched first.

**`regime.py`** — pure functions, exhaustively testable:
- ER of a monotonic series is 1.0; of a perfect zigzag, ~0.0
- flat series returns `undefined`, not a ZeroDivisionError
- short history returns `undefined`
- hysteresis: a series sitting in the dead band holds its prior regime
- a series crossing the opposite threshold does switch

**Routing** — through the real `StrategyManager`:
- trending regime consults momentum and not mean_reversion
- ranging regime requires RSI agreement
- `undefined` produces `hold`
- confidence is no longer pinned at ~0.5 (the regression this exists to fix)
- other ensemble methods still behave exactly as before

**Net exposure** — extends `tests/test_exposure.py`:
- an add beyond the net cap is refused
- a reducing order always passes, even far past the cap
- exits are never blocked
- the per-symbol cap still applies unchanged

**Universe manager:**
- a correlated candidate is rejected
- an illiquid candidate is rejected
- removals fire on the documented triggers
- nothing auto-applies — every proposal lands as an escalation
- approval reaches the trading loop's effective universe
- approval survives a simulated restart via the overlay

**End to end** — the harness itself is the integration test. The comparison
run is recorded in `data/strategy_backtest_results.json` and the acceptance
gate is checked against the OOS row.

## 9. Out of scope

- Cross-sectional market-neutral ranking (Approach B). Revisit if the router
  clears the Sharpe bar but misses the correlation bar.
- Intraday signalling. The 60s loop stays the risk layer — stops, trailing
  stops, exposure, cooldown, kill switch. Signals are daily.
- Retiring `_weighted_average_ensemble` and friends. They stay, unused by
  default, so the A/B remains runnable.
- Any change to the NSE dividend sleeve.

## 10. Risks

- **Regime classification is itself a prediction.** A wrong call routes to the
  strategy guaranteed to be wrong for that tape. The dead band limits how often
  this happens near boundaries but cannot eliminate it. The conviction scaling
  is the mitigation: marginal regimes trade smaller.
- **Both techniques are well documented and well crowded.** A realistic good
  outcome is a modest positive Sharpe with low correlation, not a step change.
  The acceptance gate is set accordingly, not aspirationally.
- **11 symbols is a thin universe**, and three of them (SPY/QQQ/DIA) are near
  duplicates. The correlation screen addresses new additions but does not fix
  the existing set; expanding the universe is the natural follow-on.
- **Daily backtest, 60s live loop.** The harness validates daily-bar logic.
  Part 4's hybrid split makes that the intended design rather than an accident,
  but the mismatch is why signals move to a daily cadence rather than being
  left ambiguous.
- **One-shot OOS.** Section 7 spends the out-of-sample window once. If the
  result is marginal there is no clean second look without fresh data.
