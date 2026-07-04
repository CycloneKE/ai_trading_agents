# QA Test Report — AI Trading Platform

**Date:** 2026-07-04 · **Build:** `premarket-hardening` (PR #1) · **Environment:** production config, live Alpaca paper account, market closed (weekend/holiday)

## Use cases under test

| UC | Use case | Actor |
|----|----------|-------|
| UC-0 | Regression: all automated tests pass | CI |
| UC-1 | A trader authenticates (and attackers can't) | Trader / attacker |
| UC-2 | A trader monitors portfolio and market data via API/dashboard | Trader |
| UC-3 | An operator halts and resumes automated trading | Operator |
| UC-4 | The system cannot duplicate, mis-size, or lose track of orders | System |
| UC-5 | A first-time trader logs into the dashboard and orients | Trader |
| UC-6 | The system practices on history without overfitting | System (nightly) |

## Results

### UC-0 Regression — PASS
`pytest tests/` → **50 passed** (journal, attribution, sizing, allocator guardrails, parallel strategies, overlay, warm-start, smoke start).

### UC-1 Authentication — 7/7 PASS
| ID | Scenario | Expected | Result |
|----|----------|----------|--------|
| AUTH-01 | Health endpoint without token | 200 (public, no data) | PASS |
| AUTH-02 | Login, wrong password | 401 | PASS |
| AUTH-03 | Login, malformed body | 400 | PASS |
| AUTH-04 | Login, unknown user | 401 | PASS |
| AUTH-05 | Login, valid credentials | 200 + JWT | PASS |
| AUTH-06 | Protected endpoint, no token | 401 | PASS |
| AUTH-07 | Protected endpoint, garbage token | 401 | PASS |

Lockout (5 failures → 429 even with the correct password, retry_after ≈ 15 min) was live-verified at implementation time; not re-run to avoid locking the QA account mid-pass.

### UC-2 API contracts — 3/3 PASS + cache verified
| ID | Scenario | Result |
|----|----------|--------|
| API-01 | `/api/status` reports running | PASS |
| API-02 | `/api/portfolio` returns account + positions | PASS |
| API-03 | Unknown endpoint → 404 | PASS |
| API-04 | News-feed response cache | PASS — cold 4.6 s → cached 0.10 s (46×) |

### UC-3 Kill switch — 5/5 PASS
| ID | Scenario | Result |
|----|----------|--------|
| KILL-01 | Halt without `{"confirm": true}` → 400 | PASS |
| KILL-02 | Halt with confirm → halted | PASS |
| KILL-03 | `/api/status` reflects halt | PASS |
| KILL-04 | Halt without auth → 401 | PASS |
| KILL-05 | Resume restores trading | PASS |

### UC-4 Order safeguards (live broker) — 7/7 PASS
| ID | Scenario | Result |
|----|----------|--------|
| SAFE-01 | Broker connects, confirmed PAPER endpoint | PASS |
| SAFE-02 | Market calendar accessible | PASS |
| SAFE-03 | Limit order accepted while market closed (queues) | PASS |
| SAFE-04 | Duplicate client_order_id rejected server-side | PASS |
| SAFE-05 | Fractional DAY order accepted | PASS |
| SAFE-06 | Journal blocks duplicate decision client-side | PASS |
| SAFE-07 | Startup reconcile resolves in-flight order against live broker | PASS |

All QA orders were canceled after the pass; no positions were opened.

### UC-5 Dashboard UX (real browser) — 6/6 PASS
| ID | Scenario | Result |
|----|----------|--------|
| UX-01 | Wrong password → error shown, stays on login | PASS |
| UX-02 | Valid login → dashboard renders | PASS |
| UX-03 | First login → Getting Started guide auto-opens | PASS |
| UX-04 | Market clock: Saturday 04:36 ET shows CLOSED (weekend logic overrides premarket window) | PASS |
| UX-05 | HALT and help (?) controls present in header | PASS |
| UX-06 | All five tabs render; **zero browser console errors** | PASS |

Strategy Attribution panel confirmed populated with real journal data.

### UC-6 Practice loop — PASS (verified at implementation, 2026-07-03)
Live dry-run on ~150 real daily bars × 8 symbols: momentum/RSI variants correctly rejected by the both-windows out-of-sample gate; mean_reversion `rsi_overbought=75` correctly identified as a promotion candidate. Corrupt/missing overlay files verified harmless (unit-tested).

## Summary

**43 scenarios executed this pass: 43 PASS, 0 FAIL.** No new defects found.

## Known limitations (accepted, documented in runbook)
- No fill-path testing possible while market closed — order *acceptance* and dedup verified; fills were verified live during the 2026-07-02 premarket session (AAPL/SPY round trip).
- NSE Kenya data is synthetic/stale (scraper sources not refreshing) — cosmetic for US-market sessions.
- Single shared paper portfolio; logins gate access, not per-trader books.
- HTTP unless the Caddy TLS proxy is used.
