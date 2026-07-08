# Decision Transparency (Group 1) — Design

## Goal

Make the agent's existing reasoning legible without hiding anything new
behind more clicks than necessary. Three changes, all operator-only
(matching the existing decision-internals redaction policy already applied
to the symbol drill-down's `decisions` field):

1. Expandable decision rows in the symbol drill-down — full LLM reasoning
   plus what the LLM was given (signals, price, news).
2. Agent Log filtering — hide plain "no signal" holds by default, surface
   only decisions with a specific reason.
3. Access control — extend the existing operator/viewer split to the
   agent-activity feed, which currently has none.

Out of scope: Group 2 (portfolio-level Agent-vs-Buy&Hold rollup, LLM
veto-effectiveness metric) — separate spec.

## 1. Expandable Decision Rows

### Data model — `src/agent/decision_journal.py`

Add a nullable `news_headlines_json TEXT` column to the `decisions` table.
The table already has real rows in `data/order_journal.db` from live
sessions, so this cannot be a fresh `CREATE TABLE` — `DecisionJournal.__init__`
must check for the column's existence (`PRAGMA table_info(decisions)`) and
`ALTER TABLE decisions ADD COLUMN news_headlines_json TEXT` if missing,
after running the existing `_SCHEMA` script. This is additive and backward
compatible: old rows read back `news_headlines: []`.

`record()` accepts an optional `news_headlines: List[str]` key on the
decision dict, storing `json.dumps(...)` (empty list if absent).
`recent()` parses it back the same way `per_strategy_json` and
`llm_verdict_json` are parsed today, defaulting to `[]` on missing/null.

### Capture point — `src/agent/main.py`

Where `dec['llm_verdict']` is already populated (inside the
`if symbol_signals and symbol_signals.get('action') != 'hold':` block,
around the `validate_trade` call), also set:

```python
dec['news_headlines'] = [
    n.get('title') for n in symbol_news[:5]
    if isinstance(n, dict) and n.get('title')
]
```

This only fires for decisions that actually reach LLM validation — the
majority of pure-hold cycles (no directional signal at all) never populate
this field, which is correct: there's no "LLM input" to show for a cycle
where the LLM was never invoked.

### API — no changes needed

`/api/symbol/<symbol>` already passes `decisions` straight from
`dj.recent()` through `build_payload()`, and already strips `decisions`
entirely for non-operators (`payload['decisions'] = []`,
`decisions_restricted = True`). The new field is covered by that existing
redaction with no additional code.

### Frontend — `frontend/components/SymbolDrilldown.js`

Each decision row gets a chevron toggle. Expanding a row reveals a panel
directly below it with:

- Full `llm_verdict.reasoning` (no truncation), or
  `"No LLM review — no directional signal generated"` if `llm_verdict` is
  empty.
- Per-strategy signals: iterate `per_strategy` entries as
  `{name}: {action} @ {confidence}`, or `"No signal generated this cycle"`
  if empty.
- `news_headlines` as a bullet list, or `"Not captured (recorded before
  this feature)"` if absent — this covers both pre-migration rows and
  non-LLM-validated holds under one message, since the two are
  indistinguishable from the frontend's perspective and the distinction
  doesn't matter to the reader.
- Price at decision time.

Collapsed rows are unchanged from today (timestamp, action, confidence,
short reasoning snippet) so the list stays scannable; expansion is opt-in
per row, one row open at a time.

### Testing

- Unit test: `DecisionJournal.record()` → `recent()` round-trips
  `news_headlines`.
- Migration test: construct a `DecisionJournal` against a temp DB file
  seeded with the *old* schema (no `news_headlines_json` column), confirm
  `__init__` adds the column without raising or losing existing rows.
- Manual verification in the browser preview: expand a row with a captured
  `llm_verdict`, expand a plain-hold row, confirm both empty states read
  as intentional rather than broken.

## 2. Agent Log Filtering + Structured Feed

### Backend — `/api/agent-activity` (`src/api/api_server.py`)

Stop flattening to a single `message` string. Return structured fields:

```json
{"timestamp": "...", "symbol": "XLV", "action": "hold",
 "executed": false, "skip_reason": "llm_veto", "message": "HOLD blocked: llm_veto"}
```

`message` is kept for simple consumers, but the frontend filters/colors on
the raw fields, not by parsing the string.

Add `@role_required('operator')` to this route, matching `/api/anomalies`.

### Frontend fetch — `frontend/components/AdvancedDashboard.js`

`agent-activity` currently sits inside the batched `fetchData()` call,
fired unconditionally before the caller's role is known on first load.
Since it's now operator-gated, pull it into its own `useEffect` keyed on
`isOperator`, identical in shape to the existing `/api/anomalies` effect:
viewers never fire a request that would 403, and operators only start
polling once `data.status.role` has resolved. Keep the 20-second poll
interval it has today (the cadence of the batched `fetchData` it's being
pulled out of), not the 30-second interval `/api/anomalies` uses.

### Filtering & styling

An entry is "interesting" (shown by default) if:
- `executed` is `true`, OR
- `skip_reason` is set and is not the literal string `"hold"`.

Plain no-signal holds (`skip_reason === "hold"` or unset with
`action === "hold"`) are filtered out by default. A "Show idle holds"
toggle in the panel header (local component state, default off) reveals
them.

Visual treatment:
- Executed entries: primary/green accent + a distinct icon.
- Blocked-with-reason entries: amber for `llm_veto`/`bias_downgrade`,
  red for `risk_limits`/`halted`.
- Idle holds (when the toggle is on): muted gray, as today.

### Viewer experience

For `isOperator === false`, the Agent Log panel renders an explicit
"Operator view only" placeholder instead of an empty box — consistent
with how the anomaly banner already handles viewers.

### Testing

- Backend test: `/api/agent-activity` returns 403 for a viewer token, 200
  with the structured fields above for an operator token.
- Manual verification: toggle "Show idle holds" on/off and confirm
  filtering; log in as a viewer-role account and confirm the placeholder
  renders instead of a silently-empty panel.

## 3. Access Control Summary

- Symbol drill-down: already redacts `decisions` for viewers; the new
  `news_headlines` field is covered automatically.
- Agent Log: newly gated `@role_required('operator')` server-side, and the
  fetch itself is skipped client-side for viewers.
- No other viewer-facing surface exposes LLM reasoning, inputs, or skip
  reasons.

## Rollout

- The DB migration is additive and safe to deploy without downtime or
  touching existing data — old rows simply read back an empty
  `news_headlines` list.
- No config or dependency changes required.
- New tests: decision-journal migration + round-trip, agent-activity role
  gating (403/200).
