"""
Self Assessment Engine: a periodic AI review of the agent's own results.

It is advisory. It used to post whatever the AI suggested to the operator's
approval queue, and to apply some suggestions by itself: settings that do
not exist (strategy_weights, which nothing reads), order sizes, placeholder
tickers such as ABC or XYZ, and contradictory stop changes, all made with no
trades to judge by. Approving one did not apply it either. Now:

- the AI is asked only once at least `self_assessment.min_closed_trades`
  (15) round trips have closed, and again only after `min_new_trades` (5)
  more have; otherwise it costs no AI call and the last review stays on
  the Notifications page. It is shown what those trades did (closed_
  trade_evidence): the win rate, average win and loss net of costs, and
  how each exit went, by whether a stop-loss, a trailing stop or a signal
  closed it;
- it may suggest changes to the stop settings only (ADJUSTABLE), within
  fixed ranges and by at most half the current value. Strategy settings
  are tuned on evidence by strategy_tuner.py; order sizes and the trading
  lists are the operator's;
- every suggestion is checked (check_proposals), the rest are kept with the
  reason, and nothing is applied: the latest review is shown on the
  Notifications page (latest_review), and adopting a suggestion is a change
  to config.json.
"""
import os
import json
import sqlite3
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from src.agent.escalation_manager import EscalationManager
from src.agent.round_trips import closed_round_trips
from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                      TEXT NOT NULL,
    cycles_reviewed         INTEGER,
    prediction_accuracy_json TEXT,
    adaptation_effectiveness_json TEXT,
    bias_trends_json TEXT,
    improvement_plan_json TEXT
);

CREATE TABLE IF NOT EXISTS improvements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id   INTEGER REFERENCES assessments(id),
    target          TEXT NOT NULL,        -- e.g. 'confidence_threshold'
    previous_value  TEXT,
    new_value       TEXT,
    reasoning       TEXT,
    applied_at      TEXT NOT NULL,
    auto_applied    INTEGER DEFAULT 0,    -- 1 if auto, 0 if operator-approved
    outcome_score   REAL,                 -- filled by future assessments
    outcome_notes   TEXT
);
"""

# setting -> (lowest, highest) a suggestion may propose
ADJUSTABLE = {
    'risk_limits.stop_loss_pct': (0.02, 0.15),
    'risk_limits.trailing_stop_pct': (0.01, 0.10),
    'risk_limits.stop_loss_atr_mult': (1.5, 4.0),
    'risk_limits.trailing_stop_atr_mult': (1.5, 5.0),
}
MAX_RELATIVE_CHANGE = 0.5      # a suggestion moves a setting by at most half its value
MAX_SUGGESTIONS = 3
DEFAULT_MIN_CLOSED = 15
DEFAULT_MIN_NEW = 5

_RETROSPECTIVE_SYSTEM_PROMPT = """
You review an algorithmic trading system's recent results and may suggest changes to its stop-loss
settings, only where the results clearly support them. Suggest nothing when they do not.

You are given the round trips the agent has closed: their win rate, average win and loss (after
costs), and how each kind of exit went (a stop-loss, a trailing stop, or a signal), plus the current
stop settings. Stops that fire often at a small gain or loss may be too tight; losses well beyond the
stop distance or a long tail of large losses may mean too loose. Only these settings
may be changed, within these ranges, and by at most half of the current value:
- risk_limits.stop_loss_pct (0.02 to 0.15): fixed stop-loss, a fraction of the entry price
- risk_limits.trailing_stop_pct (0.01 to 0.10): fixed trailing stop, a fraction of the high
- risk_limits.stop_loss_atr_mult (1.5 to 4.0): stop distance in multiples of average true range
- risk_limits.trailing_stop_atr_mult (1.5 to 5.0): trailing distance in multiples of average true range
At most three suggestions. Do not suggest anything else: no symbols, no strategy weights, no order sizes.

Reply with JSON only, in this shape:
{"parameter_adjustments": [{"target": "risk_limits.trailing_stop_pct", "proposed": 0.04,
  "reasoning": "one sentence citing the numbers given"}],
 "reasoning_summary": "two sentences at most"}
"""


class SelfAssessmentEngine:
    def __init__(self, llm_orchestrator, escalation_manager: EscalationManager, config: Dict[str, Any],
                 db_path: str = str(DATA_DIR / 'improvements.db')):
        self.llm = llm_orchestrator
        self.escalation_manager = escalation_manager
        self.config = config
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Initialize SQLite DB
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        
        logger.info(f"SelfAssessmentEngine database initialized at {db_path}")

    def run_assessment(self, decision_journal, order_journal, performance_analytics,
                       cycles_to_review: int = 390) -> Dict[str, Any]:
        """Review the agent's closed trades and, when there is new evidence
        enough, ask the AI for suggestions, which are checked and recorded.
        Nothing is applied. Returns the review (see latest_review).

        (`decision_journal`, `performance_analytics` and `cycles_to_review`
        are kept for the caller's sake; the review reads the order journal.)
        """
        fills = order_journal.filled_orders() if order_journal else []
        evidence = closed_trade_evidence(fills, self.config)
        closed = evidence['closed_trades']
        cfg = self.config.get('self_assessment') or {}
        min_closed = int(cfg.get('min_closed_trades', DEFAULT_MIN_CLOSED))
        min_new = int(cfg.get('min_new_trades', DEFAULT_MIN_NEW))
        asked_at = self._closed_when_last_asked()

        review: Dict[str, Any] = {'closed_trades': closed, 'min_trades': min_closed, 'skipped': None,
                                  'suggestions': [], 'rejected': [], 'summary': ''}
        if closed < min_closed:
            review['skipped'] = (f"{closed} closed trade{'s' if closed != 1 else ''} so far; "
                                 f"the review waits for {min_closed} before suggesting changes")
        elif asked_at is not None and closed - asked_at < min_new:
            # Nothing has changed since the AI last looked: not asked again,
            # and the review on the dashboard stays the one it gave.
            latest = self.latest_review()
            if latest is not None:
                return latest
            review['skipped'] = f"waiting for {min_new} more closed trades since the last review"
        elif not (self.llm and getattr(self.llm, "enabled", False)):
            review['skipped'] = 'no AI service is set up'
        else:
            context = {
                "current_settings": {t: (self.config.get('risk_limits') or {}).get(t.split('.', 1)[1])
                                     for t in ADJUSTABLE},
                **evidence,
            }
            plan = None
            try:
                plan = self.llm.propose_json(_RETROSPECTIVE_SYSTEM_PROMPT, json.dumps(context, default=str))
            except Exception as e:
                logger.error(f"AI self-review failed: {e}")
            review['asked_at_closed'] = closed
            if isinstance(plan, dict):
                review['suggestions'], review['rejected'] = check_proposals(plan, self.config)
                review['summary'] = str(plan.get('reasoning_summary') or '')[:500]
            else:
                review['skipped'] = 'the AI did not answer'

        # A waiting review is recorded when it says something new, not every pass.
        latest = self.latest_review()
        if (review['skipped'] and latest is not None and latest.get('skipped') == review['skipped']
                and latest.get('closed_trades') == closed):
            return latest
        review['assessment_id'] = self._record_assessment(closed, evidence, review)
        logger.info(f"AI self-review: {describe(review)}")
        return review

    def _closed_when_last_asked(self) -> Optional[int]:
        """How many trades had closed when the AI was last asked, or None."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT improvement_plan_json FROM assessments ORDER BY id DESC LIMIT 50").fetchall()
        for (raw,) in rows:
            try:
                plan = json.loads(raw or '{}')
            except ValueError:
                continue
            if isinstance(plan, dict) and plan.get('asked_at_closed') is not None:
                return int(plan['asked_at_closed'])
        return None

    def latest_review(self) -> Optional[Dict[str, Any]]:
        """The newest review for the dashboard, or None when there is none
        (reviews recorded before they were checked are not shown)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT ts, improvement_plan_json FROM assessments ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        try:
            plan = json.loads(row[1] or '{}')
        except ValueError:
            return None
        if not isinstance(plan, dict) or 'suggestions' not in plan:
            return None
        return {'at': row[0], **{k: plan.get(k) for k in
                                 ('closed_trades', 'executed_trades', 'min_trades', 'skipped',
                                  'suggestions', 'rejected', 'summary')}}

    def _record_assessment(self, closed: int, evidence: Dict[str, Any], plan: Dict[str, Any]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO assessments (ts, cycles_reviewed, prediction_accuracy_json, "
                "adaptation_effectiveness_json, bias_trends_json, improvement_plan_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), closed, json.dumps(evidence),
                 None, None, json.dumps(plan)))
            self._conn.commit()
            return cur.lastrowid

    def close(self):
        """Close connection."""
        with self._lock:
            self._conn.close()


def closed_trade_evidence(fills: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the agent's closed round trips did, from the order journal's
    filled orders (oldest first): FIFO-matched, long only, each exit tagged
    by what closed it (a stop-loss, a trailing stop, the kill switch, or a
    signal), with returns after the modelled costs a fill price leaves out
    (see round_trips.closed_round_trips), as fractions of the entry price.

    This replaces judging decisions by the next journal row's price, which
    could be minutes or hours later and counted buys as trades.
    """
    trades = closed_round_trips(fills, config)

    def pct(v):
        return round(v * 100, 2)

    n = len(trades)
    out: Dict[str, Any] = {'closed_trades': n}
    if not n:
        return out
    wins = [t['ret'] for t in trades if t['ret'] > 0]
    losses = [t['ret'] for t in trades if t['ret'] <= 0]
    days = [t['days'] for t in trades if t['days'] is not None]
    out.update({
        'win_rate_pct': round(100.0 * len(wins) / n, 1),
        'avg_win_pct': pct(sum(wins) / len(wins)) if wins else None,
        'avg_loss_pct': pct(sum(losses) / len(losses)) if losses else None,
        'expectancy_pct': pct(sum(t['ret'] for t in trades) / n),
        'avg_hold_days': round(sum(days) / len(days), 2) if days else None,
        'exits': {tag: {'n': sum(1 for t in trades if t['exit'] == tag),
                        'avg_return_pct': pct(sum(t['ret'] for t in trades if t['exit'] == tag) /
                                              sum(1 for t in trades if t['exit'] == tag))}
                  for tag in sorted({t['exit'] for t in trades})},
    })
    return out


def check_proposals(plan: Dict[str, Any], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The AI's suggestions that pass the rules (see ADJUSTABLE), with the
    current value taken from config.json rather than the AI's own claim, and
    the rest with the reason each was set aside."""
    risk = config.get('risk_limits') or {}
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for adj in (plan.get('parameter_adjustments') or [])[:10]:
        if not isinstance(adj, dict):
            continue
        target = str(adj.get('target') or '').strip()
        bounds = ADJUSTABLE.get(target)
        current = risk.get(target.split('.', 1)[-1]) if bounds else None
        try:
            proposed = float(adj.get('proposed'))
        except (TypeError, ValueError):
            proposed = None
        why = None
        if not bounds:
            why = 'not a setting the review may change'
        elif any(a['target'] == target for a in accepted):
            why = 'suggested twice'
        elif not isinstance(current, (int, float)) or current <= 0:
            why = 'not set in config.json'
        elif proposed is None:
            why = 'no number given'
        elif not bounds[0] <= proposed <= bounds[1]:
            why = f'outside the allowed range {bounds[0]} to {bounds[1]}'
        elif abs(proposed - current) < 1e-9:
            why = 'the same as the current value'
        elif abs(proposed / current - 1) > MAX_RELATIVE_CHANGE:
            why = 'a change of more than half the current value'
        elif len(accepted) >= MAX_SUGGESTIONS:
            why = f'more than {MAX_SUGGESTIONS} suggestions'
        if why:
            rejected.append({'target': target or '?', 'proposed': adj.get('proposed'), 'reason': why})
            continue
        accepted.append({'target': target, 'current': float(current), 'proposed': proposed,
                         'reasoning': str(adj.get('reasoning') or '')[:300]})
    for act in plan.get('symbol_actions') or []:
        if isinstance(act, dict):
            rejected.append({'target': f"{act.get('action', '?')} {act.get('symbol', '?')}",
                             'proposed': None, 'reason': 'symbol changes are not part of the review'})
    return accepted, rejected


def describe(review: Dict[str, Any]) -> str:
    """One line for the log."""
    if review.get('skipped'):
        return f"skipped: {review['skipped']}"
    made = [f"{s['target']} {s['current']} -> {s['proposed']}" for s in review.get('suggestions') or []]
    return (f"{len(made)} suggestion(s) {'; '.join(made)}" if made else 'no changes suggested') + \
        (f"; {len(review.get('rejected') or [])} set aside" if review.get('rejected') else '')
