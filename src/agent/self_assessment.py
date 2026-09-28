"""
Self Assessment Engine: a periodic AI review of the agent's own results.

It is advisory. It used to post whatever the AI suggested to the operator's
approval queue, and to apply some suggestions by itself: settings that do
not exist (strategy_weights, which nothing reads), order sizes, placeholder
tickers such as ABC or XYZ, and contradictory stop changes, all made with no
trades to judge by. Approving one did not apply it either. Now:

- the AI is asked only once the period reviewed holds at least
  `self_assessment.min_trades` executed trades (20); before that the review
  records that it waited, and costs no AI call;
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
DEFAULT_MIN_TRADES = 20

_RETROSPECTIVE_SYSTEM_PROMPT = """
You review an algorithmic trading system's recent results and may suggest changes to its stop-loss
settings, only where the results clearly support them. Suggest nothing when they do not.

You are given how many trades were executed, how often buys and sells were followed by a favourable
move, how often the AI's trade vetoes were right, and the current stop settings. Only these settings
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
        """Review the latest decisions and, when there is evidence enough,
        ask the AI for suggestions, which are checked and recorded. Nothing
        is applied. Returns the review (see latest_review)."""
        decisions = decision_journal.recent(limit=cycles_to_review) if decision_journal else []
        filled_orders = order_journal.filled_orders() if order_journal else []
        self._score_past_improvements(performance_analytics)
        accuracy_metrics = self._analyze_prediction_accuracy(decisions, filled_orders)
        adaptation_metrics = self._analyze_adaptation_effectiveness()
        bias_trends = self._compile_bias_trends(decisions)

        executed = sum(1 for d in decisions if d.get('executed'))
        min_trades = int((self.config.get('self_assessment') or {}).get('min_trades', DEFAULT_MIN_TRADES))
        review: Dict[str, Any] = {'executed_trades': executed, 'min_trades': min_trades, 'skipped': None,
                                  'suggestions': [], 'rejected': [], 'summary': ''}
        if executed < min_trades:
            review['skipped'] = (f"{executed} trade{'s' if executed != 1 else ''} in the period reviewed; "
                                 f"the review waits for {min_trades} before suggesting changes")
        elif not (self.llm and getattr(self.llm, "enabled", False)):
            review['skipped'] = 'no AI service is set up'
        else:
            context = {
                "executed_trades": executed,
                "current_settings": {t: (self.config.get('risk_limits') or {}).get(t.split('.', 1)[1])
                                     for t in ADJUSTABLE},
                "prediction_accuracy": accuracy_metrics,
                "adaptation_effectiveness": adaptation_metrics,
                "bias_trends": bias_trends,
            }
            plan = None
            try:
                plan = self.llm.propose_json(_RETROSPECTIVE_SYSTEM_PROMPT, json.dumps(context, default=str))
            except Exception as e:
                logger.error(f"AI self-review failed: {e}")
            if isinstance(plan, dict):
                review['suggestions'], review['rejected'] = check_proposals(plan, self.config)
                review['summary'] = str(plan.get('reasoning_summary') or '')[:500]
            else:
                review['skipped'] = 'the AI did not answer'

        review['assessment_id'] = self._record_assessment(
            cycles_reviewed=len(decisions), accuracy=accuracy_metrics,
            adaptations=adaptation_metrics, biases=bias_trends, plan=review)
        logger.info(f"AI self-review: {describe(review)}")
        return review

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
                                 ('executed_trades', 'min_trades', 'skipped', 'suggestions', 'rejected', 'summary')}}

    def _analyze_prediction_accuracy(self, decisions: List[Dict[str, Any]], fills: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates accuracy rates: how often buy/sell signals led to positive returns,
        and whether LLM vetos were correct.
        """
        # Create symbol price lookup helper from recent decisions
        price_by_symbol = {}
        for d in decisions:
            sym = d.get('symbol')
            price = d.get('price')
            if sym and price:
                if sym not in price_by_symbol:
                    price_by_symbol[sym] = []
                price_by_symbol[sym].append((d.get('ts'), price))

        # Core metrics
        total_buys = 0
        correct_buys = 0
        total_sells = 0
        correct_sells = 0
        veto_count = 0
        correct_vetos = 0

        # Review decisions
        for d in decisions:
            symbol = d.get('symbol')
            action = d.get('action')
            price = d.get('price')
            ts = d.get('ts')
            skip_reason = d.get('skip_reason')

            if not symbol or not price:
                continue

            # Compare decision price against the future prices of the symbol
            future_prices = [p for t, p in price_by_symbol.get(symbol, []) if t > ts]
            if not future_prices:
                continue
            final_future_price = future_prices[0] # next recorded cycle price


            if d.get('executed'):
                if action == 'buy':
                    total_buys += 1
                    if final_future_price > price:
                        correct_buys += 1
                elif action == 'sell':
                    total_sells += 1
                    if final_future_price < price:
                        correct_sells += 1
            elif skip_reason == 'llm_veto':
                veto_count += 1
                # Veto was correct if executing the trade would have lost money (or held flat)
                # Buy veto correct: price didn't go up
                # Sell veto correct: price didn't go down
                orig_signal_action = d.get('per_strategy_json', '') # fallbacks
                is_buy_signal = 'buy' in str(orig_signal_action).lower()
                
                if is_buy_signal:
                    if final_future_price <= price:
                        correct_vetos += 1
                else:
                    if final_future_price >= price:
                        correct_vetos += 1

        return {
            "buy_signals_count": total_buys,
            "buy_accuracy_pct": (correct_buys / total_buys * 100) if total_buys > 0 else None,
            "sell_signals_count": total_sells,
            "sell_accuracy_pct": (correct_sells / total_sells * 100) if total_sells > 0 else None,
            "llm_vetos_count": veto_count,
            "veto_accuracy_pct": (correct_vetos / veto_count * 100) if veto_count > 0 else None
        }

    def _analyze_adaptation_effectiveness(self) -> List[Dict[str, Any]]:
        """Analyzes outcome scores of previous adaptations."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT target, outcome_score, outcome_notes, applied_at FROM improvements "
                "WHERE outcome_score IS NOT NULL ORDER BY applied_at DESC LIMIT 20"
            )
            cur.row_factory = sqlite3.Row
            return [dict(r) for r in cur.fetchall()]

    def _compile_bias_trends(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate bias checks over the reviewed decisions."""
        bias_counts = {}
        total_decisions = len(decisions)
        
        for d in decisions:
            skip_reason = d.get('skip_reason')
            if skip_reason == 'bias_downgrade':
                bias_counts['directional_bias'] = bias_counts.get('directional_bias', 0) + 1

        return {
            "total_decisions": total_decisions,
            "bias_mitigations_triggered": bias_counts,
            "bias_rate_pct": (sum(bias_counts.values()) / total_decisions * 100) if total_decisions > 0 else 0
        }

    def _score_past_improvements(self, performance_analytics) -> None:
        """
        Evaluates performance after previous parameter adjustments to calculate
        an improvement outcome score (ratio of Sharpe ratio improvement).
        """
        if not performance_analytics:
            return
            
        with self._lock:
            # Find improvements applied over 12 hours ago with no outcome score
            cur = self._conn.execute(
                "SELECT id, target, new_value, applied_at FROM improvements "
                "WHERE outcome_score IS NULL AND datetime(applied_at) < datetime('now', '-12 hours')"
            )
            unscored = cur.fetchall()
            if not unscored:
                return

            # Retrieve overall performance report
            perf_report = performance_analytics.generate_performance_report(period='7D')
            metrics = perf_report.get('metrics', {})
            sharpe = metrics.get('sharpe_ratio', 1.0)
            win_rate = metrics.get('win_rate', 0.5)
            
            # Simple scoring logic: positive Sharpe or positive win rate yields a positive score
            score = max(-1.0, min(1.0, (sharpe - 1.0) / 2.0))
            
            for imp_id, target, val, applied_at in unscored:
                notes = f"Sharpe ratio at assessment: {sharpe:.2f}. Win rate: {win_rate:.1%}"
                self._conn.execute(
                    "UPDATE improvements SET outcome_score = ?, outcome_notes = ? WHERE id = ?",
                    (score, notes, imp_id)
                )
            self._conn.commit()

    def _record_assessment(self, cycles_reviewed: int, accuracy: Dict[str, Any],
                           adaptations: List[Dict[str, Any]], biases: Dict[str, Any],
                           plan: Dict[str, Any]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO assessments (ts, cycles_reviewed, prediction_accuracy_json, "
                "adaptation_effectiveness_json, bias_trends_json, improvement_plan_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    cycles_reviewed,
                    json.dumps(accuracy),
                    json.dumps(adaptations),
                    json.dumps(biases),
                    json.dumps(plan)
                )
            )
            self._conn.commit()
            return cur.lastrowid

    def close(self):
        """Close connection."""
        with self._lock:
            self._conn.close()


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
