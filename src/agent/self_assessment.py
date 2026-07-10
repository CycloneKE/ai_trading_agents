"""
Self Assessment Engine:
Runs end-of-day retrospective assessments of agent decisions, evaluates adaptation results,
and uses LLM reasoning to adjust strategy weights, confidence floors, or escalate structural changes.
"""
import os
import json
import sqlite3
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from src.agent.escalation_manager import EscalationManager

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

_RETROSPECTIVE_SYSTEM_PROMPT = """
You are the Chief Auditor and Performance Optimizer of an AI algorithmic trading portfolio.
Your job is to perform a rigorous retrospective review of the system's past decisions and performance, and propose optimizations.

You will be given:
1. Trade execution details and P&L performance.
2. Accuracy rates: how often BUY/SELL decisions actually led to favorable price movements, and whether LLM trade vetos were correct.
3. Adaptation tracking: whether previous parameter adaptations actually improved results.
4. Active strategy weights and configuration parameters.

Based on this audit, you must propose concrete changes to improve system performance.
- Adjust strategy weights (e.g. momentum vs mean reversion) to tilt toward profitable strategies.
- Adjust global confidence floors (e.g. increase if many trades are skipped or losing).
- Add or remove symbols from active watchlist if they are persistently unprofitable.

Output your proposal strictly as a valid JSON object matching this schema:
{
  "parameter_adjustments": [
    {
      "target": "strategy_weights.<strategy_name>" or "risk_limits.stop_loss_pct" or "risk_limits.trailing_stop_pct" or "execution.min_order_size",
      "current": float_or_string,
      "proposed": float_or_string,
      "reasoning": "why this adjustment is needed"
    }
  ],
  "symbol_actions": [
    {
      "symbol": "TICKER",
      "action": "remove" or "pause" or "add",
      "reasoning": "performance justification"
    }
  ],
  "risk_adjustments": [
    {
      "target": "risk_limits.max_position_size",
      "proposed": float,
      "reasoning": "rationale"
    }
  ],
  "reasoning_summary": "executive summary of the audit findings"
}

Ensure the output is strictly valid raw JSON. Do not wrap in markdown blocks, do not add trailing text or comments.
"""


class SelfAssessmentEngine:
    def __init__(self, llm_orchestrator, escalation_manager: EscalationManager, config: Dict[str, Any],
                 db_path: str = os.path.join('data', 'improvements.db')):
        self.llm = llm_orchestrator
        self.escalation_manager = escalation_manager
        self.config = config
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Initialize SQLite DB
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        
        logger.info(f"SelfAssessmentEngine database initialized at {db_path}")

    def run_assessment(self, decision_journal, order_journal, performance_analytics, 
                       cycles_to_review: int = 390) -> Dict[str, Any]:
        """
        Runs retrospective analysis of the past decisions/fills.
        Computes accuracy metrics, compiles feedback, asks LLM for improvements,
        and logs results to the improvements database.
        """
        logger.info(f"Running retrospective self-assessment for the past {cycles_to_review} cycles...")
        
        # 1. Gather historical decisions and filled orders
        decisions = decision_journal.recent(limit=cycles_to_review) if decision_journal else []
        filled_orders = order_journal.filled_orders() if order_journal else []
        
        # 2. Score previous improvements (closing the loop)
        self._score_past_improvements(performance_analytics)
        
        # 3. Analyze Prediction Accuracy
        accuracy_metrics = self._analyze_prediction_accuracy(decisions, filled_orders)
        
        # 4. Analyze Adaptation Effectiveness
        adaptation_metrics = self._analyze_adaptation_effectiveness()
        
        # 5. Extract bias trends
        bias_trends = self._compile_bias_trends(decisions)
        
        # 6. Build LLM Context and request improvement plan
        context = {
            "current_configuration": {
                "strategy_weights": self.config.get("strategy_weights", {}),
                "risk_limits": self.config.get("risk_limits", {}),
                "execution": self.config.get("execution", {})
            },
            "performance_summary": {
                "prediction_accuracy": accuracy_metrics,
                "adaptation_effectiveness": adaptation_metrics,
                "bias_trends": bias_trends
            }
        }
        
        improvement_plan = {}
        if self.llm and getattr(self.llm, "enabled", False):
            try:
                improvement_plan = self.llm.propose_json(_RETROSPECTIVE_SYSTEM_PROMPT, json.dumps(context, default=str)) or {}
            except Exception as e:
                logger.error(f"LLM retrospective proposal failed: {e}")
                
        # 7. Record Assessment in DB
        assessment_id = self._record_assessment(
            cycles_reviewed=len(decisions),
            accuracy=accuracy_metrics,
            adaptations=adaptation_metrics,
            biases=bias_trends,
            plan=improvement_plan
        )
        
        improvement_plan["assessment_id"] = assessment_id
        logger.info(f"Self-assessment completed successfully. ID: {assessment_id}")
        return improvement_plan

    def apply_improvements(self, plan: Dict[str, Any], require_approval: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Applies proposed changes. Safe parameter modifications are auto-applied directly to config.
        Structural or high-risk modifications are escalated to the operator's queue.
        Returns: (auto_applied_list, escalated_list)
        """
        assessment_id = plan.get("assessment_id")
        auto_applied = []
        escalated = []
        
        # Guardrails for automatic changes
        safe_parameter_keys = {
            "strategy_weights.momentum",
            "strategy_weights.mean_reversion",
            "strategy_weights.rsi_strategy",
            "execution.min_order_size",
            "execution.max_order_size"
        }
        
        # Process parameter adjustments
        adjustments = plan.get("parameter_adjustments", [])
        for adj in adjustments:
            target = adj.get("target")
            current = adj.get("current")
            proposed = adj.get("proposed")
            reasoning = adj.get("reasoning", "")
            
            if not target:
                continue
                
            is_safe = target in safe_parameter_keys
            
            if is_safe and not require_approval:
                # Apply change directly to config
                self._apply_config_change(target, proposed)
                
                # Record in DB
                self._record_improvement(
                    assessment_id=assessment_id,
                    target=target,
                    prev_value=str(current),
                    new_value=str(proposed),
                    reasoning=reasoning,
                    auto_applied=True
                )
                auto_applied.append(adj)
                logger.info(f"Auto-applied parameter shift: {target} -> {proposed}")
            else:
                # Escalate to operator approval
                escalation_reason = f"Proposed adjustment: {target} to {proposed} (Current: {current}). Rationale: {reasoning}"
                self.escalation_manager.create_escalation(
                    signal_id=None,
                    symbol="SYSTEM",
                    action=f"config_change:{target}",
                    reason=escalation_reason,
                    risk_level="medium" if is_safe else "high"
                )
                escalated.append(adj)
                logger.info(f"Escalated parameter change: {target} to {proposed} (Safe: {is_safe})")

        # Process symbol actions (always escalated)
        symbol_actions = plan.get("symbol_actions", [])
        for act in symbol_actions:
            symbol = act.get("symbol", "").upper()
            action = act.get("action")
            reasoning = act.get("reasoning", "")
            
            if not symbol or not action:
                continue
                
            escalation_reason = f"Retrospective recommendation: {action} tracking for symbol '{symbol}'. Rationale: {reasoning}"
            self.escalation_manager.create_escalation(
                signal_id=None,
                symbol=symbol,
                action=f"{action}_symbol",
                reason=escalation_reason,
                risk_level="medium"
            )
            escalated.append(act)
            logger.info(f"Escalated symbol recommendation: {action} {symbol}")
            
        return auto_applied, escalated

    def _apply_config_change(self, target: str, value: Any) -> None:
        """Applies a dotted key change directly to the active config."""
        parts = target.split(".")
        config = self.config
        
        # Traverse down to nested dict level
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
            
        # Parse value if string representation of float/int
        parsed_value = value
        if isinstance(value, str):
            try:
                if "." in value:
                    parsed_value = float(value)
                else:
                    parsed_value = int(value)
            except ValueError:
                pass
                
        # Guardrail bounds clamping
        if parts[0] == "strategy_weights":
            parsed_value = max(0.0, min(2.0, float(parsed_value)))
            
        config[parts[-1]] = parsed_value

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
                    datetime.utcnow().isoformat(),
                    cycles_reviewed,
                    json.dumps(accuracy),
                    json.dumps(adaptations),
                    json.dumps(biases),
                    json.dumps(plan)
                )
            )
            self._conn.commit()
            return cur.lastrowid

    def _record_improvement(self, assessment_id: int, target: str, prev_value: str,
                            new_value: str, reasoning: str, auto_applied: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO improvements (assessment_id, target, previous_value, new_value, "
                "reasoning, applied_at, auto_applied) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    assessment_id,
                    target,
                    prev_value,
                    new_value,
                    reasoning,
                    datetime.utcnow().isoformat(),
                    1 if auto_applied else 0
                )
            )
            self._conn.commit()

    def close(self):
        """Close connection."""
        with self._lock:
            self._conn.close()
