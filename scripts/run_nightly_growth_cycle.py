"""Nightly Growth & Self-Assessment Cycle.

Executes post-market closed-loop diagnostics:
1. Calculates strategy attribution and execution metrics from the order journal.
2. Says which strategies are doing better or worse and what weight it would suggest for each.
   Report only: it changes nothing. It used to write those weights to data/config_overlay.json
   after a single closed trade, +-10% a night, compounding on its own earlier changes; the
   live agent never read that file, so the dashboard showed weights that were not in effect.
   The ensemble's evidence tilt (guardrails.evidence_tilt) already re-weights strategies from
   the same order journal, inside the guardrails (nothing before five closed trades, 0.5x-1.5x).
3. Evaluates tracked symbols for persistent losses, illiquidity, or undefined regimes and escalates removal proposals.
4. Generates a comprehensive growth report for review.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.decision_journal import DecisionJournal
from src.agent.escalation_manager import EscalationManager
from src.agent.order_journal import OrderJournal
from src.agent.guardrails import evidence_tilt
from src.agent.strategy_attribution import compute_attribution
from src.agent.universe_manager import effective_universe, propose_removals, compute_symbol_stats
from src.utils.paths import DATA_DIR
from src.utils.timeutil import utc_now_iso

logger = logging.getLogger(__name__)

_compute_symbol_stats = compute_symbol_stats


def _trade_sharpe(returns: List[float]) -> float:
    """Per-trade Sharpe over the strategy's closed-trade returns, as the
    strategy manager computes it for the evidence tilt."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    return (mean / var ** 0.5) * len(returns) ** 0.5 if var > 0 else 0.0


def run_growth_cycle(config: Optional[Dict[str, Any]] = None,
                     data_dir: Optional[str] = None,
                     dry_run: bool = False) -> Dict[str, Any]:
    """Execute the full growth and learning cycle."""
    data_path = data_dir or str(DATA_DIR)
    os.makedirs(data_path, exist_ok=True)

    config_path = os.path.join(data_path, 'config.json')
    if config is None:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            with open('config/config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)


    # 1. Initialize journals & managers
    order_journal = OrderJournal(db_path=os.path.join(data_path, 'order_journal.db'))
    decision_journal = DecisionJournal(db_path=os.path.join(data_path, 'decision_journal.db'))
    escalation_mgr = EscalationManager(db_path=os.path.join(data_path, 'escalations.db'))

    # 2. Strategy Attribution
    fills = order_journal.filled_orders()
    attribution = compute_attribution(fills)

    # 3. Strategy weights: what is in effect, and what the evidence suggests.
    #    Report only. The suggestion is the same evidence tilt the ensemble
    #    applies live (no change before five closed trades,
    #    then 0.5x to 1.5x by how convincing the record is), so the report can
    #    no longer disagree with what the agent actually does.
    strategies_node = config.get('strategies', {}) or {}
    configured_weights: Dict[str, float] = {}
    suggested_weights: Dict[str, float] = {}
    for strat_name, strat_cfg in strategies_node.items():
        if not isinstance(strat_cfg, dict) or strat_name.startswith('_'):
            continue
        weight = float(strat_cfg.get('weight', 1.0))
        configured_weights[strat_name] = weight
        stat = attribution.get(strat_name) or {}
        trades = int(stat.get('closed_trades', 0) or 0)
        sharpe = _trade_sharpe(stat.get('trade_returns') or [])
        suggested_weights[strat_name] = round(weight * evidence_tilt(trades, sharpe), 3)

    # 4. Symbol Performance & Universe Pruning
    watchlist = escalation_mgr.get_active_watchlist() if escalation_mgr else []
    tracked_us = set(effective_universe(config, watchlist, 'us'))
    tracked_nse = set(effective_universe(config, watchlist, 'kenyan'))
    all_tracked = tracked_us | tracked_nse

    symbol_stats = _compute_symbol_stats(fills)
    removal_proposals = propose_removals(all_tracked, symbol_stats, config)

    # Create escalations for operator review
    if not dry_run and escalation_mgr:
        for r in removal_proposals:
            escalation_mgr.create_escalation(
                signal_id=None,
                symbol=r['symbol'],
                action='remove_symbol',
                reason=r['reason'],
                risk_level='medium'
            )

    # 5. Compile Growth Report
    report = {
        'timestamp': utc_now_iso(),
        'attribution': attribution,
        'strategies': configured_weights,
        'suggested_weights': suggested_weights,
        'weights_note': ('Weights in effect are the configured ones, tilted live by the evidence '
                         'tilt. Suggested weights are that tilt applied to the closed trades so '
                         'far; nothing is changed by this report.'),
        'symbol_stats': symbol_stats,
        'removals': removal_proposals,
        'tracked_symbols_count': len(all_tracked),
        'fills_analyzed': len(fills),
    }

    report_path = os.path.join(data_path, 'nightly_growth_report.json')
    if not dry_run:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n--- Starting Nightly Growth & Self-Assessment Cycle ---")
    report = run_growth_cycle()

    print(f"Timestamp: {report['timestamp']}")
    print(f"Fills Analyzed: {report['fills_analyzed']}")
    print("\nStrategy Attribution & Weights (suggested weights are not applied):")
    for strat, wt in report['strategies'].items():
        attr = report['attribution'].get(strat, {})
        pnl = attr.get('realized_pnl', 0.0)
        trades = attr.get('closed_trades', 0)
        win_rate = attr.get('win_rate', 0.0)
        suggested = report['suggested_weights'].get(strat, wt)
        print(f"  {strat:<16} | Weight: {wt:.2f} | Suggested: {suggested:.2f} | PnL: ${pnl:>9,.2f} "
              f"| Trades: {trades:>3} | Win Rate: {win_rate:>5.1%}")

    print("\nUniverse Removal Proposals:")
    if report['removals']:
        for rem in report['removals']:
            print(f"  [PROPOSAL] {rem['symbol']}: {rem['reason']}")
    else:
        print("  None. All tracked symbols within acceptable performance parameters.")

    print(f"\nWrote full report to data/nightly_growth_report.json")
    print("--- Nightly Growth Cycle Completed Successfully ---\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
