"""Nightly Growth & Self-Assessment Cycle.

Executes post-market closed-loop diagnostics:
1. Calculates strategy attribution and execution metrics from the order journal.
2. Identifies outperforming vs underperforming strategies and updates weights into data/config_overlay.json.
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
from src.agent.self_assessment import _write_overlay_key, apply_overlay
from src.agent.strategy_attribution import compute_attribution
from src.agent.universe_manager import effective_universe, propose_removals, compute_symbol_stats
from src.utils.paths import DATA_DIR
from src.utils.timeutil import utc_now_iso

logger = logging.getLogger(__name__)

_compute_symbol_stats = compute_symbol_stats


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

    overlay_path = os.path.join(data_path, 'config_overlay.json')
    apply_overlay(config, overlay_path)

    # 1. Initialize journals & managers
    order_journal = OrderJournal(db_path=os.path.join(data_path, 'order_journal.db'))
    decision_journal = DecisionJournal(db_path=os.path.join(data_path, 'decision_journal.db'))
    escalation_mgr = EscalationManager(db_path=os.path.join(data_path, 'escalations.db'))

    # 2. Strategy Attribution
    fills = order_journal.filled_orders()
    attribution = compute_attribution(fills)

    # 3. Dynamic Strategy Weight Adaptation
    strategies_node = config.setdefault('strategies', {})
    adjusted_weights: Dict[str, float] = {}

    for strat_name, strat_cfg in strategies_node.items():
        current_weight = float((strat_cfg or {}).get('weight', 1.0))
        strat_stat = attribution.get(strat_name)

        if strat_stat and strat_stat.get('closed_trades', 0) >= 1:
            pnl = float(strat_stat.get('realized_pnl', 0.0))
            win_rate = float(strat_stat.get('win_rate', 0.0))

            if pnl > 0 and win_rate >= 0.5:
                # Outperforming strategy: reward with 10% weight increase
                new_weight = round(min(2.0, current_weight * 1.10), 3)
            elif pnl < 0:
                # Underperforming strategy: throttle by 10%
                new_weight = round(max(0.1, current_weight * 0.90), 3)
            else:
                new_weight = current_weight

            if new_weight != current_weight and not dry_run:
                _write_overlay_key(f"strategies.{strat_name}.weight", new_weight, overlay_path)
                logger.info(f"GrowthCycle: adjusted {strat_name} weight: {current_weight} -> {new_weight}")

            adjusted_weights[strat_name] = new_weight
        else:
            adjusted_weights[strat_name] = current_weight

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
        'strategies': adjusted_weights,
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
    print("\nStrategy Attribution & Weights:")
    for strat, wt in report['strategies'].items():
        attr = report['attribution'].get(strat, {})
        pnl = attr.get('realized_pnl', 0.0)
        trades = attr.get('closed_trades', 0)
        win_rate = attr.get('win_rate', 0.0)
        print(f"  {strat:<16} | Weight: {wt:.2f} | PnL: ${pnl:>9,.2f} | Trades: {trades:>3} | Win Rate: {win_rate:>5.1%}")

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
