"""View builder for continuous growth, self-assessment, and Champion vs Challenger tracking."""

from typing import Any, Dict, List, Optional


def build_growth_view(config: Optional[Dict[str, Any]] = None,
                      growth_report: Optional[Dict[str, Any]] = None,
                      backtest_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compile Champion vs Challenger and learning metrics into a dashboard-ready view."""
    cfg = config or {}
    report = growth_report or {}
    bt = backtest_results or {}

    champion_method = cfg.get('ensemble_method', 'adaptive_confidence')
    strategies = cfg.get('strategies', {})
    weights = {k: float(v.get('weight', 1.0)) for k, v in strategies.items() if isinstance(v, dict)}

    champion = {
        'ensemble_method': champion_method,
        'status': 'active',
        'weights': weights,
    }

    challengers: List[Dict[str, Any]] = []
    results = bt.get('results', [])
    for res in results:
        window_name = res.get('window', '')
        if 'regime_routed' in window_name:
            acceptance = bt.get('acceptance', {})
            passed = bt.get('passed', False)
            challengers.append({
                'ensemble_method': 'regime_routed',
                'status': 'cleared_for_promotion' if passed else 'in_evaluation',
                'metrics': {
                    'sharpe': res.get('sharpe'),
                    'total_return': res.get('total_return'),
                    'max_drawdown': res.get('max_drawdown'),
                    'correlation_to_spy': res.get('correlation_to_spy'),
                    'trades': res.get('trades'),
                    'win_rate': res.get('win_rate'),
                },
                'acceptance_gate': {
                    'passed': passed,
                    'checks': acceptance,
                }
            })

    learning_cycle = {
        'last_run': report.get('timestamp'),
        'fills_analyzed': report.get('fills_analyzed', 0),
        'tracked_symbols_count': report.get('tracked_symbols_count', 0),
        'active_attributions': report.get('attribution', {}),
        'strategy_weights': report.get('strategies', {}),
        'removal_proposals': report.get('removals', []),
    }

    return {
        'champion': champion,
        'challengers': challengers,
        'learning_cycle': learning_cycle,
    }
