"""Universe Manager: Tracks active universe and proposes symbol removals for underperformers."""
from typing import Any, Dict, List, Optional, Set


def effective_universe(config: Optional[Dict[str, Any]],
                       watchlist: Optional[List[Dict[str, Any]]] = None,
                       market: str = 'us') -> List[str]:
    """Returns effective tracked universe combining static config and approved watchlist items."""
    config = config or {}
    watchlist = watchlist or []
    
    symbols: List[str] = []
    if market == 'us':
        dm_symbols = (config.get('data_manager') or {}).get('symbols') or []
        # Filter for US equities/ETFs/crypto/forex or standard tickers
        symbols.extend(dm_symbols)
        for item in watchlist:
            if isinstance(item, dict):
                sym = item.get('symbol')
                mkt = (item.get('market') or '').lower()
                if sym and mkt in ('us', 'crypto', 'forex'):
                    if sym not in symbols:
                        symbols.append(sym)
    elif market in ('kenyan', 'nse'):
        nse_symbols = (config.get('nse') or {}).get('symbols') or ['SCOM', 'EABL', 'KCB', 'EQTY', 'COOP', 'BAT']
        symbols.extend(nse_symbols)
        for item in watchlist:
            if isinstance(item, dict):
                sym = item.get('symbol')
                mkt = (item.get('market') or '').lower()
                if sym and mkt in ('kenyan', 'nse'):
                    if sym not in symbols:
                        symbols.append(sym)
    return symbols


def compute_symbol_stats(fills: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculates per-symbol execution stats including realized PnL, win rate, and consecutive losses."""
    stats: Dict[str, Dict[str, Any]] = {}
    
    for f in fills:
        sym = f.get('symbol')
        if not sym:
            continue
        s = stats.setdefault(sym, {
            'trades': 0, 'wins': 0, 'losses': 0, 'realized_pnl': 0.0,
            'recent_pnls': []
        })
        pnl = float(f.get('pnl') or f.get('realized_pnl') or 0.0)
        s['trades'] += 1
        s['realized_pnl'] += pnl
        s['recent_pnls'].append(pnl)
        if pnl > 0:
            s['wins'] += 1
        elif pnl < 0:
            s['losses'] += 1

    for sym, s in stats.items():
        # Calculate consecutive losses from most recent trades backwards
        consec = 0
        for pnl in reversed(s['recent_pnls']):
            if pnl < 0:
                consec += 1
            else:
                break  # a win or a break-even trade ends the streak
        s['consecutive_losses'] = consec
        s['win_rate'] = (s['wins'] / s['trades']) if s['trades'] > 0 else 0.0
        
    return stats


def propose_removals(tracked_symbols: Set[str],
                     symbol_stats: Dict[str, Dict[str, Any]],
                     config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Flags assets experiencing persistent consecutive losses or negative return regimes."""
    proposals: List[Dict[str, Any]] = []
    # Thresholds are in the book's currency; override under `universe_pruning`.
    cfg = (config or {}).get('universe_pruning') or {}
    streak_pnl = float(cfg.get('streak_loss_threshold', -100))
    severe_pnl = float(cfg.get('severe_loss_threshold', -500))

    for sym in tracked_symbols:
        stat = symbol_stats.get(sym)
        if not stat:
            continue
        consec = stat.get('consecutive_losses', 0)
        pnl = stat.get('realized_pnl', 0.0)
        trades = stat.get('trades', 0)
        
        if consec >= 4 and pnl < streak_pnl:
            proposals.append({
                'symbol': sym,
                'reason': f"Persistent drawdowns: {consec} consecutive losses with realized PnL of ${pnl:,.2f}",
                'consecutive_losses': consec,
                'realized_pnl': pnl,
                'trades': trades
            })
        elif trades >= 10 and stat.get('win_rate', 0.0) < 0.20 and pnl < severe_pnl:
            proposals.append({
                'symbol': sym,
                'reason': f"Severe underperformance: {stat.get('win_rate', 0.0):.1%} win rate over {trades} trades (${pnl:,.2f} PnL)",
                'consecutive_losses': consec,
                'realized_pnl': pnl,
                'trades': trades
            })
            
    return proposals
