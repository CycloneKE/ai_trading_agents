"""Scores past executed decisions against current prices so the LLM
prompt can carry the agent's own track record ('you were right 60% of
the time on SCOM'). Pure functions — journal in, summary out."""
from typing import Any, Callable, Dict, List, Optional


def score_decisions(decisions: List[Dict[str, Any]],
                    price_lookup: Callable[[str], Optional[float]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    prices: Dict[str, Optional[float]] = {}
    for d in decisions:
        if not d.get('executed') or not d.get('price'):
            continue
        action = (d.get('action') or '').lower()
        if action not in ('buy', 'sell'):
            continue
        sym = d.get('symbol')
        if sym not in prices:
            try:
                prices[sym] = price_lookup(sym)
            except Exception:
                prices[sym] = None
        now = prices[sym]
        if not now:
            continue
        rec = out.setdefault(sym, {'evaluated': 0, 'hits': 0, 'hit_rate': 0.0})
        rec['evaluated'] += 1
        if (action == 'buy' and now > d['price']) or (action == 'sell' and now < d['price']):
            rec['hits'] += 1
    for rec in out.values():
        rec['hit_rate'] = round(rec['hits'] / rec['evaluated'], 3) if rec['evaluated'] else 0.0
    return out


def summary_line(scores: Dict[str, Dict[str, Any]], symbol: str) -> str:
    rec = scores.get(symbol)
    if not rec or not rec['evaluated']:
        return f"No scored history for {symbol} yet — treat the ensemble signal on its merits."
    pct = round(rec['hit_rate'] * 100)
    return (f"Track record on {symbol}: {rec['hits']}/{rec['evaluated']} executed calls "
            f"({pct}%) moved in the traded direction. Weigh your confidence accordingly.")
