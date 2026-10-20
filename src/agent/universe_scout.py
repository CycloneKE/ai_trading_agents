"""Proposes new symbols for the trading universe. Proposals become
operator escalations — the agent never trades an unapproved name."""
from typing import Any, Dict, List, Optional, Set

DEFAULT_US_CANDIDATE_POOL = [
    'NVDA', 'AMD', 'META', 'AMZN', 'GOOGL', 'MSFT', 'AAPL', 'TSLA',
    'SPY', 'QQQ', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'SMH', 'IWM'
]


def propose_candidates(tracked: Set[str], movers: List[Dict[str, Any]],
                       news_texts: List[str], threshold_pct: float = 3.0,
                       known_universe: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for m in movers:
        sym = (m.get('symbol') or '').upper()
        pct = m.get('change_pct') or 0
        if sym and sym not in tracked and sym not in seen and abs(pct) >= threshold_pct:
            seen.add(sym)
            cand = {'symbol': sym, 'reason': f"Moved {'+' if pct >= 0 else ''}{pct}% today while untracked"}
            if 'price' in m or 'close' in m:
                cand['price'] = float(m.get('price') or m.get('close') or 0.0)
            if 'avg_dollar_volume' in m or 'volume' in m:
                cand['avg_dollar_volume'] = float(m.get('avg_dollar_volume') or (float(m.get('volume') or 0) * cand.get('price', 1.0)))
            out.append(cand)
    for sym, name in (known_universe or {}).items():
        if sym in tracked or sym in seen:
            continue
        for text in news_texts:
            if sym in text or (name and name.lower() in text.lower()):
                seen.add(sym)
                out.append({'symbol': sym, 'reason': f"In the news: {text[:80]}"})
                break
    return out


def scout_us_candidates(tracked: Set[str], quotes: Dict[str, Dict[str, Any]],
                        threshold_pct: float = 3.0,
                        candidate_pool: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Scan candidate US equities/ETFs for significant percentage moves."""
    pool = candidate_pool or DEFAULT_US_CANDIDATE_POOL
    movers = []
    for sym in pool:
        s = sym.upper()
        if s in tracked:
            continue
        q = quotes.get(s) or quotes.get(sym) or {}
        pct = q.get('change_pct')
        if pct is None and q.get('close') and q.get('open'):
            try:
                pct = ((float(q['close']) - float(q['open'])) / float(q['open'])) * 100.0
            except ZeroDivisionError:
                pct = 0.0
        if pct is not None and abs(pct) >= threshold_pct:
            movers.append({
                'symbol': s,
                'change_pct': round(float(pct), 2),
                'price': float(q.get('price') or q.get('close') or 0.0),
                'avg_dollar_volume': float(q.get('avg_dollar_volume') or (float(q.get('volume') or 0) * float(q.get('price') or q.get('close') or 1.0))),
            })
    return propose_candidates(tracked, movers, [], threshold_pct=threshold_pct)
