"""Proposes new symbols for the trading universe. Proposals become
operator escalations — the agent never trades an unapproved name."""
from typing import Any, Dict, List, Optional, Set


def propose_candidates(tracked: Set[str], movers: List[Dict[str, Any]],
                       news_texts: List[str], threshold_pct: float = 3.0,
                       known_universe: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    seen, out = set(), []
    for m in movers:
        sym = (m.get('symbol') or '').upper()
        pct = m.get('change_pct') or 0
        if sym and sym not in tracked and sym not in seen and abs(pct) >= threshold_pct:
            seen.add(sym)
            out.append({'symbol': sym, 'reason': f"Moved {'+' if pct >= 0 else ''}{pct}% today while untracked"})
    for sym, name in (known_universe or {}).items():
        if sym in tracked or sym in seen:
            continue
        for text in news_texts:
            if sym in text or (name and name.lower() in text.lower()):
                seen.add(sym)
                out.append({'symbol': sym, 'reason': f"In the news: {text[:80]}"})
                break
    return out
