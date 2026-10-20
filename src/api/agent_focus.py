"""Three-lane live view of agent behaviour: holding / reviewing / traded."""
from typing import Any, Dict, List


def build_agent_focus(positions: List[Dict[str, Any]], decisions: List[Dict[str, Any]],
                      cash: float, equity: float) -> Dict[str, Any]:
    holding = [{'symbol': p.get('symbol'), 'quantity': p.get('quantity'),
                'unrealized_pl_pct': p.get('unrealized_pl_pct')} for p in positions]
    reviewing, traded, seen = [], [], set()
    for d in decisions:  # journal returns newest first; keep first per symbol
        sym = d.get('symbol')
        if sym in seen:
            continue
        seen.add(sym)
        if d.get('executed'):
            traded.append({'symbol': sym, 'action': (d.get('action') or '').upper(),
                           'price': d.get('price'), 'ts': d.get('ts')})
        else:
            reviewing.append({'symbol': sym, 'action': (d.get('action') or 'hold').upper(),
                              'reason': d.get('skip_reason') or 'signal below threshold',
                              'ts': d.get('ts')})
    deployed_pct = round((1 - cash / equity) * 100, 1) if equity else 0.0
    return {'holding': holding, 'reviewing': reviewing, 'traded': traded,
            'cash': {'cash': cash, 'equity': equity, 'deployed_pct': deployed_pct}}
