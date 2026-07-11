"""Long-Term Sleeve dashboard view: holdings, pending accumulation tickets
(with veto flags surfaced), and cumulative dividend income."""
from typing import Any, Dict, List


def build_sleeve_view(holdings: Dict[str, Dict[str, Any]], pending_tickets: List[Dict[str, Any]],
                      dividend_history: List[Dict[str, Any]], nse_capital_kes: float,
                      capital_split_pct: float) -> Dict[str, Any]:
    sleeve_tickets = [t for t in pending_tickets if t.get('book') == 'long_term']
    total_received = sum(d['amount_kes'] for d in dividend_history
                         if d.get('status') in ('received', 'swept'))
    return {
        'capital_target_kes': round(nse_capital_kes * capital_split_pct, 2),
        'holdings': [{'symbol': symbol, **data} for symbol, data in holdings.items()],
        'pending_tickets': [
            {
                'symbol': t['symbol'], 'quantity': t['quantity'],
                'suggested_limit_price': t.get('suggested_limit_price'),
                'rationale': t.get('rationale'),
                'veto_flagged': 'VETO_FLAG' in (t.get('rationale') or ''),
                'veto_unavailable': 'VETO_UNAVAILABLE' in (t.get('rationale') or ''),
                'llm_reasoning': t.get('llm_reasoning'),
            }
            for t in sleeve_tickets
        ],
        'cumulative_dividends_kes': round(total_received, 2),
    }
