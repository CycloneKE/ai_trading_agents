"""Daily bars and trade markers for the dashboard's price charts.

The symbol panel used to chart the prices at which the agent happened to
make decisions: minute-by-minute points with moving averages worked out over
those points, labelled in dollars even for NSE stocks. A chart has to show
the market itself, so this serves real daily bars (yfinance for US and
crypto, the NSE's own ticker and price lists for NSE, never synthetic seed
data) with the agent's fills marked on them.
"""
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.connectors.nse_scraper import REAL_NSE_SOURCES

STOP_STRATEGIES = {'stop_loss', 'trailing_stop'}


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _bar(day: str, o, h, l, c, v=0) -> Dict[str, Any]:
    """One bar, its high and low widened to contain its open and close."""
    c = float(c)
    o = o if o else c
    h = max(x for x in (h, o, c) if x)
    l = min(x for x in (l, o, c) if x)
    return {'time': day, 'open': round(o, 4), 'high': round(h, 4), 'low': round(l, 4),
            'close': round(c, 4), 'volume': int(float(v or 0))}


def nse_bars(symbol: str, csv_dir: Path, days: int = 250) -> List[Dict[str, Any]]:
    """Real daily bars from the scraper's CSV, oldest first; no seed data."""
    path = Path(csv_dir) / f"{symbol.upper()}.csv"
    if not path.exists():
        return []
    by_day: Dict[str, Dict[str, Any]] = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            close = _num(r.get('close'))
            day = (r.get('date') or '')[:10]
            if r.get('source') not in REAL_NSE_SOURCES or not close or len(day) != 10:
                continue
            by_day[day] = _bar(day, _num(r.get('open')), _num(r.get('high')),
                               _num(r.get('low')), close, r.get('volume'))
    return [by_day[d] for d in sorted(by_day)][-days:]


def yfinance_bars(symbol: str, period: str = '2y') -> List[Dict[str, Any]]:
    """Daily bars from yfinance, the live feed's own source, oldest first."""
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval='1d', auto_adjust=True)
    if df is None or df.empty:
        return []
    return [_bar(idx.date().isoformat(), _num(r['Open']), _num(r['High']), _num(r['Low']),
                 float(r['Close']), r.get('Volume', 0))
            for idx, r in df.iterrows() if _num(r['Close'])]


def markers(fills: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What each fill did to the position: buy, add, trim, sell or stop.

    Fills are oldest first: time (ISO), side, quantity, price and optionally
    strategy. An add is a buy while already holding; a trim is a sell that
    leaves shares; a stop is a sell by a stop-loss or trailing stop.
    """
    held, out = 0.0, []
    for f in fills:
        qty, price = float(f.get('quantity') or 0), float(f.get('price') or 0)
        if qty <= 0 or price <= 0 or f.get('side') not in ('buy', 'sell'):
            continue
        if f['side'] == 'buy':
            kind = 'add' if held > 1e-9 else 'buy'
            held += qty
        else:
            if (f.get('strategy') or '') in STOP_STRATEGIES:
                kind = 'stop'
            else:
                kind = 'trim' if held - qty > 1e-9 else 'sell'
            held = max(held - qty, 0.0)
        out.append({'time': (f.get('time') or '')[:10], 'kind': kind, 'side': f['side'],
                    'price': round(price, 4), 'quantity': qty,
                    'strategy': f.get('strategy') or ''})
    return out


def journal_fills(order_journal, symbol: str) -> List[Dict[str, Any]]:
    """The US/crypto book's fills for a symbol, oldest first."""
    if order_journal is None:
        return []
    rows = [r for r in order_journal.orders_for_symbol(symbol, limit=500)
            if r.get('status') == 'filled' and r.get('filled_quantity')]
    rows.sort(key=lambda r: r.get('created_at') or '')
    return [{'time': r.get('updated_at') or r.get('created_at'), 'side': r.get('side'),
             'quantity': r.get('filled_quantity'), 'price': r.get('filled_avg_price'),
             'strategy': r.get('strategy')} for r in rows]


def tradingview_symbol(symbol: str, market: str) -> str:
    """The symbol as TradingView names it, for the 'open full chart' link."""
    s = symbol.upper()
    if market == 'nse':
        return f'NSEKE:{s}'
    if market == 'crypto':
        return f"COINBASE:{s.replace('-', '').replace('/', '')}"
    return s
