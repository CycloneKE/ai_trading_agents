"""The paper accounts, laid open for the dashboard.

Two accounts are traded on paper besides the NSE one (which has its own
page, nse_paper_account.py):

- 'us': the primary broker's account, Alpaca's paper account in this setup.
  It holds US stocks and funds (the core index funds among them) and crypto.
- 'forex': the forex paper book, a simulated account inside the agent.

`build_view` answers, for one of them, what the operator asks of a paper
account: how much is it worth against what it started with, how much of that
is cash, what does it hold and where would each holding be stopped out, what
has it traded, what has it made on closed trades, what is still working at
the broker, and what do its trades cost. Everything comes from the broker's
own figures and the order journal; nothing is estimated except the costs,
which are labelled as modelled.

`PaperEquityLog` keeps one equity reading a day per book, so an account
without a history of its own (the forex book) still gets a curve.
"""
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.agent.cost_model import classify, costs_for, fill_gap_pct
from src.agent.round_trips import closed_round_trips
from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

# `broker` is the broker manager's name for the account's broker; None is the
# primary broker. 'forex_paper' is main.FOREX_BOOK.
BOOKS: Dict[str, Dict[str, Any]] = {
    'us': {'title': 'US and crypto', 'markets': ('us_equity', 'crypto'), 'broker': None,
           'currency': 'USD', 'cost_symbols': {'us_equity': 'AAPL', 'crypto': 'BTC-USD'}},
    'forex': {'title': 'Forex', 'markets': ('forex',), 'broker': 'forex_paper',
              'currency': 'USD', 'cost_symbols': {'forex': 'EUR_USD'}},
}

REGION = {'us_equity': 'US', 'crypto': 'Crypto', 'forex': 'Forex'}
WORKING = ('intent', 'submitted')


def book_of(symbol: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Which paper account trades `symbol`, or None (the NSE account's, or
    an unknown market)."""
    market = classify(symbol, config)
    for name, spec in BOOKS.items():
        if market in spec['markets']:
            return name
    return None


# ---------------------------------------------------------------- history

class PaperEquityLog:
    """One reading a day of a book's cash and equity: the last one written
    that day. Small enough to sit in its own SQLite file in the data
    volume, so a redeploy keeps it."""

    def __init__(self, db_path: str = str(DATA_DIR / 'paper_equity.db')):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_equity ("
            " book TEXT NOT NULL, day TEXT NOT NULL, cash REAL, equity REAL NOT NULL,"
            " updated_at TEXT NOT NULL, PRIMARY KEY (book, day))")
        self._conn.commit()

    def record(self, book: str, cash: float, equity: float, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO paper_equity (book, day, cash, equity, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (book, now.date().isoformat(), float(cash), float(equity), now.isoformat()))
            self._conn.commit()

    def history(self, book: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT day, cash, equity FROM paper_equity WHERE book = ? ORDER BY day",
                (book,)).fetchall()
        return [{'day': d, 'cash': c, 'equity': e} for d, c, e in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def daily_curve(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The last reading of each day from timestamped values
    ({'timestamp': datetime or ISO text, 'value': equity}), oldest first."""
    last: Dict[str, float] = {}
    for p in points or []:
        try:
            ts, value = p['timestamp'], float(p['value'])
            day = (ts.date().isoformat() if hasattr(ts, 'date') else str(ts)[:10])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0:
            last[day] = value
    return [{'day': d, 'equity': v} for d, v in sorted(last.items())]


# ------------------------------------------------------------------- view

def _num(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if n == n else default


def _holding(p: Any, kind: str, equity: float, config: Dict[str, Any], core: set,
             stops_for: Optional[Callable[[str], Dict[str, Any]]],
             peak_for: Optional[Callable[[str], Optional[float]]]) -> Dict[str, Any]:
    symbol = str(getattr(p, 'symbol', '')).upper()
    qty = _num(getattr(p, 'quantity', 0))
    avg = _num(getattr(p, 'avg_entry_price', 0))
    last = _num(getattr(p, 'current_price', 0)) or avg
    value = _num(getattr(p, 'market_value', 0)) or qty * last
    cost = _num(getattr(p, 'cost_basis', 0)) or qty * avg
    pnl = _num(getattr(p, 'unrealized_pl', None), value - cost)
    is_core = symbol in core
    row: Dict[str, Any] = {
        'symbol': symbol, 'region': REGION.get(classify(symbol, config), 'US'),
        'quantity': qty, 'avg_cost': avg, 'last_price': last, 'market_value': value,
        'cost_basis': cost, 'unrealised_pnl': pnl,
        'unrealised_pnl_pct': (pnl / cost * 100.0) if cost else 0.0,
        'weight_pct': (value / equity * 100.0) if equity else 0.0,
        'core': is_core, 'stops': None,
    }
    if is_core or not stops_for:
        return row          # the core is held through drawdowns by design
    try:
        d = stops_for(symbol) or {}
    except Exception as e:
        logger.debug(f"No stop distances for {symbol}: {e}")
        return row
    stop_pct, trail_pct = _num(d.get('stop_loss_pct')), _num(d.get('trailing_stop_pct'))
    if not stop_pct and not trail_pct:
        return row
    high = None
    if peak_for:
        try:
            high = peak_for(symbol)
        except Exception:
            high = None
    high = _num(high) or None
    row['stops'] = {
        'stop_loss_pct': stop_pct, 'trailing_stop_pct': trail_pct, 'source': d.get('source'),
        'stop_loss_price': (avg * (1 - stop_pct)) if avg and stop_pct else None,
        'high_since_entry': high,
        'trailing_stop_price': (high * (1 - trail_pct)) if high and trail_pct else None,
    }
    return row


def _stats(trips: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trips)
    out: Dict[str, Any] = {'closed_trades': n, 'wins': 0, 'win_rate': None,
                           'realised_pnl': 0.0, 'avg_win_pct': None, 'avg_loss_pct': None}
    if not n:
        return out
    wins = [t['ret'] for t in trips if t['ret'] > 0]
    losses = [t['ret'] for t in trips if t['ret'] <= 0]
    out.update({
        'wins': len(wins), 'win_rate': len(wins) / n,
        'realised_pnl': sum(t['ret'] * t.get('cost', 0.0) for t in trips),
        'avg_win_pct': (sum(wins) / len(wins) * 100.0) if wins else None,
        'avg_loss_pct': (sum(losses) / len(losses) * 100.0) if losses else None,
    })
    return out


def build_view(kind: str, broker: Any, journal: Any, config: Optional[Dict[str, Any]],
               stops_for: Optional[Callable[[str], Dict[str, Any]]] = None,
               peak_for: Optional[Callable[[str], Optional[float]]] = None,
               core_symbols: Iterable[str] = (), curve: Optional[List[Dict[str, Any]]] = None,
               starting_capital: Optional[float] = None, session: Optional[Dict[str, Any]] = None,
               now: Optional[datetime] = None) -> Dict[str, Any]:
    """The whole page for one paper account (see the module docstring).

    Returns {'enabled': False, 'reason': ...} when the account is off or its
    broker is not connected, so a page says so instead of showing zeros."""
    spec = BOOKS[kind]
    cfg = config or {}
    now = now or datetime.now(timezone.utc)
    if broker is None or not getattr(broker, 'is_connected', False):
        return {'enabled': False, 'kind': kind, 'title': spec['title'],
                'reason': 'The account is not connected.'}
    info = broker.get_account_info()
    if info is None or info.equity is None:
        return {'enabled': False, 'kind': kind, 'title': spec['title'],
                'reason': 'The account did not answer.'}

    equity, cash = _num(info.equity), _num(getattr(info, 'cash', 0))
    core = {str(s).upper() for s in core_symbols or ()}
    start = _num(starting_capital)
    if not start:
        start = _num(getattr(broker, 'initial_cash', 0)) or _num((cfg.get('trading') or {}).get('initial_capital'))
    held = [p for p in (broker.get_positions() or [])
            if _num(getattr(p, 'quantity', 0)) != 0 and book_of(getattr(p, 'symbol', ''), cfg) == kind]
    holdings = sorted((_holding(p, kind, equity, cfg, core, stops_for, peak_for) for p in held),
                      key=lambda h: -h['market_value'])
    invested = sum(h['market_value'] for h in holdings)

    fills = [o for o in (journal.filled_orders() if journal else [])
             if book_of(o.get('symbol', ''), cfg) == kind]
    trips = closed_round_trips(fills, cfg)
    modelled = sum(_num(o.get('filled_quantity')) * _num(o.get('filled_avg_price')) *
                   fill_gap_pct(str(o.get('symbol', '')), cfg) for o in fills)
    working = [{'symbol': o['symbol'], 'side': o['side'], 'quantity': o['quantity'],
                'status': o['status'], 'since': o.get('created_at')}
               for o in (journal.unresolved() if journal else [])
               if o.get('status') in WORKING and book_of(o.get('symbol', ''), cfg) == kind]

    points = [dict(p) for p in (curve or [])]
    today = now.date().isoformat()
    if equity > 0:
        if points and points[-1].get('day') == today:
            points[-1]['equity'] = equity
        else:
            points.append({'day': today, 'equity': equity})

    risk = cfg.get('risk_limits') or {}
    trading = cfg.get('trading') or {}
    bias = cfg.get('bias_check') or {}
    core_cfg = cfg.get('core_satellite') or {}
    return {
        'enabled': True, 'kind': kind, 'title': spec['title'], 'currency': spec['currency'],
        'broker': {'name': getattr(info, 'broker_name', '') or getattr(broker, 'broker_name', ''),
                   'paper': getattr(broker, 'is_paper_trading', None)},
        'account': {
            'starting_capital': start, 'equity': equity, 'cash': cash, 'invested': invested,
            'cash_pct': (cash / equity * 100.0) if equity else 0.0,
            'pnl': (equity - start) if start else None,
            'return_pct': ((equity / start - 1.0) * 100.0) if start else None,
            'started_on': points[0]['day'] if points else None,
        },
        'holdings': holdings,
        'stats': {**_stats(trips), 'modelled_costs': modelled, 'fills': len(fills)},
        'fills': [{'id': o.get('client_order_id'), 'symbol': o['symbol'], 'side': o['side'],
                   'quantity': _num(o.get('filled_quantity')), 'price': _num(o.get('filled_avg_price')),
                   'strategy': o.get('strategy'), 'at': o.get('updated_at') or o.get('created_at')}
                  for o in reversed(fills)][:100],
        'working_orders': working,
        'equity_curve': points,
        'session': session,
        'rules': {
            'stop_loss_pct': risk.get('stop_loss_pct'), 'trailing_stop_pct': risk.get('trailing_stop_pct'),
            'stop_loss_atr_mult': risk.get('stop_loss_atr_mult'),
            'trailing_stop_atr_mult': risk.get('trailing_stop_atr_mult'),
            'min_stop_pct': risk.get('min_stop_pct'), 'max_stop_pct': risk.get('max_stop_pct'),
            'max_position_size': risk.get('max_position_size'), 'risk_per_trade': trading.get('risk_per_trade'),
            'max_new_buys_per_day': bias.get('max_new_buys_per_day', 3),
            'add_to_winners': ((trading.get('add_to_winners') or {}).get('markets')
                               if (trading.get('add_to_winners') or {}).get('enabled', True) else []),
            'core': ({'active_share': core_cfg.get('active_share'), 'symbols': core_cfg.get('core_symbols')}
                     if kind == 'us' and core_cfg.get('enabled') else None),
        },
        'costs': {market: {k: c.get(k) for k in ('commission_pct', 'slippage_pct', 'verified', 'note')}
                  for market, sym in spec['cost_symbols'].items() for c in [costs_for(sym, cfg)]},
    }


def headline(view: Dict[str, Any]) -> Dict[str, Any]:
    """The few figures that say how an account is doing, for the strip of
    accounts at the top of the Portfolio page."""
    acct = view.get('account') or {}
    return {'id': view.get('kind'), 'title': view.get('title'), 'enabled': bool(view.get('enabled')),
            'currency': view.get('currency', 'USD'), 'equity': acct.get('equity'),
            'starting_capital': acct.get('starting_capital'), 'return_pct': acct.get('return_pct'),
            'cash': acct.get('cash'), 'holdings': len(view.get('holdings') or []),
            'closed_trades': (view.get('stats') or {}).get('closed_trades'),
            'reason': view.get('reason')}
