#!/usr/bin/env python3
"""Which markets can the agent get real prices for?

Before the agent paper trades a new market, its prices have to be reachable,
fresh and long enough to warm the strategies up. This asks Yahoo Finance
(the same source the agent uses for US stocks and currencies) about a short
list of well-known shares in Europe, Japan, China and Africa, the US-listed
ETFs and receipts (ADRs) that give exposure to the same countries, and some
currency pairs, and prints one line each:

    market, symbol, currency, last price, date of the last daily bar, how old
    that is, how many daily bars the last two years hold, and a verdict.

It also flags quotes given in pence (London) or cents (Johannesburg) rather
than pounds or rand, which would misprice every trade by a factor of 100
if taken at face value. When Alpaca keys are set, the US-listed ones are
also checked for whether Alpaca will trade them.

Run it where the agent runs (Coolify, the app's terminal):

    python scripts/probe_markets.py                 # everything
    python scripts/probe_markets.py --markets japan china
    python scripts/probe_markets.py --json > markets.json

It places no orders and writes nothing.
"""
import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Well-known, liquid names. The point is to learn whether a market's data
# works, not to pick stocks.
CANDIDATES: Dict[str, List[str]] = {
    'europe': ['SAP.DE', 'SIE.DE', 'ASML.AS', 'MC.PA', 'NESN.SW', 'HSBA.L', 'SHEL.L'],
    'europe_us_listed': ['VGK', 'EWG', 'EWU', 'ASML', 'SAP', 'NVO'],
    'japan': ['7203.T', '6758.T', '9984.T'],
    'japan_us_listed': ['EWJ', 'DXJ', 'TM', 'SONY'],
    'china': ['0700.HK', '9988.HK', '1211.HK', '600519.SS', '000858.SZ'],
    'china_us_listed': ['BABA', 'PDD', 'KWEB', 'FXI', 'MCHI'],
    'africa': ['NPN.JO', 'SBK.JO', 'MTN.JO', 'DANGCEM.LG', 'COMI.CA', 'IAM.CS', 'SCOM.KE'],
    'africa_us_listed': ['EZA', 'AFK', 'EGPT', 'NGE'],
    'forex_crosses': ['USD_JPY', 'USD_CHF', 'USD_CAD', 'EUR_GBP', 'EUR_JPY', 'USD_ZAR', 'USD_KES', 'USD_NGN'],
}

MIN_BARS = 500              # about two years: what the strategies warm up on
MAX_AGE_DAYS = 5            # a weekend and a public holiday, no more
QUOTED_IN_MINOR = {'GBp': 'pence', 'GBX': 'pence', 'ZAc': 'cents', 'ZAC': 'cents', 'ILA': 'agorot'}


def verdict(bars: int, age_days: Optional[int], price: Optional[float]) -> str:
    """OK, THIN (data, but too little or too old to rely on) or NO DATA."""
    if not price or not bars or age_days is None:
        return 'NO DATA'
    if bars < MIN_BARS or age_days > MAX_AGE_DAYS:
        return 'THIN'
    return 'OK'


def business_days_between(start: date, end: date) -> int:
    """Weekdays from `start` to `end` (0 on the same day), no holiday calendar."""
    if end <= start:
        return 0
    days, day = 0, start
    while day < end:
        day = date.fromordinal(day.toordinal() + 1)
        if day.weekday() < 5:
            days += 1
    return days


def assess(symbol: str, currency: Optional[str], price: Optional[float],
           bar_dates: List[date], today: Optional[date] = None) -> Dict[str, Any]:
    """One line of the report from what Yahoo answered."""
    today = today or datetime.now(timezone.utc).date()
    last = max(bar_dates) if bar_dates else None
    age = business_days_between(last, today) if last else None
    minor = QUOTED_IN_MINOR.get(currency or '')
    notes = []
    if minor:
        notes.append(f'quoted in {minor}, not the main unit: divide by 100')
    if last and age is not None and age > MAX_AGE_DAYS:
        notes.append(f'last bar is {age} weekdays old')
    if bar_dates and len(bar_dates) < MIN_BARS:
        notes.append(f'only {len(bar_dates)} daily bars')
    return {'symbol': symbol, 'currency': currency, 'price': price, 'bars': len(bar_dates),
            'last_bar': last.isoformat() if last else None, 'age_weekdays': age,
            'verdict': verdict(len(bar_dates), age, price), 'notes': notes}


def probe(symbol: str) -> Dict[str, Any]:
    from src.utils.real_price_feed import yahoo_symbol
    import yfinance as yf
    name = yahoo_symbol(symbol)
    try:
        ticker = yf.Ticker(name)
        hist = ticker.history(period='2y', interval='1d', auto_adjust=False)
        dates = [d.date() for d in hist.index] if hist is not None and len(hist) else []
        price, currency = None, None
        try:
            info = ticker.fast_info
            price, currency = info.get('last_price'), info.get('currency')
        except Exception:
            pass
        if price is None and len(hist):
            price = float(hist['Close'].iloc[-1])
        return assess(symbol, currency, float(price) if price else None, dates)
    except Exception as e:
        return {**assess(symbol, None, None, []), 'notes': [f'{type(e).__name__}: {str(e)[:80]}']}


def alpaca_tradable(symbols: List[str]) -> Dict[str, Optional[bool]]:
    """Whether Alpaca lists each symbol as tradable; None when it cannot be
    asked (no keys, or the answer was an error other than 'not found')."""
    if not (os.environ.get('TRADING_ALPACA_API_KEY') and os.environ.get('TRADING_ALPACA_API_SECRET')):
        return {s: None for s in symbols}
    try:
        from src.connectors.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker({'paper': True})
        if not broker.connect():
            return {s: None for s in symbols}
    except Exception:
        return {s: None for s in symbols}
    out: Dict[str, Optional[bool]] = {}
    for s in symbols:
        try:
            asset = broker._request('GET', f'assets/{s}')
            out[s] = bool(asset and asset.get('tradable'))
        except Exception as e:
            out[s] = False if getattr(e, 'status_code', None) == 404 else None
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--markets', nargs='*', choices=sorted(CANDIDATES), help='only these groups')
    ap.add_argument('--json', action='store_true', help='print JSON instead of a table')
    args = ap.parse_args(argv)

    groups = args.markets or list(CANDIDATES)
    rows: List[Dict[str, Any]] = []
    for group in groups:
        symbols = CANDIDATES[group]
        tradable = alpaca_tradable(symbols) if group.endswith('_us_listed') else {}
        for symbol in symbols:
            row = probe(symbol)
            row['group'] = group
            if group.endswith('_us_listed'):
                row['alpaca_tradable'] = tradable.get(symbol)
            rows.append(row)
            time.sleep(0.5)                 # Yahoo throttles quick repeats

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    print(f"{'GROUP':17} {'SYMBOL':11} {'CCY':4} {'LAST':>11} {'LAST BAR':10} {'AGE':>4} {'BARS':>5}  {'VERDICT':8} NOTES")
    for r in rows:
        alp = {True: 'Alpaca: tradable', False: 'Alpaca: NOT tradable', None: ''}[r.get('alpaca_tradable')] \
            if 'alpaca_tradable' in r else ''
        notes = '; '.join(filter(None, [alp] + r['notes']))
        price = f"{r['price']:,.4f}" if r['price'] else '-'
        print(f"{r['group']:17} {r['symbol']:11} {(r['currency'] or '-'):4} {price:>11} {(r['last_bar'] or '-'):10} "
              f"{(r['age_weekdays'] if r['age_weekdays'] is not None else '-'):>4} {r['bars']:>5}  {r['verdict']:8} {notes}")
    ok = sum(1 for r in rows if r['verdict'] == 'OK')
    print(f"\n{ok} of {len(rows)} usable. OK means at least {MIN_BARS} daily bars and a last bar no more than "
          f"{MAX_AGE_DAYS} weekdays old.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
