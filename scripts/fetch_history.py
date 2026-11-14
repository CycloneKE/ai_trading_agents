#!/usr/bin/env python3
"""Download daily price history for the backtest (scripts/run_backtest.py).

The backtest reads one <SYMBOL>.csv per symbol from a folder; this fills that
folder from Yahoo Finance (prices adjusted for splits and dividends), for
every symbol the agent trades in the US account: the technical strategies'
list, the core funds, the momentum sleeve's funds and SPY as the benchmark.

NSE stocks are not downloaded: Yahoo's tickers for them can name other
companies. Their history is the agent's own (data/nse_historical).

Usage:
    python scripts/fetch_history.py                    # 6 years into data/history
    python scripts/fetch_history.py --years 10 --out data/history
    python scripts/fetch_history.py --symbols AAPL,MSFT
"""
import argparse
import csv
import json
import os
import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Row = Tuple[str, float, float, float, float, float]      # date, open, high, low, close, volume


def symbols_from_config(cfg: Dict) -> List[str]:
    """Every US-account symbol the config trades, plus SPY, without NSE stocks."""
    from src.agent.cost_model import classify
    out = list((cfg.get('data_manager') or {}).get('symbols') or [])
    out += list((cfg.get('core_satellite') or {}).get('core_symbols') or [])
    out += list((cfg.get('momentum_sleeve') or {}).get('universe') or [])
    out.append('SPY')
    seen, keep = set(), []
    for s in out:
        s = str(s).upper()
        if s in seen or classify(s, cfg) == 'nse':
            continue
        seen.add(s)
        keep.append(s)
    return keep


def yahoo_rows(symbol: str, years: int) -> List[Row]:
    """Adjusted daily bars from Yahoo Finance, oldest first."""
    import yfinance as yf
    from src.utils.real_price_feed import yahoo_symbol
    df = yf.Ticker(yahoo_symbol(symbol)).history(period=f'{years}y', interval='1d', auto_adjust=True)
    if df is None or df.empty:
        return []
    return [(idx.date().isoformat(), float(r['Open']), float(r['High']), float(r['Low']),
             float(r['Close']), float(r.get('Volume', 0) or 0))
            for idx, r in df.iterrows() if r['Close'] and r['Close'] > 0]


def write_csv(path: str, rows: Iterable[Row]) -> int:
    rows = list(rows)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
        for r in rows:
            w.writerow([r[0]] + [round(x, 6) for x in r[1:5]] + [int(r[5])])
    return len(rows)


def fetch(symbols: Iterable[str], out_dir: str, years: int = 6,
          get: Optional[Callable[[str, int], List[Row]]] = None) -> Dict[str, int]:
    """Bars written per symbol (0 for a symbol that could not be fetched)."""
    get = get or yahoo_rows
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    for sym in symbols:
        try:
            rows = get(sym, years)
        except Exception as e:                    # one bad symbol must not stop the rest
            print(f"  {sym}: failed ({e})")
            rows = []
        done[sym] = write_csv(os.path.join(out_dir, f'{sym}.csv'), rows) if rows else 0
        print(f"  {sym}: {done[sym]} daily bars" if rows else f"  {sym}: no data")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--symbols', help='comma-separated; default: every US-account symbol in the config')
    ap.add_argument('--years', type=int, default=6)
    ap.add_argument('--out', default='data/history')
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = json.load(f)
    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols else symbols_from_config(cfg)
    print(f"Downloading {args.years} years of daily prices for {len(symbols)} symbols into {args.out}")
    done = fetch(symbols, args.out, args.years)
    missing = [s for s, n in done.items() if not n]
    print(f"Done: {len(done) - len(missing)} of {len(done)} symbols."
          + (f" No data for: {', '.join(missing)}." if missing else ''))
    return 0 if len(missing) < len(done) else 1


if __name__ == '__main__':
    sys.exit(main())
