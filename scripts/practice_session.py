"""Nightly practice session: sharpen strategy parameters on recent history.

For each enabled technical strategy, simulates a small grid of parameter
variants over recent daily bars, split into train (first 70%) and test
(last 30%) windows. A variant is promoted ONLY if it beats the current
parameters on BOTH windows — the out-of-sample gate that keeps this from
overfitting yesterday's noise.

Promotions are written to data/strategy_params.json, which StrategyManager
overlays onto config at startup, and every decision is logged to
data/practice_log.jsonl for audit.

Run after the close (manually or via Task Scheduler):
    .venv\\Scripts\\python.exe scripts\\practice_session.py
    ... practice_session.py --dry-run     # evaluate but promote nothing
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('practice')

from src.utils.config_validator import load_config
from src.agent.technical_strategy import TechnicalStrategy

PARAMS_FILE = os.path.join('data', 'strategy_params.json')
LOG_FILE = os.path.join('data', 'practice_log.jsonl')
TRAIN_FRACTION = 0.7

# Parameter grid per tunable knob; variants are single-knob deviations from
# current params (keeps the search honest and the runtime tiny).
GRID = {
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75],
    'lookback_period': [30, 50],
}


def fetch_bars(symbols):
    from src.connectors.alpaca_broker import AlpacaBroker
    broker = AlpacaBroker({'paper': True})
    if not broker.connect():
        raise SystemExit('Cannot connect to Alpaca for historical bars')
    start = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    bars = {}
    for sym in symbols:
        try:
            data = broker.api.get_bars(sym, '1Day', limit=150, feed='iex', start=start)
            closes = [float(b.c) for b in data]
            if len(closes) >= 80:
                bars[sym] = closes
        except Exception as e:
            logger.warning(f'No bars for {sym}: {e}')
    return bars


def simulate(params, closes):
    """Walk the bars; score = confidence-weighted next-day return captured.
    Positive = the signals pointed the right way on average."""
    strat = TechnicalStrategy('practice', params)
    score = 0.0
    for i in range(len(closes) - 1):
        sig = strat.generate_signals({'symbol': 'X', 'price': closes[i]})
        next_ret = (closes[i + 1] - closes[i]) / closes[i]
        if sig['action'] == 'buy':
            score += sig['confidence'] * next_ret
        elif sig['action'] == 'sell':
            score -= sig['confidence'] * next_ret
    return score


def evaluate(params, bars):
    """Average train/test scores across symbols."""
    train_total, test_total = 0.0, 0.0
    for closes in bars.values():
        split = int(len(closes) * TRAIN_FRACTION)
        train_total += simulate(params, closes[:split])
        test_total += simulate(params, closes[split:])
    n = max(len(bars), 1)
    return train_total / n, test_total / n


def variants_of(current):
    """Single-knob deviations from the current parameters."""
    out = []
    for knob, values in GRID.items():
        for v in values:
            if current.get(knob) == v:
                continue
            variant = dict(current)
            variant[knob] = v
            out.append((f'{knob}={v}', variant))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    config = load_config('config/config.json')
    symbols = config.get('data_manager', {}).get('symbols', [])[:8]
    strategies = {
        name: cfg for name, cfg in config.get('strategies', {}).items()
        if cfg.get('enabled') and cfg.get('type') == 'technical'
    }

    overlay = {}
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            overlay = json.load(f)

    print(f'Fetching bars for {len(symbols)} symbols...')
    bars = fetch_bars(symbols)
    print(f'{len(bars)} symbols with sufficient history\n')

    promotions = {}
    for name, base_cfg in strategies.items():
        current = {**base_cfg, **overlay.get(name, {})}
        cur_train, cur_test = evaluate(current, bars)
        print(f'{name}: current params train={cur_train:+.4f} test={cur_test:+.4f}')

        best = None
        for label, variant in variants_of(current):
            v_train, v_test = evaluate(variant, bars)
            promoted = v_train > cur_train and v_test > cur_test
            marker = ' <-- beats current on BOTH windows' if promoted else ''
            print(f'  {label:24s} train={v_train:+.4f} test={v_test:+.4f}{marker}')
            if promoted and (best is None or v_test > best[2]):
                best = (label, variant, v_test, v_train)

        record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': name,
            'current': {k: current.get(k) for k in GRID},
            'current_scores': {'train': cur_train, 'test': cur_test},
            'promoted': None,
        }
        if best:
            label, variant, v_test, v_train = best
            # Only record knobs the variant actually carries — strategies not
            # defining a grid knob in config fall back to their defaults.
            new_params = {k: variant[k] for k in GRID if k in variant}
            record['promoted'] = {'change': label, 'params': new_params,
                                  'scores': {'train': v_train, 'test': v_test}}
            if not args.dry_run:
                promotions[name] = new_params
            print(f'  PROMOTE {label} (test {cur_test:+.4f} -> {v_test:+.4f})'
                  + (' [dry-run: not applied]' if args.dry_run else ''))
        else:
            print('  no variant beat current parameters out-of-sample; keeping current')
        print()

        os.makedirs('data', exist_ok=True)
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')

    if promotions and not args.dry_run:
        merged = {**overlay, **{k: {**overlay.get(k, {}), **v} for k, v in promotions.items()}}
        with open(PARAMS_FILE, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f'Wrote {len(promotions)} promotion(s) to {PARAMS_FILE} '
              f'(picked up at next agent start)')
    elif not promotions:
        print('No promotions this session.')


if __name__ == '__main__':
    main()
