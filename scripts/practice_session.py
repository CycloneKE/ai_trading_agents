"""Practice session: how would small changes to the technical strategies' settings have done?

For each enabled technical strategy, simulates single-setting variants over
the last two years of daily bars, split into train (first 70%) and test (last
30%) windows, and reports which variants beat the current settings on both.

Report only. It used to write its "promotions" to data/strategy_params.json,
the same file the weekly strategy tuner (src/agent/strategy_tuner.py) keeps
its state in, in a different layout: whichever ran second wiped out the
other's record. It also scored variants on next-day moves with no trading
costs, and could apply RSI settings to strategies that do not use them. The
weekly tuner is the one process that changes strategy settings: it costs
trades per market, tunes each strategy only on its own settings, and adopts
a change only on held-out evidence. This script is kept as a quick look,
written to data/practice_log.jsonl.

Prices are split- and dividend-adjusted (as the live warm-start's are):
the raw closes it used to read showed a stock split as a crash.

Run after the close:
    python scripts/practice_session.py
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
from src.utils.paths import DATA_DIR

LOG_FILE = str(DATA_DIR / 'practice_log.jsonl')
TRAIN_FRACTION = 0.7

# Parameter grid per tunable knob; variants are single-knob deviations from
# current params (keeps the search honest and the runtime tiny).
GRID = {
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75],
    'lookback_period': [30, 50],
}


def fetch_bars(symbols):
    # Yahoo, split- and dividend-adjusted, ~500 daily bars; Alpaca's IEX
    # feed (~150 bars, also adjusted) for any symbol Yahoo cannot supply.
    from src.agent.history_warmstart import _yfinance_daily
    bars = {}
    missing = []
    for sym in symbols:
        try:
            closes = [row[3] for row in _yfinance_daily(sym, period='2y')]
        except Exception as e:
            logger.warning(f'No Yahoo history for {sym}: {e}')
            closes = []
        if len(closes) >= 80:
            bars[sym] = closes
        else:
            missing.append(sym)
    if missing:
        try:
            from src.connectors.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker({'paper': True})
            if broker.connect():
                start = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
                for sym in missing:
                    try:
                        data = broker.api.get_bars(sym, '1Day', limit=150, feed='iex', start=start,
                                                   adjustment='all')
                        closes = [float(b.c) for b in data]
                        if len(closes) >= 80:
                            bars[sym] = closes
                    except Exception as e:
                        logger.warning(f'No bars for {sym}: {e}')
        except Exception as e:
            logger.warning(f'Alpaca fallback unavailable: {e}')
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
    # Accepted for old scheduled commands; the script is always report-only now.
    parser.add_argument('--dry-run', action='store_true', help=argparse.SUPPRESS)
    parser.parse_args()

    config = load_config('config/config.json')
    symbols = config.get('data_manager', {}).get('symbols', [])[:8]
    strategies = {
        name: cfg for name, cfg in config.get('strategies', {}).items()
        if cfg.get('enabled') and cfg.get('type') == 'technical'
    }

    # The settings the agent runs with now: config plus what the weekly tuner adopted.
    tuned = {}
    tuner_file = DATA_DIR / 'strategy_params.json'
    if os.path.exists(tuner_file):
        try:
            with open(tuner_file) as f:
                saved = json.load(f)
            tuned = saved.get('params', {}) if isinstance(saved.get('params'), dict) else {}
        except (OSError, ValueError) as e:
            logger.warning(f'Could not read the tuned settings: {e}')

    print(f'Fetching bars for {len(symbols)} symbols...')
    bars = fetch_bars(symbols)
    print(f'{len(bars)} symbols with sufficient history\n')

    better = 0
    for name, base_cfg in strategies.items():
        current = {**base_cfg, **tuned.get(name, {})}
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
                                  'scores': {'train': v_train, 'test': v_test},
                                  'applied': False}
            better += 1
            print(f'  BETTER ON BOTH: {label} (test {cur_test:+.4f} -> {v_test:+.4f}); '
                  f'report only, nothing applied')
        else:
            print('  no variant beat current parameters out-of-sample; keeping current')
        print()

        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')

    print(f'{better} strateg{"y" if better == 1 else "ies"} had a variant better on both windows. '
          f'Nothing was changed: the weekly strategy tuner decides settings. Logged to {LOG_FILE}.')


if __name__ == '__main__':
    main()
