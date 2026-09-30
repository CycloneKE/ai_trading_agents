#!/usr/bin/env python3
"""Where the agent is now, against where the paper run needs it to be.

Reads the journals read-only (safe against a live agent), scores the agent on
health, evidence, edge, returns and risk, forecasts and learning, and says
plainly what is short of its target. The dashboard shows the same scorecard
(GET /api/scorecard); this prints it for a terminal.

The forecast check and the S&P 500 comparison need price history from the
internet. With --offline they are skipped and marked too early.

Usage:
    python scripts/agent_scorecard.py
    python scripts/agent_scorecard.py --offline
    python scripts/agent_scorecard.py --json
    python scripts/agent_scorecard.py --out reports/scorecard.md
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import scorecard                                   # noqa: E402
from src.agent.alerts import AlertLog, EmailAlerter               # noqa: E402
from src.agent.signal_ledger import closes_provider              # noqa: E402
from src.utils.config_validator import load_config                # noqa: E402
from src.utils.paths import DATA_DIR                              # noqa: E402

MARK = {'pass': '[PASS] ', 'watch': '[WATCH]', 'fail': '[FAIL] ', 'too_early': '[early]', 'info': '[info] '}


def price_history(offline: bool, config=None):
    """(closes_for, spy) for the forecast and S&P 500 checks; (None, None) offline.
    A Kenyan stock is read from the NSE's own price files, never from Yahoo,
    where its ticker can name a different company."""
    if offline:
        return None, None
    import logging
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)      # a missing symbol is not news here
    closes_for = closes_provider(config)
    return closes_for, (closes_for('SPY') or None)


def render(card):
    lines = ['# Agent scorecard', '',
             f"Generated {card['generated_at']} | day {card['run']['day']} of {card['run']['planned_days']} "
             f"({card['run']['percent_through']}% through)", '',
             f"## {card['verdict']['title']}", '', card['verdict']['text'], '']
    c = card['counts']
    lines.append(f"{c['pass']} pass, {c['watch']} watch, {c['fail']} fail, {c['too_early']} too early, {c['info']} info")
    for sec in card['sections']:
        lines += ['', f"## {sec['title']}", f"{sec['question']}", '']
        for m in sec['metrics']:
            target = f" (target: {m['target']})" if m['target'] else ''
            lines.append(f"{MARK[m['status']]} {m['label']}: {m['display']}{target}")
            if m['note']:
                lines.append(f"          {m['note']}")
            if m['id'] == 'strategy_skill' and isinstance(m.get('detail'), list):
                for r in m['detail']:
                    few = '' if r['enough'] else ' (too few signals to say)'
                    lines.append(f"          {r['source']:<15} {r['market']:<10} {r['hit_rate']:.0%} of {r['n']} right "
                                 f"(chance {r['chance']:.0%}, z {r['z']:+.1f}, weight x{r['tilt']:.2f}){few}")
    lines += ['', 'Targets are proposals kept in config.json under scorecard.targets. Nothing here predicts the '
              'future: every figure is measured from what the agent did and what the market then did.']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='config/config.json')
    p.add_argument('--data-dir', default=str(DATA_DIR))
    p.add_argument('--offline', action='store_true', help='skip the price-history checks')
    p.add_argument('--json', action='store_true')
    p.add_argument('--out', help='also write the report to this file')
    args = p.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    closes_for, spy = price_history(args.offline, config)
    alerter = EmailAlerter(config, log=AlertLog(os.path.join(args.data_dir, 'alerts.jsonl'))).status()
    inputs = scorecard.read_inputs(args.data_dir, config, closes_for=closes_for, spy=spy, alerter=alerter)
    card = scorecard.build(inputs, config)
    text = json.dumps(card, indent=2, default=str) if args.json else render(card)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
