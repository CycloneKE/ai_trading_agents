#!/usr/bin/env python3
"""Turn a running paper trade into evidence instead of a black box.

Ninety days of an agent running is only useful if someone can answer four
questions from it without reading logs:

    1. Is it actually trading, or has it quietly stopped?
    2. What did it make or lose, per symbol and per strategy?
    3. When it holds, why does it hold?
    4. Is anything silently degraded?

Reads the order journal and the decision journal, which are the run's only
durable record, and the run manifest written by scripts/start_paper_run.py,
which is what makes "no trades in three weeks" distinguishable from "no
trades yet, it started an hour ago".

Both journals are opened read-only, so this is safe to run against a live
agent.

Usage:
    python scripts/paper_run_report.py
    python scripts/paper_run_report.py --since 7 --out reports/week1.md
    python scripts/paper_run_report.py --json
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.paths import DATA_DIR  # noqa: E402

RUNS_DIR = DATA_DIR / 'paper_runs'

# Plain-language readings of the journal's skip reasons. A histogram of bare
# codes tells an operator nothing; what they need is which ones mean the
# system is working as designed and which mean something is broken.
SKIP_MEANING = {
    'hold': ('by design', 'no directional signal; the normal resting state'),
    'below_confidence': ('by design', 'signal too weak to act on'),
    'dissent': ('by design', 'strategies disagreed on direction'),
    'llm_veto': ('by design', 'the validation layer overruled the ensemble'),
    'bias_downgrade': ('by design', 'the bias detector pushed the signal to hold'),
    'duplicate': ('by design', 'the journal blocked a repeated decision'),
    'fallback_price': ('DEGRADED', 'price was synthetic, so the trade was refused; '
                                   'the vendor feed was down for this symbol'),
    'halted': ('DEGRADED', 'the kill switch was engaged; the run was not trading'),
    'no_account_info': ('DEGRADED', 'the broker did not return account state, '
                                    'so nothing could be sized'),
    'min_notional': ('check sizing', 'position sized below the broker minimum; '
                                     'capital or risk-per-trade may be too small'),
    'pdt_guard': ('check sizing', 'pattern-day-trader rule blocked the trade'),
}


# --------------------------------------------------------------- data access

def read_only(path):
    """Open a journal without any chance of disturbing a live run."""
    if not os.path.exists(path):
        return None
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    except sqlite3.Error:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def rows(conn, sql, params=()):
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []


def latest_manifest(path=None):
    if path:
        with open(path, encoding='utf-8') as f:
            return json.load(f), path
    if not os.path.isdir(RUNS_DIR):
        return None, None
    files = sorted(f for f in os.listdir(RUNS_DIR) if f.endswith('.json'))
    if not files:
        return None, None
    full = os.path.join(RUNS_DIR, files[-1])
    try:
        with open(full, encoding='utf-8') as f:
            return json.load(f), full
    except Exception:
        return None, full


def parse_ts(value):
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------- P&L attribution

def round_trips(fills):
    """FIFO-match fills into closed round trips.

    Attribution follows the opening order's strategy, not the closing one: a
    trailing stop closes the position but it did not choose it, and crediting
    the exit would make every stopped-out trade look like the stop's fault
    rather than the entry's.
    """
    lots = defaultdict(deque)
    trips = []
    for o in fills:
        sym = (o.get('symbol') or '').upper()
        qty = float(o.get('filled_quantity') or 0)
        price = o.get('filled_avg_price')
        if qty <= 0 or price is None:
            continue
        price = float(price)
        side = (o.get('side') or '').lower()
        if side in ('buy', 'buy_to_cover'):
            lots[sym].append({'qty': qty, 'price': price,
                              'strategy': o.get('strategy') or 'unknown',
                              'opened_at': o.get('updated_at') or o.get('created_at')})
            continue
        if side not in ('sell', 'sell_short'):
            continue
        remaining = qty
        while remaining > 1e-9 and lots[sym]:
            lot = lots[sym][0]
            matched = min(remaining, lot['qty'])
            trips.append({
                'symbol': sym,
                'strategy': lot['strategy'],
                'quantity': matched,
                'entry_price': lot['price'],
                'exit_price': price,
                'pnl': (price - lot['price']) * matched,
                'return_pct': (price / lot['price'] - 1.0) if lot['price'] else 0.0,
                'opened_at': lot['opened_at'],
                'closed_at': o.get('updated_at') or o.get('created_at'),
            })
            lot['qty'] -= matched
            remaining -= matched
            if lot['qty'] <= 1e-9:
                lots[sym].popleft()
    open_lots = {s: [dict(l) for l in q] for s, q in lots.items() if q}
    return trips, open_lots


def aggregate(trips, key):
    out = {}
    for t in trips:
        k = t[key]
        a = out.setdefault(k, {'trips': 0, 'wins': 0, 'pnl': 0.0,
                               'gross_win': 0.0, 'gross_loss': 0.0})
        a['trips'] += 1
        a['pnl'] += t['pnl']
        if t['pnl'] > 0:
            a['wins'] += 1
            a['gross_win'] += t['pnl']
        else:
            a['gross_loss'] += abs(t['pnl'])
    for a in out.values():
        a['win_rate'] = a['wins'] / a['trips'] if a['trips'] else 0.0
        a['profit_factor'] = (a['gross_win'] / a['gross_loss']
                              if a['gross_loss'] > 0 else float('inf'))
    return out


# --------------------------------------------------------------- the report

def build(args):
    orders_db = args.orders or str(DATA_DIR / 'order_journal.db')
    decisions_db = args.decisions or str(DATA_DIR / 'decision_journal.db')
    manifest, manifest_path = latest_manifest(args.manifest)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=args.since) if args.since else None
    # Both journals write datetime.utcnow().isoformat(), which is naive. Compare
    # against a naive string so a timezone suffix does not sort rows out.
    cutoff_iso = cutoff.replace(tzinfo=None).isoformat() if cutoff else None

    oc = read_only(orders_db)
    dc = read_only(decisions_db)

    if cutoff_iso:
        all_orders = rows(oc, 'SELECT * FROM orders WHERE created_at >= ?'
                              ' ORDER BY created_at', (cutoff_iso,))
        decisions = rows(dc, 'SELECT * FROM decisions WHERE ts >= ?'
                             ' ORDER BY ts', (cutoff_iso,))
    else:
        all_orders = rows(oc, 'SELECT * FROM orders ORDER BY created_at')
        decisions = rows(dc, 'SELECT * FROM decisions ORDER BY ts')

    fills = [o for o in all_orders if (o.get('status') or '') == 'filled'
             and float(o.get('filled_quantity') or 0) > 0]
    trips, open_lots = round_trips(fills)

    statuses = Counter((o.get('status') or 'unknown') for o in all_orders)
    skips = Counter((d.get('skip_reason') or ('executed' if d.get('executed')
                                              else 'unrecorded'))
                    for d in decisions)

    last_decision = max((parse_ts(d.get('ts')) for d in decisions), default=None)
    last_order = max((parse_ts(o.get('created_at')) for o in all_orders), default=None)

    llm_missing = sum(1 for d in decisions
                      if not (d.get('llm_verdict_json') or '').strip('{} \n'))

    started = parse_ts((manifest or {}).get('started_at'))
    elapsed_days = ((now - started).total_seconds() / 86400.0) if started else None

    return {
        'generated_at': now.isoformat(),
        'window_days': args.since,
        'manifest_path': manifest_path,
        'manifest': manifest,
        'elapsed_days': elapsed_days,
        'orders': {
            'total': len(all_orders),
            'by_status': dict(statuses),
            'last_order_at': last_order.isoformat() if last_order else None,
            'hours_since_last_order': ((now - last_order).total_seconds() / 3600.0
                                       if last_order else None),
        },
        'decisions': {
            'total': len(decisions),
            'symbols': len({d.get('symbol') for d in decisions}),
            'by_skip_reason': dict(skips),
            'last_decision_at': last_decision.isoformat() if last_decision else None,
            'hours_since_last_decision': ((now - last_decision).total_seconds() / 3600.0
                                          if last_decision else None),
            'without_llm_verdict': llm_missing,
        },
        'pnl': {
            'closed_trips': len(trips),
            'realised': sum(t['pnl'] for t in trips),
            'by_symbol': aggregate(trips, 'symbol'),
            'by_strategy': aggregate(trips, 'strategy'),
            'open_positions': {s: {'lots': len(q),
                                   'quantity': sum(l['qty'] for l in q),
                                   'cost_basis': sum(l['qty'] * l['price'] for l in q)}
                               for s, q in open_lots.items()},
        },
    }



def concentration_note(by_symbol, threshold=0.7):
    """Flag P&L that rests on one symbol.

    A backtest in this repository once showed a convincing edge that was
    entirely one stock; the other eleven symbols net to nothing. A headline
    P&L hides that completely, and the distinction matters because one symbol
    over one period is an anecdote, not an edge.
    """
    winners = {s: a['pnl'] for s, a in by_symbol.items() if a['pnl'] > 0}
    total = sum(winners.values())
    if not winners or total <= 0:
        return ''
    top, top_pnl = max(winners.items(), key=lambda x: x[1])
    share = top_pnl / total
    if share < threshold:
        return ''
    return (f"**{share:.0%} of the gross profit is {top} alone.** Strip it out "
            f"and the rest of the book made {total - top_pnl:,.2f}. One symbol "
            f"over one period is an anecdote; check whether the edge survives "
            f"without it before trusting the headline number.")


def fmt_money(v):
    return f'{v:>12,.2f}'


def render(rep, config_path):
    L = []
    add = L.append
    add('# Paper run report')
    add('')
    add(f"Generated {rep['generated_at'][:19]}Z"
        + (f" | window: last {rep['window_days']} days" if rep['window_days']
           else ' | window: entire journal'))
    add('')

    # 1. The run itself
    add('## The run')
    add('')
    m = rep['manifest']
    if not m:
        add('No run manifest found. Without one this report cannot tell a run '
            'that has traded nothing from a run that started an hour ago. '
            'Start runs with `python scripts/start_paper_run.py --start` so a '
            'manifest is written.')
    else:
        elapsed = rep['elapsed_days'] or 0
        planned = m.get('planned_days') or 0
        add(f"- Started: {m.get('started_at', '?')[:19]}Z")
        add(f"- Elapsed: {elapsed:.1f} of {planned} planned days "
            f"({elapsed / planned:.0%} through)" if planned else
            f"- Elapsed: {elapsed:.1f} days")
        add(f"- Commit: {str(m.get('git_commit', '?'))[:12]}")
        uni = m.get('universe', {})
        add(f"- Universe: {len(uni.get('loop_symbols', []))} every cycle, "
            f"{len(uni.get('slow_pass_symbols', []))} on the slow pass")
        if m.get('started_with_blockers'):
            add('')
            add(f"**This run was started with known blockers:** "
                f"{', '.join(m['started_with_blockers'])}. Its results are "
                f"suspect until those are understood.")
        if m.get('warnings_at_start'):
            add(f"- Warnings at start: {', '.join(m['warnings_at_start'])}")
    add('')

    # 2. Is it trading?
    o = rep['orders']
    d = rep['decisions']
    add('## Is it trading?')
    add('')
    add(f"- Decisions recorded: {d['total']:,} across {d['symbols']} symbols")
    add(f"- Orders journaled: {o['total']:,}")
    if o['by_status']:
        for status, n in sorted(o['by_status'].items(), key=lambda x: -x[1]):
            add(f"    - {status}: {n:,}")
    if d['hours_since_last_decision'] is not None:
        add(f"- Last decision: {d['hours_since_last_decision']:.1f} hours ago")
    if o['hours_since_last_order'] is not None:
        add(f"- Last order: {o['hours_since_last_order']:.1f} hours ago")
    add('')

    if d['total'] == 0 and o['total'] == 0:
        add('**The agent has recorded nothing at all.** It is not running, or '
            'it is running and the journals are not being written. Check the '
            'process is alive and that `data/` is writable.')
    elif d['total'] == 0:
        add('**Orders exist but no decisions were journaled.** The trades are '
            'real, but the reasoning behind them was not recorded, so this '
            'report can say what happened and not why.')
    elif d['hours_since_last_decision'] is not None and d['hours_since_last_decision'] > 24:
        add(f"**Nothing recorded for {d['hours_since_last_decision']:.0f} hours.** "
            f"The agent has most likely stopped. A live run writes a heartbeat "
            f"row per symbol every 30 cycles even when nothing changes.")
    elif o['total'] == 0:
        add('**The agent is running and deciding, but has placed no orders.** '
            'The reasons below say why. If they are all "by design" the '
            'configuration is simply not finding trades; if any are DEGRADED, '
            'something is broken.')
    add('')

    # 3. Why it holds
    add('## Why it holds')
    add('')
    if not d['by_skip_reason']:
        add('No decisions in the window.')
    else:
        total = sum(d['by_skip_reason'].values())
        add('| reason | count | share | reading |')
        add('|---|---:|---:|---|')
        for reason, n in sorted(d['by_skip_reason'].items(), key=lambda x: -x[1]):
            verdict, meaning = SKIP_MEANING.get(
                reason, ('executed', 'the trade went out')
                if reason == 'executed' else ('?', 'unrecognised reason'))
            add(f"| {reason} | {n:,} | {n / total:.1%} | {verdict}: {meaning} |")
        degraded = {r: n for r, n in d['by_skip_reason'].items()
                    if SKIP_MEANING.get(r, ('', ''))[0] == 'DEGRADED'}
        if degraded:
            add('')
            add(f"**{sum(degraded.values()):,} decisions were blocked by "
                f"degradation, not by the strategy:** "
                f"{', '.join(sorted(degraded))}. Those are lost opportunities, "
                f"not a verdict on the strategy.")
    add('')

    # 4. P&L
    p = rep['pnl']
    add('## What it made')
    add('')
    add(f"- Closed round trips: {p['closed_trips']:,}")
    add(f"- Realised P&L: {p['realised']:,.2f}")
    add('')
    add('Realised P&L is computed from actual fill prices, FIFO matched. '
        'Broker commission and the slippage already embedded in those fills '
        'are *not* subtracted again here, because doing so would double-count '
        'them. Compare each return against its market round-trip cost hurdle '
        'from `src/agent/cost_model.py` before calling an edge real.')
    add('')
    concentration = concentration_note(p['by_symbol'])
    if concentration:
        add(concentration)
        add('')
    if p['by_symbol']:
        add('### By symbol')
        add('')
        add('| symbol | trips | win rate | profit factor | P&L |')
        add('|---|---:|---:|---:|---:|')
        for sym, a in sorted(p['by_symbol'].items(), key=lambda x: -x[1]['pnl']):
            pf = ('inf' if a['profit_factor'] == float('inf')
                  else f"{a['profit_factor']:.2f}")
            add(f"| {sym} | {a['trips']} | {a['win_rate']:.0%} | {pf} "
                f"| {a['pnl']:,.2f} |")
        add('')
    if p['by_strategy']:
        add('### By strategy')
        add('')
        add('Attributed to the strategy that *opened* the position, so a '
            'stop-loss exit is charged to the entry that needed stopping.')
        add('')
        add('| strategy | trips | win rate | profit factor | P&L |')
        add('|---|---:|---:|---:|---:|')
        for name, a in sorted(p['by_strategy'].items(), key=lambda x: -x[1]['pnl']):
            pf = ('inf' if a['profit_factor'] == float('inf')
                  else f"{a['profit_factor']:.2f}")
            add(f"| {name} | {a['trips']} | {a['win_rate']:.0%} | {pf} "
                f"| {a['pnl']:,.2f} |")
        add('')
    if p['open_positions']:
        add('### Still open')
        add('')
        add('Cost basis only. Marking these to market needs a live price feed, '
            'which this report deliberately does not require.')
        add('')
        add('| symbol | quantity | cost basis |')
        add('|---|---:|---:|')
        for sym, a in sorted(p['open_positions'].items()):
            add(f"| {sym} | {a['quantity']:,.4f} | {a['cost_basis']:,.2f} |")
        add('')

    # 5. Degradation
    add('## Silent degradation')
    add('')
    findings = []
    stuck = (o['by_status'].get('intent', 0) + o['by_status'].get('submitted', 0))
    if stuck:
        findings.append(f"{stuck} order(s) never reached a final status. Either "
                        f"the broker never answered or reconciliation is not "
                        f"running at startup.")
    failed = o['by_status'].get('failed', 0) + o['by_status'].get('aborted', 0)
    if failed:
        findings.append(f"{failed} order(s) failed or were aborted. Each one is "
                        f"a decision the agent made and could not act on.")
    rejected = o['by_status'].get('rejected', 0)
    if rejected:
        findings.append(f"{rejected} order(s) were rejected by the broker. "
                        f"Usually sizing, buying power, or a tradability rule.")
    if d['total'] and d['without_llm_verdict'] == d['total']:
        findings.append('No decision carries an LLM verdict. Validation is '
                        'either disabled or has been rate-limited off for the '
                        'whole window, so trades went out unvalidated.')
    elif d['total'] and d['without_llm_verdict'] > 0.5 * d['total']:
        findings.append(f"{d['without_llm_verdict']:,} of {d['total']:,} "
                        f"decisions ({d['without_llm_verdict'] / d['total']:.0%}) "
                        f"have no LLM verdict, which is what quota exhaustion "
                        f"looks like from the journal.")
    fb = d['by_skip_reason'].get('fallback_price', 0)
    if fb:
        findings.append(f"{fb:,} decisions saw synthetic fallback prices. The "
                        f"vendor feed was down for those; the agent correctly "
                        f"refused to trade, and correctly learned nothing.")
    if findings:
        for f in findings:
            add(f'- {f}')
    else:
        add('Nothing detected. Orders reached final statuses, validation ran, '
            'and prices were real.')
    add('')

    # 6. Verdict
    add('## Verdict')
    add('')
    if d['total'] == 0 and o['total'] == 0:
        add('No evidence. The run is not recording anything.')
    elif p['closed_trips'] == 0 and o['total'] == 0:
        add('No evidence yet. The agent is alive and deciding but has not '
            'traded, so there is nothing to judge. Read "Why it holds" to see '
            'whether that is the strategy being selective or something broken.')
    elif p['closed_trips'] < 20:
        add(f"Too early to judge. {p['closed_trips']} closed round trips is not "
            f"enough to separate skill from luck; thirty or more per strategy "
            f"is the point at which win rate and profit factor start meaning "
            f"something.")
    else:
        wins = sum(a['gross_win'] for a in p['by_symbol'].values())
        losses = sum(a['gross_loss'] for a in p['by_symbol'].values())
        pf = (wins / losses) if losses > 0 else float('inf')
        pf_txt = 'inf (no losing trade yet)' if pf == float('inf') else f'{pf:.2f}'
        add(f"{p['closed_trips']} closed round trips, {p['realised']:,.2f} "
            f"realised, profit factor {pf_txt}. Judge each strategy against "
            f"its own round-trip cost hurdle rather than against zero, and "
            f"against what the cash would have earned sitting in a money "
            f"market fund over the same period.")
    add('')
    add(f'Config: `{config_path}` | Orders and decisions read read-only from '
        f'`{DATA_DIR}`.')
    return '\n'.join(L) + '\n'


def main():
    p = argparse.ArgumentParser(description='Report on a running paper trade.')
    p.add_argument('--since', type=float, default=None,
                   help='only the last N days (default: the entire journal)')
    p.add_argument('--orders', help='path to order_journal.db')
    p.add_argument('--decisions', help='path to decision_journal.db')
    p.add_argument('--manifest', help='a specific run manifest')
    p.add_argument('--config', default='config/config.json')
    p.add_argument('--out', help='write the report to a file')
    p.add_argument('--json', action='store_true')
    args = p.parse_args()

    rep = build(args)
    if args.json:
        out = json.dumps(rep, indent=2, default=str)
    else:
        out = render(rep, args.config)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(out)
        print(f'Wrote {args.out}')
    else:
        print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
