"""The scorecard (src/agent/scorecard.py): where the agent is now against where
the paper run needs it to be.

What matters: a measure never passes on too little evidence, a rising market
does not make every buy look clever, a failing health check outranks
everything else, and the numbers come out of the real journals.
"""
import json
import random
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from src.agent import scorecard
from src.agent.decision_journal import DecisionJournal
from src.agent.order_journal import OrderJournal
from src.agent.scorecard import build, targets, wilson

NOW = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)
CONFIG = {'risk_management': {'max_drawdown': 0.10}}


def metric(card, mid):
    return next(m for s in card['sections'] for m in s['metrics'] if m['id'] == mid)


def status(card, mid):
    return metric(card, mid)['status']


def hourly_decisions(start, hours, symbol='BTC-USD', reason='hold'):
    return [{'ts': (start + timedelta(hours=h)).replace(tzinfo=None).isoformat(), 'symbol': symbol,
             'action': 'hold', 'skip_reason': reason, 'executed': 0, 'price': 100.0} for h in range(hours)]


def fills_for(trades):
    """Filled orders that close as round trips: [(symbol, buy, sell)], one day apart."""
    out, t = [], datetime(2026, 9, 1, 15, 0)
    for i, (sym, buy, sell) in enumerate(trades):
        for side, price, when in (('buy', buy, t), ('sell', sell, t + timedelta(days=1))):
            out.append({'symbol': sym, 'side': side, 'filled_quantity': 10, 'filled_avg_price': price,
                        'created_at': when.isoformat(), 'strategy': 'momentum'})
        t += timedelta(days=2)
    return out


def inputs(**kw):
    start = NOW - timedelta(days=kw.pop('days', 2))
    base = {'run_started_at': start.isoformat(), 'planned_days': 90,
            'decisions': hourly_decisions(start, int((NOW - start).total_seconds() // 3600)),
            'fills': [], 'equity': [], 'spy': None, 'closes_for': None, 'healing': None,
            'alerter': None, 'alerts': [], 'tuner': None}
    base.update(kw)
    return base


# ------------------------------------------------------------- statistics

def test_the_win_rate_range_is_wide_when_there_are_few_trades():
    lo, hi = wilson(6, 10)
    assert lo < 0.5 < hi and hi - lo > 0.4                      # 6 of 10 proves nothing
    lo, hi = wilson(600, 1000)
    assert lo > 0.5 and hi - lo < 0.07
    assert wilson(0, 0) == (0.0, 1.0)


def test_the_drawdown_target_is_the_accounts_own_limit_unless_overridden():
    assert targets(CONFIG)['max_drawdown_pct'] == 10.0
    assert targets({'risk_management': {'max_drawdown': 0.2}})['max_drawdown_pct'] == 20.0
    assert targets({'scorecard': {'targets': {'max_drawdown_pct': 7, '_c': 'x'}}})['max_drawdown_pct'] == 7
    assert targets({})['min_closed_trades'] == 30


# ---------------------------------------------------------- a fresh run

def test_a_two_day_old_run_with_no_trades_is_too_early_and_says_what_is_missing():
    card = build(inputs(alerter={'configured': False}), CONFIG, NOW)
    assert card['verdict']['level'] == 'too_early'
    assert card['run']['day'] == 2.0 and card['run']['percent_through'] == 2.2
    assert status(card, 'closed_trades') == 'too_early'
    assert '30 more closed trades needed' in metric(card, 'closed_trades')['note']
    for mid in ('win_rate', 'avg_return', 'profit_factor', 'concentration'):
        assert status(card, mid) == 'too_early'                  # never a pass on no evidence
    assert status(card, 'coverage') == 'pass'
    assert status(card, 'email') == 'watch' and 'ALERT_EMAIL_TO' in metric(card, 'email')['note']
    assert status(card, 'learning_gate') == 'too_early'
    assert 'closed trades' in metric(card, 'learning_gate')['note']


def test_the_scorecard_lists_all_six_questions_in_order():
    card = build(inputs(), CONFIG, NOW)
    assert [s['id'] for s in card['sections']] == ['health', 'evidence', 'edge', 'returns', 'forecast', 'learning']
    assert sum(card['counts'].values()) == sum(len(s['metrics']) for s in card['sections'])
    json.dumps(card)                                             # it must serialise for the API


# ---------------------------------------------------------------- health

def test_hours_the_agent_was_not_running_show_as_a_coverage_gap():
    start = NOW - timedelta(days=3)
    rows = hourly_decisions(start, 72)
    gone = [r for i, r in enumerate(rows) if not 20 <= i < 44]     # a 24 hour outage
    card = build(inputs(days=3, decisions=gone), CONFIG, NOW)
    m = metric(card, 'coverage')
    assert m['value'] == pytest.approx(66.7, abs=0.2) and m['status'] == 'fail'
    assert '24 of the last 72 hours have no record' in m['note']
    assert card['verdict']['level'] == 'needs_attention'


def test_decisions_blocked_by_missing_data_are_counted_by_reason_and_symbol():
    start = NOW - timedelta(days=2)
    rows = hourly_decisions(start, 48)
    for r in rows[:12]:
        r['skip_reason'], r['symbol'] = 'fallback_price', 'EUR_USD'
    for r in rows[12:16]:
        r['skip_reason'] = 'market_closed'                        # by design: not degraded
    m = metric(build(inputs(decisions=rows), CONFIG, NOW), 'degraded')
    assert m['value'] == 25.0 and m['status'] == 'fail'
    assert m['detail']['by_reason'] == {'fallback_price': 12} and m['detail']['by_symbol'] == {'EUR_USD': 12}
    assert 'EUR_USD (12)' in m['note']


def test_a_halt_outranks_every_other_result_in_the_verdict():
    healing = {'halted': True, 'halt_reason': 'daily loss limit', 'halt_clears_itself': False, 'workers': []}
    card = build(inputs(healing=healing), CONFIG, NOW)
    assert status(card, 'halted') == 'fail'
    assert 'until you press Resume' in metric(card, 'halted')['note']
    assert card['verdict']['level'] == 'needs_attention' and 'Trading halted right now' in card['verdict']['text']
    healing.update(halt_reason='HEARTBEAT_TIMEOUT: x', halt_clears_itself=True)
    assert 'lifts itself' in metric(build(inputs(healing=healing), CONFIG, NOW), 'halted')['note']


def test_a_dead_worker_and_a_failing_loop_are_failures():
    healing = {'halted': False, 'workers': [{'worker': 'nse_scraper', 'alive': False, 'gave_up': False}],
               'consecutive_loop_errors': 4, 'last_loop_error': 'KeyError: x', 'symbols_without_price': {'SCOM': 41.0}}
    card = build(inputs(healing=healing), CONFIG, NOW)
    assert status(card, 'workers') == 'fail' and 'nse_scraper' in metric(card, 'workers')['note']
    assert status(card, 'loop_errors') == 'fail' and status(card, 'no_price') == 'watch'


def test_self_repairs_in_the_last_week_are_counted_from_the_alert_log():
    week_ago = (NOW - timedelta(days=9)).isoformat()
    alerts = [{'ts': NOW.isoformat(), 'key': 'heal:api_server:restarted', 'severity': 'warning', 'status': 'sent'},
              {'ts': NOW.isoformat(), 'key': 'heal:auto_resumed', 'severity': 'warning', 'status': 'sent'},
              {'ts': NOW.isoformat(), 'key': 'heal:halt', 'severity': 'critical', 'status': 'sent'},
              {'ts': week_ago, 'key': 'heal:unclean_restart', 'severity': 'warning', 'status': 'sent'}]
    m = metric(build(inputs(alerts=alerts), CONFIG, NOW), 'healing')
    assert m['display'] == '1 restarts, 1 auto-resumes, 0 crashes' and '1 critical' in m['note']


def test_email_on_is_a_pass_and_names_where_it_goes():
    alerter = {'configured': True, 'recipients': ['b***@example.com'], 'sent_24h': 2, 'last_error': None}
    m = metric(build(inputs(alerter=alerter), CONFIG, NOW), 'email')
    assert m['status'] == 'pass' and 'b***@example.com' in m['note']


# --------------------------------------------------------------- evidence

def test_the_trade_count_is_judged_against_the_pace_needed_to_reach_thirty():
    def st(days, trades):
        return status(build(inputs(days=days, fills=fills_for([('AAPL', 100, 105)] * trades)), CONFIG,
                            NOW + timedelta(days=0)), 'closed_trades')
    assert st(10, 0) == 'too_early'                                # day 10 of 90: too soon to worry
    assert st(60, 3) == 'fail'                                     # day 60, expected 20, has 3
    assert st(60, 12) == 'watch'
    assert st(60, 30) == 'pass'


# ------------------------------------------------------------------- edge

def test_a_real_edge_needs_enough_trades_and_a_win_rate_beyond_luck():
    winners = [('AAPL', 100, 104)] * 34 + [('MSFT', 100, 97)] * 6      # 85% winners, net of costs
    card = build(inputs(days=40, fills=fills_for(winners)), CONFIG, NOW)
    assert status(card, 'win_rate') == 'pass' and status(card, 'avg_return') == 'pass'
    assert status(card, 'profit_factor') == 'pass'
    assert status(card, 'concentration') in ('pass', 'watch')


def test_a_losing_record_fails_and_a_coin_flip_is_only_a_watch():
    losers = [('AAPL', 100, 96)] * 32 + [('MSFT', 100, 104)] * 8
    card = build(inputs(days=40, fills=fills_for(losers)), CONFIG, NOW)
    assert status(card, 'win_rate') == 'fail' and status(card, 'avg_return') == 'fail'
    flip = [('AAPL', 100, 103)] * 21 + [('MSFT', 100, 97)] * 19
    card = build(inputs(days=40, fills=fills_for(flip)), CONFIG, NOW)
    assert status(card, 'win_rate') == 'watch'
    assert 'luck cannot be ruled out' in metric(card, 'win_rate')['note']


def test_gains_from_a_single_symbol_are_flagged():
    trades = [('NVDA', 100, 110)] * 30 + [('AAPL', 100, 101)] * 2 + [('MSFT', 100, 98)] * 8
    m = metric(build(inputs(days=40, fills=fills_for(trades)), CONFIG, NOW), 'concentration')
    assert m['status'] == 'watch' and 'NVDA' in m['display'] and 'anecdote' in m['note']


# ------------------------------------------------------ returns and risk

def curve(values, start=NOW - timedelta(days=40)):
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


def spy_closes(start, n, daily):
    return {(start + timedelta(days=i)).date(): 400 * (1 + daily) ** i for i in range(n)}


def test_the_account_is_compared_with_the_sp500_over_the_same_days():
    start = NOW - timedelta(days=40)
    equity = curve([100_000 * 1.0005 ** i for i in range(41)])           # about +2%
    behind = build(inputs(days=40, equity=equity, spy=spy_closes(start, 41, 0.002)), CONFIG, NOW)
    m = metric(behind, 'vs_spy')
    assert m['status'] == 'fail' and 'SPY +8' in m['display']
    ahead = build(inputs(days=40, equity=equity, spy=spy_closes(start, 41, 0.0001)), CONFIG, NOW)
    assert status(ahead, 'vs_spy') == 'pass'


def test_a_young_account_is_not_judged_against_the_sp500():
    start = NOW - timedelta(days=5)
    card = build(inputs(days=5, equity=curve([100_000, 101_000, 99_500], start), spy=spy_closes(start, 6, 0.001)),
                 CONFIG, NOW)
    assert status(card, 'vs_spy') == 'too_early' and 'judged from 14' in metric(card, 'vs_spy')['note']


def test_drawdown_is_measured_peak_to_trough_against_the_limit():
    def dd(values):
        return metric(build(inputs(days=40, equity=curve(values)), CONFIG, NOW), 'drawdown')
    assert dd([100, 110, 99, 105])['value'] == pytest.approx(10.0)      # 110 -> 99
    assert dd([100, 110, 108])['status'] == 'pass'
    assert dd([100, 110, 101])['status'] == 'watch'                     # 8.2% of a 10% limit
    assert dd([100, 110, 95])['status'] == 'fail'
    assert dd([100])['status'] == 'too_early'


# --------------------------------------------------------------- forecasts

def closes_from(values, start=date(2026, 1, 5)):
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


def signal_rows(symbol, days, action, start=date(2026, 1, 5)):
    return [{'ts': datetime.combine(start + timedelta(days=d), datetime.min.time()).replace(hour=15).isoformat(),
             'symbol': symbol, 'action': action, 'skip_reason': 'hold', 'executed': 0, 'price': 100.0}
            for d in days]


def forecast_status(dec, closes, horizon=5):
    card = build(inputs(days=200, decisions=dec, closes_for=lambda s: closes), CONFIG,
                 datetime(2026, 12, 1, tzinfo=timezone.utc))
    return metric(card, f'forecast_{horizon}d')


def walk(n=300, seed=3):
    rng = random.Random(seed)
    v, out = 100.0, []
    for _ in range(n):
        v *= 1 + rng.gauss(0, 0.01)
        out.append(v)
    return out


def test_signals_that_call_the_price_move_beat_chance_and_pass():
    vals = walk()
    closes = closes_from(vals)
    days = [i for i in range(0, 290) if vals[i + 5] > vals[i]]               # buys placed before rises only
    m = forecast_status(signal_rows('AAPL', days, 'buy'), closes)
    assert m['status'] == 'pass' and m['value'] > 0.95
    wrong = [i for i in range(0, 290) if vals[i + 5] < vals[i]]              # buys placed before falls only
    assert forecast_status(signal_rows('AAPL', wrong, 'buy'), closes)['status'] == 'fail'


def test_a_rising_market_does_not_make_every_buy_look_skilful():
    """A buy in a market that only rises is right every time: that is the
    market's doing, so it must not pass."""
    closes = closes_from([100 * 1.004 ** i for i in range(300)])
    m = forecast_status(signal_rows('AAPL', range(0, 290), 'buy'), closes)
    assert m['value'] == 1.0 and m['status'] == 'watch'
    assert 'above 100%' in m['target']


def test_too_few_signals_with_a_known_outcome_is_too_early_and_says_how_many():
    vals = walk()
    m = forecast_status(signal_rows('AAPL', range(0, 20, 2), 'buy'), closes_from(vals))
    assert m['status'] == 'too_early' and '10 with a known outcome' in m['display']
    m = forecast_status(signal_rows('AAPL', range(0, 120, 2), 'buy'), closes_from(vals))     # 60: some evidence, no verdict
    assert m['status'] == 'watch' and 'a verdict needs 100' in m['note']


def test_signals_are_counted_once_a_day_whether_or_not_they_traded():
    rows = signal_rows('AAPL', [1, 1, 1, 2], 'buy') + signal_rows('AAPL', [1], 'sell')
    rows[0]['executed'] = 1
    sigs = scorecard.signals_from(rows, CONFIG)
    assert sorted((s['day'].day, s['action']) for s in sigs) == [(6, 'buy'), (6, 'sell'), (7, 'buy')]


def test_without_price_history_the_forecast_says_so_rather_than_guessing():
    card = build(inputs(decisions=signal_rows('AAPL', [1, 2, 3], 'buy'), closes_for=None), CONFIG, NOW)
    assert status(card, 'forecast_5d') == 'too_early'
    assert 'not available' in metric(card, 'forecast_5d')['note']


# ---------------------------------------------------------------- learning

def test_learning_starts_once_a_strategy_has_five_closed_trades():
    few = build(inputs(days=40, fills=fills_for([('AAPL', 100, 104)] * 4)), CONFIG, NOW)
    assert status(few, 'learning_gate') == 'watch'                       # day 40 and still nothing to learn from
    enough = build(inputs(days=40, fills=fills_for([('AAPL', 100, 104)] * 6)), CONFIG, NOW)
    assert status(enough, 'learning_gate') == 'pass'
    assert metric(enough, 'strategies_with_evidence')['detail'] == {'momentum': 6}


def test_the_weekly_settings_review_is_checked_for_staleness():
    def run(last):
        tuner = {'last_run': last, 'log': [{'outcome': 'adopted'}, {'outcome': 'kept: nothing beat it'}]}
        return metric(build(inputs(days=40, tuner=tuner), CONFIG, NOW), 'tuner')
    assert run(NOW.timestamp() - 3 * 86400)['status'] == 'pass'
    assert '1 changes adopted' in run(NOW.timestamp() - 3 * 86400)['note']
    assert run(NOW.timestamp() - 30 * 86400)['status'] == 'watch'
    assert status(build(inputs(days=40, tuner=None), CONFIG, NOW), 'tuner') == 'watch'
    assert status(build(inputs(days=3, tuner=None), CONFIG, NOW), 'tuner') == 'too_early'
    off = build(inputs(days=40), {**CONFIG, 'strategy_tuning': {'enabled': False}}, NOW)
    assert status(off, 'tuner') == 'info'


# ------------------------------------------------------ the whole verdict

def test_a_run_where_everything_passes_meets_the_bar_but_does_not_promise_anything():
    start = NOW - timedelta(days=90)
    winners = [('AAPL', 100, 104)] * 20 + [('MSFT', 100, 103)] * 20 + [('NVDA', 100, 97)] * 4
    vals = walk(400)
    days = [i for i in range(0, 380) if vals[i + 5] > vals[i]][:130]
    later = datetime(2027, 1, 20, tzinfo=timezone.utc)
    inp = inputs(days=90, fills=fills_for(winners), equity=curve([100_000 * 1.001 ** i for i in range(91)], start),
                 spy=spy_closes(start, 91, 0.0002), closes_for=lambda s: closes_from(vals),
                 decisions=hourly_decisions(start, 90 * 24) + signal_rows('AAPL', days, 'buy'),
                 healing={'halted': False, 'workers': [{'worker': 'w', 'alive': True, 'gave_up': False}]},
                 alerter={'configured': True, 'recipients': ['a***@x.com'], 'sent_24h': 0},
                 tuner={'last_run': later.timestamp() - 86400, 'log': []})
    card = build(inp, CONFIG, NOW)
    fails = [m['id'] for s in card['sections'] for m in s['metrics'] if m['status'] == 'fail']
    assert fails == [], fails
    assert card['verdict']['level'] in ('on_track', 'meets_bar', 'too_early')


# ------------------------------------------------- reading the real journals

def test_the_scorecard_is_built_from_the_journals_and_never_writes_to_them(tmp_path):
    started = datetime(2026, 9, 28, 11, 31, 30, tzinfo=timezone.utc)
    (tmp_path / 'paper_runs').mkdir()
    (tmp_path / 'paper_runs' / 'run_1.json').write_text(json.dumps(
        {'started_at': started.isoformat(), 'planned_days': 90}))

    dj = DecisionJournal(str(tmp_path / 'decision_journal.db'))
    for i in range(30):
        dj._conn.execute(
            "INSERT INTO decisions (ts, cycle, symbol, action, executed, skip_reason, price) VALUES (?,?,?,?,?,?,?)",
            ((started + timedelta(hours=i)).replace(tzinfo=None).isoformat(), i, 'AAPL', 'hold', 0,
             'fallback_price' if i < 3 else 'hold', 190.0))
    dj._conn.execute("INSERT INTO decisions (ts, cycle, symbol, action, executed, skip_reason, price) VALUES (?,?,?,?,?,?,?)",
                     ('2026-09-20T10:00:00', 1, 'OLD', 'hold', 0, 'hold', 1.0))   # before the run began
    dj._conn.commit()

    oj = OrderJournal(str(tmp_path / 'order_journal.db'))
    for coid, side, price in (('b1', 'buy', 100.0), ('s1', 'sell', 106.0)):
        oj.record_intent(coid, 'AAPL', side, 5, 'market', strategy='momentum')
        oj.mark_final(coid, 'filled', 5, price)

    conn = sqlite3.connect(tmp_path / 'escalations.db')
    conn.execute("CREATE TABLE portfolio_history (timestamp TEXT PRIMARY KEY, value REAL NOT NULL)")
    for i, v in enumerate((100_000, 101_000, 99_000)):
        conn.execute("INSERT INTO portfolio_history VALUES (?, ?)", ((started + timedelta(days=i)).isoformat(), v))
    conn.execute("INSERT INTO portfolio_history VALUES (?, ?)", ('2026-09-01T00:00:00+00:00', 50_000))   # before
    conn.commit()
    conn.close()
    (tmp_path / 'strategy_params.json').write_text(json.dumps({'last_run': 1.0, 'log': []}))

    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir() if p.suffix in ('.db', '.json')}
    now = started + timedelta(days=2)
    inp = scorecard.read_inputs(str(tmp_path), CONFIG, now=now)
    card = build(inp, CONFIG, now)
    assert len(inp['decisions']) == 30 and len(inp['fills']) == 2 and len(inp['equity']) == 3   # only the run
    assert card['run']['planned_days'] == 90 and card['run']['day'] == 2.0
    assert metric(card, 'closed_trades')['value'] == 1
    assert metric(card, 'degraded')['value'] == 10.0                                             # 3 of 30
    assert metric(card, 'drawdown')['value'] == pytest.approx(100 * 2000 / 101000, abs=0.01)
    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir() if p.suffix in ('.db', '.json')} == before


def test_an_empty_data_folder_gives_a_scorecard_not_an_error(tmp_path):
    card = build(scorecard.read_inputs(str(tmp_path), CONFIG, now=NOW), CONFIG, NOW)
    assert card['verdict']['level'] in ('too_early', 'needs_attention')
    json.dumps(card)


# ---------------------------------------------------------------- the API

def test_the_dashboard_endpoint_serves_the_scorecard(tmp_path, monkeypatch):
    import os
    os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')
    import bcrypt
    import src.api.auth as auth
    from types import SimpleNamespace
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'viewer': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'viewer'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    monkeypatch.setattr('src.utils.paths.DATA_DIR', tmp_path)
    healing = SimpleNamespace(status=lambda: {'halted': False, 'workers': [], 'symbols_without_price': {}})
    alerter = SimpleNamespace(status=lambda: {'configured': False})
    agent = SimpleNamespace(components={}, config={'data_manager': {'test_mode': True, 'symbols': []},
                                                    'test_mode': True, **CONFIG},
                            self_healing=healing, alerter=alerter)
    client = TradingAPI(agent, {}).app.test_client()
    headers = {'Authorization': f"Bearer {create_token('viewer', 'viewer')}"}
    r = client.get('/api/scorecard', headers=headers)                 # a viewer may read it: it has no decision internals
    assert r.status_code == 200
    card = r.get_json()
    assert [s['id'] for s in card['sections']][0] == 'health' and card['verdict']['level']
    assert next(m for s in card['sections'] for m in s['metrics'] if m['id'] == 'email')['status'] == 'watch'
    assert client.get('/api/scorecard').status_code in (401, 403)      # but not anonymously


def test_only_the_rows_the_scorecard_needs_are_loaded(tmp_path):
    started = datetime(2026, 9, 1, tzinfo=timezone.utc)
    (tmp_path / 'paper_runs').mkdir()
    (tmp_path / 'paper_runs' / 'run_1.json').write_text(json.dumps({'started_at': started.isoformat()}))
    dj = DecisionJournal(str(tmp_path / 'decision_journal.db'))
    rows = [('2026-09-05T10:00:00', 'hold'), ('2026-09-05T11:00:00', 'buy'), ('2026-09-29T10:00:00', 'hold')]
    for ts, action in rows:
        dj._conn.execute("INSERT INTO decisions (ts, cycle, symbol, action, executed, skip_reason, price) "
                         "VALUES (?,?,?,?,?,?,?)", (ts, 1, 'AAPL', action, 0, action, 1.0))
    dj._conn.commit()
    got = scorecard.read_inputs(str(tmp_path), CONFIG, now=datetime(2026, 9, 30, tzinfo=timezone.utc))['decisions']
    assert [(d['ts'][:10], d['action']) for d in got] == [('2026-09-05', 'buy'), ('2026-09-29', 'hold')]


# ------------------------------------------------- the approved targets

def test_the_approved_targets_in_config_json_match_what_the_scorecard_uses():
    from src.utils.config_validator import load_config, validate_config
    config = load_config('config/config.json')
    assert validate_config(config) == []
    written = {k: v for k, v in config['scorecard']['targets'].items()}
    unknown = set(written) - set(scorecard.DEFAULT_TARGETS)
    assert not unknown, f"not a real target: {unknown}"                       # a typo would be silently ignored
    used = targets(config)
    assert all(used[k] == v for k, v in written.items())
    assert used['max_drawdown_pct'] == 100 * config['risk_management']['max_drawdown']
    assert {**scorecard.DEFAULT_TARGETS, 'max_drawdown_pct': used['max_drawdown_pct']} == used
