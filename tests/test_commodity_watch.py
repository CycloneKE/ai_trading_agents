"""Watch-only commodities (src/agent/commodity_watch.py): the three patterns,
that none of them can peek at the future, how a pattern earns the word
CANDIDATE, how the live job keeps backfill and live evidence apart, and that
nothing here can place an order."""
import json
import os
import random
from datetime import date, datetime, timedelta, timezone

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import pytest

from src.agent import commodity_watch as cw


def weekdays(start, n):
    d, out = start, []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def series(days, fn):
    return {d: fn(i, d) for i, d in enumerate(days)}


def walk(days, drift=0.0, vol=0.01, seed=1, start=100.0):
    rng, price, out = random.Random(seed), start, {}
    for d in days:
        price *= 1 + drift + rng.gauss(0, vol)
        out[d] = price
    return out


DAYS = weekdays(date(2005, 1, 3), 252 * 16)


# ----------------------------------------------------------------- patterns

def test_the_trend_pattern_signals_only_on_the_day_the_price_crosses_its_200_day_average():
    # 300 days falling, then 300 rising: one cross up, no other signal.
    v = [200 - i * 0.2 for i in range(300)] + [140 + i * 0.5 for i in range(300)]
    days = weekdays(date(2020, 1, 6), len(v))
    sigs = cw.trend_signals(days, v)
    assert [s['action'] for s in sigs] == ['buy']
    i = days.index(sigs[0]['day'])
    assert v[i] > sum(v[i - 199:i + 1]) / 200 and v[i - 1] <= sum(v[i - 200:i]) / 200


def test_no_signal_is_given_before_there_is_enough_history():
    short = walk(weekdays(date(2024, 1, 1), 150))
    assert cw.signals_for('GC=F', 'metals', short) == []


def test_momentum_goes_with_the_past_twelve_months_once_a_month():
    days = weekdays(date(2018, 1, 1), 252 * 3)
    up = [cw.momentum_signals(days, [100 + i * 0.1 for i in range(len(days))])]
    down = cw.momentum_signals(days, [300 - i * 0.1 for i in range(len(days))])
    assert up[0] and {s['action'] for s in up[0]} == {'buy'}
    assert down and {s['action'] for s in down} == {'sell'}
    firsts = {d for d in days[1:] if d.month != days[days.index(d) - 1].month}
    assert {s['day'] for s in up[0]} <= firsts                         # only on a month's first trading day


def seasonal_world():
    """March always rises 8%, every other month is flat noise."""
    days = weekdays(date(2005, 1, 3), 252 * 16)
    rng, price, out, prev = random.Random(3), 100.0, {}, None
    for d in days:
        drift = 0.08 / 21 if d.month == 3 else 0.0
        price *= 1 + drift + rng.gauss(0, 0.002)
        out[d] = price
    return days, out


def test_the_seasonal_pattern_needs_five_earlier_years_and_then_buys_the_strong_month():
    days, closes = seasonal_world()
    d, v = cw._series(closes)
    sigs = [s for s in cw.seasonal_signals(d, v) if s['day'].month == 3]
    assert sigs and all(s['action'] == 'buy' for s in sigs)
    assert min(s['day'].year for s in sigs) == 2010                    # 2005..2009 are the five earlier years
    assert all(s['day'].year >= 2010 for s in cw.seasonal_signals(d, v))


def test_the_seasonal_pattern_cannot_see_the_future():
    # Signals computed on a series cut at a date equal the full series' signals up to that date,
    # even when what comes after is changed completely.
    days, closes = seasonal_world()
    cut = date(2014, 6, 30)
    kept = {d: c for d, c in closes.items() if d <= cut}
    changed = {**kept, **{d: 1.0 for d in closes if d > cut}}
    for f in (cw.seasonal_signals, cw.momentum_signals, cw.trend_signals):
        d1, v1 = cw._series(kept)
        d2, v2 = cw._series(changed)
        early = [s for s in f(d2, v2) if s['day'] <= cut]
        assert early == f(d1, v1)


def test_signals_are_given_in_the_shape_the_learning_ledger_scores():
    days, closes = seasonal_world()
    sigs = cw.signals_for('ZW=F', 'agriculture', closes, 'backfill')
    assert sigs and {'symbol', 'day', 'source', 'action', 'market', 'version'} <= set(sigs[0])
    assert {s['source'] for s in sigs} <= set(cw.SOURCES) and {s['market'] for s in sigs} == {'agriculture'}


def test_the_snapshot_says_what_each_pattern_says_now():
    days, closes = seasonal_world()
    snap = cw.snapshot('GC=F', closes)
    assert snap['enough_history'] and isinstance(snap['above_200d'], bool)
    assert snap['seasonal']['years'] >= 5
    assert cw.snapshot('GC=F', {d: 1.0 for d in DAYS[:50]})['enough_history'] is False


# -------------------------------------------------------------------- study

def rows_for(closes_by, meta):
    return cw.run_study(closes_by, meta, min_signals=30)


def regime_walk(days, seed, run=500, drift=0.0008, vol=0.007):
    """Trends that last about two years and then reverse, so chance is near 50%
    and following the past year's direction genuinely pays."""
    rng, price, out = random.Random(seed), 100.0, {}
    for i, d in enumerate(days):
        sign = 1 if (i // run + seed) % 2 == 0 else -1
        price *= 1 + sign * drift + rng.gauss(0, vol)
        out[d] = price
    return out


def test_a_pattern_that_really_works_is_a_candidate_and_luck_is_not():
    trending = {f'T{i}': regime_walk(DAYS, seed=i) for i in range(8)}
    result = rows_for(trending, {s: {'group': 'metals'} for s in trending})
    mom = next(r for r in result['table'] if r['source'] == 'momentum_12m' and r['market'] == 'all_commodities')
    assert 0.35 < mom['chance'] < 0.65                                 # not just a market that always rises
    assert mom['verdict'] == 'candidate' and mom['z'] >= 2 and mom['early_lift'] > 0 and mom['late_lift'] > 0



def test_pure_noise_is_almost_never_called_a_candidate():
    # 30 made-up markets with no pattern in them at all: a two-standard-error test with the
    # both-halves rule should be fooled rarely (here, at most 2 in 90 verdicts).
    fooled = total = 0
    for k in range(30):
        noise = {f'N{i}': walk(DAYS, drift=0.0, vol=0.01, seed=5000 + k * 10 + i) for i in range(6)}
        result = rows_for(noise, {s: {'group': 'energy'} for s in noise})
        for r in result['table']:
            if r['market'] == 'all_commodities':
                total += 1
                fooled += r['verdict'] == 'candidate'
    assert total == 90 and fooled <= 2


def test_a_market_that_only_rises_gives_no_credit_for_buying_it():
    always_up = {'U': walk(DAYS, drift=0.0006, vol=0.004, seed=1)}
    table = cw.run_study(always_up, {'U': {'group': 'metals'}})['table']
    mom = next(r for r in table if r['source'] == 'momentum_12m' and r['market'] == 'all_commodities')
    assert mom['hit_rate'] > 0.7 and abs(mom['hit_rate'] - mom['chance']) < 0.05   # right often, but chance was too
    assert mom['verdict'] != 'candidate'


def test_a_pattern_that_only_worked_in_one_half_of_history_is_not_a_candidate():
    from src.agent.commodity_watch import _verdict
    assert _verdict(200, 3.0, 0.10, -0.02, 30) == 'no_clear_edge'
    assert _verdict(200, 3.0, 0.10, 0.04, 30) == 'candidate'
    assert _verdict(200, -3.0, -0.05, -0.04, 30) == 'wrong_way'
    assert _verdict(10, 5.0, 0.2, 0.2, 30) == 'too_few'


def test_the_assumed_cost_comes_off_the_average_move():
    noise = {'N': walk(DAYS, seed=7)}
    table = cw.run_study(noise, {'N': {'group': 'metals'}}, min_signals=5, cost_pct=0.25)['table']
    r = table[0]
    assert r['net_move_pct'] == pytest.approx(r['avg_move_pct'] - 0.25, abs=0.01)


# ---------------------------------------------------------------- the job

def watcher(tmp_path, closes_by, clock=None):
    clock = clock or (lambda: datetime(2020, 6, 1, tzinfo=timezone.utc).timestamp())
    cfg = {'watch_only': {'commodities': {'instruments': [
        {'symbol': s, 'name': s, 'group': 'metals', 'kind': 'futures'} for s in closes_by]}}}
    return cw.CommodityWatch(cfg, data_dir=str(tmp_path), closes_for=lambda s: closes_by.get(s, {}), clock=clock)


def test_the_watch_keeps_replayed_history_and_live_signals_apart(tmp_path):
    closes = {'GC=F': walk(DAYS, drift=0.0004, vol=0.006, seed=11)}
    last = max(closes['GC=F'])
    started = last - timedelta(days=200)
    w = watcher(tmp_path, closes, clock=lambda: datetime(started.year, started.month, started.day,
                                                        tzinfo=timezone.utc).timestamp())
    assert w.started == started
    result = w.run()
    assert result['instruments'] == 1 and result['new_outcomes'] > 100
    rep = cw.report(str(tmp_path), {})
    assert rep['started'] == started.isoformat() and rep['backfill']['signals'] > rep['live']['signals'] >= 0
    rows = w.ledger.rows()
    assert {r['version'] for r in rows} <= {'backfill', 'live'}
    assert all((r['day'] < started.isoformat()) == (r['version'] == 'backfill') for r in rows)
    assert rep['instruments'][0]['symbol'] == 'GC=F' and rep['instruments'][0]['group'] == 'metals'


def test_running_again_adds_nothing_and_the_start_date_survives_a_restart(tmp_path):
    closes = {'GC=F': walk(DAYS, seed=5)}
    w = watcher(tmp_path, closes)
    first = w.run()['new_outcomes']
    assert first > 0 and w.run()['new_outcomes'] == 0
    w.close()
    again = watcher(tmp_path, closes, clock=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())
    assert again.started == w.started                                  # read back, not reset


def test_a_commodity_with_no_prices_is_skipped_not_fatal(tmp_path):
    w = watcher(tmp_path, {'GC=F': walk(DAYS, seed=2)})
    w.items.append({'symbol': 'XX=F', 'name': 'x', 'group': 'energy', 'kind': 'futures'})
    assert w.run()['instruments'] == 1


def test_the_watch_updates_in_the_background_when_due_and_can_be_switched_off(tmp_path):
    t = {'now': 1_000_000.0}
    w = watcher(tmp_path, {'GC=F': walk(DAYS, seed=2)}, clock=lambda: t['now'])
    assert w.maybe_update() is True
    w._thread.join(timeout=30)
    assert w.maybe_update() is False                                   # not due again for hours
    t['now'] += 13 * 3600
    assert w.due()
    off = cw.CommodityWatch({'watch_only': {'commodities': {'enabled': False}, 'equity_markets': {'enabled': False}}},
                            data_dir=str(tmp_path / 'off'), closes_for=lambda s: {})
    assert off.maybe_update() is False and off.items == []


# ------------------------------------------------------------------ safety

def test_nothing_watched_is_traded_and_the_module_cannot_reach_a_broker():
    config = json.load(open('config/config.json', encoding='utf-8'))
    traded = set(config['data_manager']['symbols']) | set(config['data_manager'].get('nse_symbols', []))
    assert not {i['symbol'] for i in cw.instruments(config)} & traded
    source = open(cw.__file__, encoding='utf-8').read()
    for forbidden in ('broker', 'submit_order', 'place_order', 'order_journal'):
        code = [ln for ln in source.splitlines() if not ln.strip().startswith(('#', '"""')) and forbidden in ln
                and 'import' in ln]
        assert code == []


def test_the_shipped_watch_list_is_sound():
    items = cw.instruments(json.load(open('config/config.json', encoding='utf-8')))
    symbols = [i['symbol'] for i in items]
    assert len(symbols) == len(set(symbols)) >= 20
    assert {i['group'] for i in items} == {'metals', 'energy', 'agriculture', 'uk_equities', 'japan_equities',
                                           'france_equities', 'south_africa_equities'}
    assert {i['kind'] for i in items} == {'futures', 'fund', 'stock', 'index'}


# --------------------------------------------------------------------- API

def test_the_dashboard_can_read_the_watch_report(tmp_path, monkeypatch):
    import bcrypt
    from types import SimpleNamespace
    import src.api.auth as auth
    import src.api.api_server as api_server
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'operator': {'password': bcrypt.hashpw(b'x', bcrypt.gensalt()).decode(),
                                              'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    import src.utils.paths as paths
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    w = watcher(tmp_path, {'GC=F': walk(DAYS, seed=9)})
    w.run()
    client = api_server.TradingAPI(SimpleNamespace(components={}, config={}), {}).app.test_client()
    r = client.get('/api/watch/commodities', headers={'Authorization': f"Bearer {create_token('operator', 'operator')}"})
    data = r.get_json()
    assert r.status_code == 200 and data['enabled'] is True and data['horizon'] == 20
    assert data['backfill']['signals'] > 0 and data['instruments'][0]['symbol'] == 'GC=F'
    assert client.get('/api/watch/commodities').status_code in (401, 403)


# ------------------------------------------------------------ stock markets

def test_the_four_stock_markets_each_have_an_index_a_fund_and_several_companies_with_a_unit():
    items = cw.instruments({})
    for group, suffix in (('uk_equities', '.L'), ('japan_equities', '.T'), ('france_equities', '.PA'),
                          ('south_africa_equities', '.JO')):
        mine = [i for i in items if i['group'] == group]
        assert {'index', 'fund', 'stock'} <= {i['kind'] for i in mine} and len(mine) >= 9
        assert all(i['quote'] for i in mine)
        assert all(i['symbol'].endswith(suffix) or i['kind'] == 'index' for i in mine)
    assert {i['quote'] for i in items if i['group'] == 'uk_equities'} == {'points', 'pence'}
    assert {i['quote'] for i in items if i['group'] == 'south_africa_equities'} == {'points', 'cents'}


def test_either_family_can_be_switched_off_on_its_own():
    only_stocks = cw.instruments({'watch_only': {'commodities': {'enabled': False}}})
    assert only_stocks and all(cw.family(i['group']) == 'equities' for i in only_stocks)
    only_commodities = cw.instruments({'watch_only': {'equity_markets': {'enabled': False}}})
    assert only_commodities and all(cw.family(i['group']) == 'commodities' for i in only_commodities)
    assert cw.settings({'watch_only': {'commodities': {'enabled': False}}})['enabled'] is True
    both_off = {'watch_only': {'commodities': {'enabled': False}, 'equity_markets': {'enabled': False}}}
    assert cw.settings(both_off)['enabled'] is False


def test_a_listed_market_can_be_replaced_from_the_config():
    cfg = {'watch_only': {'equity_markets': {'instruments': [
        {'symbol': 'XYZ.L', 'name': 'X', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'}]}}}
    uk = [i for i in cw.instruments(cfg) if cw.family(i['group']) == 'equities']
    assert [i['symbol'] for i in uk] == ['XYZ.L']


def test_a_hundredfold_unit_change_in_the_price_is_undone_and_real_moves_are_left_alone():
    # London pence that Yahoo briefly serves as pounds: 100 -> 1 -> 100 over a stretch.
    v = [100.0, 101.0, 1.02, 1.03, 1.01, 102.0, 103.0]
    fixed, steps = cw.fix_unit_steps(v)
    assert steps == 2
    assert fixed[-2:] == [102.0, 103.0]                                    # the latest segment is untouched
    assert all(98 < x < 106 for x in fixed)                                # the dip segment matches its neighbours
    # a genuine 30% fall and 40% rise are not unit changes
    real = [100.0, 70.0, 98.0, 97.0]
    assert cw.fix_unit_steps(real) == (real, 0)
    # a series that moves permanently from pounds to pence is joined up
    joined, n = cw.fix_unit_steps([2.0, 2.1, 2.05, 205.0, 207.0])
    assert n == 1 and joined == pytest.approx([200.0, 210.0, 205.0, 205.0, 207.0])


def test_a_unit_glitch_cannot_fake_a_hundredfold_return_in_the_scoring():
    days = weekdays(date(2015, 1, 5), 252 * 8)
    clean = walk(days, drift=0.0, vol=0.01, seed=21, start=1000.0)
    glitched = dict(clean)
    for d in days[900:930]:
        glitched[d] = clean[d] / 100                                       # a month served in pounds, not pence
    fixed, n = cw.clean_closes(glitched)
    assert n == 2
    assert all(fixed[d] == pytest.approx(clean[d], rel=1e-9) for d in days)
    result = cw.run_study({'ULVR.L': glitched}, {'ULVR.L': {'group': 'uk_equities'}}, min_signals=5)
    assert result['unit_fixes'] == {'ULVR.L': 2}
    assert max(abs(r['avg_move_pct']) for r in result['table']) < 20        # not the 9,900% a raw glitch would give


def test_stock_markets_are_pooled_apart_from_commodities_and_costed_per_market():
    uk = {f'U{i}': regime_walk(DAYS, seed=i) for i in range(4)}
    gold = {f'G{i}': regime_walk(DAYS, seed=10 + i) for i in range(4)}
    meta = {**{s: {'group': 'uk_equities'} for s in uk}, **{s: {'group': 'metals'} for s in gold}}
    result = cw.run_study({**uk, **gold}, meta, min_signals=5, cost_pct=0.1, group_costs={'uk_equities': 0.7})
    markets = {r['market'] for r in result['table']}
    assert {'uk_equities', 'metals', 'all_equities', 'all_commodities'} <= markets
    by = {(r['source'], r['market']): r for r in result['table']}
    uk_row, metal_row = by[('momentum_12m', 'uk_equities')], by[('momentum_12m', 'metals')]
    assert uk_row['assumed_cost_pct'] == 0.7 and metal_row['assumed_cost_pct'] == 0.1
    assert uk_row['net_move_pct'] == pytest.approx(uk_row['avg_move_pct'] - 0.7, abs=0.01)
    pooled_eq = by[('momentum_12m', 'all_equities')]
    assert pooled_eq['n'] == uk_row['n'] and pooled_eq['assumed_cost_pct'] == 0.7


def test_each_stock_market_gets_its_own_live_start_day(tmp_path):
    days = weekdays(date(2015, 1, 5), 252 * 8)
    closes = {'GC=F': walk(days, seed=3), 'HSBA.L': walk(days, seed=4), 'MC.PA': walk(days, seed=5)}
    cfg = {'watch_only': {'commodities': {'instruments': [
               {'symbol': 'GC=F', 'name': 'g', 'group': 'metals', 'kind': 'futures'}]},
           'equity_markets': {'instruments': [
               {'symbol': 'HSBA.L', 'name': 'h', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'},
               {'symbol': 'MC.PA', 'name': 'm', 'group': 'france_equities', 'kind': 'stock', 'quote': 'euro'}]}}}
    day1 = datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp()
    w = cw.CommodityWatch(cfg, data_dir=str(tmp_path), closes_for=lambda s: closes.get(s, {}), clock=lambda: day1)
    assert w.started == date(2024, 3, 1) and w.group_started == {}
    w.run()
    assert w.group_started == {'uk_equities': date(2024, 3, 1), 'france_equities': date(2024, 3, 1)}
    # France is dropped, then comes back later: its start must not move earlier, and a market that
    # never returned prices has no start at all.
    day2 = datetime(2024, 9, 1, tzinfo=timezone.utc).timestamp()
    cfg2 = {**cfg, 'watch_only': {**cfg['watch_only'], 'equity_markets': {'instruments': cfg['watch_only']['equity_markets']['instruments'] + [
        {'symbol': '7203.T', 'name': 't', 'group': 'japan_equities', 'kind': 'stock', 'quote': 'yen'}]}}}
    again = cw.CommodityWatch(cfg2, data_dir=str(tmp_path), closes_for=lambda s: closes.get(s, {}), clock=lambda: day2)
    again.run()
    assert again.group_started == {'uk_equities': date(2024, 3, 1), 'france_equities': date(2024, 3, 1)}
    closes['7203.T'] = walk(days, seed=6)
    again.run()
    assert again.group_started['japan_equities'] == date(2024, 9, 1)
    rep = cw.report(str(tmp_path), cfg2)
    assert rep['started_by_group']['japan_equities'] == '2024-09-01' and rep['costs_pct']['uk_equities'] == 0.7
    rows = again.ledger.rows()
    jp = [r for r in rows if r['symbol'] == '7203.T']
    assert jp and all((r['day'] < '2024-09-01') == (r['version'] == 'backfill') for r in jp)
    uk = [r for r in rows if r['symbol'] == 'HSBA.L']
    assert all((r['day'] < '2024-03-01') == (r['version'] == 'backfill') for r in uk)


def test_the_snapshot_carries_each_markets_unit_and_the_cleaning_it_needed(tmp_path):
    days = weekdays(date(2015, 1, 5), 252 * 3)
    base = walk(days, seed=8, start=2000.0)
    glitched = {d: (c / 100 if i in (400, 401) else c) for i, (d, c) in enumerate(base.items())}
    cfg = {'watch_only': {'commodities': {'enabled': False}, 'equity_markets': {'instruments': [
        {'symbol': 'ULVR.L', 'name': 'Unilever', 'group': 'uk_equities', 'kind': 'stock', 'quote': 'pence'}]}}}
    w = cw.CommodityWatch(cfg, data_dir=str(tmp_path), closes_for=lambda s: glitched)
    w.run()
    snap = cw.report(str(tmp_path), cfg)['instruments'][0]
    assert snap['quote'] == 'pence' and snap['group'] == 'uk_equities' and snap['kind'] == 'stock'
    assert snap['unit_fixes'] == 2


def test_no_stock_market_instrument_is_traded_by_the_agent():
    config = json.load(open('config/config.json', encoding='utf-8'))
    traded = set(config['data_manager']['symbols']) | set(config['data_manager'].get('nse_symbols', []))
    watched = {i['symbol'] for i in cw.instruments(config)}
    assert not watched & traded
