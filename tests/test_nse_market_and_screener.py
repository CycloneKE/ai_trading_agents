"""Step B, the whole NSE in Market Watch, and step C, the screener and the
short list the agent trades (nse_universe.py, nse_isin.py,
nse_pricelist.learn_isins, nse_screener.py, TradingAgent's NSE list)."""
import csv
import json
import types
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.agent import nse_screener as scr
from src.connectors import nse_isin, nse_pricelist as pl, nse_scraper, nse_universe
from src.connectors.nse_pricelist import Box
from tests.test_nse_pricelist import HEADINGS, WIDTH, _line


# ------------------------------------------------------------- the feed

def _payload(rows, day='25/09/2026'):
    return {'message': [{'snapshot': rows}, {'updated_at': {'date': day}}]}


FEED = [{'issuer': 'SCOM', 'price': 36.2, 'prev_price': 36.6, 'volume': 100},
        {'issuer': 'JUB', 'price': 180.0, 'prev_price': 178.0, 'volume': 50, 'isin': 'KE0000000273'},
        {'issuer': 'HAFR', 'price': 0.4, 'prev_price': 0.4, 'board': 'KE0000000999'},
        # A bond and a path: never stocks, never stored.
        {'issuer': 'IFB1/2023/17', 'price': 100.0, 'prev_price': 100.0},
        {'issuer': '../X', 'price': 1.0, 'prev_price': 1.0}]


def test_the_feed_gives_every_listed_stock_when_no_list_is_asked_for():
    today = date(2026, 9, 25)
    assert set(nse_scraper.parse_ticker_reply(_payload(FEED), None, today)) == {'SCOM', 'JUB', 'HAFR'}
    assert set(nse_scraper.parse_ticker_reply(_payload(FEED), ['SCOM'], today)) == {'SCOM'}


def test_an_isin_in_a_feed_row_is_read_beside_its_ticker():
    # HAFR's ISIN-shaped value is in some other field: not taken as its ISIN.
    assert nse_scraper.ticker_isins(_payload(FEED)) == {'KE0000000273': 'JUB'}


def test_one_issuer_that_cannot_be_stored_costs_the_others_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_scraper, 'DATA_DIR', tmp_path)
    today = date(2026, 9, 25)
    bars = nse_scraper.parse_ticker_reply(_payload(FEED), None, today)
    bars['ODD'] = bars['HAFR'].__class__(**{**bars['HAFR'].__dict__, 'symbol': 'ODD'})
    monkeypatch.setattr(nse_scraper, 'scrape_nse_ticker', lambda symbols=None: dict(bars))
    monkeypatch.setattr(nse_scraper, 'scrape_afx_kwayisi', lambda symbols: {})
    real_save = nse_scraper.save_bars_csv

    def save(symbol, rows):
        if symbol == 'ODD':
            raise OSError('disk said no')
        real_save(symbol, rows)
    monkeypatch.setattr(nse_scraper, 'save_bars_csv', save)
    results = nse_scraper.NSEPeriodicScraper(symbols=['SCOM']).run_once()
    assert results == {'SCOM': 1}
    assert {p.stem for p in tmp_path.glob('*.csv')} == {'SCOM', 'JUB', 'HAFR'}
    assert not (tmp_path.parent / 'X.csv').exists()
    with pytest.raises(ValueError):
        nse_scraper.save_bars_csv('../X', [])


def test_the_scraper_stores_a_bar_for_every_listed_stock(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_scraper, 'DATA_DIR', tmp_path)
    bars = nse_scraper.parse_ticker_reply(_payload(FEED), None, date(2026, 9, 25))
    monkeypatch.setattr(nse_scraper, 'scrape_nse_ticker', lambda symbols=None: dict(bars))
    monkeypatch.setattr(nse_scraper, 'scrape_afx_kwayisi', lambda symbols: {})
    s = nse_scraper.NSEPeriodicScraper(symbols=['SCOM', 'KCB'])
    results = s.run_once()
    assert results == {'SCOM': 1, 'KCB': 0}                      # the watched ones, as before
    assert {p.stem for p in tmp_path.glob('*.csv')} == {'SCOM', 'JUB', 'HAFR'}
    assert s.get_status()['last_cycle_listed'] == 3


# ----------------------------------------------------------- the universe

def _csv(folder, symbol, rows):
    with open(folder / f'{symbol}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['date', 'open', 'high', 'low', 'close', 'volume',
                                          'change_pct', 'source'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _history(close_fn, days=80, volume=100_000, end=date(2026, 9, 25), source='nse_ticker'):
    out, d = [], end
    while len(out) < days:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return [{'date': x.isoformat(), 'close': round(close_fn(i), 2), 'volume': volume,
             'change_pct': 0, 'source': source} for i, x in enumerate(sorted(out))]


def test_market_watch_lists_real_data_and_the_traded_list(tmp_path):
    _csv(tmp_path, 'JUB', _history(lambda i: 180, days=1))
    _csv(tmp_path, 'SMER', _history(lambda i: 3, days=1, source='synthetic'))
    assert nse_universe.market_symbols(tmp_path, ['KCB']) == ['JUB', 'KCB']
    assert nse_universe.name_of('JUB') == 'Jubilee Holdings'
    assert nse_universe.sector_of('XYZ') == nse_universe.UNCLASSIFIED


def test_any_listed_nse_stock_is_costed_as_nse_but_not_a_us_fund(tmp_path, monkeypatch):
    import src.connectors.nse_connector as nse_connector
    from src.agent.cost_model import classify
    monkeypatch.setattr(nse_universe, '_seen', set())
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', tmp_path)
    cfg = {'data_manager': {'symbols': ['SPY'], 'nse_symbols': ['SCOM']},
           'core_satellite': {'core_symbols': ['VOO']}}
    assert classify('JUB', cfg) == 'nse'
    assert classify('SPY', cfg) == 'us_equity'
    nse_universe.register(['VOO'])       # even if the feed once listed a same-named code
    assert classify('VOO', cfg) == 'us_equity'
    # A stock outside the built-in list is known from its history on disk,
    # so the answer is the same straight after a restart.
    assert classify('NEWCO', cfg) == 'us_equity'
    _csv(tmp_path, 'NEWCO', _history(lambda i: 10, days=1))
    assert classify('NEWCO', cfg) == 'nse'


# --------------------------------------------------------- learning ISINs

def _row(isin, vwap, previous, y):
    return _line(y, [(isin, 676), ("100.00", 911), ("90.00", 1008), (f"{vwap:.2f}", 1117),
                     (f"{previous:.2f}", 1258), ("10,000", 1366)])


def test_every_isin_row_is_read_known_or_not():
    page = HEADINGS + _row('KE0000000273', 180.0, 178.0, 600)
    rows, _ = pl.read_isin_rows(page, WIDTH)
    assert rows['KE0000000273']['vwap'] == 180.0
    assert pl.read_page(page, WIDTH)[0] == {}                 # not in the verified nine


def _sessions(n=12):
    return [date(2026, 9, 1) + timedelta(days=i) for i in range(n)]


def _moving(days):
    """A price that rises 3% every other session and holds in between."""
    px, out = 100.0, {}
    for i, d in enumerate(days):
        prev = px
        px = round(px * 1.03, 2) if i % 2 else px
        out[d] = (px, prev)
    return out


def test_an_unknown_isin_is_learned_when_one_stock_matches_session_after_session():
    days = _sessions()
    jub = _moving(days)
    readings = {d: {'KE0000000273': {'vwap': c, 'previous': p, 'volume': 1000}} for d, (c, p) in jub.items()}
    recorded = {'JUB': jub, 'SCOM': {d: (36.0, 36.0) for d in days}}
    volumes = {'JUB': {d: 900 for d in days}}                   # the feed's last reading, before the close
    assert pl.learn_isins(readings, recorded, nse_isin.VERIFIED, volumes=volumes) == {'KE0000000273': 'JUB'}


def test_an_isin_is_not_learned_on_too_little_ambiguous_or_contradicted_evidence():
    days = _sessions()
    moving = _moving(days)
    readings = {d: {'KE0000000273': {'vwap': c, 'previous': p, 'volume': 1000}} for d, (c, p) in moving.items()}
    twins = {'AAA': dict(moving), 'BBB': dict(moving)}
    assert pl.learn_isins(readings, twins, {}) == {}                          # two match
    short = {'AAA': {d: moving[d] for d in days[:9]}}
    assert pl.learn_isins(readings, short, {}) == {}                          # nine sessions
    bad = {'AAA': {**moving, days[5]: (60.0, 60.0)}}
    assert pl.learn_isins(readings, bad, {}) == {}                            # contradicted
    taken = {'SCOM': dict(moving)}
    assert pl.learn_isins(readings, taken, nse_isin.VERIFIED) == {}           # SCOM has one
    busier = {'AAA': dict(moving)}
    assert pl.learn_isins(readings, busier, {}, volumes={'AAA': {days[3]: 5000}}) == {}  # traded more


def test_a_flat_price_never_teaches_an_isin():
    # An ISIN the feed does not record (a preference share, say) printing a
    # flat 6.00, and a stock that happens to sit at 6.02 every day.
    days = _sessions(30)
    readings = {d: {'KE0000000364': {'vwap': 6.0, 'previous': 6.0}} for d in days}
    assert pl.learn_isins(readings, {'NSE': {d: (6.02, 6.02) for d in days}}, {}) == {}


def test_bars_that_disagree_with_the_feed_record_are_caught():
    from src.connectors.nse_scraper import DailyBar
    bar = lambda d, c: DailyBar(date=d.isoformat(), symbol='JUB', open=c, high=c, low=c, close=c,
                                volume=1, change_pct=0.0, source='nse_pricelist')
    days = _sessions(4)
    recorded = {d: (100.0, 100.0) for d in days[:3]}
    assert pl.feed_agreement([bar(d, 100.4) for d in days], recorded) == (3, 0)
    assert pl.feed_agreement([bar(d, 6.0) for d in days], recorded) == (0, 3)


def test_learned_isins_are_kept_without_overwriting_or_doubling(tmp_path):
    path = tmp_path / 'map.json'
    assert nse_isin.remember({'KE0000000273': 'JUB'}, 'test', path) == {'KE0000000273': 'JUB'}
    assert nse_isin.remember({'KE0000000273': 'KQ'}, 'test', path) == {}      # ISIN taken
    assert nse_isin.remember({'KE9999999999': 'JUB'}, 'test', path) == {}     # ticker taken
    assert nse_isin.remember({'KE1000001402': 'XX'}, 'test', path) == {}      # a verified one
    m = nse_isin.load(path)
    assert m['KE0000000273'] == 'JUB' and m['KE1000001402'] == 'SCOM'
    assert json.loads(path.read_text())['KE0000000273']['how'] == 'test'


def test_the_feed_corrects_a_mapping_inferred_from_prices(tmp_path):
    path = tmp_path / 'map.json'
    nse_isin.remember({'KE0000000364': 'NSE'}, 'price lists', path)
    # The feed says that ISIN is another stock, and NSE has its own.
    assert nse_isin.remember({'KE0000000364': 'KQ'}, 'feed', path, from_feed=True) == {'KE0000000364': 'KQ'}
    assert nse_isin.remember({'KE0000000999': 'NSE'}, 'feed', path, from_feed=True) == {'KE0000000999': 'NSE'}
    m = nse_isin.load(path)
    assert m['KE0000000364'] == 'KQ' and m['KE0000000999'] == 'NSE'
    # A feed mapping is not replaced by another, nor a verified one by anything.
    assert nse_isin.remember({'KE0000000364': 'JUB'}, 'feed', path, from_feed=True) == {}
    assert nse_isin.remember({'KE1000001402': 'XX'}, 'feed', path, from_feed=True) == {}


def test_a_damaged_map_is_never_overwritten(tmp_path):
    path = tmp_path / 'map.json'
    path.write_text('{"KE0000000273": {"symbol": "JUB"')         # cut off mid-write
    assert nse_isin.remember({'KE0000000364': 'KQ'}, 'test', path) == {}
    assert path.read_text() == '{"KE0000000273": {"symbol": "JUB"'
    assert nse_isin.load(path)['KE1000001402'] == 'SCOM'           # the verified nine still work


# --------------------------------------------------------------- screener

CFG = {**scr.DEFAULTS, 'enabled': True, 'min_history_days': 50, 'min_avg_value_kes': 500_000,
       'max_symbols': 5, 'max_per_sector': 2, 'exploration_slots': 1}
TODAY = date(2026, 9, 26)


def _exchange(tmp_path):
    # Banks: KCB best, then EQTY, COOP, ABSA; SCOM strong; JUB rising;
    # KAPC too thin to trade; NBV too new.
    spec = {'KCB': 0.004, 'EQTY': 0.003, 'COOP': 0.002, 'ABSA': 0.001, 'SCOM': 0.0035,
            'JUB': 0.0025, 'EABL': -0.001, 'BAT': 0.0005}
    for sym, drift in spec.items():
        _csv(tmp_path, sym, _history(lambda i, g=drift: 100 * (1 + g) ** i))
    _csv(tmp_path, 'KAPC', _history(lambda i: 200 * 1.01 ** i, volume=10))
    _csv(tmp_path, 'NBV', _history(lambda i: 5 + i * 0.1, days=20))
    return list(spec) + ['KAPC', 'NBV']


def test_the_screen_ranks_eligible_stocks_and_says_why_others_are_not(tmp_path):
    ranked = scr.screen(_exchange(tmp_path), tmp_path, CFG, TODAY)
    by = {m.symbol: m for m in ranked}
    assert ranked[0].symbol == 'KCB' and ranked[0].rank == 1
    assert all(0 <= m.score <= 1 for m in ranked if m.eligible)
    assert all(m.score is None for m in ranked if not m.eligible)
    assert not by['KAPC'].eligible and 'a day' in by['KAPC'].reason
    assert not by['NBV'].eligible and '20 of 50 days' in by['NBV'].reason
    assert by['EABL'].eligible and by['EABL'].above_sma50 is False


def test_the_short_list_keeps_holdings_caps_sectors_and_explores(tmp_path):
    ranked = scr.screen(_exchange(tmp_path), tmp_path, CFG, TODAY)
    sl = scr.build(ranked, holdings=['BAT'], cfg=CFG, week=0)
    assert sl.roles['BAT'] == 'holding'
    assert len(sl.symbols) == 5
    banks = [s for s in sl.symbols if nse_universe.sector_of(s) == 'Banking']
    assert banks == ['KCB', 'EQTY']                                  # two banks at most
    assert [s for s, r in sl.roles.items() if r == 'exploration'] and 'KAPC' not in sl.symbols
    other_week = scr.build(ranked, holdings=['BAT'], cfg=CFG, week=1)
    explore = lambda s: [x for x, r in s.roles.items() if r == 'exploration']
    assert explore(sl) != explore(other_week)                        # it rotates


def test_the_configured_list_fills_in_while_few_stocks_qualify():
    sl = scr.build([], holdings=[], cfg=CFG, fallback=['SCOM', 'KCB', 'EQTY', 'COOP'])
    assert sl.symbols == ['SCOM', 'KCB', 'EQTY']                     # third bank capped
    assert set(sl.roles.values()) == {'fallback'}


def test_the_configured_list_never_fills_in_with_a_suspended_or_thin_stock(tmp_path):
    _csv(tmp_path, 'SCOM', _history(lambda i: 30, days=10))                          # new, liquid
    _csv(tmp_path, 'KCB', _history(lambda i: 40, days=60, end=date(2026, 8, 28)))   # no price for a month
    _csv(tmp_path, 'EGAD', _history(lambda i: 15, days=30, volume=10))              # a few shares a day
    ranked = scr.screen(['SCOM', 'KCB', 'EGAD', 'EQTY'], tmp_path, CFG, TODAY)
    by = {m.symbol: m for m in ranked}
    assert by['KCB'].stale and by['EGAD'].thin and not by['SCOM'].thin
    sl = scr.build(ranked, holdings=[], cfg=CFG, fallback=['SCOM', 'KCB', 'EGAD', 'EQTY'])
    assert sl.symbols == ['SCOM', 'EQTY']                             # EQTY: no data yet, still tried


def test_history_counts_only_finished_sessions():
    rows = [{'date': (date(2026, 7, 1) + timedelta(days=i)).isoformat(), 'close': 10.0,
             'volume': 1e6, 'change_pct': 0.0} for i in range(50)]
    today = date.fromisoformat(rows[-1]['date'])                      # the last bar is today's
    m = scr.metrics_for('SCOM', rows, CFG, today)
    assert m.days == 49 and not m.eligible and '49 of 50' in m.reason


def test_one_damaged_history_file_does_not_break_the_screen(tmp_path):
    _csv(tmp_path, 'SCOM', _history(lambda i: 30, days=60))
    (tmp_path / 'KCB.csv').write_bytes(b'date,close,volume,source\n2026-09-25,40,lots,nse_ticker\n\xff\xfe')
    ranked = scr.screen(['SCOM', 'KCB'], tmp_path, CFG, TODAY)
    assert {m.symbol for m in ranked} == {'SCOM', 'KCB'}
    assert nse_universe.symbols_with_real_data(tmp_path) == ['SCOM']


class _Llm:
    enabled = True

    def __init__(self, reply):
        self.reply, self.asked = reply, []

    def propose_json(self, system, user):
        self.asked.append(user)
        return self.reply


def test_the_ai_may_remove_a_name_for_a_reason_but_never_a_holding(tmp_path):
    ranked = scr.screen(_exchange(tmp_path), tmp_path, CFG, TODAY)
    llm = _Llm({'remove': [{'symbol': 'KCB', 'reason': 'suspended from trading'},
                           {'symbol': 'BAT', 'reason': 'held'},
                           {'symbol': 'SCOM', 'reason': ''}], 'notes': 'one suspension'})
    sl = scr.refresh(ranked, ['BAT'], CFG, llm=llm, now=datetime(2026, 9, 28, tzinfo=timezone.utc))
    assert 'KCB' not in sl.symbols and 'BAT' in sl.symbols and 'SCOM' in sl.symbols
    assert sl.removed == [{'symbol': 'KCB', 'reason': 'suspended from trading'}]
    assert sl.reviewed_by_ai and sl.notes == 'one suspension' and len(sl.symbols) == 5
    silent = scr.refresh(ranked, [], CFG, llm=_Llm(None))
    assert not silent.reviewed_by_ai and 'KCB' in silent.symbols


def test_the_short_list_is_kept_and_rebuilt_weekly(tmp_path):
    store = scr.ShortlistStore(tmp_path / 'sl.json')
    assert store.due(CFG)
    store.save(scr.Shortlist(symbols=['KCB'], roles={'KCB': 'top'},
                             built_at=datetime(2026, 9, 21, tzinfo=timezone.utc).isoformat()))
    assert store.load().symbols == ['KCB']
    assert not store.due(CFG, now=datetime(2026, 9, 25, tzinfo=timezone.utc))
    assert store.due(CFG, now=datetime(2026, 9, 28, tzinfo=timezone.utc))


# ------------------------------------------------------------ in the agent

def _agent(tmp_path, monkeypatch, enabled=True, held=()):
    import src.agent.main as main
    monkeypatch.setattr(main, 'DATA_DIR', tmp_path)
    paper = SimpleNamespace(enabled=True, positions=lambda: {s: {} for s in held})
    agent = SimpleNamespace(config={'data_manager': {'nse_symbols': ['SCOM', 'KCB']},
                                    'nse_screener': {'enabled': enabled}},
                            components={'nse_paper_account': paper})
    for name in ('nse_trading_symbols', '_refresh_nse_shortlist'):
        setattr(agent, name, types.MethodType(getattr(main.TradingAgent, name), agent))
    return agent


def test_with_the_screener_off_the_agent_trades_the_configured_list(tmp_path, monkeypatch):
    assert _agent(tmp_path, monkeypatch, enabled=False).nse_trading_symbols() == ['SCOM', 'KCB']


def test_a_holding_is_always_evaluated_even_with_the_screener_off(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, enabled=False, held=['JUB'])
    assert agent.nse_trading_symbols() == ['SCOM', 'KCB', 'JUB']


def test_without_the_paper_account_the_recorded_trading_fills_are_the_holdings(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    queue = SimpleNamespace(positions=lambda book=None: {
        'JUB': {'quantity': 300}, 'EABL': {'quantity': 0}} if book == 'trading' else {})
    agent.components = {'nse_paper_account': SimpleNamespace(enabled=False), 'nse_order_queue': queue}
    assert agent.nse_trading_symbols() == ['SCOM', 'KCB', 'JUB']       # EABL was sold


def test_a_stock_leaving_the_list_is_forgotten_and_one_returning_starts_afresh(monkeypatch):
    import src.agent.main as main
    from src.agent.slow_strategies import NseRotationStrategy
    from src.agent.strategy_manager import StrategyManager
    from src.agent.volatility import VolatilityTracker
    sm = StrategyManager({'strategies': {'rot': {'type': 'rotation', 'lookback': 5, 'skip': 1,
                                                 'top_n': 1, 'min_universe': 1}}})
    rot = sm.strategies['rot']
    assert isinstance(rot, NseRotationStrategy)
    for sym, g in (('KCB', 1.02), ('EQTY', 1.01)):
        rot.seed_history(sym, [100 * g ** i for i in range(10)])
    sm.regime_detector.warm_start('KCB', [100.0] * 60)
    vol = VolatilityTracker()
    vol.warm_start('KCB', [100.0] * 20)
    warmed = []
    agent = SimpleNamespace(components={'strategy_manager': sm}, volatility=vol,
                            _history_seeded={'KCB', 'EQTY', 'AAPL'}, _nse_last_price={'KCB': 1.0},
                            _nse_active={'KCB', 'EQTY'})
    agent._warm_start_history = lambda: warmed.append(True)
    main.TradingAgent._sync_nse_state(agent, ['EQTY'])
    assert [s for s, _ in rot.ranking()] == ['EQTY']                  # KCB no longer ranked
    assert sm.regime_detector.bars('KCB') == 0 and vol.bars('KCB') == 0
    assert agent._history_seeded == {'EQTY', 'AAPL'} and 'KCB' not in agent._nse_last_price
    assert not warmed
    main.TradingAgent._sync_nse_state(agent, ['EQTY', 'KCB'])        # back next week
    assert warmed == [True]


def test_an_old_price_is_never_acted_on():
    from src.agent.main import TradingAgent
    today = date(2026, 9, 26)
    assert TradingAgent._nse_price_age_days({'timestamp': '2026-09-25'}, today) == 1
    assert TradingAgent._nse_price_age_days({'timestamp': '2026-08-20T15:00:00+03:00'}, today) == 37
    assert TradingAgent._nse_price_age_days({'timestamp': ''}, today) is None


def test_the_agent_trades_the_short_list_and_every_holding(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, held=['JUB'])
    assert agent.nse_trading_symbols() == ['SCOM', 'KCB', 'JUB']     # not built yet
    scr.ShortlistStore(tmp_path / 'nse_shortlist.json').save(
        scr.Shortlist(symbols=['EQTY', 'SCOM'], roles={'EQTY': 'top', 'SCOM': 'top'},
                      built_at=datetime.now(timezone.utc).isoformat()))
    assert agent.nse_trading_symbols() == ['EQTY', 'SCOM', 'JUB']


def test_the_agent_builds_the_short_list_when_it_is_due(tmp_path, monkeypatch):
    import src.connectors.nse_connector as nse_connector
    csvs = tmp_path / 'csv'
    csvs.mkdir()
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', csvs)
    _exchange(csvs)
    agent = _agent(tmp_path, monkeypatch)
    sl = agent._refresh_nse_shortlist(now=datetime(2026, 9, 26, tzinfo=timezone.utc), llm=_Llm(None))
    assert sl.symbols and sl.symbols[0] == 'KCB'
    assert agent._refresh_nse_shortlist(now=datetime(2026, 9, 27, tzinfo=timezone.utc)) is None
    assert agent.nse_trading_symbols()[:len(sl.symbols)] == sl.symbols


# ------------------------------------------------------------------- API

@pytest.fixture
def _api_user(tmp_path, monkeypatch):
    import bcrypt
    import src.api.auth as auth
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    from src.api.auth import create_token
    return {'Authorization': f"Bearer {create_token('tester', 'operator')}"}


def test_the_scan_endpoint_shows_the_screen_and_the_traded_list(tmp_path, monkeypatch, _api_user):
    import src.api.api_server as api_server
    import src.connectors.nse_connector as nse_connector
    csvs = tmp_path / 'csv'
    csvs.mkdir()
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', csvs)
    monkeypatch.setattr(api_server, 'DATA_DIR', tmp_path)
    _exchange(csvs)
    agent = SimpleNamespace(components={}, config={'data_manager': {'nse_symbols': ['SCOM', 'KCB']},
                                                   'nse_screener': {'enabled': True}})
    client = api_server.TradingAPI(agent, {}).app.test_client()
    data = client.get('/api/nse/scan', headers=_api_user).get_json()
    assert data['enabled'] is True and data['traded'] == ['SCOM', 'KCB']
    assert data['roles'] == {'SCOM': 'configured', 'KCB': 'configured'}   # not built yet
    assert data['stocks'][0]['symbol'] == 'KCB' and data['eligible'] == 8
    assert data['rules']['min_history_days'] == 50


def test_the_learning_endpoint_shows_where_each_strategy_votes(_api_user):
    import src.api.api_server as api_server
    from src.agent.strategy_manager import StrategyManager
    from src.utils.config_validator import load_config
    cfg = load_config('config/config.json')
    agent = SimpleNamespace(components={'strategy_manager': StrategyManager(cfg)}, config=cfg,
                            order_journal=None,
                            strategy_tuner=SimpleNamespace(state={'last_run': 0, 'params': {}, 'log': [
                                {'strategy': 'momentum', 'at': '2026-09-20T05:00:00', 'outcome': 'adopted',
                                 'before': {'threshold': 0.02}, 'after': {'threshold': 0.025}}]}))
    client = api_server.TradingAPI(agent, {}).app.test_client()
    data = client.get('/api/strategies/learning', headers=_api_user).get_json()
    by = {s['name']: s for s in data['strategies']}
    assert by['crypto_trend']['markets'] == ['crypto']
    assert by['momentum']['markets'] == ['us_equity', 'nse']
    assert by['nse_rotation']['history_days'] == 141
    assert data['strategy_markets'] == {'crypto': ['crypto_trend']}
    assert data['tuner']['log'][0]['outcome'] == 'adopted'
    import src.api.auth as auth
    users = json.loads(open(auth.USERS_FILE).read())
    users['guest'] = {**users['tester'], 'role': 'viewer'}
    open(auth.USERS_FILE, 'w').write(json.dumps(users))
    viewer = {'Authorization': f"Bearer {auth.create_token('guest', 'viewer')}"}
    seen = client.get('/api/strategies/learning', headers=viewer).get_json()
    assert seen['tuner']['log'] == [] and seen['tuner']['restricted']
    assert {s['name'] for s in seen['strategies']} == set(by)
