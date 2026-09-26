"""Reading AIB-AXYS's daily Market Pulse (src/agent/market_pulse.py) and
what the agent does with it. The text below is copied from the 23 September
2026 report as pypdf reads it; no report is stored in the repository."""
import json
from datetime import date

import pytest

import src.utils.paths as paths
from src.agent import market_pulse as mp

PAGE1 = """AXYS Market Pulse – 23rd Sept 2026
Equities Highlights
    Capital News Update
❖ Public Announcement-Quick Mart Plc Intention_to_Float on the NSE (here)
❖ East African Breweries Plc - AGM Notice & Agenda (here)
❖ Kenya Power and Lighting Company Plc – Audited Financial Results for the Year Ended 30
June 2026 (here)
❖ Absa Bank Kenya Plc – Appointment of Managing Director and CEO (here)

 Top Foreigner Buys  Top Foreigner Sales
"""
PAGE2 = """Fixed Income Stats
91-day rate 8.78% 8.77% ▲ 1.63bps
182-day rate 8.91% 8.93% ▼ (1.97bps)
364-day rate 9.06% 9.07% ▼ (0.99bps)
Interbank Rate 8.75% 8.75% ▲ 0.16bps
Exchange Rates
US Dollar 129.45 129.46 ▲ 1bps
Euro 147.92 148.46 ▲ 37bps
"""
PAGE3 = """MARKET SCORECARD As of: 23/Sep/26
BANKING Current
Price
ABSA Bank Kenya Plc 33.00 ▼ (0.2%) ▲ 33.6% 213,157 18.04 179,240.7 4.60% 4.01             2.35             8.2x              1.8x              7.1% 55.7% 22.2% 3.9%
KCB Group Plc 93.00 - ▲ 41.4% 743,476 111.08 298,852.0 7.67% 23.64           6.00             3.9x              0.8x              6.5% 28.8% 21.3% 3.3%
Family Bank Limited 29.25 ▼ (1.3%) ▲ 62.5% 213,142 19.98 48,632.7 1.25% 4.41             1.20             6.6x              1.5x              4.1% 30.5% 20.4% 2.8%
Industry Median - ▲ 41.4% 1,749,696.7 44.91% 6.5x             1.2x             6.7% 49.6% 19.6% 3.3%
COMMERCIAL AND SERVICES Current
Kenya Airways Ltd 5.70 ▲ 0.4% ▲ 61.5% 814,623 -25.39 33,196.8 0.85% (2.82)            -              (0.2x)             0.0% 0.0% 0.0% 0.0%
ENERGY & PETROLEUM Current
Umeme Ltd 6.10 ▲ 0.7% ▼ (22.0%) 139,525 7.55 9,905.7 0.25% (3.55)            7.68             (1.7x)             0.8x              126.0% (161.2%) 0.0% 0.0%
MANUFACTURING & ALLIED Current
Shri Krishana Overseas Plc 16.45 ▼ (1.8%) ▲ 101.6% 16,604 1.53 830.7 0.02% 4.14             -              4.0x              10.8x             0.0% 0.00 2.71 0.39
Market Average ▼ (0.2%) ▲ 32.5% 9.4x 1.6x 3.2% 14.3% 7.1%
"""
TEXTS = [PAGE1, PAGE2, PAGE3]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    import src.connectors.nse_connector as nse_connector
    import src.connectors.nse_scraper as nse_scraper
    csvs = tmp_path / 'nse_historical'
    csvs.mkdir()
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', csvs)
    monkeypatch.setattr(nse_scraper, 'DATA_DIR', csvs)
    import src.agent.benchmarks as bm
    bm._rate_cache.clear()
    return tmp_path


def test_the_report_is_recognised_and_dated():
    assert mp.is_market_pulse(TEXTS)
    assert not mp.is_market_pulse(['Equity research: KCB Group, BUY, target 120'])
    assert mp.report_date(TEXTS) == date(2026, 9, 23)


def test_every_scorecard_row_is_read_and_ratios_are_recomputed():
    rows = {r.name: r for r in mp.parse_scorecard(TEXTS)}
    assert set(rows) == {'ABSA Bank Kenya Plc', 'KCB Group Plc', 'Family Bank Limited',
                         'Kenya Airways Ltd', 'Umeme Ltd', 'Shri Krishana Overseas Plc'}
    absa = rows['ABSA Bank Kenya Plc']
    assert (absa.price, absa.change_pct, absa.ytd_pct, absa.volume) == (33.0, -0.2, 33.6, 213157)
    assert (absa.eps, absa.dps, absa.roe_pct) == (4.01, 2.35, 22.2)
    assert absa.pe == pytest.approx(8.23, abs=0.01) and absa.dividend_yield_pct == pytest.approx(7.12)
    assert rows['KCB Group Plc'].change_pct is None                    # printed "-"
    kq = rows['Kenya Airways Ltd']
    assert kq.eps == -2.82 and kq.pe is None and kq.pb is None         # losses, negative book
    assert rows['Umeme Ltd'].eps == -3.55 and rows['Umeme Ltd'].dps == 7.68
    assert rows['Shri Krishana Overseas Plc'].roe_pct is None          # malformed tail: not guessed
    assert rows['ABSA Bank Kenya Plc'].sector == 'Banking'


def test_rates_fx_and_announcements_are_read():
    assert mp.parse_rates(TEXTS) == {'tbill_91': 0.0878, 'tbill_182': 0.0891,
                                     'tbill_364': 0.0906, 'interbank': 0.0875}
    assert mp.parse_fx(TEXTS)['usd_kes'] == 129.45
    news = mp.parse_announcements(TEXTS, mp._alias_table())
    assert [(a['symbol'], a['kind']) for a in news] == [
        (None, 'listing'), ('EABL', 'agm'), ('KPLC', 'results'), ('ABSA', 'board')]
    assert news[2]['text'].endswith('30 June 2026')                    # wrapped line joined


def test_names_become_tickers_and_an_unknown_one_is_left_unmatched():
    table = mp._alias_table()
    assert table[mp.normalise('ABSA Bank Kenya Plc')] == 'ABSA'
    assert table[mp.normalise('Kenya Airways Ltd')] == 'KQ'
    assert mp.normalise('Family Bank Limited') not in table


def test_an_unknown_name_is_learned_from_the_ticker_feeds_record():
    rows = [r for r in mp.parse_scorecard(TEXTS) if r.name == 'Family Bank Limited']
    on = date(2026, 9, 23)
    recorded = {'FAML': {on: (29.25, 29.63)}, 'OTHER': {on: (29.30, 29.0)}}
    volumes = {'FAML': {on: 213142}, 'OTHER': {on: 5000}}
    assert mp.learn_names(rows, on, recorded, volumes, set()) == {'Family Bank Limited': 'FAML'}
    assert mp.learn_names(rows, on, recorded, {'FAML': {on: 1}, 'OTHER': {on: 5000}}, set()) == {}


def test_ingest_stores_fundamentals_rates_news_and_prices(data_dir):
    s = mp.ingest('report.pdf', TEXTS)
    assert s['as_of'] == '2026-09-23' and s['stocks_mapped'] == 5
    assert s['unmapped'] == ['Family Bank Limited'] and s['price_bars_added'] == 5
    f = mp.fundamentals('ABSA', today=date(2026, 9, 24))
    assert f['eps'] == 4.01 and f['dividend_yield_pct'] == pytest.approx(7.12)
    assert mp.latest_rates(today=date(2026, 9, 24))['tbill_91'] == 0.0878
    assert mp.announcements('EABL', today=date(2026, 9, 24))[0]['kind'] == 'agm'
    csv_text = (data_dir / 'nse_historical' / 'KCB.csv').read_text()
    assert '2026-09-23' in csv_text and 'aib_market_pulse' in csv_text
    # Ingesting the same report again changes nothing and adds no bar.
    again = mp.ingest('report.pdf', TEXTS)
    assert again['price_bars_added'] == 0 and again['price_bars_already_recorded'] == 5
    assert len(json.loads((data_dir / 'market_pulse' / 'announcements.json').read_text())) == 4


def test_a_live_price_already_stored_is_never_overwritten(data_dir):
    (data_dir / 'nse_historical' / 'KCB.csv').write_text(
        'date,symbol,open,high,low,close,volume,change_pct,source\n'
        '2026-09-23,KCB,93,94,92,93.25,700000,0.3,nse_ticker\n')
    s = mp.ingest('report.pdf', TEXTS)
    assert s['price_bars_added'] == 4
    assert 'aib_market_pulse' not in (data_dir / 'nse_historical' / 'KCB.csv').read_text()


def test_the_agent_uses_what_the_report_says(data_dir):
    mp.ingest('report.pdf', TEXTS)
    today = date(2026, 9, 24)
    ctx = mp.research_context('ABSA', today=today)
    assert 'P/E 8.23x' in ctx['rationale'] and 'Appointment of Managing Director' in ctx['rationale']
    assert mp.research_context('SCOM', today=today) is None
    # The T-bill rate the benchmark and the idle cash use.
    import src.agent.benchmarks as bm
    bm._rate_cache.clear()
    rate, source = bm.tbill_rate({'benchmarks': {'tbill_rate_pct': 0.07}})
    assert rate == 0.0878 and 'Market Pulse' in source


def test_the_dividend_sleeve_ranks_from_the_report(data_dir, monkeypatch):
    mp.ingest('report.pdf', TEXTS)
    monkeypatch.setattr(mp, 'fundamentals',
                        lambda symbol=None, **kw: mp._read('fundamentals.json', {}).get(symbol.upper(), {}))
    from src.agent.sleeve.fundamentals_store import FundamentalsStore
    store = FundamentalsStore(dividends_path=data_dir / 'none.json')
    f = store.get('ABSA')
    assert f.dividend_per_share_kes == 2.35 and f.eps_kes == 4.01
    assert f.yield_ttm_pct == pytest.approx(7.12) and f.payout_ratio == pytest.approx(0.586, abs=0.001)
    assert store.get('KQ') is None                                       # pays no dividend


def test_a_market_report_upload_no_longer_invents_hold_ratings(data_dir):
    from src.agent.broker_research_ingest import BrokerResearchIngest
    ing = BrokerResearchIngest(None, None, {'data_manager': {'nse_symbols': ['KCB', 'ABSA']}})
    commentary = ('The market was weighed down by KCB Group and ABSA Bank which lost 1.1% and 0.7%. '
                  'Analysts rate Safaricom a BUY with a target of KES 45.')
    signals = ing._extract_signals_fallback(commentary)
    assert [s['symbol'] for s in signals] == ['SCOM']
    assert signals[0]['recommendation'] == 'BUY'


def test_the_upload_route_reads_a_market_pulse_without_the_ai(data_dir, monkeypatch):
    from src.agent.broker_research_ingest import BrokerResearchIngest

    class _Esc:
        def record_upload(self, *a):
            return 7

        def update_upload_status(self, *a):
            self.status = a

    class _NoAi:
        enabled = True

        def propose_json(self, *a, **k):
            raise AssertionError('the AI must not be asked about a Market Pulse')

    monkeypatch.setattr(mp, 'page_texts', lambda path: TEXTS)
    esc = _Esc()
    out = BrokerResearchIngest(_NoAi(), esc, {}).process_pdf('pulse.pdf')
    assert out['document_type'] == 'market_pulse' and out['status'] == 'completed'
    assert out['stocks_mapped'] == 5 and esc.status == (7, 'completed', 5)
