"""Which NSE source wins a day, when sources disagree, and a licensed feed
(nse_scraper.save_bars_csv / SOURCE_RANK, nse_licensed_feed.py).

Before: the last write won, so a web-page price scraped during the session
could replace the exchange's own end-of-day price list, two real sources
that disagreed were never noticed, and there was no way to load a licensed
feed's files.
"""
import csv
import json

import pytest

from src.connectors import nse_licensed_feed as feed
from src.connectors import nse_scraper as ns
from src.connectors.nse_scraper import DailyBar


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, 'DATA_DIR', tmp_path / 'hist')
    return tmp_path / 'hist'


def bar(close, source, day='2026-11-13', symbol='KCB'):
    return DailyBar(date=day, symbol=symbol, open=close, high=close, low=close, close=close,
                    volume=100, change_pct=0.0, source=source)


def stored(store, symbol='KCB'):
    with open(store / f'{symbol}.csv', newline='') as f:
        return {r['date']: (float(r['close']), r['source']) for r in csv.DictReader(f)}


def test_a_better_source_is_not_replaced_by_a_worse_one(store):
    ns.save_bars_csv('KCB', [bar(41.50, 'nse_pricelist')])
    result = ns.save_bars_csv('KCB', [bar(41.60, 'afx_kwayisi')])
    assert stored(store)['2026-11-13'] == (41.50, 'nse_pricelist')
    assert result['kept'] == ['2026-11-13'] and result['written'] == []
    ns.save_bars_csv('KCB', [bar(41.55, 'nse_ticker')])             # the ticker is below the price list too
    assert stored(store)['2026-11-13'] == (41.50, 'nse_pricelist')


def test_a_better_source_replaces_a_worse_one_and_a_source_updates_its_own_bar(store):
    ns.save_bars_csv('KCB', [bar(41.0, 'nse_ticker')])
    ns.save_bars_csv('KCB', [bar(41.2, 'nse_ticker')])                 # later in the session
    assert stored(store)['2026-11-13'] == (41.2, 'nse_ticker')
    ns.save_bars_csv('KCB', [bar(41.5, 'nse_pricelist')])
    assert stored(store)['2026-11-13'] == (41.5, 'nse_pricelist')
    ns.save_bars_csv('KCB', [bar(41.4, 'licensed_eod')])
    assert stored(store)['2026-11-13'] == (41.4, 'licensed_eod')


def test_seed_data_never_buries_a_real_price(store):
    ns.save_bars_csv('KCB', [bar(41.5, 'nse_ticker')])
    ns.save_bars_csv('KCB', [bar(30.0, 'synthetic')])
    assert stored(store)['2026-11-13'] == (41.5, 'nse_ticker')
    ns.save_bars_csv('KCB', [bar(30.0, 'synthetic', day='2026-11-12')])
    ns.save_bars_csv('KCB', [bar(40.0, 'afx_kwayisi', day='2026-11-12')])
    assert stored(store)['2026-11-12'] == (40.0, 'afx_kwayisi')


def test_two_real_sources_far_apart_are_reported(store):
    ns.save_bars_csv('KCB', [bar(41.5, 'nse_ticker')])
    close = ns.save_bars_csv('KCB', [bar(42.0, 'afx_kwayisi')])            # 1.2%: normal noise
    assert close['disagreements'] == []
    far = ns.save_bars_csv('KCB', [bar(37.0, 'afx_kwayisi')])               # 10.8% apart
    assert far['disagreements'] == [{'symbol': 'KCB', 'date': '2026-11-13', 'gap_pct': 10.84,
                                     'nse_ticker': 41.5, 'afx_kwayisi': 37.0}]
    assert ns.save_bars_csv('KCB', [bar(30.0, 'synthetic')])['disagreements'] == []   # seed data is no source


def test_the_scraper_status_lists_recent_disagreements():
    scraper = ns.NSEPeriodicScraper(symbols=['KCB'])
    assert scraper.get_status()['source_disagreements'] == []
    assert scraper.get_status()['licensed_feed'] == {'enabled': False}
    scraper._note_disagreements([{'symbol': 'KCB', 'date': '2026-11-13', 'gap_pct': 10.8}])
    assert scraper.get_status()['source_disagreements'][0]['symbol'] == 'KCB'


# ------------------------------------------------------------- licensed feed

VENDOR = ('Ticker;Trade Date;Open Price;High;Low;Last;Shares\n'
          'KCB;13/11/2026;4100;41.90;40.80;41.40;1,200\n'
          'SCOM;13/11/2026;27.10;27.60;27.00;27.45;350000\n'
          'BAD!;13/11/2026;1;1;1;1;1\n'
          'EQTY;not a date;50;50;50;50;1\n')
VENDOR_CFG = {'enabled': True,
              'columns': {'symbol': 'Ticker', 'date': 'Trade Date', 'open': 'Open Price', 'close': 'Last',
                          'volume': 'Shares'},
              'date_format': '%d/%m/%Y', 'delimiter': ';'}


def test_a_vendor_file_is_read_with_its_own_column_names(tmp_path, store):
    (tmp_path / 'feed').mkdir()
    (tmp_path / 'feed' / 'eod_20261113.csv').write_text(VENDOR)
    cfg = {**VENDOR_CFG, 'folder': str(tmp_path / 'feed')}
    summary = feed.import_folder(cfg)
    assert summary['files'] == 1 and summary['bars'] == 2 and summary['skipped_rows'] == 2
    assert stored(store, 'KCB')['2026-11-13'] == (41.4, 'licensed_eod')
    assert stored(store, 'SCOM')['2026-11-13'] == (27.45, 'licensed_eod')
    with open(store / 'KCB.csv', newline='') as f:
        kcb = next(csv.DictReader(f))
    assert float(kcb['open']) == 41.4 and float(kcb['high']) == 41.9    # an open 100x the close is not believed
    assert feed.import_folder(cfg)['files'] == 0                              # imported once
    (tmp_path / 'feed' / 'eod_20261113.csv').write_text(VENDOR.replace('41.40', '41.45'))
    assert feed.import_folder(cfg)['files'] == 1                              # a corrected file again
    assert stored(store, 'KCB')['2026-11-13'] == (41.45, 'licensed_eod')
    assert json.loads((tmp_path / 'feed' / '.imported.json').read_text())


def test_the_feed_outranks_the_scrapers_and_is_checked_against_them(tmp_path, store):
    ns.save_bars_csv('KCB', [bar(37.0, 'nse_ticker')])
    (tmp_path / 'feed').mkdir()
    (tmp_path / 'feed' / 'eod.csv').write_text(VENDOR)
    summary = feed.import_folder({**VENDOR_CFG, 'folder': str(tmp_path / 'feed')})
    assert stored(store, 'KCB')['2026-11-13'] == (41.4, 'licensed_eod')
    assert summary['disagreements'][0]['symbol'] == 'KCB'                     # 11.9% from the ticker
    ns.save_bars_csv('KCB', [bar(41.3, 'nse_ticker')])                         # the scraper cannot replace it
    assert stored(store, 'KCB')['2026-11-13'] == (41.4, 'licensed_eod')


def test_off_or_without_a_folder_nothing_happens(tmp_path):
    assert feed.import_folder({'folder': str(tmp_path / 'missing')})['files'] == 0
    s = feed.settings(None)
    assert s['enabled'] is False and s['folder'].endswith('nse_feed') and s['columns']['close'] == 'Close'
    assert 'licensed_eod' in ns.REAL_NSE_SOURCES


def test_the_scraper_imports_the_feed_before_scraping(tmp_path, store, monkeypatch):
    (tmp_path / 'feed').mkdir()
    (tmp_path / 'feed' / 'eod.csv').write_text(VENDOR)
    monkeypatch.setattr(ns, 'scrape_nse_ticker', lambda symbols=None: {})
    monkeypatch.setattr(ns, 'scrape_afx_kwayisi', lambda symbols: {})
    scraper = ns.NSEPeriodicScraper(symbols=['KCB'], record_all_listed=False,
                                    licensed_feed={**VENDOR_CFG, 'folder': str(tmp_path / 'feed')})
    scraper.run_once()
    assert stored(store, 'KCB')['2026-11-13'] == (41.4, 'licensed_eod')
    assert scraper.get_status()['licensed_feed']['last_import']['bars'] == 2
