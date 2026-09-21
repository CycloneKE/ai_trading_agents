"""Trading lists versus classification lists, and the scraper reading config.

`data_manager` holds two kinds of list and conflating them causes two
separate mistakes. Counting a classification list toward the universe
inflates it and reports an intentional overlap as a bug. Ignoring config in
the scraper means trimming the universe changes nothing about what is
fetched.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'manage_universe', pathlib.Path(__file__).parent.parent / 'scripts' / 'manage_universe.py')
mu = importlib.util.module_from_spec(_SPEC)
sys.modules['manage_universe'] = mu
_SPEC.loader.exec_module(mu)


CFG = {'data_manager': {
    'symbols': ['AAPL', 'SPY', 'BTC-USD', 'ETH-USD'],
    'nse_symbols': ['SCOM', 'EQTY'],
    'crypto_symbols': ['BTC-USD', 'ETH-USD'],
}}


def test_classification_overlap_is_not_counted_toward_the_universe():
    """BTC-USD is in `symbols` so it trades, and in `crypto_symbols` so it is
    priced as crypto. That is one symbol, not two."""
    assert mu.all_symbols(CFG) == {'AAPL', 'SPY', 'BTC-USD', 'ETH-USD', 'SCOM', 'EQTY'}
    assert len(mu.all_symbols(CFG)) == 6


def test_classification_overlap_is_not_reported_as_a_conflict():
    assert mu.conflicts(CFG) == {}


def test_a_symbol_in_two_trading_lists_is_a_conflict():
    """Two execution paths on different cadences would both act on it."""
    cfg = {'data_manager': {**CFG['data_manager'],
                            'nse_symbols': ['SCOM', 'AAPL']}}
    assert mu.conflicts(cfg) == {'AAPL': ['symbols', 'nse_symbols']}


def test_traded_crypto_missing_from_the_classification_list_is_flagged():
    cfg = {'data_manager': {**CFG['data_manager'], 'crypto_symbols': ['BTC-USD']}}
    assert mu.unclassified_crypto(cfg) == ['ETH-USD']


def test_fully_declared_crypto_is_not_flagged():
    assert mu.unclassified_crypto(CFG) == []


def test_non_crypto_symbols_are_never_flagged_as_undeclared_crypto():
    cfg = {'data_manager': {'symbols': ['AAPL', 'SPY'], 'nse_symbols': [],
                            'crypto_symbols': []}}
    assert mu.unclassified_crypto(cfg) == []


# --- the shipped config -----------------------------------------------------

def _shipped():
    with open('config/config.json') as f:
        return json.load(f)


def test_the_shipped_config_has_no_trading_conflicts():
    assert mu.conflicts(_shipped()) == {}


def test_the_shipped_config_declares_all_its_crypto():
    assert mu.unclassified_crypto(_shipped()) == []


def test_the_shipped_universe_is_the_trimmed_twenty_two():
    assert len(mu.all_symbols(_shipped())) == 22


def test_the_sleeve_only_accumulates_names_the_scraper_fetches():
    """The sleeve needs prices for every name it buys, and prices come from
    the scraper, which now walks data_manager.nse_symbols."""
    cfg = _shipped()
    scraped = {s.upper() for s in cfg['data_manager']['nse_symbols']}
    sleeve = {s.upper() for s in cfg['sleeve']['universe']}
    assert sleeve <= scraped, f"sleeve wants prices for {sleeve - scraped}"


def test_config_documents_the_two_kinds_of_list():
    note = _shipped()['data_manager'].get('_symbol_lists', '')
    assert 'TRADING' in note and 'CLASSIFICATION' in note


# --- scraper honours config -------------------------------------------------

def test_scraper_uses_supplied_symbols_over_its_own_default():
    from src.connectors.nse_scraper import DEFAULT_SYMBOLS, NSEPeriodicScraper
    s = NSEPeriodicScraper(symbols=['SCOM', 'EQTY'])
    assert s.symbols == ['SCOM', 'EQTY']
    assert len(s.symbols) < len(DEFAULT_SYMBOLS)


def test_scraper_falls_back_when_given_nothing():
    """The CLI entry point passes no symbols and must still work."""
    from src.connectors.nse_scraper import DEFAULT_SYMBOLS, NSEPeriodicScraper
    assert NSEPeriodicScraper().symbols == DEFAULT_SYMBOLS
    assert NSEPeriodicScraper(symbols=[]).symbols == DEFAULT_SYMBOLS


def test_scraper_interval_is_configurable():
    from src.connectors.nse_scraper import NSEPeriodicScraper
    assert NSEPeriodicScraper(interval_minutes=15).interval == 15 * 60
