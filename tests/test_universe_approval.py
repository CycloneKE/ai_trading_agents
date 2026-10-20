"""An approved symbol must actually get traded.

Verified against the code before this change: resolve_escalation appends to
config['data_manager']['nse_symbols'] only, in memory, and the trading loop
reads config['data_manager']['symbols'] at main.py:633 while
get_active_watchlist() is used solely to build LLM research context. So
operator approval had no effect on what the agent traded.
"""

import json
import os
import tempfile

import pytest

from src.agent.universe_manager import effective_universe, persist_universe, remove_from_universe


def base_config():
    return {'data_manager': {'symbols': ['AAPL', 'MSFT'],
                             'nse_symbols': ['SCOM', 'EQTY']}}


def test_effective_universe_includes_config_symbols():
    cfg = base_config()
    assert set(effective_universe(cfg, [], 'us')) == {'AAPL', 'MSFT'}


def test_an_approved_watchlist_symbol_joins_the_universe():
    """THE regression: this was written to the watchlist and never read."""
    cfg = base_config()
    watchlist = [{'symbol': 'NVDA', 'market': 'us', 'status': 'active'}]
    assert 'NVDA' in effective_universe(cfg, watchlist, 'us')


def test_a_removed_watchlist_symbol_leaves_the_universe():
    cfg = base_config()
    watchlist = [{'symbol': 'AAPL', 'market': 'us', 'status': 'removed'}]
    assert 'AAPL' not in effective_universe(cfg, watchlist, 'us')


def test_a_paused_watchlist_symbol_leaves_the_universe():
    cfg = base_config()
    watchlist = [{'symbol': 'AAPL', 'market': 'us', 'status': 'paused'}]
    assert 'AAPL' not in effective_universe(cfg, watchlist, 'us')


def test_markets_do_not_leak_into_each_other():
    cfg = base_config()
    watchlist = [{'symbol': 'NVDA', 'market': 'us', 'status': 'active'}]
    assert 'NVDA' not in effective_universe(cfg, watchlist, 'kenyan')
    assert set(effective_universe(cfg, watchlist, 'kenyan')) == {'SCOM', 'EQTY'}


def test_the_universe_has_no_duplicates():
    cfg = base_config()
    watchlist = [{'symbol': 'AAPL', 'market': 'us', 'status': 'active'}]
    universe = effective_universe(cfg, watchlist, 'us')
    assert len(universe) == len(set(universe))


def test_a_us_symbol_persists_to_the_us_list():
    """US symbols previously had no approval path at all."""
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'NVDA', 'us', overlay)
        assert 'NVDA' in cfg['data_manager']['symbols']
        saved = json.loads(open(overlay, encoding='utf-8').read())
        assert 'NVDA' in saved['data_manager.symbols']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_a_kenyan_symbol_persists_to_the_nse_list():
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'KCB', 'kenyan', overlay)
        assert 'KCB' in cfg['data_manager']['nse_symbols']
        assert 'KCB' not in cfg['data_manager']['symbols']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_an_approval_survives_a_restart():
    """The in-memory write reverted on every restart."""
    from src.agent.self_assessment import apply_overlay
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'NVDA', 'us', overlay)

        fresh = base_config()
        apply_overlay(fresh, overlay)
        assert 'NVDA' in fresh['data_manager']['symbols']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_persisting_the_same_symbol_twice_is_idempotent():
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        persist_universe(cfg, 'NVDA', 'us', overlay)
        persist_universe(cfg, 'NVDA', 'us', overlay)
        assert cfg['data_manager']['symbols'].count('NVDA') == 1
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_remove_from_universe_removes_symbol_and_persists_overlay():
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        remove_from_universe(cfg, 'AAPL', 'us', overlay)
        assert 'AAPL' not in cfg['data_manager']['symbols']
        saved = json.loads(open(overlay, encoding='utf-8').read())
        assert 'AAPL' not in saved['data_manager.symbols']
        assert 'MSFT' in saved['data_manager.symbols']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)


def test_remove_from_universe_nonexistent_symbol_is_safe():
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)
    try:
        cfg = base_config()
        remove_from_universe(cfg, 'NONEXISTENT', 'us', overlay)
        assert cfg['data_manager']['symbols'] == ['AAPL', 'MSFT']
    finally:
        if os.path.exists(overlay):
            os.unlink(overlay)
