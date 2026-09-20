"""Per-market transaction costs."""
import json
import pytest

from src.agent.cost_model import (classify, costs_for, round_trip_pct,
                                  unverified_markets)

CFG = {
    'data_manager': {
        'nse_symbols': ['SCOM', 'EQTY', 'KCB'],
        'crypto_symbols': ['BTC-USD', 'ETH-USD'],
    },
    'costs': {
        '_comment': 'documentation, not a market',
        'us_equity': {'commission_pct': 0.0, 'slippage_pct': 0.0005, 'verified': True},
        'nse': {'commission_pct': 0.017, 'slippage_pct': 0.003, 'verified': False},
        'crypto': {'commission_pct': 0.006, 'slippage_pct': 0.001, 'verified': False},
    },
}


@pytest.mark.parametrize('symbol,market', [
    ('AAPL', 'us_equity'), ('SPY', 'us_equity'),
    ('SCOM', 'nse'), ('scom', 'nse'), ('EQTY', 'nse'),
    ('BTC-USD', 'crypto'), ('SOL-USD', 'crypto'),
])
def test_symbols_map_to_their_market(symbol, market):
    assert classify(symbol, CFG) == market


def test_unknown_symbol_defaults_to_us_equity():
    assert classify('NVDA', CFG) == 'us_equity'
    assert classify('', CFG) == 'us_equity'
    assert classify(None, CFG) == 'us_equity'


def test_nse_round_trip_is_an_order_of_magnitude_above_us():
    """The gap this module exists to represent."""
    assert round_trip_pct('SCOM', CFG) > 10 * round_trip_pct('AAPL', CFG)
    assert round_trip_pct('SCOM', CFG) == pytest.approx(0.04)
    assert round_trip_pct('AAPL', CFG) == pytest.approx(0.001)


def test_documentation_keys_are_not_treated_as_markets():
    # A bare string under `costs` (the `_comment` key) used to reach
    # dict.update and raise "dictionary update sequence element #0 has
    # length 1; 2 is required".
    flagged = unverified_markets(CFG)
    assert '_comment' not in flagged
    assert costs_for('AAPL', CFG)['market'] == 'us_equity'   # resolution still works


def test_unverified_markets_flags_placeholders_only():
    flagged = unverified_markets(CFG)
    assert set(flagged) == {'nse', 'crypto'}
    assert 'us_equity' not in flagged


def test_config_overrides_the_builtin_default():
    cfg = {**CFG, 'costs': {**CFG['costs'], 'nse': {'commission_pct': 0.005,
                                                    'slippage_pct': 0.0,
                                                    'verified': True}}}
    assert round_trip_pct('SCOM', cfg) == pytest.approx(0.01)
    assert 'nse' not in unverified_markets(cfg)


def test_works_with_no_config_at_all():
    assert classify('AAPL') == 'us_equity'
    assert round_trip_pct('AAPL') >= 0
    assert costs_for('AAPL')['market'] == 'us_equity'


def test_shipped_config_parses_and_flags_nse_as_unverified():
    with open('config/config.json') as f:
        cfg = json.load(f)
    assert costs_for('SCOM', cfg)['market'] == 'nse'
    assert round_trip_pct('SCOM', cfg) > 0.02
    assert 'nse' in unverified_markets(cfg)
