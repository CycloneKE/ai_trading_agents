import pytest
from src.agent.technical_strategy import TechnicalStrategy


def test_momentum_multi_timeframe_blocks_bear_market_rally():
    # Strategy with multi-timeframe confirmation enabled
    strat = TechnicalStrategy('momentum', {
        'lookback_period': 50,
        'threshold': 0.02,
        'multi_timeframe_confirm': True
    })

    # Historical prices: severe downtrend over 20 days (from 150 down to 100),
    # but a small short-term bounce in the last 10 days (from 95 to 102).
    # Closes for 49 bars:
    closes = [150.0 - i * 1.5 for i in range(30)] + [100.0] * 10 + [95.0 + i * 0.7 for i in range(9)]
    strat.seed_history('SPY', closes)

    # Latest price is 102.0. 10-day momentum > 2%, but 20-day momentum is negative (-15%).
    # Multi-timeframe should refuse to buy a bear-market rally into resistance.
    sig = strat.generate_signals({'symbol': 'SPY', 'price': 102.0})
    assert sig['action'] == 'hold'


def test_momentum_multi_timeframe_allows_aligned_trend():
    strat = TechnicalStrategy('momentum', {
        'lookback_period': 50,
        'threshold': 0.02,
        'multi_timeframe_confirm': True
    })

    # Continuous healthy uptrend: both 10-day and 20-day momentum are positive
    closes = [100.0 + i * 1.0 for i in range(50)]
    strat.seed_history('SPY', closes[:-1])

    sig = strat.generate_signals({'symbol': 'SPY', 'price': closes[-1]})
    assert sig['action'] == 'buy'
    assert sig['confidence'] > 0.5


def test_mean_reversion_normal_volume_buys_dip():
    strat = TechnicalStrategy('mean_reversion', {
        'lookback_period': 20,
        'band': 0.02,
        'volume_knife_filter': True
    })

    # 19 bars with price 100.0 and steady volume 10,000
    closes = [100.0] * 19
    strat.seed_history('AAPL', closes, volumes=[10000.0] * 19)

    # Dip by 3% (below the 2% band) with normal volume (10,500)
    sig = strat.generate_signals({
        'symbol': 'AAPL',
        'price': 97.0,
        'volume': 10500.0
    })
    assert sig['action'] == 'buy'
    assert sig['confidence'] > 0.5


def test_mean_reversion_falling_knife_filter_blocks_panic_dump():
    strat = TechnicalStrategy('mean_reversion', {
        'lookback_period': 20,
        'band': 0.02,
        'volume_knife_filter': True
    })

    # 19 bars with price 100.0 and steady volume 10,000
    closes = [100.0] * 19
    strat.seed_history('AAPL', closes, volumes=[10000.0] * 19)

    # Dip by 3% on massive panic liquidation volume (35,000, 3.5x normal)
    # The falling-knife filter must suppress the buy to prevent catching a falling knife
    sig = strat.generate_signals({
        'symbol': 'AAPL',
        'price': 97.0,
        'volume': 35000.0
    })
    assert sig['action'] == 'hold'
    assert sig['confidence'] == 0.0
