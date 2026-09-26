"""The slower strategies from the gap analysis (src/agent/slow_strategies.py),
how the ensemble scopes them to their markets, and the dividend sleeve's
value score and T-bill switch."""
import os
import tempfile
from datetime import date

import pytest

from src.agent.slow_strategies import (CryptoTrendStrategy, EarningsCalendar,
                                       EarningsDriftStrategy, NseRotationStrategy)
from src.agent.strategy_manager import StrategyManager
from src.utils.config_validator import load_config


def _feed(strategy, symbol, closes):
    """One close per call, as a backtest feeds bars (no bar_date)."""
    out = None
    for c in closes:
        out = strategy.generate_signals({'symbol': symbol, 'price': c})
    return out


# ------------------------------------------------------------ crypto trend

def test_the_trend_strategy_abstains_until_it_has_its_average():
    s = CryptoTrendStrategy('crypto_trend', {'sma_period': 20})
    assert _feed(s, 'BTC-USD', [100.0] * 19)['abstain'] is True
    assert _feed(s, 'BTC-USD', [100.0]).get('abstain') is None


def test_above_the_band_buys_below_it_sells_and_inside_it_has_no_view():
    s = CryptoTrendStrategy('crypto_trend', {'sma_period': 20, 'band': 0.02})
    s.seed_history('BTC-USD', [100.0] * 20)
    up = s.generate_signals({'symbol': 'BTC-USD', 'price': 110.0})
    assert up['action'] == 'buy' and 0.5 <= up['confidence'] <= 1.0
    s.seed_history('ETH-USD', [100.0] * 20)
    assert s.generate_signals({'symbol': 'ETH-USD', 'price': 101.0})['action'] == 'hold'
    s.seed_history('SOL-USD', [100.0] * 20)
    assert s.generate_signals({'symbol': 'SOL-USD', 'price': 90.0})['action'] == 'sell'


# ------------------------------------------------------------ NSE rotation

def _rotation(**kw):
    return NseRotationStrategy('nse_rotation', {'lookback': 10, 'skip': 2, 'top_n': 2,
                                                'exit_rank': 3, 'min_universe': 4, **kw})


def _path(ret, n=13):
    """n closes rising by `ret` in total up to two bars ago, flat since."""
    step = (1 + ret) ** (1 / 10)
    closes = [100.0 * step ** min(i, 10) for i in range(n)]
    return closes


def test_rotation_waits_for_enough_stocks_to_rank():
    s = _rotation()
    for sym, ret in (('A', 0.3), ('B', 0.2), ('C', 0.1)):
        s.seed_history(sym, _path(ret))
    assert s.generate_signals({'symbol': 'A', 'price': _path(0.3)[-1]})['abstain'] is True


def test_rotation_buys_the_leaders_and_sells_the_laggards():
    s = _rotation()
    rets = {'A': 0.30, 'B': 0.20, 'C': 0.10, 'D': 0.05, 'E': -0.10}
    # Every stock has its history before the day's first evaluation, as in
    # the live loop; each then gets the day's price.
    for sym, ret in rets.items():
        s.seed_history(sym, _path(ret))
    sig = {sym: s.generate_signals({'symbol': sym, 'price': _path(ret)[-1]})
           for sym, ret in rets.items()}
    assert sig['A']['action'] == 'buy' and sig['A']['confidence'] == 1.0
    assert sig['B']['action'] == 'buy' and sig['B']['confidence'] < 1.0
    assert sig['C'].get('abstain') is True           # ranked 3rd: between buy and exit
    assert sig['D']['action'] == 'sell'              # outside the exit rank
    assert sig['E']['action'] == 'sell'              # falling
    assert sig['A']['indicators']['rank'] == 1


# ------------------------------------------------------- earnings drift

def _calendar(tmp_path, rows):
    calls = []

    def fetch(symbol, start, end):
        calls.append(symbol)
        return rows.get(symbol, [])
    cal = EarningsCalendar(None, tmp_path / 'earnings.json', fetch=fetch)
    cal.calls = calls
    return cal


def test_a_clear_beat_is_a_buy_that_fades_and_a_miss_is_a_sell(tmp_path):
    cal = _calendar(tmp_path, {
        'AAPL': [{'date': '2026-09-01', 'epsActual': 1.20, 'epsEstimate': 1.00}],
        'MSFT': [{'date': '2026-09-01', 'epsActual': 0.80, 'epsEstimate': 1.00}],
        'NVDA': [{'date': '2026-09-01', 'epsActual': 1.02, 'epsEstimate': 1.00}]})
    s = EarningsDriftStrategy('us_earnings_drift', {'min_surprise': 0.05, 'hold_days': 40},
                              calendar=cal)
    early = s.generate_signals({'symbol': 'AAPL'}, today=date(2026, 9, 2))
    late = s.generate_signals({'symbol': 'AAPL'}, today=date(2026, 10, 5))
    assert early['action'] == 'buy' and early['confidence'] > late['confidence']
    assert early['indicators']['surprise'] == pytest.approx(0.2)
    assert s.generate_signals({'symbol': 'MSFT'}, today=date(2026, 9, 2))['action'] == 'sell'
    assert s.generate_signals({'symbol': 'NVDA'}, today=date(2026, 9, 2))['abstain'] is True
    assert s.generate_signals({'symbol': 'AAPL'}, today=date(2026, 10, 20))['abstain'] is True


def test_earnings_are_fetched_once_a_day_and_without_a_key_it_abstains(tmp_path):
    cal = _calendar(tmp_path, {'AAPL': []})
    s = EarningsDriftStrategy('us_earnings_drift', {}, calendar=cal)
    for _ in range(3):
        s.generate_signals({'symbol': 'AAPL'}, today=date(2026, 9, 2))
    assert cal.calls == ['AAPL']
    keyless = EarningsDriftStrategy('us_earnings_drift', {},
                                    calendar=EarningsCalendar(None, tmp_path / 'x.json'))
    assert keyless.generate_signals({'symbol': 'AAPL'})['reason'] == 'no Finnhub API key'


# ------------------------------------------------------ in the ensemble

@pytest.fixture(scope='module')
def manager():
    return StrategyManager(load_config('config/config.json'))


def test_each_market_hears_only_its_own_strategies(manager):
    assert set(manager.strategies) >= {'crypto_trend', 'nse_rotation', 'us_earnings_drift'}
    voters = lambda sym: {n for n in manager.strategies if manager.in_scope(n, sym)}
    assert voters('BTC-USD') == {'crypto_trend'}
    assert voters('SCOM') == {'momentum', 'mean_reversion', 'rsi_strategy', 'nse_rotation'}
    assert voters('AAPL') == {'momentum', 'mean_reversion', 'rsi_strategy', 'us_earnings_drift'}


def test_history_needs_are_per_market_and_slow_strategies_are_optional(manager):
    assert manager.history_needed('BTC-USD') == 200
    assert manager.history_needed('BTC-USD', required_only=True) == 1
    assert manager.history_needed('SCOM') == 141
    assert manager.history_needed('SCOM', required_only=True) == 50


def test_an_abstaining_strategy_does_not_water_down_the_others(manager):
    votes = {'momentum': {'action': 'buy', 'confidence': 0.8, 'position_size': 0.04},
             'rsi_strategy': {'action': 'buy', 'confidence': 0.7, 'position_size': 0.035}}
    data = {'symbol': 'AAPL', 'price': 100.0}
    without = manager._combine_signals(dict(votes), data)
    with_abstain = manager._combine_signals(
        {**votes, 'us_earnings_drift': {'action': 'hold', 'confidence': 0.0,
                                        'position_size': 0.0, 'abstain': True}}, data)
    assert with_abstain['action'] == without['action'] == 'buy'
    assert with_abstain['confidence'] == pytest.approx(without['confidence'])
    only = manager._combine_signals({'us_earnings_drift': {'action': 'hold', 'abstain': True}}, data)
    assert only['action'] == 'hold'


def test_a_disabled_strategy_is_not_loaded():
    cfg = load_config('config/config.json')
    cfg['strategies']['nse_rotation']['enabled'] = False
    assert 'nse_rotation' not in StrategyManager(cfg).strategies


# -------------------------------------------- dividend sleeve: value, bills

from src.agent.sleeve.dividend_scorer import ScoringConfig, earnings_yield_pct, score_symbol  # noqa: E402
from src.agent.sleeve.fundamentals_store import Fundamentals  # noqa: E402
from src.agent.sleeve.sleeve_manager import tbill_switch_reason  # noqa: E402


def _fund(sym, yield_pct, dps, eps):
    return Fundamentals(sym, yield_ttm_pct=yield_pct, dividend_per_share_kes=dps, eps_kes=eps,
                        payout_ratio=0.5, years_consecutive_paid=6, eps_trend='positive',
                        avg_daily_volume=100_000, last_updated=date.today().isoformat())


def test_earnings_yield_comes_from_the_price_behind_the_dividend_yield():
    # A 2.00 dividend yielding 5% means a 40.00 price; 6.00 of earnings is 15%.
    assert earnings_yield_pct(_fund('X', 5.0, 2.0, 6.0)) == pytest.approx(15.0)
    assert earnings_yield_pct(_fund('X', 0.0, 2.0, 6.0)) is None


def test_the_cheaper_of_two_equal_payers_scores_higher_with_a_value_weight():
    cfg = ScoringConfig(value_weight=0.3, min_avg_daily_volume=1000)
    cheap = score_symbol(_fund('CHEAP', 6.0, 2.0, 4.0), cfg)    # earnings yield 12%
    dear = score_symbol(_fund('DEAR', 6.0, 2.0, 2.5), cfg)      # earnings yield 7.5%
    assert cheap.combined_score > dear.combined_score
    assert cheap.earnings_yield_pct == pytest.approx(12.0)


def test_the_sleeve_holds_bills_when_shares_earn_less_than_bills_after_tax():
    dear = [_fund(s, 4.0, 2.0, 3.0) for s in 'ABC']      # earnings yields of 6%
    cheap = [_fund(s, 6.0, 2.0, 4.0) for s in 'ABC']     # 12%
    assert 'below the T-bill' in tbill_switch_reason(dear, 8.78)   # 7.46% after tax
    assert tbill_switch_reason(cheap, 8.78) is None
    assert tbill_switch_reason(dear[:2], 8.78) is None   # too few to judge


def test_a_switched_month_places_no_tickets_and_keeps_the_dividend_cash():
    from tests.test_sleeve_manager import FakeDividendLedger, make_config, make_manager
    cfg = make_config()
    cfg['sleeve']['universe'] = ['A', 'B', 'C']
    cfg['sleeve']['tbill_switch'] = {'enabled': True}
    ledger = FakeDividendLedger(unswept=5000)
    mgr = make_manager(os.path.join(tempfile.mkdtemp(), 's.db'), config=cfg,
                       dividend_ledger=ledger,
                       fundamentals=[_fund(s, 4.0, 2.0, 3.0) for s in 'ABC'])
    assert mgr.run_monthly_cycle({'A': 50.0, 'B': 50.0, 'C': 50.0}, today=date(2026, 10, 1)) == []
    assert ledger.swept_calls == 0
    assert 'below the T-bill' in mgr._get_state('last_skip_reason')
    mgr.close()
