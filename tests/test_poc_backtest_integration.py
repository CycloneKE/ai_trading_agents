import pytest
from integrations.poc_backtest import generate_synthetic_data, strategy_factory
from backtest_engine import BacktestEngine


def test_poc_backtest_runs_and_produces_sane_output():
    symbols = ['AAA', 'BBB', 'CCC']
    data = generate_synthetic_data(symbols, periods=30, start_price=100.0)

    # Initialize engine with config dict and add market data
    config = {'initial_capital': 100000.0}
    engine = BacktestEngine(config)
    engine.add_market_data(data)

    strategy = strategy_factory(price_series_window=5, max_weight=0.5, turnover_limit=0.3)
    results = engine.run_backtest(strategy)

    # Basic sanity checks
    assert isinstance(results, dict)
    assert 'portfolio_history' in results
    ph = results['portfolio_history']
    assert len(ph) > 0

    # final portfolio value is finite and positive
    final = ph[-1]
    assert final.get('portfolio_value', 0) > 0