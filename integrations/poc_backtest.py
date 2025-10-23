"""
Integration harness: run BacktestEngine with poc_portfolio_optimizer to produce deterministic results.
"""
from backtest_engine import BacktestEngine, Order, OrderSide, OrderType
from poc_portfolio_optimizer import inverse_vol_weights, cap_weights, rebalance_with_turnover
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_data(symbols, periods=30, start_price=100.0):
    dates = [datetime(2020,1,1) + timedelta(days=i) for i in range(periods)]
    data = {}
    rng = np.random.RandomState(42)
    for s in symbols:
        price = start_price + rng.randn(periods).cumsum()
        df = pd.DataFrame({'close': price, 'volume': np.ones(periods)*1000}, index=dates)
        data[s] = df
    return data


def strategy_factory(price_series_window=10, max_weight=0.4, turnover_limit=0.2):
    # This closure will be passed to BacktestEngine.run_backtest
    history = {}

    def strategy(timestamp, prices: Dict[str, float], portfolio_snapshot: Dict[str, Any]):
        # Accumulate history
        for s, p in prices.items():
            history.setdefault(s, []).append(float(p))
        # Only start allocating after we have enough history
        for s in list(history.keys()):
            if len(history[s]) > price_series_window:
                history[s] = history[s][-price_series_window-1:]
        if any(len(v) < price_series_window+1 for v in history.values()):
            return []

        # Build target weights from inverse vol
        targets = inverse_vol_weights(history, window=price_series_window)
        targets = cap_weights(targets, max_weight)

        # Turn targets into orders by comparing to current positions
        current = {}
        for s in targets.keys():
            current[s] = 0.0
        if portfolio_snapshot and 'positions' in portfolio_snapshot:
            for s, p in portfolio_snapshot['positions'].items():
                current[s] = p.get('market_value', 0.0) / (portfolio_snapshot.get('portfolio_value', 1.0) or 1.0)

        new_weights = rebalance_with_turnover(current, targets, turnover_limit=turnover_limit)

        # Convert weight changes to orders (very simplified: assume price-based quantity)
        orders = []
        pv = portfolio_snapshot.get('portfolio_value', 100000.0)
        for s in new_weights:
            desired_value = new_weights[s] * pv
            current_value = current.get(s, 0.0) * pv
            delta = desired_value - current_value
            if abs(delta) / pv < 0.001:
                continue
            price = prices.get(s)
            if not price:
                continue
            qty = abs(delta) / price
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(symbol=s, side=side, order_type=OrderType.MARKET, quantity=qty))
        return orders
"""
Integration harness: run BacktestEngine with poc_portfolio_optimizer to produce deterministic results.
"""
from backtest_engine import BacktestEngine, Order, OrderSide, OrderType
from poc_portfolio_optimizer import inverse_vol_weights, cap_weights, rebalance_with_turnover
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_data(symbols, periods=30, start_price=100.0):
    dates = [datetime(2020,1,1) + timedelta(days=i) for i in range(periods)]
    data = {}
    rng = np.random.RandomState(42)
    for s in symbols:
        price = start_price + rng.randn(periods).cumsum()
        df = pd.DataFrame({'close': price, 'volume': np.ones(periods)*1000}, index=dates)
        data[s] = df
    return data


def strategy_factory(price_series_window=10, max_weight=0.4, turnover_limit=0.2):
    # This closure will be passed to BacktestEngine.run_backtest
    history = {}

    def strategy(timestamp, prices: Dict[str, float], portfolio_snapshot: Dict[str, Any]):
        # Accumulate history
        for s, p in prices.items():
            history.setdefault(s, []).append(float(p))
        # Only start allocating after we have enough history
        for s in list(history.keys()):
            if len(history[s]) > price_series_window:
                history[s] = history[s][-price_series_window-1:]
        if any(len(v) < price_series_window+1 for v in history.values()):
            return []

        # Build target weights from inverse vol
        targets = inverse_vol_weights(history, window=price_series_window)
        targets = cap_weights(targets, max_weight)

        # Turn targets into orders by comparing to current positions
        current = {}
        for s in targets.keys():
            current[s] = 0.0
        if portfolio_snapshot and 'positions' in portfolio_snapshot:
            for s, p in portfolio_snapshot['positions'].items():
                current[s] = p.get('market_value', 0.0) / (portfolio_snapshot.get('portfolio_value', 1.0) or 1.0)

        new_weights = rebalance_with_turnover(current, targets, turnover_limit=turnover_limit)

        # Convert weight changes to orders (very simplified: assume price-based quantity)
        orders = []
        pv = portfolio_snapshot.get('portfolio_value', 100000.0)
        for s in new_weights:
            desired_value = new_weights[s] * pv
            current_value = current.get(s, 0.0) * pv
            delta = desired_value - current_value
            if abs(delta) / pv < 0.001:
                continue
            price = prices.get(s)
            if not price:
                continue
            qty = abs(delta) / price
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(symbol=s, side=side, order_type=OrderType.MARKET, quantity=qty))
        return orders

    return strategy