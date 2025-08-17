"""
Backtesting Engine with realistic market conditions.
Includes slippage, commissions, market impact, and various order types.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation for backtesting."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None


@dataclass
class Trade:
    """Trade execution representation."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    order_id: str


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


class BacktestEngine:
    """
    High-speed backtesting engine with realistic market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Trading parameters
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.min_commission = config.get('min_commission', 1.0)
        self.slippage_model = config.get('slippage_model', 'linear')
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 0.05%
        self.market_impact_model = config.get('market_impact_model', 'sqrt')
        self.market_impact_rate = config.get('market_impact_rate', 0.0001)
        
        # Portfolio state
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.orders = []
        
        # Performance tracking
        self.portfolio_history = []
        self.returns_history = []
        self.drawdown_history = []
        
        # Market data
        self.current_prices = {}
        self.current_timestamp = None
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        logger.info("Backtest engine initialized")
    
    def add_market_data(self, data: Dict[str, pd.DataFrame]):
        """
        Add market data for backtesting.
        
        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
        """
        try:
            self.market_data = {}
            
            for symbol, df in data.items():
                if not df.empty and 'close' in df.columns:
                    # Ensure datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if 'timestamp' in df.columns:
                            df = df.set_index('timestamp')
                        elif 'date' in df.columns:
                            df = df.set_index('date')
                    
                    # Sort by timestamp
                    df = df.sort_index()
                    
                    self.market_data[symbol] = df
                    logger.info(f"Added market data for {symbol}: {len(df)} bars")
            
        except Exception as e:
            logger.error(f"Error adding market data: {str(e)}")
    
    def calculate_slippage(self, symbol: str, side: OrderSide, quantity: float, 
                          price: float) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Order price
            
        Returns:
            Slippage amount
        """
        try:
            if self.slippage_model == 'linear':
                # Linear slippage model
                slippage = price * self.slippage_rate
                
            elif self.slippage_model == 'sqrt':
                # Square root model (more realistic for large orders)
                volume_factor = np.sqrt(quantity / 1000)  # Normalize by 1000 shares
                slippage = price * self.slippage_rate * volume_factor
                
            elif self.slippage_model == 'market_impact':
                # Market impact model
                if symbol in self.market_data:
                    # Use average volume for impact calculation
                    avg_volume = self.market_data[symbol]['volume'].mean()
                    volume_ratio = quantity / avg_volume if avg_volume > 0 else 0.01
                    slippage = price * self.market_impact_rate * np.sqrt(volume_ratio)
                else:
                    slippage = price * self.slippage_rate
                    
            else:
                slippage = price * self.slippage_rate
            
            # Apply direction (buy orders pay slippage, sell orders receive negative slippage)
            if side == OrderSide.BUY:
                return slippage
            else:
                return -slippage
                
        except Exception as e:
            logger.error(f"Error calculating slippage: {str(e)}")
            return 0.0
    
    def calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Commission amount
        """
        try:
            trade_value = quantity * price
            commission = max(trade_value * self.commission_rate, self.min_commission)
            return commission
            
        except Exception as e:
            logger.error(f"Error calculating commission: {str(e)}")
            return self.min_commission
    
    def execute_order(self, order: Order, current_price: float) -> Optional[Trade]:
        """
        Execute an order at current market conditions.
        
        Args:
            order: Order to execute
            current_price: Current market price
            
        Returns:
            Trade if executed, None otherwise
        """
        try:
            # Check if order can be executed
            can_execute = False
            execution_price = current_price
            
            if order.order_type == OrderType.MARKET:
                can_execute = True
                execution_price = current_price
                
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    can_execute = True
                    execution_price = order.price
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    can_execute = True
                    execution_price = order.price
                    
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    can_execute = True
                    execution_price = current_price
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    can_execute = True
                    execution_price = current_price
            
            if not can_execute:
                return None
            
            # Calculate slippage and commission
            slippage = self.calculate_slippage(order.symbol, order.side, order.quantity, execution_price)
            commission = self.calculate_commission(order.quantity, execution_price)
            
            # Adjust execution price for slippage
            final_price = execution_price + slippage
            
            # Check if we have enough cash/shares
            trade_value = order.quantity * final_price
            
            if order.side == OrderSide.BUY:
                total_cost = trade_value + commission
                if self.cash < total_cost:
                    logger.warning(f"Insufficient cash for order: {total_cost} > {self.cash}")
                    return None
            else:  # SELL
                current_position = self.positions.get(order.symbol, Position(order.symbol, 0, 0, 0, 0, 0))
                if current_position.quantity < order.quantity:
                    logger.warning(f"Insufficient shares for sell order: {order.quantity} > {current_position.quantity}")
                    return None
            
            # Create trade
            trade = Trade(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=final_price,
                commission=commission,
                slippage=abs(slippage),
                timestamp=self.current_timestamp,
                order_id=order.order_id or f"order_{len(self.trades)}"
            )
            
            # Update portfolio
            self._update_portfolio(trade)
            
            # Track statistics
            self.total_trades += 1
            self.total_commission += commission
            self.total_slippage += abs(slippage)
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return None
    
    def _update_portfolio(self, trade: Trade):
        """
        Update portfolio state after trade execution.
        
        Args:
            trade: Executed trade
        """
        try:
            symbol = trade.symbol
            
            # Get or create position
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol, 0, 0, 0, 0, 0)
            
            position = self.positions[symbol]
            
            if trade.side == OrderSide.BUY:
                # Update cash
                self.cash -= (trade.quantity * trade.price + trade.commission)
                
                # Update position
                if position.quantity >= 0:  # Long position or new position
                    total_cost = position.quantity * position.avg_price + trade.quantity * trade.price
                    total_quantity = position.quantity + trade.quantity
                    position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                    position.quantity = total_quantity
                else:  # Covering short position
                    if trade.quantity <= abs(position.quantity):
                        # Partial cover
                        realized_pnl = trade.quantity * (position.avg_price - trade.price)
                        position.realized_pnl += realized_pnl
                        position.quantity += trade.quantity
                    else:
                        # Full cover and go long
                        cover_quantity = abs(position.quantity)
                        realized_pnl = cover_quantity * (position.avg_price - trade.price)
                        position.realized_pnl += realized_pnl
                        
                        remaining_quantity = trade.quantity - cover_quantity
                        position.quantity = remaining_quantity
                        position.avg_price = trade.price
                        
            else:  # SELL
                # Update cash
                self.cash += (trade.quantity * trade.price - trade.commission)
                
                # Update position
                if position.quantity > 0:  # Selling long position
                    if trade.quantity <= position.quantity:
                        # Partial or full sale
                        realized_pnl = trade.quantity * (trade.price - position.avg_price)
                        position.realized_pnl += realized_pnl
                        position.quantity -= trade.quantity
                    else:
                        # Sell more than we have (go short)
                        long_quantity = position.quantity
                        realized_pnl = long_quantity * (trade.price - position.avg_price)
                        position.realized_pnl += realized_pnl
                        
                        short_quantity = trade.quantity - long_quantity
                        position.quantity = -short_quantity
                        position.avg_price = trade.price
                else:  # Adding to short position
                    total_cost = abs(position.quantity) * position.avg_price + trade.quantity * trade.price
                    total_quantity = abs(position.quantity) + trade.quantity
                    position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                    position.quantity = -total_quantity
            
            # Clean up zero positions
            if abs(position.quantity) < 1e-8:
                position.quantity = 0
                if symbol in self.current_prices:
                    position.market_value = 0
                    position.unrealized_pnl = 0
            
            # Store trade
            self.trades.append(trade)
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {str(e)}")
    
    def update_portfolio_values(self):
        """
        Update portfolio values based on current market prices.
        """
        try:
            total_value = self.cash
            
            for symbol, position in self.positions.items():
                if symbol in self.current_prices and position.quantity != 0:
                    current_price = self.current_prices[symbol]
                    position.market_value = position.quantity * current_price
                    
                    if position.quantity > 0:  # Long position
                        position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
                    else:  # Short position
                        position.unrealized_pnl = abs(position.quantity) * (position.avg_price - current_price)
                    
                    total_value += position.market_value
                else:
                    position.market_value = 0
                    position.unrealized_pnl = 0
            
            self.portfolio_value = total_value
            
        except Exception as e:
            logger.error(f"Error updating portfolio values: {str(e)}")
    
    def run_backtest(self, strategy_func: Callable, start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run backtest with given strategy function.
        
        Args:
            strategy_func: Strategy function that takes (timestamp, prices, portfolio) and returns orders
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results
        """
        try:
            if not self.market_data:
                raise ValueError("No market data available for backtesting")
            
            # Get date range
            all_dates = set()
            for df in self.market_data.values():
                all_dates.update(df.index)
            
            all_dates = sorted(all_dates)
            
            if start_date:
                all_dates = [d for d in all_dates if d >= start_date]
            if end_date:
                all_dates = [d for d in all_dates if d <= end_date]
            
            if not all_dates:
                raise ValueError("No data available for specified date range")
            
            logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} periods)")
            
            # Reset portfolio state
            self.cash = self.initial_capital
            self.positions = {}
            self.portfolio_value = self.initial_capital
            self.trades = []
            self.portfolio_history = []
            self.returns_history = []
            
            # Run backtest
            for i, timestamp in enumerate(all_dates):
                self.current_timestamp = timestamp
                
                # Update current prices
                self.current_prices = {}
                for symbol, df in self.market_data.items():
                    if timestamp in df.index:
                        self.current_prices[symbol] = df.loc[timestamp, 'close']
                
                # Update portfolio values
                prev_portfolio_value = self.portfolio_value
                self.update_portfolio_values()
                
                # Calculate return
                if i > 0 and prev_portfolio_value > 0:
                    period_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
                    self.returns_history.append(period_return)
                
                # Store portfolio snapshot
                portfolio_snapshot = {
                    'timestamp': timestamp,
                    'portfolio_value': self.portfolio_value,
                    'cash': self.cash,
                    'positions': {symbol: {
                        'quantity': pos.quantity,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl
                    } for symbol, pos in self.positions.items() if pos.quantity != 0}
                }
                self.portfolio_history.append(portfolio_snapshot)
                
                # Generate strategy signals
                try:
                    orders = strategy_func(timestamp, self.current_prices.copy(), portfolio_snapshot.copy())
                    
                    if orders:
                        # Execute orders
                        for order in orders:
                            if order.symbol in self.current_prices:
                                trade = self.execute_order(order, self.current_prices[order.symbol])
                                if trade:
                                    if trade.side == OrderSide.BUY:
                                        logger.debug(f"Executed BUY: {trade.quantity} {trade.symbol} @ {trade.price:.2f}")
                                    else:
                                        logger.debug(f"Executed SELL: {trade.quantity} {trade.symbol} @ {trade.price:.2f}")
                
                except Exception as e:
                    logger.error(f"Error in strategy function at {timestamp}: {str(e)}")
                
                # Progress logging
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{len(all_dates)} periods, Portfolio: ${self.portfolio_value:.2f}")
            
            # Calculate final results
            results = self._calculate_backtest_results()
            
            logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio_value:.2f}")
            logger.info(f"Total return: {results['performance_metrics']['total_return']:.2%}")
            logger.info(f"Sharpe ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {}
    
    def _calculate_backtest_results(self) -> Dict[str, Any]:
        """
        Calculate comprehensive backtest results.
        
        Returns:
            Dict with backtest results
        """
        try:
            if not self.portfolio_history:
                return {}
            
            # Basic metrics
            initial_value = self.initial_capital
            final_value = self.portfolio_value
            total_return = (final_value - initial_value) / initial_value
            
            # Returns analysis
            returns = np.array(self.returns_history)
            
            if len(returns) > 0:
                # Performance metrics
                annual_return = np.mean(returns) * 252
                annual_volatility = np.std(returns) * np.sqrt(252)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                
                # Drawdown analysis
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
                # Win rate
                winning_periods = len(returns[returns > 0])
                win_rate = winning_periods / len(returns) if len(returns) > 0 else 0
                
            else:
                annual_return = 0
                annual_volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                win_rate = 0
            
            # Trade analysis
            if self.trades:
                trade_pnls = []
                for trade in self.trades:
                    # Calculate P&L for each trade (simplified)
                    if trade.side == OrderSide.SELL:
                        # For sells, we need to match with previous buys
                        # This is simplified - in reality we'd need proper position tracking
                        trade_pnls.append(0)  # Placeholder
                
                winning_trades = len([t for t in self.trades if getattr(t, 'pnl', 0) > 0])
                trade_win_rate = winning_trades / len(self.trades) if self.trades else 0
            else:
                trade_win_rate = 0
            
            results = {
                'performance_metrics': {
                    'initial_capital': initial_value,
                    'final_value': final_value,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'annual_volatility': annual_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate
                },
                'trade_metrics': {
                    'total_trades': len(self.trades),
                    'trade_win_rate': trade_win_rate,
                    'total_commission': self.total_commission,
                    'total_slippage': self.total_slippage,
                    'avg_commission_per_trade': self.total_commission / len(self.trades) if self.trades else 0,
                    'avg_slippage_per_trade': self.total_slippage / len(self.trades) if self.trades else 0
                },
                'portfolio_history': self.portfolio_history,
                'trades': [
                    {
                        'symbol': t.symbol,
                        'side': t.side.value,
                        'quantity': t.quantity,
                        'price': t.price,
                        'commission': t.commission,
                        'slippage': t.slippage,
                        'timestamp': t.timestamp.isoformat()
                    } for t in self.trades
                ],
                'final_positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'avg_price': pos.avg_price,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl
                    } for symbol, pos in self.positions.items() if pos.quantity != 0
                },
                'returns_history': self.returns_history,
                'backtest_config': self.config
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating backtest results: {str(e)}")
            return {}

