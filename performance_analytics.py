"""
Performance Analytics Engine
Comprehensive performance analysis and reporting for trading strategies
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    SPY = "SPY"
    QQQ = "QQQ"
    CUSTOM = "CUSTOM"

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int

class PerformanceAnalytics:
    """Comprehensive performance analytics engine"""
    
    def __init__(self, config: Dict[str, Any], database_manager=None):
        self.config = config
        self.database = database_manager
        
        # Analytics configuration
        self.benchmark = config.get('benchmark', BenchmarkType.SPY.value)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        self.trading_days_per_year = config.get('trading_days_per_year', 252)
        
        # Data storage
        self.returns_history = []
        self.trades_history = []
        self.portfolio_values = []
        self.benchmark_returns = []
        
        # Cached metrics
        self.cached_metrics = {}
        self.last_calculation = None
        
    def record_portfolio_value(self, value: float, timestamp: Optional[datetime] = None):
        """Record portfolio value for performance tracking"""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            self.portfolio_values.append({
                'timestamp': timestamp,
                'value': value
            })
            
            # Calculate return if we have previous value
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]['value']
                if prev_value > 0:
                    return_pct = (value - prev_value) / prev_value
                    self.returns_history.append({
                        'timestamp': timestamp,
                        'return': return_pct
                    })
            
            # Keep only recent data (configurable)
            max_history = self.config.get('max_history_days', 365) * 24  # Assuming hourly data
            if len(self.portfolio_values) > max_history:
                self.portfolio_values = self.portfolio_values[-max_history:]
            
            if len(self.returns_history) > max_history:
                self.returns_history = self.returns_history[-max_history:]
                
        except Exception as e:
            logger.error(f"Error recording portfolio value: {e}")
    
    def record_trade(self, symbol: str, side: str, quantity: float, price: float, 
                    timestamp: Optional[datetime] = None, strategy: Optional[str] = None):
        """Record trade execution for performance analysis"""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'strategy': strategy or 'unknown'
            }
            
            self.trades_history.append(trade)
            
            # Keep only recent trades
            max_trades = self.config.get('max_trades_history', 10000)
            if len(self.trades_history) > max_trades:
                self.trades_history = self.trades_history[-max_trades:]
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def calculate_performance_metrics(self, start_date: Optional[datetime] = None, 
                                    end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Filter data by date range
            returns = self._filter_returns_by_date(start_date, end_date)
            trades = self._filter_trades_by_date(start_date, end_date)
            
            if not returns:
                return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # Convert to numpy array for calculations
            returns_array = np.array([r['return'] for r in returns])
            
            # Basic return metrics
            total_return = np.prod(1 + returns_array) - 1
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns_array)) - 1
            volatility = np.std(returns_array) * np.sqrt(self.trading_days_per_year)
            
            # Risk-adjusted metrics
            excess_returns = returns_array - (self.risk_free_rate / self.trading_days_per_year)
            sharpe_ratio = np.mean(excess_returns) / np.std(returns_array) * np.sqrt(self.trading_days_per_year) if np.std(returns_array) > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(self.trading_days_per_year) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year) if downside_deviation > 0 else 0
            
            # Drawdown metrics
            max_drawdown = self._calculate_max_drawdown(returns_array)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade-based metrics
            trade_metrics = self._calculate_trade_metrics(trades)
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=trade_metrics['win_rate'],
                profit_factor=trade_metrics['profit_factor'],
                avg_win=trade_metrics['avg_win'],
                avg_loss=trade_metrics['avg_loss'],
                total_trades=trade_metrics['total_trades'],
                winning_trades=trade_metrics['winning_trades'],
                losing_trades=trade_metrics['losing_trades']
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _filter_returns_by_date(self, start_date: Optional[datetime], 
                               end_date: Optional[datetime]) -> List[Dict[str, Any]]:
        """Filter returns by date range"""
        filtered_returns = self.returns_history
        
        if start_date:
            filtered_returns = [r for r in filtered_returns if r['timestamp'] >= start_date]
        
        if end_date:
            filtered_returns = [r for r in filtered_returns if r['timestamp'] <= end_date]
        
        return filtered_returns
    
    def _filter_trades_by_date(self, start_date: Optional[datetime], 
                              end_date: Optional[datetime]) -> List[Dict[str, Any]]:
        """Filter trades by date range"""
        filtered_trades = self.trades_history
        
        if start_date:
            filtered_trades = [t for t in filtered_trades if t['timestamp'] >= start_date]
        
        if end_date:
            filtered_trades = [t for t in filtered_trades if t['timestamp'] <= end_date]
        
        return filtered_trades
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            return float(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade-based performance metrics"""
        try:
            if not trades:
                return {
                    'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
                    'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0
                }
            
            # Group trades by symbol to calculate P&L
            positions = {}
            trade_pnls = []
            
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in positions:
                    positions[symbol] = {'quantity': 0, 'avg_price': 0}
                
                position = positions[symbol]
                
                if trade['side'] == 'buy':
                    # Update average price for buys
                    if position['quantity'] >= 0:
                        total_cost = position['quantity'] * position['avg_price'] + trade['value']
                        total_quantity = position['quantity'] + trade['quantity']
                        position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
                        position['quantity'] = total_quantity
                    else:
                        # Covering short position
                        pnl = (position['avg_price'] - trade['price']) * min(abs(position['quantity']), trade['quantity'])
                        trade_pnls.append(pnl)
                        position['quantity'] += trade['quantity']
                
                else:  # sell
                    if position['quantity'] > 0:
                        # Selling long position
                        pnl = (trade['price'] - position['avg_price']) * min(position['quantity'], trade['quantity'])
                        trade_pnls.append(pnl)
                        position['quantity'] -= trade['quantity']
                    else:
                        # Opening short position
                        if position['quantity'] <= 0:
                            total_value = abs(position['quantity']) * position['avg_price'] + trade['value']
                            total_quantity = abs(position['quantity']) + trade['quantity']
                            position['avg_price'] = total_value / total_quantity if total_quantity > 0 else 0
                            position['quantity'] = -total_quantity
            
            # Calculate metrics from P&L
            if not trade_pnls:
                return {
                    'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
                    'total_trades': len(trades), 'winning_trades': 0, 'losing_trades': 0
                }
            
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_trades': len(trade_pnls),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {
                'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0
            }
    
    def generate_performance_report(self, period: str = '1M') -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            if period == '1D':
                start_date = end_date - timedelta(days=1)
            elif period == '1W':
                start_date = end_date - timedelta(weeks=1)
            elif period == '1M':
                start_date = end_date - timedelta(days=30)
            elif period == '3M':
                start_date = end_date - timedelta(days=90)
            elif period == '1Y':
                start_date = end_date - timedelta(days=365)
            else:
                start_date = None  # All time
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(start_date, end_date)
            
            # Get strategy breakdown
            strategy_performance = self._calculate_strategy_performance(start_date, end_date)
            
            # Get recent portfolio values for chart data
            recent_values = self._get_recent_portfolio_values(start_date, end_date)
            
            return {
                'period': period,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat(),
                'generated_at': datetime.utcnow().isoformat(),
                'metrics': {
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'volatility': metrics.volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'calmar_ratio': metrics.calmar_ratio,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'total_trades': metrics.total_trades
                },
                'strategy_performance': strategy_performance,
                'portfolio_chart': recent_values,
                'summary': {
                    'current_value': recent_values[-1]['value'] if recent_values else 0,
                    'period_return': metrics.total_return,
                    'best_day': self._get_best_day(start_date, end_date),
                    'worst_day': self._get_worst_day(start_date, end_date),
                    'total_trades': metrics.total_trades,
                    'winning_trades': metrics.winning_trades,
                    'losing_trades': metrics.losing_trades
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _calculate_strategy_performance(self, start_date: Optional[datetime], 
                                      end_date: Optional[datetime]) -> Dict[str, Any]:
        """Calculate performance breakdown by strategy"""
        try:
            trades = self._filter_trades_by_date(start_date, end_date)
            
            strategy_stats = {}
            
            for trade in trades:
                strategy = trade.get('strategy', 'unknown')
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        'trades': 0,
                        'volume': 0,
                        'symbols': set()
                    }
                
                strategy_stats[strategy]['trades'] += 1
                strategy_stats[strategy]['volume'] += trade['value']
                strategy_stats[strategy]['symbols'].add(trade['symbol'])
            
            # Convert sets to counts
            for strategy in strategy_stats:
                strategy_stats[strategy]['unique_symbols'] = len(strategy_stats[strategy]['symbols'])
                del strategy_stats[strategy]['symbols']
            
            return strategy_stats
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return {}
    
    def _get_recent_portfolio_values(self, start_date: Optional[datetime], 
                                   end_date: Optional[datetime]) -> List[Dict[str, Any]]:
        """Get portfolio values for charting"""
        try:
            values = self.portfolio_values
            
            if start_date:
                values = [v for v in values if v['timestamp'] >= start_date]
            
            if end_date:
                values = [v for v in values if v['timestamp'] <= end_date]
            
            # Format for charting
            chart_data = [
                {
                    'timestamp': v['timestamp'].isoformat(),
                    'value': v['value']
                }
                for v in values
            ]
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error getting portfolio values: {e}")
            return []
    
    def _get_best_day(self, start_date: Optional[datetime], 
                     end_date: Optional[datetime]) -> Dict[str, Any]:
        """Get best performing day"""
        try:
            returns = self._filter_returns_by_date(start_date, end_date)
            
            if not returns:
                return {'date': None, 'return': 0}
            
            best_return = max(returns, key=lambda x: x['return'])
            
            return {
                'date': best_return['timestamp'].isoformat(),
                'return': best_return['return']
            }
            
        except Exception as e:
            logger.error(f"Error getting best day: {e}")
            return {'date': None, 'return': 0}
    
    def _get_worst_day(self, start_date: Optional[datetime], 
                      end_date: Optional[datetime]) -> Dict[str, Any]:
        """Get worst performing day"""
        try:
            returns = self._filter_returns_by_date(start_date, end_date)
            
            if not returns:
                return {'date': None, 'return': 0}
            
            worst_return = min(returns, key=lambda x: x['return'])
            
            return {
                'date': worst_return['timestamp'].isoformat(),
                'return': worst_return['return']
            }
            
        except Exception as e:
            logger.error(f"Error getting worst day: {e}")
            return {'date': None, 'return': 0}
    
    def get_status(self) -> Dict[str, Any]:
        """Get analytics engine status"""
        return {
            'returns_history_count': len(self.returns_history),
            'trades_history_count': len(self.trades_history),
            'portfolio_values_count': len(self.portfolio_values),
            'last_calculation': self.last_calculation.isoformat() if self.last_calculation else None,
            'benchmark': self.benchmark,
            'risk_free_rate': self.risk_free_rate
        }