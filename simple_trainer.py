#!/usr/bin/env python3
"""
Simplified AI Trading Agent Trainer and Evaluator
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAITrainer:
    """Simplified trainer for AI trading agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transaction_cost = 0.001  # 0.1%
        self.initial_capital = 100000
        
        # Mock strategy for demonstration
        self.model_trained = False
        self.trades = []
        self.performance_metrics = {}
    
    def generate_mock_data(self, symbol: str, days: int = 500) -> pd.DataFrame:
        """Generate realistic mock market data"""
        np.random.seed(42)
        
        # Generate price data with trend and volatility
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Simulate realistic price movement
        initial_price = 150.0
        returns = np.random.normal(0.0005, 0.02, days)  # Small positive drift, 2% daily vol
        
        prices = [initial_price]
        for i in range(1, days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        # Generate OHLC from close prices
        data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Volume': np.random.randint(1000000, 10000000, days)
        })
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = self._calculate_macd(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std()
        
        data.set_index('Date', inplace=True)
        return data.ffill().dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def simple_strategy(self, data: pd.DataFrame, i: int) -> str:
        """Simple trading strategy for demonstration"""
        if i < 20:  # Need enough data for indicators
            return 'hold'
        
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        
        # Simple momentum + mean reversion strategy
        rsi = row['RSI']
        macd = row['MACD']
        price_ma_ratio = row['Close'] / row['SMA_20']
        
        # Buy conditions
        if (rsi < 30 and macd > 0 and price_ma_ratio > 1.02):
            return 'buy'
        
        # Sell conditions  
        elif (rsi > 70 or price_ma_ratio < 0.98):
            return 'sell'
        
        return 'hold'
    
    def backtest(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Run backtest with simple strategy"""
        logger.info("Running backtest...")
        
        # Initialize portfolio
        cash = self.initial_capital
        position = 0
        self.trades = []
        equity_curve = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            current_price = row['Close']
            
            # Get trading signal
            signal = self.simple_strategy(data, i)
            
            # Update position value
            position_value = position * current_price if position > 0 else 0
            total_value = cash + position_value
            
            # Execute trades
            if signal == 'buy' and position == 0 and cash > current_price * 100:
                # Buy with 90% of available cash
                shares_to_buy = int((cash * 0.9) / current_price)
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= cash:
                    cash -= cost
                    position = shares_to_buy
                    
                    self.trades.append({
                        'date': data.index[i],
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'value': cost
                    })
            
            elif signal == 'sell' and position > 0:
                # Sell all shares
                proceeds = position * current_price * (1 - self.transaction_cost)
                
                # Calculate P&L
                last_buy = next((t for t in reversed(self.trades) if t['action'] == 'BUY'), None)
                if last_buy:
                    pnl = proceeds - last_buy['value']
                    pnl_pct = (pnl / last_buy['value']) * 100
                else:
                    pnl = pnl_pct = 0
                
                cash += proceeds
                
                self.trades.append({
                    'date': data.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                position = 0
            
            # Track equity
            equity_curve.append({
                'date': data.index[i],
                'total_value': total_value,
                'price': current_price
            })
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_metrics(data, equity_curve)
        return self.performance_metrics
    
    def _calculate_metrics(self, data: pd.DataFrame, equity_curve: List[Dict]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not equity_curve:
            return {}
        
        # Basic performance
        initial_value = self.initial_capital
        final_value = equity_curve[-1]['total_value']
        total_return = (final_value - initial_value) / initial_value
        
        # Trade statistics
        completed_trades = [t for t in self.trades if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_profit = np.mean([t['pnl'] for t in completed_trades]) if completed_trades else 0
        
        # Risk metrics
        values = [eq['total_value'] for eq in equity_curve]
        returns = pd.Series(values).pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-10) if volatility > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = [(v - p) / p for v, p in zip(values, peak)]
        max_drawdown = min(drawdown) if drawdown else 0
        
        # Benchmark (buy and hold)
        buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        
        return {
            'total_return': total_return,
            'total_trades': len(completed_trades),
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'final_value': final_value
        }
    
    def generate_report(self) -> str:
        """Generate performance report"""
        if not self.performance_metrics:
            return "No performance data available"
        
        m = self.performance_metrics
        
        report = f"""
{'='*60}
AI TRADING AGENT PERFORMANCE REPORT
{'='*60}

OVERALL PERFORMANCE:
  Total Return:           {m['total_return']:.2%}
  Buy & Hold Return:      {m['buy_hold_return']:.2%}
  Excess Return:          {m['excess_return']:.2%}
  Final Portfolio Value:  ${m['final_value']:,.2f}

TRADE STATISTICS:
  Total Trades:           {m['total_trades']}
  Win Rate:               {m['win_rate']:.1%}
  Average Profit/Trade:   ${m['avg_profit_per_trade']:.2f}

RISK METRICS:
  Maximum Drawdown:       {m['max_drawdown']:.2%}
  Volatility (Annual):    {m['volatility']:.2%}
  Sharpe Ratio:           {m['sharpe_ratio']:.2f}

RECENT TRADES:
"""
        
        # Add recent trades
        recent_trades = self.trades[-10:] if len(self.trades) > 10 else self.trades
        for trade in recent_trades:
            pnl_str = f"P&L: ${trade.get('pnl', 0):.2f}" if 'pnl' in trade else ""
            report += f"  {trade['date'].strftime('%Y-%m-%d')} | {trade['action']} | ${trade['price']:.2f} | {pnl_str}\n"
        
        # Performance insights
        report += f"\nPERFORMANCE INSIGHTS:\n"
        
        if m['excess_return'] > 0:
            report += f"  [+] Strategy outperformed buy-and-hold by {m['excess_return']:.2%}\n"
        else:
            report += f"  [-] Strategy underperformed buy-and-hold by {abs(m['excess_return']):.2%}\n"
        
        if m['sharpe_ratio'] > 1.0:
            report += f"  [+] Good risk-adjusted returns (Sharpe: {m['sharpe_ratio']:.2f})\n"
        else:
            report += f"  [!] Poor risk-adjusted returns (Sharpe: {m['sharpe_ratio']:.2f})\n"
        
        if m['win_rate'] > 0.5:
            report += f"  [+] Positive win rate: {m['win_rate']:.1%}\n"
        else:
            report += f"  [!] Low win rate: {m['win_rate']:.1%}\n"
        
        report += f"\n{'='*60}\n"
        return report
    
    def export_results(self, filename: str = "backtest_results.json"):
        """Export results to JSON file"""
        results = {
            'performance_metrics': self.performance_metrics,
            'trades': [
                {
                    'date': trade['date'].isoformat(),
                    'action': trade['action'],
                    'price': trade['price'],
                    'pnl': trade.get('pnl', 0)
                }
                for trade in self.trades
            ],
            'config': self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filename}")

def main():
    """Main function to run training and evaluation"""
    # Configuration
    config = {
        'strategy_name': 'simple_momentum_mean_reversion',
        'lookback_period': 20,
        'transaction_cost': 0.001,
        'initial_capital': 100000
    }
    
    # Initialize trainer
    trainer = SimpleAITrainer(config)
    
    # Generate mock data (replace with real data loading)
    print("Generating market data...")
    symbol = 'AAPL'
    data = trainer.generate_mock_data(symbol, days=365)
    
    print(f"Generated {len(data)} days of market data")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Split data for training and testing
    split_point = int(len(data) * 0.7)  # 70% for training, 30% for testing
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Run backtest on test data
    print("Running backtest...")
    performance = trainer.backtest(test_data, symbol)
    
    # Generate and display report
    print(trainer.generate_report())
    
    # Export results
    trainer.export_results("ai_agent_backtest_results.json")
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()