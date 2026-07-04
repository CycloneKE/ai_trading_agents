#!/usr/bin/env python3
"""
AI Trading Agent Trainer and Evaluator
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging
from supervised_learning import SupervisedLearningStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgentTrainer:
    """Train and evaluate AI trading agent with comprehensive backtesting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 0.1%
        self.initial_capital = config.get('initial_capital', 100000)
        
        # Initialize strategy
        self.strategy = SupervisedLearningStrategy("ai_agent", config)
        
        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data with technical indicators"""
        try:
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date)
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
            data['Volatility'] = data['Close'].pct_change().rolling(20).std()
            
            # Forward fill missing values
            data = data.fillna(method='ffill').dropna()
            
            logger.info(f"Loaded {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def train_agent(self, data: pd.DataFrame, symbol: str) -> bool:
        """Train the AI agent on historical data"""
        logger.info("Training AI agent...")
        
        training_data = []
        returns = data['Close'].pct_change().shift(-1)  # Next day return
        
        for i in range(len(data) - 1):
            row = data.iloc[i]
            market_data = {
                'symbol': symbol,
                'close': row['Close'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'volume': row['Volume'],
                'rsi': row['RSI'],
                'macd': row['MACD'],
                'bb_upper': row['BB_Upper'],
                'bb_lower': row['BB_Lower'],
                'volatility': row['Volatility'],
                'sma_50': row['SMA_50'],
                'pe_ratio': 15.0,  # Mock fundamental data
                'debt_ratio': 0.3,
                'current_ratio': 1.5,
                'market_cap': 1000000000
            }
            
            training_data.append(market_data)
        
        # Train with feedback
        feedback = {
            'returns': returns.dropna().tolist(),
            'win_rate': 0.55,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.15
        }
        
        success = self.strategy.update_model(training_data[-1], feedback)
        logger.info(f"Training {'successful' if success else 'failed'}")
        return success
    
    def backtest(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        logger.info("Starting backtest...")
        
        # Initialize portfolio
        cash = self.initial_capital
        position = 0
        position_value = 0
        
        self.trades = []
        self.equity_curve = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            current_price = row['Close']
            
            # Prepare market data
            market_data = {
                'symbol': symbol,
                'close': current_price,
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'volume': row['Volume'],
                'rsi': row['RSI'] if not pd.isna(row['RSI']) else 50,
                'macd': row['MACD'] if not pd.isna(row['MACD']) else 0,
                'bb_upper': row['BB_Upper'] if not pd.isna(row['BB_Upper']) else current_price * 1.02,
                'bb_lower': row['BB_Lower'] if not pd.isna(row['BB_Lower']) else current_price * 0.98,
                'volatility': row['Volatility'] if not pd.isna(row['Volatility']) else 0.02,
                'account_value': cash + position_value,
                'pe_ratio': 15.0,
                'debt_ratio': 0.3,
                'current_ratio': 1.5,
                'market_cap': 1000000000,
                'sector': 'technology'
            }
            
            # Get signal from AI agent
            signal = self.strategy.generate_signals(market_data)
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)
            
            # Update position value
            if position != 0:
                position_value = position * current_price
            
            # Execute trades
            if action == 'buy' and position <= 0 and cash > current_price:
                # Buy signal
                shares_to_buy = int((cash * 0.95) / current_price)  # Use 95% of cash
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    if cost <= cash:
                        cash -= cost
                        position += shares_to_buy
                        
                        self.trades.append({
                            'date': data.index[i],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': cost,
                            'confidence': confidence
                        })
            
            elif action == 'sell' and position > 0:
                # Sell signal
                proceeds = position * current_price * (1 - self.transaction_cost)
                cash += proceeds
                
                # Calculate P&L
                if self.trades:
                    last_buy = next((t for t in reversed(self.trades) if t['action'] == 'BUY'), None)
                    if last_buy:
                        pnl = proceeds - last_buy['value']
                        pnl_pct = (pnl / last_buy['value']) * 100
                    else:
                        pnl = pnl_pct = 0
                else:
                    pnl = pnl_pct = 0
                
                self.trades.append({
                    'date': data.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': confidence
                })
                
                position = 0
            
            # Update equity curve
            total_value = cash + (position * current_price if position > 0 else 0)
            self.equity_curve.append({
                'date': data.index[i],
                'total_value': total_value,
                'cash': cash,
                'position_value': position * current_price if position > 0 else 0,
                'price': current_price
            })
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(data)
        
        logger.info("Backtest completed")
        return self.performance_metrics
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = equity_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Trade metrics
        completed_trades = [t for t in self.trades if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        returns = equity_df['total_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-10)
        
        # Drawdown
        rolling_max = equity_df['total_value'].expanding().max()
        drawdown = (equity_df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Benchmark comparison (buy and hold)
        buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        
        return {
            'total_return': total_return,
            'total_trades': len(completed_trades),
            'win_rate': win_rate,
            'avg_profit_per_trade': np.mean([t['pnl'] for t in completed_trades]) if completed_trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'final_value': final_value
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.performance_metrics:
            return "No performance data available"
        
        metrics = self.performance_metrics
        
        report = f"""
{'='*60}
AI TRADING AGENT PERFORMANCE REPORT
{'='*60}

OVERALL PERFORMANCE:
  Total Return:           {metrics['total_return']:.2%}
  Buy & Hold Return:      {metrics['buy_hold_return']:.2%}
  Excess Return:          {metrics['excess_return']:.2%}
  Final Portfolio Value:  ${metrics['final_value']:,.2f}

TRADE STATISTICS:
  Total Trades:           {metrics['total_trades']}
  Win Rate:               {metrics['win_rate']:.1%}
  Average Profit/Trade:   ${metrics['avg_profit_per_trade']:.2f}
  Average Win:            ${metrics['avg_win']:.2f}
  Average Loss:           ${metrics['avg_loss']:.2f}

RISK METRICS:
  Maximum Drawdown:       {metrics['max_drawdown']:.2%}
  Volatility (Annual):    {metrics['volatility']:.2%}
  Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}

TRADE LOG (Last 10 trades):
"""
        
        # Add recent trades
        recent_trades = self.trades[-10:] if len(self.trades) > 10 else self.trades
        for trade in recent_trades:
            pnl_str = f"P&L: ${trade.get('pnl', 0):.2f}" if 'pnl' in trade else ""
            report += f"  {trade['date'].strftime('%Y-%m-%d')} | {trade['action']} | ${trade['price']:.2f} | {pnl_str}\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def plot_results(self):
        """Generate performance charts"""
        if not self.equity_curve:
            print("No data to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price chart with trades
        ax1.plot(equity_df.index, equity_df['price'], label='Price', alpha=0.7)
        
        # Mark buy/sell points
        for trade in self.trades:
            color = 'green' if trade['action'] == 'BUY' else 'red'
            marker = '^' if trade['action'] == 'BUY' else 'v'
            ax1.scatter(trade['date'], trade['price'], color=color, marker=marker, s=50)
        
        ax1.set_title('Price Chart with Trades')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        
        # Equity curve
        ax2.plot(equity_df.index, equity_df['total_value'], label='Portfolio Value')
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        
        # Drawdown chart
        rolling_max = equity_df['total_value'].expanding().max()
        drawdown = (equity_df['total_value'] - rolling_max) / rolling_max
        ax3.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        
        # Trade distribution
        completed_trades = [t for t in self.trades if 'pnl' in t]
        if completed_trades:
            pnls = [t['pnl'] for t in completed_trades]
            ax4.hist(pnls, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--')
            ax4.set_title('Trade P&L Distribution')
            ax4.set_xlabel('P&L ($)')
            ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main training and evaluation function"""
    # Configuration
    config = {
        'model_type': 'xgboost',
        'lookback_period': 20,
        'threshold': 0.02,
        'features': ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'pe_ratio', 'debt_ratio'],
        'transaction_cost': 0.001,
        'initial_capital': 100000,
        'bias_threshold': 0.15
    }
    
    # Initialize trainer
    trainer = AIAgentTrainer(config)
    
    # Load data
    symbol = 'AAPL'
    train_start = '2020-01-01'
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    test_end = '2024-12-31'
    
    print("Loading training data...")
    train_data = trainer.load_data(symbol, train_start, train_end)
    
    print("Loading test data...")
    test_data = trainer.load_data(symbol, test_start, test_end)
    
    if train_data.empty or test_data.empty:
        print("Failed to load data")
        return
    
    # Train agent
    print("Training AI agent...")
    trainer.train_agent(train_data, symbol)
    
    # Run backtest
    print("Running backtest...")
    performance = trainer.backtest(test_data, symbol)
    
    # Generate report
    print(trainer.generate_report())
    
    # Plot results
    trainer.plot_results()

if __name__ == "__main__":
    main()