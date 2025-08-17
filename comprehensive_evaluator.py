#!/usr/bin/env python3
"""
Comprehensive AI Trading Agent Evaluator
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAgentEvaluator:
    """Comprehensive evaluation system for AI trading agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.initial_capital = config.get('initial_capital', 100000)
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        self.feature_importance = {}
        
    def generate_realistic_data(self, symbol: str, days: int = 500) -> pd.DataFrame:
        """Generate realistic market data with various market conditions"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Create different market regimes
        regime_length = days // 4
        regimes = ['bull', 'bear', 'sideways', 'volatile']
        
        prices = []
        volumes = []
        initial_price = 150.0
        current_price = initial_price
        
        for i in range(days):
            regime = regimes[min(i // regime_length, 3)]
            
            if regime == 'bull':
                drift = 0.001  # Positive trend
                vol = 0.015    # Moderate volatility
            elif regime == 'bear':
                drift = -0.0008  # Negative trend
                vol = 0.025      # Higher volatility
            elif regime == 'sideways':
                drift = 0.0002   # Minimal trend
                vol = 0.012      # Low volatility
            else:  # volatile
                drift = 0.0005   # Small positive trend
                vol = 0.035      # High volatility
            
            # Generate price movement
            return_val = np.random.normal(drift, vol)
            current_price *= (1 + return_val)
            current_price = max(current_price, 10.0)  # Floor price
            
            prices.append(current_price)
            volumes.append(np.random.randint(500000, 5000000))
        
        # Create OHLC data
        data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            'Volume': volumes
        })
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        data.set_index('Date', inplace=True)
        
        return data.ffill().dropna()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Moving averages
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        sma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        data['BB_Upper'] = sma20 + (std20 * 2)
        data['BB_Lower'] = sma20 - (std20 * 2)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / sma20
        
        # Volatility measures
        data['Volatility'] = data['Close'].pct_change().rolling(20).std()
        data['ATR'] = self._calculate_atr(data)
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price position indicators
        data['Price_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def advanced_strategy(self, data: pd.DataFrame, i: int) -> Tuple[str, float]:
        """Advanced multi-factor trading strategy"""
        if i < 50:  # Need enough data
            return 'hold', 0.0
        
        row = data.iloc[i]
        
        # Technical signals
        rsi = row['RSI']
        macd = row['MACD']
        macd_signal = row['MACD_Signal']
        bb_position = row['Price_Position']
        volume_ratio = row['Volume_Ratio']
        
        # Trend signals
        sma_10 = row['SMA_10']
        sma_20 = row['SMA_20']
        sma_50 = row['SMA_50']
        price = row['Close']
        
        # Calculate signal strength
        signals = []
        
        # RSI signals
        if rsi < 30:
            signals.append(('buy', 0.8))  # Oversold
        elif rsi > 70:
            signals.append(('sell', 0.8))  # Overbought
        
        # MACD signals
        if macd > macd_signal and macd > 0:
            signals.append(('buy', 0.6))
        elif macd < macd_signal and macd < 0:
            signals.append(('sell', 0.6))
        
        # Trend signals
        if price > sma_10 > sma_20 > sma_50:
            signals.append(('buy', 0.7))  # Strong uptrend
        elif price < sma_10 < sma_20 < sma_50:
            signals.append(('sell', 0.7))  # Strong downtrend
        
        # Bollinger Band signals
        if bb_position < 0.2 and volume_ratio > 1.2:
            signals.append(('buy', 0.5))  # Near lower band with volume
        elif bb_position > 0.8:
            signals.append(('sell', 0.5))  # Near upper band
        
        # Aggregate signals
        buy_strength = sum(strength for action, strength in signals if action == 'buy')
        sell_strength = sum(strength for action, strength in signals if action == 'sell')
        
        # Decision logic
        if buy_strength > sell_strength and buy_strength > 1.0:
            return 'buy', buy_strength
        elif sell_strength > buy_strength and sell_strength > 1.0:
            return 'sell', sell_strength
        else:
            return 'hold', 0.0
    
    def backtest_with_analysis(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Comprehensive backtest with detailed analysis"""
        logger.info("Running comprehensive backtest...")
        
        # Initialize portfolio
        cash = self.initial_capital
        position = 0
        self.trades = []
        self.equity_curve = []
        
        # Track feature performance
        feature_performance = {
            'rsi_signals': [],
            'macd_signals': [],
            'trend_signals': [],
            'bb_signals': []
        }
        
        for i in range(len(data)):
            row = data.iloc[i]
            current_price = row['Close']
            
            # Get trading signal and confidence
            signal, confidence = self.advanced_strategy(data, i)
            
            # Update position value
            position_value = position * current_price if position > 0 else 0
            total_value = cash + position_value
            
            # Execute trades based on signal and confidence
            if signal == 'buy' and position == 0 and cash > current_price * 100:
                # Position size based on confidence
                position_size = min(0.8, confidence / 2.0)  # Max 80% of capital
                shares_to_buy = int((cash * position_size) / current_price)
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= cash and shares_to_buy > 0:
                    cash -= cost
                    position = shares_to_buy
                    
                    self.trades.append({
                        'date': data.index[i],
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'value': cost,
                        'confidence': confidence,
                        'rsi': row['RSI'],
                        'macd': row['MACD']
                    })
            
            elif signal == 'sell' and position > 0:
                # Sell all shares
                proceeds = position * current_price * (1 - self.transaction_cost)
                
                # Calculate P&L
                last_buy = next((t for t in reversed(self.trades) if t['action'] == 'BUY'), None)
                if last_buy:
                    pnl = proceeds - last_buy['value']
                    pnl_pct = (pnl / last_buy['value']) * 100
                    hold_days = (data.index[i] - last_buy['date']).days
                else:
                    pnl = pnl_pct = hold_days = 0
                
                cash += proceeds
                
                self.trades.append({
                    'date': data.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': confidence,
                    'hold_days': hold_days,
                    'rsi': row['RSI'],
                    'macd': row['MACD']
                })
                
                position = 0
            
            # Track equity curve
            self.equity_curve.append({
                'date': data.index[i],
                'total_value': total_value,
                'cash': cash,
                'position_value': position_value,
                'price': current_price,
                'signal': signal,
                'confidence': confidence
            })
        
        # Calculate comprehensive metrics
        self.performance_metrics = self._calculate_comprehensive_metrics(data)
        self._analyze_feature_importance()
        
        return self.performance_metrics
    
    def _calculate_comprehensive_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic performance
        initial_value = self.initial_capital
        final_value = equity_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Trade analysis
        completed_trades = [t for t in self.trades if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        # Win/Loss statistics
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if losing_trades else float('inf')
        
        # Risk metrics
        returns = equity_df['total_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-10) if volatility > 0 else 0
        
        # Drawdown analysis
        rolling_max = equity_df['total_value'].expanding().max()
        drawdown = (equity_df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if any(drawdown < 0) else 0
        
        # Time-based metrics
        total_days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        avg_hold_time = np.mean([t.get('hold_days', 0) for t in completed_trades]) if completed_trades else 0
        
        # Benchmark comparison
        buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        
        # Advanced metrics
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365 / total_days) - 1,
            'total_trades': len(completed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': np.mean([t['pnl'] for t in completed_trades]) if completed_trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'final_value': final_value,
            'avg_hold_time': avg_hold_time,
            'total_days': total_days
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return (returns.mean() * 252) / downside_deviation
    
    def _analyze_feature_importance(self):
        """Analyze which features contributed most to performance"""
        if not self.trades:
            return
        
        completed_trades = [t for t in self.trades if 'pnl' in t]
        if not completed_trades:
            return
        
        # Analyze RSI performance
        rsi_trades = [(t['rsi'], t['pnl']) for t in completed_trades if 'rsi' in t]
        if rsi_trades:
            oversold_trades = [pnl for rsi, pnl in rsi_trades if rsi < 30]
            overbought_trades = [pnl for rsi, pnl in rsi_trades if rsi > 70]
            
            self.feature_importance['rsi'] = {
                'oversold_avg_pnl': np.mean(oversold_trades) if oversold_trades else 0,
                'overbought_avg_pnl': np.mean(overbought_trades) if overbought_trades else 0,
                'oversold_count': len(oversold_trades),
                'overbought_count': len(overbought_trades)
            }
        
        # Analyze confidence vs performance
        confidence_pnl = [(t.get('confidence', 0), t['pnl']) for t in completed_trades]
        if confidence_pnl:
            high_conf_trades = [pnl for conf, pnl in confidence_pnl if conf > 1.5]
            low_conf_trades = [pnl for conf, pnl in confidence_pnl if conf <= 1.5]
            
            self.feature_importance['confidence'] = {
                'high_confidence_avg_pnl': np.mean(high_conf_trades) if high_conf_trades else 0,
                'low_confidence_avg_pnl': np.mean(low_conf_trades) if low_conf_trades else 0,
                'high_confidence_count': len(high_conf_trades),
                'low_confidence_count': len(low_conf_trades)
            }
    
    def generate_comprehensive_report(self) -> str:
        """Generate detailed performance report"""
        if not self.performance_metrics:
            return "No performance data available"
        
        m = self.performance_metrics
        
        report = f"""
{'='*80}
COMPREHENSIVE AI TRADING AGENT EVALUATION REPORT
{'='*80}

OVERALL PERFORMANCE:
  Total Return:              {m['total_return']:.2%}
  Annualized Return:         {m['annualized_return']:.2%}
  Buy & Hold Return:         {m['buy_hold_return']:.2%}
  Excess Return:             {m['excess_return']:.2%}
  Final Portfolio Value:     ${m['final_value']:,.2f}

TRADE STATISTICS:
  Total Trades:              {m['total_trades']}
  Win Rate:                  {m['win_rate']:.1%}
  Profit Factor:             {m['profit_factor']:.2f}
  Average Profit/Trade:      ${m['avg_profit_per_trade']:.2f}
  Average Win:               ${m['avg_win']:.2f}
  Average Loss:              ${m['avg_loss']:.2f}
  Average Hold Time:         {m['avg_hold_time']:.1f} days

RISK METRICS:
  Maximum Drawdown:          {m['max_drawdown']:.2%}
  Average Drawdown:          {m['avg_drawdown']:.2%}
  Volatility (Annual):       {m['volatility']:.2%}
  Sharpe Ratio:              {m['sharpe_ratio']:.2f}
  Sortino Ratio:             {m['sortino_ratio']:.2f}
  Calmar Ratio:              {m['calmar_ratio']:.2f}

FEATURE ANALYSIS:
"""
        
        # Add feature importance analysis
        if self.feature_importance:
            if 'rsi' in self.feature_importance:
                rsi_data = self.feature_importance['rsi']
                report += f"  RSI Oversold Trades:       {rsi_data['oversold_count']} (Avg P&L: ${rsi_data['oversold_avg_pnl']:.2f})\n"
                report += f"  RSI Overbought Trades:     {rsi_data['overbought_count']} (Avg P&L: ${rsi_data['overbought_avg_pnl']:.2f})\n"
            
            if 'confidence' in self.feature_importance:
                conf_data = self.feature_importance['confidence']
                report += f"  High Confidence Trades:    {conf_data['high_confidence_count']} (Avg P&L: ${conf_data['high_confidence_avg_pnl']:.2f})\n"
                report += f"  Low Confidence Trades:     {conf_data['low_confidence_count']} (Avg P&L: ${conf_data['low_confidence_avg_pnl']:.2f})\n"
        
        # Performance insights
        report += f"\nPERFORMANCE INSIGHTS:\n"
        
        if m['excess_return'] > 0:
            report += f"  [+] Strategy outperformed buy-and-hold by {m['excess_return']:.2%}\n"
        else:
            report += f"  [-] Strategy underperformed buy-and-hold by {abs(m['excess_return']):.2%}\n"
        
        if m['sharpe_ratio'] > 1.0:
            report += f"  [+] Excellent risk-adjusted returns (Sharpe: {m['sharpe_ratio']:.2f})\n"
        elif m['sharpe_ratio'] > 0.5:
            report += f"  [+] Good risk-adjusted returns (Sharpe: {m['sharpe_ratio']:.2f})\n"
        else:
            report += f"  [!] Poor risk-adjusted returns (Sharpe: {m['sharpe_ratio']:.2f})\n"
        
        if m['win_rate'] > 0.6:
            report += f"  [+] High win rate: {m['win_rate']:.1%}\n"
        elif m['win_rate'] > 0.4:
            report += f"  [+] Reasonable win rate: {m['win_rate']:.1%}\n"
        else:
            report += f"  [!] Low win rate: {m['win_rate']:.1%}\n"
        
        if m['profit_factor'] > 1.5:
            report += f"  [+] Strong profit factor: {m['profit_factor']:.2f}\n"
        elif m['profit_factor'] > 1.0:
            report += f"  [+] Positive profit factor: {m['profit_factor']:.2f}\n"
        else:
            report += f"  [!] Poor profit factor: {m['profit_factor']:.2f}\n"
        
        if abs(m['max_drawdown']) < 0.1:
            report += f"  [+] Low maximum drawdown: {m['max_drawdown']:.2%}\n"
        elif abs(m['max_drawdown']) < 0.2:
            report += f"  [+] Moderate maximum drawdown: {m['max_drawdown']:.2%}\n"
        else:
            report += f"  [!] High maximum drawdown: {m['max_drawdown']:.2%}\n"
        
        # Recent trades
        report += f"\nRECENT TRADES (Last 10):\n"
        recent_trades = self.trades[-10:] if len(self.trades) > 10 else self.trades
        for trade in recent_trades:
            pnl_str = f"P&L: ${trade.get('pnl', 0):.2f}" if 'pnl' in trade else ""
            conf_str = f"Conf: {trade.get('confidence', 0):.1f}" if 'confidence' in trade else ""
            report += f"  {trade['date'].strftime('%Y-%m-%d')} | {trade['action']} | ${trade['price']:.2f} | {pnl_str} | {conf_str}\n"
        
        report += f"\n{'='*80}\n"
        return report

def main():
    """Main evaluation function"""
    config = {
        'strategy_name': 'advanced_multi_factor',
        'transaction_cost': 0.001,
        'initial_capital': 100000
    }
    
    evaluator = TradingAgentEvaluator(config)
    
    print("Generating comprehensive market data...")
    symbol = 'AAPL'
    data = evaluator.generate_realistic_data(symbol, days=500)
    
    print(f"Generated {len(data)} days of market data")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Split for training/testing
    split_point = int(len(data) * 0.7)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Run comprehensive backtest
    print("Running comprehensive evaluation...")
    performance = evaluator.backtest_with_analysis(test_data, symbol)
    
    # Generate detailed report
    print(evaluator.generate_comprehensive_report())
    
    # Export results
    results = {
        'performance_metrics': performance,
        'feature_importance': evaluator.feature_importance,
        'config': config,
        'data_summary': {
            'total_days': len(test_data),
            'price_range': [float(test_data['Close'].min()), float(test_data['Close'].max())],
            'avg_volume': float(test_data['Volume'].mean())
        }
    }
    
    with open('comprehensive_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Comprehensive evaluation completed!")
    print("Results exported to comprehensive_evaluation_results.json")

if __name__ == "__main__":
    main()