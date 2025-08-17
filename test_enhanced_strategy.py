#!/usr/bin/env python3
"""
Test Enhanced Trading Strategy with Stop-Loss and Trend Following
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from supervised_learning import SupervisedLearningStrategy

def generate_test_data(days=200):
    """Generate test data with trend periods"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Create trending market data
    initial_price = 150.0
    prices = [initial_price]
    
    for i in range(1, days):
        # Add trend component
        if i < days // 3:  # Uptrend
            drift = 0.002
            vol = 0.015
        elif i < 2 * days // 3:  # Downtrend  
            drift = -0.001
            vol = 0.020
        else:  # Sideways
            drift = 0.0005
            vol = 0.012
        
        return_val = np.random.normal(drift, vol)
        new_price = prices[-1] * (1 + return_val)
        prices.append(max(new_price, 10.0))
    
    # Create OHLC data
    data = pd.DataFrame({
        'Date': dates,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'volume': np.random.randint(1000000, 5000000, days)
    })
    
    # Add technical indicators
    data['rsi'] = calculate_rsi(pd.Series(prices))
    data['macd'] = calculate_macd(pd.Series(prices))
    data['ema_12'] = pd.Series(prices).ewm(span=12).mean()
    data['ema_26'] = pd.Series(prices).ewm(span=26).mean()
    data['adx'] = [25 + np.random.normal(0, 10) for _ in range(days)]  # Mock ADX
    data['atr'] = [p * 0.02 + np.random.normal(0, 0.005) for p in prices]  # Mock ATR
    
    data.set_index('Date', inplace=True)
    return data.ffill().dropna()

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD"""
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    return ema12 - ema26

def test_enhanced_strategy():
    """Test the enhanced strategy with new features"""
    
    # Enhanced configuration
    config = {
        'model_type': 'xgboost',
        'lookback_period': 20,
        'threshold': 0.01,  # Lowered threshold
        'features': ['close', 'volume', 'rsi', 'macd', 'ema_12', 'ema_26', 'adx', 'atr'],
        'stop_loss_atr_multiplier': 2.0,
        'trailing_stop_enabled': True,
        'trend_following_enabled': True,
        'ema_fast': 12,
        'ema_slow': 26,
        'adx_threshold': 25,
        'bias_threshold': 0.15,
        'max_position_size': 0.1
    }
    
    # Create strategy
    strategy = SupervisedLearningStrategy("enhanced_strategy", config)
    
    # Generate test data
    print("Generating test market data...")
    data = generate_test_data(200)
    print(f"Generated {len(data)} days of data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Simulate trading
    print("\nTesting enhanced strategy...")
    trades = []
    equity = [100000]  # Starting capital
    
    for i, (date, row) in enumerate(data.iterrows()):
        if i < 25:  # Need enough data for indicators
            continue
            
        # Prepare market data
        market_data = {
            'symbol': 'TEST',
            'close': row['close'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume'],
            'rsi': row['rsi'],
            'macd': row['macd'],
            'ema_12': row['ema_12'],
            'ema_26': row['ema_26'],
            'adx': row['adx'],
            'atr': row['atr'],
            'pe_ratio': 15.0,
            'debt_ratio': 0.3,
            'current_ratio': 1.5,
            'market_cap': 1000000000,
            'sector': 'technology'
        }
        
        # Get signal
        signal = strategy.generate_signals(market_data)
        
        # Track trades
        if signal['action'] != 'hold':
            trades.append({
                'date': date,
                'action': signal['action'],
                'price': row['close'],
                'confidence': signal['confidence'],
                'stop_price': signal.get('stop_price', 0),
                'stop_loss_triggered': signal.get('stop_loss_triggered', False)
            })
        
        # Simple equity tracking (mock)
        if signal['action'] == 'buy':
            equity.append(equity[-1] * 0.99)  # Small cost
        elif signal['action'] == 'sell':
            equity.append(equity[-1] * 1.01)  # Small gain
        else:
            equity.append(equity[-1])
    
    # Results
    print(f"\nENHANCED STRATEGY RESULTS:")
    print(f"Total Trades: {len(trades)}")
    print(f"Final Equity: ${equity[-1]:,.2f}")
    print(f"Total Return: {(equity[-1] / equity[0] - 1) * 100:.2f}%")
    
    # Show recent trades
    print(f"\nRecent Trades:")
    for trade in trades[-10:]:
        stop_str = f"Stop: ${trade['stop_price']:.2f}" if trade['stop_price'] > 0 else ""
        trigger_str = "[STOP-LOSS]" if trade['stop_loss_triggered'] else ""
        print(f"  {trade['date'].strftime('%Y-%m-%d')} | {trade['action'].upper()} | "
              f"${trade['price']:.2f} | Conf: {trade['confidence']:.2f} | {stop_str} {trigger_str}")
    
    # Strategy status
    status = strategy.get_status()
    print(f"\nSTRATEGY STATUS:")
    print(f"  Stop-Loss Enabled: {status.get('stop_loss_enabled', False)}")
    print(f"  Trailing Stop: {status.get('trailing_stop_enabled', False)}")
    print(f"  Trend Following: {status.get('trend_following_enabled', False)}")
    print(f"  Active Positions: {status.get('active_positions', 0)}")
    print(f"  Signal Threshold: {status.get('signal_threshold', 0)}")
    
    # Feature analysis
    stop_loss_trades = [t for t in trades if t['stop_loss_triggered']]
    print(f"\nRISK MANAGEMENT:")
    print(f"  Stop-Loss Triggered: {len(stop_loss_trades)} times")
    print(f"  Average Confidence: {np.mean([t['confidence'] for t in trades]):.2f}")
    
    return {
        'total_trades': len(trades),
        'stop_loss_triggers': len(stop_loss_trades),
        'final_return': (equity[-1] / equity[0] - 1) * 100,
        'trades': trades[-5:]  # Last 5 trades
    }

if __name__ == "__main__":
    try:
        results = test_enhanced_strategy()
        print(f"\n{'='*50}")
        print("ENHANCED STRATEGY TEST COMPLETED")
        print(f"{'='*50}")
        
        # Export results
        with open('enhanced_strategy_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Results exported to enhanced_strategy_test_results.json")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()