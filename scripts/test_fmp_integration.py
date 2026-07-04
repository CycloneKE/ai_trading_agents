#!/usr/bin/env python3
"""Test FMP integration with supervised learning strategy"""

import os
import sys
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from supervised_learning import SupervisedLearningStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fmp_integration():
    """Test FMP data integration in supervised learning"""
    
    # Test configuration
    config = {
        'lookback_period': 20,
        'threshold': 0.02,
        'model_type': 'random_forest',
        'features': ['close', 'volume', 'rsi', 'pe_ratio', 'debt_ratio'],
        'max_position_size': 0.1
    }
    
    # Create strategy
    strategy = SupervisedLearningStrategy("test_strategy", config)
    
    # Test data with basic market data
    test_data = {
        'symbol': 'AAPL',
        'close': 150.0,
        'open': 149.0,
        'high': 151.0,
        'low': 148.0,
        'volume': 1000000,
        'timestamp': datetime.now()
    }
    
    print("Testing FMP integration...")
    
    # Test feature extraction
    try:
        # Add some historical data first
        for i in range(25):  # Add enough data points
            data_point = test_data.copy()
            data_point['close'] = 150.0 + i * 0.5
            strategy._update_historical_data('AAPL', data_point)
        
        # Extract features
        features = strategy._extract_features('AAPL')
        
        if features:
            print(f"✓ Feature extraction successful: {len(features)} features")
            print(f"Features: {dict(zip(strategy.features, features))}")
            
            # Test signal generation
            signals = strategy.generate_signals(test_data)
            print(f"✓ Signal generation successful: {signals}")
            
        else:
            print("✗ Feature extraction failed")
            
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fmp_integration()