#!/usr/bin/env python3
"""Test XGBoost integration"""

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    print("[SUCCESS] XGBoost available")
    
    # Create sample trading data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: price, volume, rsi, pe_ratio, debt_ratio
    X = np.random.rand(n_samples, 5)
    X[:, 0] = X[:, 0] * 100 + 50  # price 50-150
    X[:, 1] = X[:, 1] * 1000000   # volume
    X[:, 2] = X[:, 2] * 100       # rsi 0-100
    X[:, 3] = X[:, 3] * 30 + 5    # pe_ratio 5-35
    X[:, 4] = X[:, 4] * 0.8       # debt_ratio 0-0.8
    
    # Target: future returns
    y = np.random.randn(n_samples) * 0.02  # 2% volatility
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='rmse'
    )
    
    print("Training XGBoost model...")
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X[:10])
    print(f"Sample predictions: {predictions[:5]}")
    
    # Feature importance
    importance = model.feature_importances_
    features = ['price', 'volume', 'rsi', 'pe_ratio', 'debt_ratio']
    
    print("\nFeature Importance:")
    for feature, imp in zip(features, importance):
        print(f"  {feature}: {imp:.3f}")
    
    print("\n[SUCCESS] XGBoost integration working!")
    print("You can now use 'xgboost' as model_type in your config")
    
except ImportError:
    print("[ERROR] XGBoost not available - install with: pip install xgboost")
except Exception as e:
    print(f"[ERROR] XGBoost test failed: {e}")

if __name__ == "__main__":
    pass