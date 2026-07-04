import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TechnicalStrategy(BaseStrategy):
    """
    Real technical strategy implementing Momentum, RSI, and Mean Reversion.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.lookback_period = config.get('lookback_period', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.historical_data = {}

    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get('symbol', 'UNKNOWN')
        price = data.get('price') or data.get('close')
        
        if not price:
            return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}

        if symbol not in self.historical_data:
            self.historical_data[symbol] = []
        
        self.historical_data[symbol].append(price)
        
        if len(self.historical_data[symbol]) < self.lookback_period:
            return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}

        # Keep history manageable
        self.historical_data[symbol] = self.historical_data[symbol][-self.lookback_period:]
        
        df = pd.DataFrame(self.historical_data[symbol], columns=['close'])
        
        # Calculate Indicators
        rsi = ta.rsi(df['close'], length=self.rsi_period)
        rsi_val = rsi.iloc[-1] if rsi is not None and not rsi.empty else 50
        
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
        
        momentum = (price - self.historical_data[symbol][-10]) / self.historical_data[symbol][-10]

        # Signal Logic
        buy_signals = 0
        sell_signals = 0
        
        # RSI Logic
        if rsi_val < self.rsi_oversold: buy_signals += 1
        if rsi_val > self.rsi_overbought: sell_signals += 1
        
        # Momentum Logic (Trend following)
        if price > sma_50 and momentum > 0: buy_signals += 1
        if price < sma_50 and momentum < 0: sell_signals += 1
        
        # Mean Reversion Logic
        if price < sma_20 * 0.98: buy_signals += 1  # 2% below mean
        if price > sma_20 * 1.02: sell_signals += 1  # 2% above mean

        action = 'hold'
        confidence = 0.0
        
        if buy_signals > sell_signals:
            action = 'buy'
            confidence = (buy_signals - sell_signals) / 3.0
        elif sell_signals > buy_signals:
            action = 'sell'
            confidence = (sell_signals - buy_signals) / 3.0
            
        return {
            'action': action,
            'confidence': float(confidence),
            'position_size': float(confidence * self.config.get('max_position_size', 0.05)),
            'indicators': {
                'rsi': float(rsi_val),
                'momentum': float(momentum),
                'sma_50': float(sma_50)
            }
        }

    def seed_history(self, symbol: str, closes: List[float]) -> int:
        """Warm-start the price buffer from historical bars so the strategy
        can signal immediately instead of being blind for lookback_period
        live cycles after every restart."""
        cleaned = [float(c) for c in closes if c and c > 0]
        if not cleaned:
            return 0
        self.historical_data[symbol] = cleaned[-self.lookback_period:]
        return len(self.historical_data[symbol])

    def update_model(self, data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        pass # Technical strategies are rule-based, but we could tune thresholds here
