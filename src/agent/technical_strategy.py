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
        self.momentum_threshold = config.get('threshold', 0.02)
        self.mean_reversion_band = config.get('band', 0.02)
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

        # Each configured strategy name evaluates ONLY its own signal — these
        # used to all run the same combined RSI+momentum+mean-reversion logic
        # regardless of name, making "momentum"/"mean_reversion"/"rsi_strategy"
        # near-identical clones that always agreed with each other. Real
        # diversity requires them to actually measure different things.
        if 'momentum' in self.name:
            action, confidence = self._momentum_signal(price, sma_50, momentum)
        elif 'reversion' in self.name:
            action, confidence = self._mean_reversion_signal(price, sma_20)
        elif 'rsi' in self.name:
            action, confidence = self._rsi_signal(rsi_val)
        else:
            action, confidence = self._combined_signal(price, sma_20, sma_50, momentum, rsi_val)

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

    @staticmethod
    def _scale(distance_beyond: float, span: float) -> float:
        """Smooth 0.5-1.0 confidence ramp once a threshold is crossed, instead
        of a hard binary jump. distance_beyond/span == 0 at the threshold,
        1.0 at double the threshold's distance, capped at 1.0."""
        return min(0.5 + 0.5 * max(distance_beyond, 0.0) / span, 1.0) if span else 1.0

    def _momentum_signal(self, price, sma_50, momentum):
        t = self.momentum_threshold
        if price > sma_50 and momentum > t:
            return 'buy', self._scale(momentum - t, t)
        if price < sma_50 and momentum < -t:
            return 'sell', self._scale(-momentum - t, t)
        return 'hold', 0.0

    def _mean_reversion_signal(self, price, sma_20):
        band = self.mean_reversion_band
        lower, upper = sma_20 * (1 - band), sma_20 * (1 + band)
        if price < lower:
            return 'buy', self._scale(lower - price, sma_20 * band)
        if price > upper:
            return 'sell', self._scale(price - upper, sma_20 * band)
        return 'hold', 0.0

    def _rsi_signal(self, rsi_val):
        if rsi_val < self.rsi_oversold:
            return 'buy', self._scale(self.rsi_oversold - rsi_val, self.rsi_oversold)
        if rsi_val > self.rsi_overbought:
            return 'sell', self._scale(rsi_val - self.rsi_overbought, 100 - self.rsi_overbought)
        return 'hold', 0.0

    def _combined_signal(self, price, sma_20, sma_50, momentum, rsi_val):
        """Fallback for a strategy name that isn't one of the three above —
        the original all-in-one vote, kept so a custom/unrecognized strategy
        name still gets a reasonable signal instead of always holding."""
        buy_signals = 0
        sell_signals = 0
        if rsi_val < self.rsi_oversold: buy_signals += 1
        if rsi_val > self.rsi_overbought: sell_signals += 1
        if price > sma_50 and momentum > 0: buy_signals += 1
        if price < sma_50 and momentum < 0: sell_signals += 1
        if price < sma_20 * 0.98: buy_signals += 1
        if price > sma_20 * 1.02: sell_signals += 1

        if buy_signals > sell_signals:
            return 'buy', (buy_signals - sell_signals) / 3.0
        if sell_signals > buy_signals:
            return 'sell', (sell_signals - buy_signals) / 3.0
        return 'hold', 0.0

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
