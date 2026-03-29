"""
Mocked Connector for safe integration-like tests.
Provides the same external interface used by DataManager but produces deterministic mock data
that is safe to run in CI without external credentials.
"""
from typing import Dict, List, Any
from datetime import datetime
import random
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MockedConnector:
    """Simple mocked connector that returns deterministic-ish market data for symbols."""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_connected = False
        self.cache = {}

    def connect(self) -> bool:
        self.is_connected = True
        logger.info("MockedConnector connected")
        return True

    def disconnect(self) -> None:
        self.is_connected = False
        logger.info("MockedConnector disconnected")

    def get_status(self) -> Dict[str, Any]:
        return {
            'status': 'mock',
            'connected': self.is_connected,
            'cache_size': len(self.cache)
        }

    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Return a mapping of symbol -> market data dict. Accepts a list of symbols.
        """
        data = {}
        for symbol in symbols:
            base = 100 + (abs(hash(symbol)) % 400)
            # deterministic random-ish values seeded by symbol hash
            rnd = (abs(hash(symbol)) % 1000) / 1000.0
            price = base + (rnd - 0.5) * 10
            data[symbol] = {
                'symbol': symbol,
                'price': float(round(price, 2)),
                'close': float(round(price, 2)),
                'open': float(round(price * (1 - 0.01 * rnd), 2)),
                'high': float(round(price * (1 + 0.01 * rnd), 2)),
                'low': float(round(price * (1 - 0.01 * rnd), 2)),
                'volume': int(1000000 * (0.5 + rnd)),
                'change_percent': float(round((rnd - 0.5) * 2, 4)),
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'mocked_connector'
            }
        return data

    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Return a pandas DataFrame with mock historical data."""
        dates = pd.date_range(end=datetime.utcnow(), periods=days, freq='D')
        base_price = 100 + (abs(hash(symbol)) % 400)
        prices = []
        price = base_price
        for _ in range(days):
            change = (abs(hash((symbol, _))) % 100) / 10000.0 - 0.005
            price = price * (1 + change)
            open_price = price * (1 - 0.001)
            high_price = price * (1 + 0.005)
            low_price = price * (1 - 0.005)
            volume = int(1000000 * (0.5 + (abs(hash((symbol, _))) % 100) / 200.0))
            prices.append({'open': open_price, 'high': high_price, 'low': low_price, 'close': price, 'volume': volume})
        df = pd.DataFrame(prices, index=dates)
        return df
