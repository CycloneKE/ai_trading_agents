"""
Fallback data generator for when API data sources fail.
Generates realistic market data for testing and development.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FallbackDataGenerator:
    """Generate realistic market data when APIs fail."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD', 'EUR_USD', 'USD_JPY']
        self.base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 330.0, 'TSLA': 250.0, 'NVDA': 450.0,
            'BTC-USD': 43000.0, 'ETH-USD': 2600.0, 'EUR_USD': 1.08, 'USD_JPY': 149.0
        }
        
    def generate_ohlcv_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate OHLCV data for a symbol."""
        try:
            base_price = self.base_prices.get(symbol, 100.0)
            dates = pd.date_range(end=datetime.now(), periods=days * 24, freq='H')
            
            # Generate price series with realistic volatility
            returns = np.random.normal(0, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Prevent negative prices
            
            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                volatility = abs(np.random.normal(0, 0.01))
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = max(int(np.random.normal(1000000, 500000)), 1000)
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Generated {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def generate_all_symbols_data(self, days: int = 30) -> dict:
        """Generate data for all symbols."""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.generate_ohlcv_data(symbol, days)
        return data
    
    def get_latest_prices(self) -> dict:
        """Get latest prices for all symbols."""
        prices = {}
        for symbol in self.symbols:
            base = self.base_prices[symbol]
            change = np.random.normal(0, 0.02)
            prices[symbol] = {
                'price': base * (1 + change),
                'change': change,
                'volume': max(int(np.random.normal(1000000, 500000)), 1000)
            }
        return prices