import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading

logger = logging.getLogger(__name__)

class RealPriceFeed:
    """
    Fetch real market prices using yfinance with caching.
    Supports US stocks, Kenyan NSE stocks, and Crypto.
    """
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 60  # seconds
        self.lock = threading.Lock()

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        with self.lock:
            cached = self.cache.get(symbol)
            # total_seconds(), not .seconds: .seconds drops the days component,
            # so an entry aged 1 day + 10s reads as 10s "fresh" — a real order
            # could then be sized off a days-old price on a long-running host.
            if cached and (datetime.now() - cached['timestamp']).total_seconds() < self.cache_ttl:
                return cached['price']

        try:
            # Map NSE symbols to yfinance format (e.g. SCOM -> SCOM.KE)
            yf_symbol = symbol.upper()
            
            # Known Kenyan NSE symbol mapping
            kenyan_symbols = {
                "SCOM", "EQTY", "KCB", "COOP", "SCBK", "SBIC", "ABSA",
                "BAT", "EABL", "KEGN", "KNRE", "BAMB", "TOTL", "CTUM",
                "NMG", "NCBA", "BRIT", "CIC"
            }
            if yf_symbol in kenyan_symbols:
                yf_symbol = f"{yf_symbol}.KE"

            ticker = yf.Ticker(yf_symbol)
            data = ticker.fast_info
            price = data.last_price
            
            if price:
                with self.lock:
                    self.cache[symbol] = {
                        'price': price,
                        'timestamp': datetime.now()
                    }
                return price
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
        
        return None

    def get_batch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch prices for multiple symbols at once."""
        results = {}
        # yfinance doesn't have a great batch 'last price' for FastInfo, 
        # so we'll do individual fetches but we could optimize later with download()
        for symbol in symbols:
            price = self.get_price(symbol)
            if price:
                results[symbol] = price
        return results

# Global instance
price_feed = RealPriceFeed()
