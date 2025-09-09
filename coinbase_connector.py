"""
Coinbase Connector for Crypto Market Data
"""
import requests
import time
from typing import Dict, Any, List
from datetime import datetime
from .base_connector import BaseConnector

class CoinbaseConnector(BaseConnector):
    # Circuit breaker state
    _failure_count = 0
    _failure_threshold = 5
    _circuit_open = False
    _circuit_reset_time = 300  # seconds
    _last_failure_time = None

    def _check_circuit_breaker(self):
        if self._circuit_open:
            if self._last_failure_time and (time.time() - self._last_failure_time > self._circuit_reset_time):
                self._failure_count = 0
                self._circuit_open = False
                print("Circuit breaker reset for CoinbaseConnector.")
            else:
                print("Circuit breaker is open. Using fallback data.")
                return True
        return False

    def _record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._circuit_open = True
            print("Circuit breaker triggered for CoinbaseConnector. Too many failures.")

    def _retry_api_call(self, func, *args, **kwargs):
        max_retries = self.config.get('max_retries', 3)
        delay = 2
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(delay)
        self._record_failure()
        return None
    def connect(self) -> bool:
        # Coinbase API is HTTP-based, so connection is stateless
        self.is_connected = True
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        return True
    """Connector for Coinbase crypto data."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("coinbase", config)
        self.base_url = "https://api.coinbase.com/v2/"

    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        if self._check_circuit_breaker():
            # Fallback: return mock data
            return {symbol: {'price': 0, 'timestamp': datetime.utcnow().isoformat(), 'source': 'mock'} for symbol in symbols}
        data = {}
        for symbol in symbols:
            def api_call():
                pair = symbol.replace("/", "-")
                url = f"{self.base_url}prices/{pair}/spot"
                resp = requests.get(url)
                if resp.status_code == 200:
                    price = float(resp.json()['data']['amount'])
                    return {'price': price, 'timestamp': datetime.utcnow().isoformat(), 'source': 'coinbase'}
                else:
                    raise Exception(f"Coinbase API error: {resp.status_code}")
            result = self._retry_api_call(api_call)
            if result:
                self._failure_count = 0
                data[symbol] = result
            else:
                data[symbol] = {'price': 0, 'timestamp': datetime.utcnow().isoformat(), 'source': 'mock'}
        return data

    def get_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Dict[str, Any]:
        # Coinbase API does not provide free historical candles; placeholder for paid/other API
        if self._check_circuit_breaker():
            return {'error': 'Circuit breaker open. No historical data.'}
        return {'error': 'Historical data not supported in free Coinbase API.'}
