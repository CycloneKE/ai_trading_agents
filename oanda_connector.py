"""
OANDA Connector for Forex Market Data
"""
import requests
import time
from typing import Dict, Any, List
from datetime import datetime
from .base_connector import BaseConnector

class OandaConnector(BaseConnector):
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
                print("Circuit breaker reset for OandaConnector.")
            else:
                print("Circuit breaker is open. Using fallback data.")
                return True
        return False

    def _record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._circuit_open = True
            print("Circuit breaker triggered for OandaConnector. Too many failures.")

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
        # OANDA API is HTTP-based, so connection is stateless
        self.is_connected = True
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        return True
    """Connector for OANDA forex data."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("oanda", config)
        self.api_key = config.get('api_key', '')
        self.account_id = config.get('account_id', '')
        self.base_url = "https://api-fxpractice.oanda.com/v3/"

    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        if self._check_circuit_breaker():
            # Fallback: return mock data
            return {symbol: {'bid': 0, 'ask': 0, 'timestamp': datetime.utcnow().isoformat(), 'source': 'mock'} for symbol in symbols}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {}
        for symbol in symbols:
            def api_call():
                url = f"{self.base_url}accounts/{self.account_id}/pricing?instruments={symbol}"
                resp = requests.get(url, headers=headers)
                if resp.status_code == 200:
                    prices = resp.json()['prices'][0]
                    return {
                        'bid': float(prices['bids'][0]['price']),
                        'ask': float(prices['asks'][0]['price']),
                        'timestamp': prices['time'],
                        'source': 'oanda'
                    }
                else:
                    raise Exception(f"OANDA API error: {resp.status_code}")
            result = self._retry_api_call(api_call)
            if result:
                self._failure_count = 0
                data[symbol] = result
            else:
                data[symbol] = {'bid': 0, 'ask': 0, 'timestamp': datetime.utcnow().isoformat(), 'source': 'mock'}
        return data

    def get_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Dict[str, Any]:
        if self._check_circuit_breaker():
            return {'error': 'Circuit breaker open. No historical data.'}
        def api_call():
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}instruments/{symbol}/candles?from={start_date}T00:00:00Z&to={end_date}T23:59:59Z&granularity={interval.upper()}"
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            else:
                raise Exception(f"OANDA API error: {resp.status_code}")
        result = self._retry_api_call(api_call)
        if result:
            self._failure_count = 0
            return result
        return {'error': 'Failed to fetch historical data'}
