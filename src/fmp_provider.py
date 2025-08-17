import os
import requests
from typing import Dict, List, Optional

class FMPProvider:
    def __init__(self):
        self.api_key = os.getenv('TRADING_FMP_API_KEY')
        self.base_url = 'https://financialmodelingprep.com/api/v3'
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        url = f"{self.base_url}/quote/{symbol}?apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if isinstance(data, dict) and 'Error Message' in data:
            return {}
        return data[0] if isinstance(data, list) and len(data) > 0 else {}
    
    def get_financial_ratios(self, symbol: str) -> Dict:
        """Get financial ratios for fundamental analysis"""
        url = f"{self.base_url}/ratios/{symbol}?apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if isinstance(data, dict) and 'Error Message' in data:
            return {}
        return data[0] if isinstance(data, list) and len(data) > 0 else {}
    
    def get_income_statement(self, symbol: str, period: str = 'annual') -> List[Dict]:
        """Get income statement data"""
        url = f"{self.base_url}/income-statement/{symbol}?period={period}&apikey={self.api_key}"
        response = requests.get(url)
        return response.json()
    
    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market capitalization"""
        quote = self.get_quote(symbol)
        return quote.get('marketCap')