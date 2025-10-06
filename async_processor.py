import asyncio
import aiohttp
import time
from typing import List, Dict

class AsyncDataProcessor:
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=self.max_concurrent)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_symbol_data(self, symbol: str, api_key: str) -> Dict:
        """Fetch data for single symbol"""
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return {symbol: data[0]}
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        
        return {symbol: None}
    
    async def fetch_multiple_symbols(self, symbols: List[str], api_key: str) -> Dict:
        """Fetch data for multiple symbols concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_with_semaphore(symbol):
            async with semaphore:
                return await self.fetch_symbol_data(symbol, api_key)
        
        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined_results = {}
        for result in results:
            if isinstance(result, dict):
                combined_results.update(result)
        
        return combined_results

async def get_market_data_async(symbols: List[str], api_key: str) -> Dict:
    """Get market data asynchronously"""
    async with AsyncDataProcessor() as processor:
        return await processor.fetch_multiple_symbols(symbols, api_key)

def run_async_fetch(symbols: List[str], api_key: str) -> Dict:
    """Synchronous wrapper for async fetch"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_market_data_async(symbols, api_key))
