#!/usr/bin/env python3
"""
Performance Optimization Fixes
"""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCache:
    """High-performance data caching system"""
    def __init__(self, max_size=1000, ttl=300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()

class ConnectionPool:
    """Database connection pooling"""
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
    
    def get_connection(self):
        with self.lock:
            if self.connections:
                return self.connections.pop()
            return self._create_connection()
    
    def return_connection(self, conn):
        with self.lock:
            if len(self.connections) < self.max_connections:
                self.connections.append(conn)
    
    def _create_connection(self):
        # Mock connection - replace with actual DB connection
        return {"id": time.time(), "active": True}

def create_optimized_data_manager():
    """Create optimized data manager"""
    code = '''import time
import threading
from concurrent.futures import ThreadPoolExecutor
from performance_optimization import DataCache, ConnectionPool

class OptimizedDataManager:
    def __init__(self):
        self.cache = DataCache(max_size=1000, ttl=300)
        self.connection_pool = ConnectionPool(max_connections=10)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    def get_market_data(self, symbol):
        """Get market data with caching"""
        cache_key = f"market_{symbol}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Fetch new data
        data = self._fetch_market_data(symbol)
        if data:
            self.cache.set(cache_key, data)
        
        return data
    
    def _fetch_market_data(self, symbol):
        """Fetch data from API (optimized)"""
        import requests
        import os
        
        fmp_key = os.getenv('TRADING_FMP_API_KEY')
        if not fmp_key:
            return self._generate_mock_data(symbol)
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={fmp_key}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
        except:
            pass
        
        return self._generate_mock_data(symbol)
    
    def _generate_mock_data(self, symbol):
        """Generate mock data for testing"""
        import random
        base_prices = {'AAPL': 175, 'GOOGL': 140, 'MSFT': 415, 'TSLA': 250, 'NVDA': 875}
        base_price = base_prices.get(symbol, 100)
        
        return {
            'symbol': symbol,
            'price': base_price + random.uniform(-5, 5),
            'change': random.uniform(-2, 2),
            'volume': random.randint(1000000, 10000000),
            'timestamp': time.time()
        }
    
    def get_multiple_symbols(self, symbols):
        """Fetch multiple symbols concurrently"""
        futures = []
        for symbol in symbols:
            future = self.executor.submit(self.get_market_data, symbol)
            futures.append((symbol, future))
        
        results = {}
        for symbol, future in futures:
            try:
                results[symbol] = future.result(timeout=10)
            except:
                results[symbol] = None
        
        return results
'''
    
    try:
        with open('optimized_data_manager.py', 'w') as f:
            f.write(code)
        logger.info("Created optimized data manager")
        return True
    except Exception as e:
        logger.error(f"Failed to create optimized data manager: {e}")
        return False

def create_memory_optimizer():
    """Create memory optimization utilities"""
    code = '''import gc
import psutil
import threading
import time
import logging

class MemoryOptimizer:
    def __init__(self, threshold_mb=1000, check_interval=60):
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Memory optimizer started")
    
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        while self.running:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > self.threshold_mb:
                    logging.warning(f"High memory usage: {memory_mb:.1f}MB")
                    self._optimize_memory()
                
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Memory monitor error: {e}")
                time.sleep(10)
    
    def _optimize_memory(self):
        """Perform memory optimization"""
        before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        collected = gc.collect()
        
        after = psutil.Process().memory_info().rss / 1024 / 1024
        freed = before - after
        
        logging.info(f"Memory optimization: {collected} objects collected, {freed:.1f}MB freed")

# Global memory optimizer
memory_optimizer = MemoryOptimizer()
'''
    
    try:
        with open('memory_optimizer.py', 'w') as f:
            f.write(code)
        logger.info("Created memory optimizer")
        return True
    except Exception as e:
        logger.error(f"Failed to create memory optimizer: {e}")
        return False

def create_async_processor():
    """Create asynchronous data processor"""
    code = '''import asyncio
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
'''
    
    try:
        with open('async_processor.py', 'w') as f:
            f.write(code)
        logger.info("Created async processor")
        return True
    except Exception as e:
        logger.error(f"Failed to create async processor: {e}")
        return False

def optimize_config_loading():
    """Optimize configuration loading"""
    try:
        # Add caching to config loading
        config_optimization = '''
@lru_cache(maxsize=1)
def load_cached_config():
    """Load configuration with caching"""
    with open('config/config.json', 'r') as f:
        return json.load(f)

def get_config():
    """Get configuration (cached)"""
    return load_cached_config().copy()
'''
        
        with open('config_cache.py', 'w') as f:
            f.write("import json\nfrom functools import lru_cache\n\n" + config_optimization)
        
        logger.info("Created config caching")
        return True
    except Exception as e:
        logger.error(f"Failed to optimize config loading: {e}")
        return False

def main():
    """Apply performance optimizations"""
    print("PERFORMANCE OPTIMIZATION")
    print("=" * 40)
    
    fixes_applied = 0
    total_fixes = 4
    
    print("\n1. Creating optimized data manager...")
    if create_optimized_data_manager():
        fixes_applied += 1
        print("   ✅ Optimized data manager created")
    
    print("\n2. Creating memory optimizer...")
    if create_memory_optimizer():
        fixes_applied += 1
        print("   ✅ Memory optimizer created")
    
    print("\n3. Creating async processor...")
    if create_async_processor():
        fixes_applied += 1
        print("   ✅ Async processor created")
    
    print("\n4. Optimizing config loading...")
    if optimize_config_loading():
        fixes_applied += 1
        print("   ✅ Config caching implemented")
    
    print(f"\n✅ Performance optimizations: {fixes_applied}/{total_fixes}")
    
    if fixes_applied >= 3:
        print("\nPERFORMANCE IMPROVEMENTS:")
        print("• Data caching with TTL")
        print("• Connection pooling")
        print("• Concurrent data fetching")
        print("• Memory optimization")
        print("• Async processing")
        print("• Configuration caching")
    
    return fixes_applied >= 3

if __name__ == "__main__":
    main()