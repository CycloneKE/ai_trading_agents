"""
News data connectors for fetching news and sentiment data.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class BaseNewsConnector:
    """Base class for all news connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
        self.name = "base_news"
    
    def connect(self) -> bool:
        """Connect to the news source."""
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from the news source."""
        self.is_connected = False
        return True
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time news data for symbols."""
        raise NotImplementedError
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical news data for a symbol."""
        raise NotImplementedError
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status."""
        return {
            'name': self.name,
            'is_connected': self.is_connected,
            'config': {k: v for k, v in self.config.items() if k != 'api_key'}
        }

class NewsAPIConnector(BaseNewsConnector):
    # Circuit breaker state
    _failure_count = 0
    _failure_threshold = 5
    _circuit_open = False
    _circuit_reset_time = 300  # seconds
    _last_failure_time = None

    def _check_circuit_breaker(self):
        import time
        if self._circuit_open:
            if self._last_failure_time and (time.time() - self._last_failure_time > self._circuit_reset_time):
                self._failure_count = 0
                self._circuit_open = False
                logger.info("Circuit breaker reset for NewsAPIConnector.")
            else:
                logger.warning("Circuit breaker is open. Using fallback data.")
                return True
        return False

    def _record_failure(self):
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._circuit_open = True
            logger.error("Circuit breaker triggered for NewsAPIConnector. Too many failures.")

    def _retry_api_call(self, func, *args, **kwargs):
        import time
        max_retries = self.config.get('max_retries', 3)
        delay = 2
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(delay)
        self._record_failure()
        return None
    """Connector for NewsAPI data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "news_api"
        import os
        self.api_key = os.getenv('TRADING_NEWS_API_KEY', config.get('api_key', ''))
        self.base_url = "https://newsapi.org/v2"
        self.rate_limit = config.get('rate_limit', 100)  # Requests per day
        self.last_request_time = 0
    
    def connect(self) -> bool:
        """Connect to NewsAPI."""
        if not self.api_key:
            logger.error("NewsAPI API key not provided")
            return False
        
        self.is_connected = True
        return True
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time news data from NewsAPI."""
        try:
            # Implement rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < 86400 / self.rate_limit:
                time.sleep(86400 / self.rate_limit - (current_time - self.last_request_time))
            
            self.last_request_time = time.time()
            
            import requests
            data = {}
            for symbol in symbols:
                try:
                    url = f"{self.base_url}/everything?q={symbol}&apiKey={self.api_key}&sortBy=publishedAt&pageSize=5"
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    res_data = resp.json()
                    
                    if res_data.get('status') == 'ok':
                        data[symbol] = {
                            'articles': res_data.get('articles', []),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    else:
                        logger.warning(f"NewsAPI returned error for {symbol}: {res_data.get('message')}")
                except Exception as e:
                    logger.warning(f"Failed to fetch NewsAPI data for {symbol}: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting real-time data from NewsAPI: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical news data from NewsAPI."""
        try:
            import requests
            start_iso = datetime.strptime(start_date, "%Y-%m-%d").isoformat()
            end_iso = datetime.strptime(end_date, "%Y-%m-%d").isoformat()
            url = f"{self.base_url}/everything?q={symbol}&from={start_iso}&to={end_iso}&apiKey={self.api_key}&sortBy=relevancy&pageSize=10"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            res_data = resp.json()
            
            if res_data.get('status') == 'ok':
                return {
                    'articles': res_data.get('articles', []),
                    'count': res_data.get('totalResults', 0)
                }
            logger.warning(f"NewsAPI historical fetch failed for {symbol}: {res_data.get('message')}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting historical data from NewsAPI: {str(e)}")
            return {}

class RedditConnector(BaseNewsConnector):
    """Connector for Reddit data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "reddit"
        self.client_id = config.get('client_id', '')
        self.client_secret = config.get('client_secret', '')
        self.user_agent = config.get('user_agent', 'AITradingAgent/1.0')
        self.rate_limit = config.get('rate_limit', 60)  # Requests per minute
        self.last_request_time = 0
    
    def connect(self) -> bool:
        """Connect to Reddit API."""
        if not self.client_id or not self.client_secret:
            logger.error("Reddit API credentials not provided")
            return False
        
        self.is_connected = True
        return True
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from Reddit."""
        try:
            # Implement rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < 60 / self.rate_limit:
                time.sleep(60 / self.rate_limit - (current_time - self.last_request_time))
            
            self.last_request_time = time.time()
            
            import requests
            data = {}
            headers = {'User-Agent': self.user_agent}
            for symbol in symbols:
                try:
                    url = f"https://www.reddit.com/r/stocks/search.json?q={symbol}&restrict_sr=1&sort=new&limit=5"
                    resp = requests.get(url, headers=headers, timeout=10)
                    resp.raise_for_status()
                    res_data = resp.json()
                    
                    posts = []
                    if 'data' in res_data and 'children' in res_data['data']:
                        for child in res_data['data']['children']:
                            post_data = child.get('data', {})
                            posts.append({
                                'title': post_data.get('title', ''),
                                'body': post_data.get('selftext', ''),
                                'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                'created_utc': post_data.get('created_utc', 0),
                                'score': post_data.get('score', 0),
                                'num_comments': post_data.get('num_comments', 0),
                                'sentiment': 0.5  # placeholder until NLP applies score
                            })
                    data[symbol] = {
                        'posts': posts,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Failed to fetch Reddit data for {symbol}: {str(e)}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting real-time data from Reddit: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical data from Reddit."""
        try:
            logger.warning("Historical data parsing for public Reddit endpoint not fully supported. Returning empty.")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting historical data from Reddit: {str(e)}")
            return {}

class FinnhubNewsConnector(BaseNewsConnector):
    """Connector for Finnhub news data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "finnhub_news"
        self.api_key = config.get('api_key', '')
        self.base_url = 'https://finnhub.io/api/v1/'
        self.rate_limit = config.get('rate_limit', 100)  # Requests per day
        self.last_request_time = 0
    
    def connect(self) -> bool:
        """Connect to Finnhub."""
        if not self.api_key:
            logger.error("Finnhub API key not provided")
            return False
        
        self.is_connected = True
        return True
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time news data from Finnhub."""
        try:
            # Implement rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < 86400 / self.rate_limit:
                time.sleep(86400 / self.rate_limit - (current_time - self.last_request_time))
            
            self.last_request_time = time.time()
            
            results = {}
            today = datetime.utcnow().date()
            week_ago = today - timedelta(days=7)
            for symbol in symbols:
                url = f"{self.base_url}company-news?symbol={symbol}&from={week_ago}&to={today}&token={self.api_key}"
                try:
                    resp = requests.get(url)
                    if resp.status_code == 200:
                        results[symbol] = resp.json()
                    else:
                        results[symbol] = None
                except Exception:
                    results[symbol] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting real-time data from Finnhub: {str(e)}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status."""
        return {
            'name': self.name,
            'is_connected': self.is_connected,
            'config': {k: v for k, v in self.config.items() if k != 'api_key'}
        }