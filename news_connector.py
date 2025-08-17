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
            
            # In a real implementation, use NewsAPI
            # For now, return dummy data
            data = {}
            for symbol in symbols:
                data[symbol] = {
                    'articles': [
                        {
                            'title': f"News about {symbol}",
                            'description': f"This is a sample news article about {symbol}",
                            'url': f"https://example.com/news/{symbol}",
                            'publishedAt': datetime.utcnow().isoformat(),
                            'sentiment': 0.5 + (hash(symbol) % 100) / 200  # Random sentiment between 0 and 1
                        }
                    ],
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting real-time data from NewsAPI: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical news data from NewsAPI."""
        try:
            # In a real implementation, use NewsAPI
            # For now, return dummy data
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days
            
            articles = []
            for i in range(min(days, 10)):  # Limit to 10 articles
                date = start + timedelta(days=i)
                articles.append({
                    'title': f"Historical news about {symbol} on {date.strftime('%Y-%m-%d')}",
                    'description': f"This is a sample historical news article about {symbol}",
                    'url': f"https://example.com/news/{symbol}/{date.strftime('%Y-%m-%d')}",
                    'publishedAt': date.isoformat(),
                    'sentiment': 0.5 + (hash(symbol + date.isoformat()) % 100) / 200
                })
            
            data = {
                'articles': articles,
                'count': len(articles)
            }
            
            return data
            
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
            
            # In a real implementation, use Reddit API
            # For now, return dummy data
            data = {}
            for symbol in symbols:
                data[symbol] = {
                    'posts': [
                        {
                            'title': f"Discussion about {symbol}",
                            'body': f"This is a sample Reddit post about {symbol}",
                            'url': f"https://reddit.com/r/stocks/comments/{symbol}",
                            'created_utc': datetime.utcnow().timestamp(),
                            'score': hash(symbol) % 1000,
                            'num_comments': hash(symbol) % 100,
                            'sentiment': 0.5 + (hash(symbol) % 100) / 200
                        }
                    ],
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting real-time data from Reddit: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical data from Reddit."""
        try:
            # In a real implementation, use Reddit API
            # For now, return dummy data
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days
            
            posts = []
            for i in range(min(days, 10)):  # Limit to 10 posts
                date = start + timedelta(days=i)
                posts.append({
                    'title': f"Historical discussion about {symbol} on {date.strftime('%Y-%m-%d')}",
                    'body': f"This is a sample historical Reddit post about {symbol}",
                    'url': f"https://reddit.com/r/stocks/comments/{symbol}/{date.strftime('%Y%m%d')}",
                    'created_utc': date.timestamp(),
                    'score': hash(symbol + date.isoformat()) % 1000,
                    'num_comments': hash(symbol + date.isoformat()) % 100,
                    'sentiment': 0.5 + (hash(symbol + date.isoformat()) % 100) / 200
                })
            
            data = {
                'posts': posts,
                'count': len(posts)
            }
            
            return data
            
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