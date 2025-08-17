"""
Aletheia API connector for alternative data and sentiment analysis.
"""

import os
import logging
import requests
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AletheiaConnector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv('TRADING_ALETHEIA_API_KEY', config.get('api_key', ''))
        self.base_url = config.get('base_url', 'https://api.aletheia.com')
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to Aletheia API."""
        if not self.api_key:
            logger.error("Aletheia API key not provided")
            return False
        
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(f"{self.base_url}/v1/status", headers=headers)
            
            if response.status_code == 200:
                self.is_connected = True
                logger.info("Connected to Aletheia API")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Aletheia: {str(e)}")
        
        return False
    
    def get_sentiment_analysis(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get sentiment analysis for a symbol."""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            params = {
                'symbol': symbol,
                'days': days,
                'sources': 'news,social,earnings'
            }
            
            response = requests.get(
                f"{self.base_url}/v1/sentiment",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'overall_sentiment': data.get('overall_sentiment', 0),
                    'news_sentiment': data.get('news_sentiment', 0),
                    'social_sentiment': data.get('social_sentiment', 0),
                    'earnings_sentiment': data.get('earnings_sentiment', 0),
                    'sentiment_trend': data.get('sentiment_trend', 'neutral'),
                    'confidence': data.get('confidence', 0),
                    'volume_mentions': data.get('volume_mentions', 0)
                }
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
        
        return self._mock_sentiment(symbol)  # Fallback to mock data
    
    def get_alternative_data(self, symbol: str, data_type: str = 'all') -> Dict[str, Any]:
        """Get alternative data signals."""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            params = {
                'symbol': symbol,
                'data_type': data_type
            }
            
            response = requests.get(
                f"{self.base_url}/v1/alternative-data",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'satellite_data': data.get('satellite_data', {}),
                    'web_traffic': data.get('web_traffic', {}),
                    'app_usage': data.get('app_usage', {}),
                    'supply_chain': data.get('supply_chain', {}),
                    'esg_scores': data.get('esg_scores', {}),
                    'insider_trading': data.get('insider_trading', {}),
                    'patent_filings': data.get('patent_filings', {})
                }
        except Exception as e:
            logger.error(f"Error getting alternative data for {symbol}: {str(e)}")
        
        return self._mock_alternative_data(symbol)
    
    def get_earnings_predictions(self, symbol: str) -> Dict[str, Any]:
        """Get AI-powered earnings predictions."""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            params = {'symbol': symbol}
            
            response = requests.get(
                f"{self.base_url}/v1/earnings/predictions",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'predicted_eps': data.get('predicted_eps', 0),
                    'predicted_revenue': data.get('predicted_revenue', 0),
                    'surprise_probability': data.get('surprise_probability', 0),
                    'direction_confidence': data.get('direction_confidence', 0),
                    'next_earnings_date': data.get('next_earnings_date', ''),
                    'analyst_consensus': data.get('analyst_consensus', {}),
                    'ai_vs_consensus': data.get('ai_vs_consensus', 0)
                }
        except Exception as e:
            logger.error(f"Error getting earnings predictions for {symbol}: {str(e)}")
        
        return self._mock_earnings_prediction(symbol)
    
    def get_market_regime(self) -> Dict[str, Any]:
        """Get current market regime analysis."""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            response = requests.get(
                f"{self.base_url}/v1/market/regime",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'current_regime': data.get('current_regime', 'normal'),
                    'regime_probability': data.get('regime_probability', 0),
                    'volatility_regime': data.get('volatility_regime', 'normal'),
                    'trend_regime': data.get('trend_regime', 'sideways'),
                    'risk_on_off': data.get('risk_on_off', 'neutral'),
                    'regime_change_probability': data.get('regime_change_probability', 0)
                }
        except Exception as e:
            logger.error(f"Error getting market regime: {str(e)}")
        
        return self._mock_market_regime()
    
    def _mock_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Mock sentiment data for testing."""
        import random
        return {
            'symbol': symbol,
            'overall_sentiment': random.uniform(-1, 1),
            'news_sentiment': random.uniform(-1, 1),
            'social_sentiment': random.uniform(-1, 1),
            'earnings_sentiment': random.uniform(-1, 1),
            'sentiment_trend': random.choice(['bullish', 'bearish', 'neutral']),
            'confidence': random.uniform(0.5, 0.9),
            'volume_mentions': random.randint(100, 1000)
        }
    
    def _mock_alternative_data(self, symbol: str) -> Dict[str, Any]:
        """Mock alternative data for testing."""
        import random
        return {
            'symbol': symbol,
            'satellite_data': {'activity_score': random.uniform(0, 1)},
            'web_traffic': {'traffic_growth': random.uniform(-0.2, 0.3)},
            'app_usage': {'usage_trend': random.uniform(-0.1, 0.2)},
            'supply_chain': {'disruption_risk': random.uniform(0, 0.5)},
            'esg_scores': {'overall_esg': random.uniform(0.3, 0.9)},
            'insider_trading': {'net_buying': random.uniform(-1, 1)},
            'patent_filings': {'innovation_score': random.uniform(0, 1)}
        }
    
    def _mock_earnings_prediction(self, symbol: str) -> Dict[str, Any]:
        """Mock earnings prediction for testing."""
        import random
        return {
            'symbol': symbol,
            'predicted_eps': round(random.uniform(1, 5), 2),
            'predicted_revenue': random.uniform(1e9, 10e9),
            'surprise_probability': random.uniform(0.3, 0.8),
            'direction_confidence': random.uniform(0.6, 0.9),
            'next_earnings_date': '2024-01-15',
            'analyst_consensus': {'eps': random.uniform(1, 5)},
            'ai_vs_consensus': random.uniform(-0.5, 0.5)
        }
    
    def _mock_market_regime(self) -> Dict[str, Any]:
        """Mock market regime for testing."""
        import random
        return {
            'current_regime': random.choice(['bull', 'bear', 'sideways', 'volatile']),
            'regime_probability': random.uniform(0.6, 0.9),
            'volatility_regime': random.choice(['low', 'normal', 'high']),
            'trend_regime': random.choice(['uptrend', 'downtrend', 'sideways']),
            'risk_on_off': random.choice(['risk_on', 'risk_off', 'neutral']),
            'regime_change_probability': random.uniform(0.1, 0.4)
        }