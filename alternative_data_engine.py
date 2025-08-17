#!/usr/bin/env python3
"""
Alternative Data Engine

Advanced alternative data integration including satellite imagery,
social media sentiment, economic indicators, and ESG factors.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict, deque

try:
    import aiohttp
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    from textblob import TextBlob
    import nltk
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

logger = logging.getLogger(__name__)


class SatelliteDataProcessor:
    """Process satellite imagery data for economic indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('satellite_api_key', '')
        self.data_cache = {}
        
    async def get_economic_activity(self, region: str, date: str) -> Dict[str, float]:
        """Get economic activity indicators from satellite data"""
        try:
            # Simulate satellite data processing
            # In production, this would connect to actual satellite data APIs
            
            indicators = {
                'shipping_activity': np.random.uniform(0.5, 1.5),  # Port activity
                'manufacturing_activity': np.random.uniform(0.7, 1.3),  # Factory emissions
                'construction_activity': np.random.uniform(0.6, 1.4),  # Construction sites
                'agricultural_activity': np.random.uniform(0.8, 1.2),  # Crop health
                'energy_consumption': np.random.uniform(0.9, 1.1),  # Night lights
                'traffic_density': np.random.uniform(0.7, 1.3)  # Road usage
            }
            
            # Add noise and trends
            for key in indicators:
                indicators[key] += np.random.normal(0, 0.05)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Satellite data processing error: {e}")
            return {}


class SocialMediaAnalyzer:
    """Analyze social media sentiment and trends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.twitter_api = None
        self.sentiment_cache = deque(maxlen=1000)
        
        if TWITTER_AVAILABLE and config.get('twitter_api_key'):
            try:
                auth = tweepy.OAuthHandler(
                    config.get('twitter_api_key', ''),
                    config.get('twitter_api_secret', '')
                )
                auth.set_access_token(
                    config.get('twitter_access_token', ''),
                    config.get('twitter_access_secret', '')
                )
                self.twitter_api = tweepy.API(auth)
            except Exception as e:
                logger.warning(f"Twitter API initialization failed: {e}")
    
    def analyze_sentiment(self, symbol: str, timeframe: str = '1h') -> Dict[str, float]:
        """Analyze sentiment for a specific symbol"""
        try:
            if not NLP_AVAILABLE:
                return self._mock_sentiment_data(symbol)
            
            # Collect tweets (mocked for demo)
            tweets = self._collect_tweets(symbol, timeframe)
            
            if not tweets:
                return self._mock_sentiment_data(symbol)
            
            # Analyze sentiment
            sentiments = []
            for tweet in tweets:
                blob = TextBlob(tweet)
                sentiments.append(blob.sentiment.polarity)
            
            # Calculate metrics
            avg_sentiment = np.mean(sentiments)
            sentiment_volatility = np.std(sentiments)
            positive_ratio = sum(1 for s in sentiments if s > 0.1) / len(sentiments)
            negative_ratio = sum(1 for s in sentiments if s < -0.1) / len(sentiments)
            
            return {
                'sentiment_score': float(avg_sentiment),
                'sentiment_volatility': float(sentiment_volatility),
                'positive_ratio': float(positive_ratio),
                'negative_ratio': float(negative_ratio),
                'tweet_volume': len(tweets),
                'confidence': min(1.0, len(tweets) / 100)  # More tweets = higher confidence
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._mock_sentiment_data(symbol)
    
    def _collect_tweets(self, symbol: str, timeframe: str) -> List[str]:
        """Collect tweets about a symbol"""
        # Mock tweet collection
        mock_tweets = [
            f"{symbol} looking strong today! Great earnings report.",
            f"Concerned about {symbol} recent performance. Market volatility affecting all stocks.",
            f"{symbol} innovation in AI is impressive. Long-term bullish.",
            f"Technical analysis shows {symbol} breaking resistance levels.",
            f"Macro environment challenging for {symbol} and similar companies."
        ]
        
        return mock_tweets[:np.random.randint(3, 8)]
    
    def _mock_sentiment_data(self, symbol: str) -> Dict[str, float]:
        """Generate mock sentiment data"""
        return {
            'sentiment_score': np.random.uniform(-0.3, 0.3),
            'sentiment_volatility': np.random.uniform(0.1, 0.5),
            'positive_ratio': np.random.uniform(0.3, 0.7),
            'negative_ratio': np.random.uniform(0.2, 0.5),
            'tweet_volume': np.random.randint(50, 200),
            'confidence': np.random.uniform(0.6, 0.9)
        }


class ESGDataProvider:
    """Environmental, Social, and Governance data provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.esg_cache = {}
        self.update_frequency = config.get('esg_update_frequency', 24)  # hours
        
    def get_esg_scores(self, symbol: str) -> Dict[str, float]:
        """Get ESG scores for a symbol"""
        try:
            # Check cache
            if symbol in self.esg_cache:
                cache_entry = self.esg_cache[symbol]
                if (datetime.now() - cache_entry['timestamp']).hours < self.update_frequency:
                    return cache_entry['data']
            
            # Generate ESG scores (in production, fetch from ESG data providers)
            esg_scores = {
                'environmental_score': np.random.uniform(20, 95),
                'social_score': np.random.uniform(25, 90),
                'governance_score': np.random.uniform(30, 95),
                'overall_esg_score': 0,
                'esg_risk_rating': '',
                'carbon_intensity': np.random.uniform(50, 500),  # tons CO2/million revenue
                'water_usage': np.random.uniform(10, 100),  # cubic meters/million revenue
                'waste_generation': np.random.uniform(5, 50),  # tons/million revenue
                'employee_satisfaction': np.random.uniform(60, 95),
                'diversity_score': np.random.uniform(40, 90),
                'board_independence': np.random.uniform(50, 95),
                'executive_compensation_ratio': np.random.uniform(50, 500)
            }
            
            # Calculate overall score
            esg_scores['overall_esg_score'] = (
                esg_scores['environmental_score'] * 0.4 +
                esg_scores['social_score'] * 0.3 +
                esg_scores['governance_score'] * 0.3
            )
            
            # Risk rating
            if esg_scores['overall_esg_score'] >= 80:
                esg_scores['esg_risk_rating'] = 'Low'
            elif esg_scores['overall_esg_score'] >= 60:
                esg_scores['esg_risk_rating'] = 'Medium'
            else:
                esg_scores['esg_risk_rating'] = 'High'
            
            # Cache results
            self.esg_cache[symbol] = {
                'data': esg_scores,
                'timestamp': datetime.now()
            }
            
            return esg_scores
            
        except Exception as e:
            logger.error(f"ESG data retrieval error: {e}")
            return {}


class EconomicIndicatorTracker:
    """Track and analyze economic indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.indicators = {}
        self.api_keys = {
            'fred': config.get('fred_api_key', ''),
            'alpha_vantage': config.get('alpha_vantage_api_key', ''),
            'quandl': config.get('quandl_api_key', '')
        }
        
    async def get_economic_indicators(self) -> Dict[str, float]:
        """Get current economic indicators"""
        try:
            # Mock economic indicators (in production, fetch from FRED, etc.)
            indicators = {
                'gdp_growth': np.random.uniform(-2, 4),  # %
                'unemployment_rate': np.random.uniform(3, 8),  # %
                'inflation_rate': np.random.uniform(0, 6),  # %
                'interest_rate': np.random.uniform(0, 5),  # %
                'consumer_confidence': np.random.uniform(80, 120),
                'manufacturing_pmi': np.random.uniform(45, 60),
                'services_pmi': np.random.uniform(48, 58),
                'retail_sales_growth': np.random.uniform(-5, 10),  # %
                'housing_starts': np.random.uniform(1000, 1800),  # thousands
                'oil_price': np.random.uniform(60, 120),  # USD/barrel
                'gold_price': np.random.uniform(1800, 2200),  # USD/oz
                'dollar_index': np.random.uniform(95, 110),
                'vix': np.random.uniform(12, 35),
                'yield_curve_spread': np.random.uniform(-0.5, 2.5)  # 10Y-2Y
            }
            
            # Add some correlation and trends
            if 'previous_indicators' in self.indicators:
                prev = self.indicators['previous_indicators']
                for key in indicators:
                    if key in prev:
                        # Add momentum
                        momentum = np.random.uniform(-0.1, 0.1)
                        indicators[key] = prev[key] * (1 + momentum) + np.random.normal(0, 0.05)
            
            self.indicators['current'] = indicators
            self.indicators['previous_indicators'] = indicators.copy()
            self.indicators['timestamp'] = datetime.now()
            
            return indicators
            
        except Exception as e:
            logger.error(f"Economic indicators error: {e}")
            return {}


class AlternativeDataEngine:
    """Main alternative data engine coordinating all data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.satellite_processor = SatelliteDataProcessor(config.get('satellite', {}))
        self.social_analyzer = SocialMediaAnalyzer(config.get('social_media', {}))
        self.esg_provider = ESGDataProvider(config.get('esg', {}))
        self.economic_tracker = EconomicIndicatorTracker(config.get('economic', {}))
        
        # Data storage
        self.alternative_data_cache = defaultdict(dict)
        self.data_history = defaultdict(lambda: deque(maxlen=100))
        
        # Update frequencies
        self.update_frequencies = {
            'satellite': config.get('satellite_update_hours', 24),
            'social': config.get('social_update_hours', 1),
            'esg': config.get('esg_update_hours', 24),
            'economic': config.get('economic_update_hours', 6)
        }
        
        self.last_updates = {source: datetime.now() - timedelta(days=1) 
                           for source in self.update_frequencies}
        
        logger.info("Alternative Data Engine initialized")
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive alternative data for a symbol"""
        try:
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_sources': {}
            }
            
            # Collect data from all sources
            tasks = []
            
            if self._should_update('satellite'):
                tasks.append(self._get_satellite_data(symbol))
            
            if self._should_update('social'):
                tasks.append(self._get_social_data(symbol))
            
            if self._should_update('esg'):
                tasks.append(self._get_esg_data(symbol))
            
            if self._should_update('economic'):
                tasks.append(self._get_economic_data())
            
            # Execute all tasks concurrently
            if ASYNC_AVAILABLE and tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                source_names = ['satellite', 'social', 'esg', 'economic']
                for i, result in enumerate(results):
                    if not isinstance(result, Exception) and i < len(source_names):
                        data['data_sources'][source_names[i]] = result
            else:
                # Fallback to synchronous execution
                if self._should_update('social'):
                    data['data_sources']['social'] = self.social_analyzer.analyze_sentiment(symbol)
                if self._should_update('esg'):
                    data['data_sources']['esg'] = self.esg_provider.get_esg_scores(symbol)
            
            # Generate composite scores
            data['composite_scores'] = self._calculate_composite_scores(data['data_sources'])
            
            # Store in cache and history
            self.alternative_data_cache[symbol] = data
            self.data_history[symbol].append(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Comprehensive data collection error: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def _get_satellite_data(self, symbol: str) -> Dict[str, Any]:
        """Get satellite data for symbol"""
        try:
            # Map symbol to relevant regions/industries
            region = self._symbol_to_region(symbol)
            date = datetime.now().strftime('%Y-%m-%d')
            
            satellite_data = await self.satellite_processor.get_economic_activity(region, date)
            self.last_updates['satellite'] = datetime.now()
            
            return satellite_data
            
        except Exception as e:
            logger.error(f"Satellite data error: {e}")
            return {}
    
    async def _get_social_data(self, symbol: str) -> Dict[str, Any]:
        """Get social media data for symbol"""
        try:
            social_data = self.social_analyzer.analyze_sentiment(symbol)
            self.last_updates['social'] = datetime.now()
            
            return social_data
            
        except Exception as e:
            logger.error(f"Social data error: {e}")
            return {}
    
    async def _get_esg_data(self, symbol: str) -> Dict[str, Any]:
        """Get ESG data for symbol"""
        try:
            esg_data = self.esg_provider.get_esg_scores(symbol)
            self.last_updates['esg'] = datetime.now()
            
            return esg_data
            
        except Exception as e:
            logger.error(f"ESG data error: {e}")
            return {}
    
    async def _get_economic_data(self) -> Dict[str, Any]:
        """Get economic indicators"""
        try:
            economic_data = await self.economic_tracker.get_economic_indicators()
            self.last_updates['economic'] = datetime.now()
            
            return economic_data
            
        except Exception as e:
            logger.error(f"Economic data error: {e}")
            return {}
    
    def _should_update(self, source: str) -> bool:
        """Check if data source should be updated"""
        hours_since_update = (datetime.now() - self.last_updates[source]).total_seconds() / 3600
        return hours_since_update >= self.update_frequencies[source]
    
    def _symbol_to_region(self, symbol: str) -> str:
        """Map trading symbol to geographic region"""
        # Simplified mapping
        region_map = {
            'AAPL': 'california',
            'TSLA': 'california',
            'MSFT': 'washington',
            'GOOGL': 'california',
            'AMZN': 'washington'
        }
        return region_map.get(symbol, 'united_states')
    
    def _calculate_composite_scores(self, data_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate composite scores from all data sources"""
        try:
            composite = {}
            
            # Alternative data sentiment score
            sentiment_components = []
            
            if 'social' in data_sources:
                social = data_sources['social']
                sentiment_components.append(social.get('sentiment_score', 0) * 0.4)
            
            if 'satellite' in data_sources:
                satellite = data_sources['satellite']
                # Convert economic activity to sentiment
                activity_score = np.mean(list(satellite.values())) - 1.0  # Center around 0
                sentiment_components.append(activity_score * 0.3)
            
            if 'economic' in data_sources:
                economic = data_sources['economic']
                # Create economic sentiment from key indicators
                econ_sentiment = (
                    (economic.get('gdp_growth', 0) / 4) * 0.3 +
                    (5 - economic.get('unemployment_rate', 5)) / 5 * 0.3 +
                    (economic.get('consumer_confidence', 100) - 100) / 20 * 0.4
                )
                sentiment_components.append(econ_sentiment * 0.3)
            
            composite['alternative_sentiment'] = float(np.sum(sentiment_components))
            
            # ESG risk score
            if 'esg' in data_sources:
                esg = data_sources['esg']
                esg_score = esg.get('overall_esg_score', 50)
                composite['esg_risk_factor'] = float((esg_score - 50) / 50)  # -1 to 1 scale
            else:
                composite['esg_risk_factor'] = 0.0
            
            # Data quality score
            available_sources = len([s for s in data_sources.values() if s])
            composite['data_quality'] = float(available_sources / 4)  # 4 total sources
            
            # Confidence score
            confidence_components = []
            for source_data in data_sources.values():
                if isinstance(source_data, dict) and 'confidence' in source_data:
                    confidence_components.append(source_data['confidence'])
            
            composite['confidence'] = float(np.mean(confidence_components)) if confidence_components else 0.5
            
            return composite
            
        except Exception as e:
            logger.error(f"Composite score calculation error: {e}")
            return {}
    
    def get_data_summary(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get summary of alternative data for a symbol"""
        try:
            if symbol not in self.data_history:
                return {'error': 'No data available for symbol'}
            
            history = list(self.data_history[symbol])
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent data
            recent_data = [
                entry for entry in history
                if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) > cutoff_date
            ]
            
            if not recent_data:
                return {'error': 'No recent data available'}
            
            # Calculate trends
            summary = {
                'symbol': symbol,
                'data_points': len(recent_data),
                'date_range': {
                    'start': recent_data[0]['timestamp'],
                    'end': recent_data[-1]['timestamp']
                },
                'trends': {},
                'latest_scores': recent_data[-1].get('composite_scores', {})
            }
            
            # Calculate trends for composite scores
            if len(recent_data) >= 2:
                first_scores = recent_data[0].get('composite_scores', {})
                last_scores = recent_data[-1].get('composite_scores', {})
                
                for key in first_scores:
                    if key in last_scores:
                        trend = last_scores[key] - first_scores[key]
                        summary['trends'][key] = float(trend)
            
            return summary
            
        except Exception as e:
            logger.error(f"Data summary error: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get alternative data engine status"""
        return {
            'status': 'active',
            'async_available': ASYNC_AVAILABLE,
            'twitter_available': TWITTER_AVAILABLE,
            'nlp_available': NLP_AVAILABLE,
            'data_sources_active': len([s for s in self.last_updates if 
                                      (datetime.now() - self.last_updates[s]).hours < 48]),
            'symbols_tracked': len(self.alternative_data_cache),
            'last_updates': {k: v.isoformat() for k, v in self.last_updates.items()}
        }