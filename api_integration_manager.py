"""
API Integration Manager - Coordinates all external APIs.
"""

import logging
from typing import Dict, Any, List
from polygon_connector import PolygonConnector
from portfolio_optimizer_api import PortfolioOptimizerAPI
from aletheia_connector import AletheiaConnector

logger = logging.getLogger(__name__)

class APIIntegrationManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.apis = {}
        
        # Initialize API connectors
        if config.get('polygon', {}).get('enabled', False):
            self.apis['polygon'] = PolygonConnector(config.get('polygon', {}))
            
        if config.get('portfolio_optimizer', {}).get('enabled', False):
            self.apis['portfolio_optimizer'] = PortfolioOptimizerAPI(config.get('portfolio_optimizer', {}))
            
        if config.get('aletheia', {}).get('enabled', False):
            self.apis['aletheia'] = AletheiaConnector(config.get('aletheia', {}))
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect to all enabled APIs."""
        results = {}
        
        for name, api in self.apis.items():
            try:
                results[name] = api.connect()
                if results[name]:
                    logger.info(f"✅ Connected to {name}")
                else:
                    logger.warning(f"❌ Failed to connect to {name}")
            except Exception as e:
                logger.error(f"Error connecting to {name}: {str(e)}")
                results[name] = False
        
        return results
    
    def get_enhanced_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get enhanced market data from multiple sources."""
        enhanced_data = {}
        
        for symbol in symbols:
            symbol_data = {'symbol': symbol}
            
            # Get Polygon data if available
            if 'polygon' in self.apis:
                try:
                    quote = self.apis['polygon'].get_real_time_quote(symbol)
                    if quote:
                        symbol_data['polygon'] = quote
                        symbol_data['price'] = quote.get('price', 0)
                        symbol_data['volume'] = quote.get('size', 0)
                except Exception as e:
                    logger.error(f"Polygon error for {symbol}: {str(e)}")
            
            # Get Aletheia sentiment if available
            if 'aletheia' in self.apis:
                try:
                    sentiment = self.apis['aletheia'].get_sentiment_analysis(symbol)
                    if sentiment:
                        symbol_data['sentiment'] = sentiment
                        symbol_data['sentiment_score'] = sentiment.get('overall_sentiment', 0)
                except Exception as e:
                    logger.error(f"Aletheia sentiment error for {symbol}: {str(e)}")
            
            # Get alternative data
            if 'aletheia' in self.apis:
                try:
                    alt_data = self.apis['aletheia'].get_alternative_data(symbol)
                    if alt_data:
                        symbol_data['alternative_data'] = alt_data
                except Exception as e:
                    logger.error(f"Aletheia alt data error for {symbol}: {str(e)}")
            
            enhanced_data[symbol] = symbol_data
        
        return enhanced_data
    
    def optimize_portfolio(self, symbols: List[str], returns_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Optimize portfolio using external API or local methods."""
        
        if 'portfolio_optimizer' in self.apis:
            try:
                result = self.apis['portfolio_optimizer'].optimize_portfolio(returns_data)
                logger.info(f"Portfolio optimized using {result.get('method', 'unknown')} method")
                return result
            except Exception as e:
                logger.error(f"Portfolio optimization error: {str(e)}")
        
        # Fallback to equal weighting
        if symbols:
            equal_weight = 1.0 / len(symbols)
            return {
                'weights': {symbol: equal_weight for symbol in symbols},
                'expected_return': 0.08,
                'expected_risk': 0.15,
                'method': 'equal_weight_fallback'
            }
        
        return {'weights': {}, 'expected_return': 0, 'expected_risk': 0}
    
    def get_market_regime(self) -> Dict[str, Any]:
        """Get current market regime analysis."""
        
        if 'aletheia' in self.apis:
            try:
                return self.apis['aletheia'].get_market_regime()
            except Exception as e:
                logger.error(f"Market regime error: {str(e)}")
        
        # Default regime
        return {
            'current_regime': 'normal',
            'regime_probability': 0.7,
            'volatility_regime': 'normal',
            'trend_regime': 'sideways',
            'risk_on_off': 'neutral'
        }
    
    def get_earnings_insights(self, symbols: List[str]) -> Dict[str, Any]:
        """Get earnings predictions and insights."""
        insights = {}
        
        if 'aletheia' in self.apis:
            for symbol in symbols:
                try:
                    prediction = self.apis['aletheia'].get_earnings_predictions(symbol)
                    if prediction:
                        insights[symbol] = prediction
                except Exception as e:
                    logger.error(f"Earnings prediction error for {symbol}: {str(e)}")
        
        return insights
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all APIs."""
        status = {}
        
        for name, api in self.apis.items():
            status[name] = {
                'connected': getattr(api, 'is_connected', False),
                'api_key_configured': bool(getattr(api, 'api_key', '')),
                'base_url': getattr(api, 'base_url', 'Unknown')
            }
        
        return status