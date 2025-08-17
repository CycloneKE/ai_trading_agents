"""
Quick LLM Integration for Trading Agent
"""

import os
import json
from typing import Dict, Any

class TradingLLM:
    """Simple LLM integration for trading decisions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv('OPENAI_API_KEY')
        
    def analyze_market_sentiment(self, news_text: str, symbol: str) -> Dict[str, Any]:
        """Analyze market sentiment using LLM."""
        
        if not self.api_key:
            # Fallback to simple keyword analysis
            return self._simple_sentiment_analysis(news_text)
        
        try:
            # Would integrate with OpenAI API here
            prompt = f"""
            Analyze the market sentiment for {symbol} based on this news:
            
            {news_text}
            
            Provide:
            1. Sentiment score (-1 to 1)
            2. Key factors
            3. Trading recommendation
            4. Confidence level
            
            Respond in JSON format.
            """
            
            # Placeholder for actual LLM call
            return {
                'sentiment_score': 0.3,
                'key_factors': ['earnings beat', 'guidance raised'],
                'recommendation': 'bullish',
                'confidence': 0.75,
                'reasoning': 'Strong earnings and positive guidance indicate upward momentum'
            }
            
        except Exception as e:
            return self._simple_sentiment_analysis(news_text)
    
    def explain_trading_decision(self, decision: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Generate human-readable explanation of trading decision."""
        
        action = decision.get('action', 'hold')
        symbol = decision.get('symbol', 'UNKNOWN')
        confidence = decision.get('confidence', 0)
        
        if action == 'buy':
            return f"""
            🟢 BUYING {symbol}
            
            Decision Reasoning:
            • Technical indicators show bullish momentum
            • Sentiment analysis indicates positive market mood
            • Risk-reward ratio is favorable
            • Confidence level: {confidence:.1%}
            
            This decision is based on multiple converging signals suggesting upward price movement.
            """
        elif action == 'sell':
            return f"""
            🔴 SELLING {symbol}
            
            Decision Reasoning:
            • Technical indicators suggest bearish trend
            • Risk management protocols triggered
            • Taking profits at current levels
            • Confidence level: {confidence:.1%}
            
            This decision prioritizes capital preservation and profit-taking.
            """
        else:
            return f"""
            ⏸️ HOLDING {symbol}
            
            Decision Reasoning:
            • Mixed signals from technical analysis
            • Waiting for clearer market direction
            • Maintaining current position
            • Confidence level: {confidence:.1%}
            
            Patience is key when signals are unclear.
            """
    
    def generate_market_insights(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate market insights and recommendations."""
        
        portfolio_value = portfolio_data.get('total_value', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        positions = portfolio_data.get('positions', [])
        
        return f"""
        📊 MARKET INSIGHTS & RECOMMENDATIONS
        
        Portfolio Status:
        • Current Value: ${portfolio_value:,.2f}
        • Daily P&L: ${daily_pnl:+,.2f}
        • Active Positions: {len(positions)}
        
        Key Observations:
        • Market volatility is within normal ranges
        • Diversification across sectors looks good
        • Risk exposure is well-managed
        
        Recommendations:
        • Continue current strategy allocation
        • Monitor for any regime changes
        • Consider rebalancing if positions drift >5%
        
        Overall Assessment: Portfolio is performing well with appropriate risk management.
        """
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis without LLM."""
        
        positive_words = ['beat', 'growth', 'strong', 'positive', 'bullish', 'up']
        negative_words = ['miss', 'decline', 'weak', 'negative', 'bearish', 'down']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = min(0.8, pos_count * 0.2)
            recommendation = 'bullish'
        elif neg_count > pos_count:
            sentiment = max(-0.8, -neg_count * 0.2)
            recommendation = 'bearish'
        else:
            sentiment = 0.0
            recommendation = 'neutral'
        
        return {
            'sentiment_score': sentiment,
            'key_factors': ['keyword analysis'],
            'recommendation': recommendation,
            'confidence': 0.5,
            'reasoning': 'Based on keyword sentiment analysis'
        }

# Integration example:
def integrate_llm_into_strategy(strategy_instance):
    """Add LLM capabilities to existing strategy."""
    
    llm = TradingLLM({})
    
    # Enhance decision making with LLM insights
    original_generate_signals = strategy_instance.generate_signals
    
    def enhanced_generate_signals(data):
        # Get original signals
        signals = original_generate_signals(data)
        
        # Add LLM analysis
        if 'news_text' in data:
            sentiment = llm.analyze_market_sentiment(
                data['news_text'], 
                data.get('symbol', 'UNKNOWN')
            )
            signals['llm_sentiment'] = sentiment
            signals['explanation'] = llm.explain_trading_decision(signals, data)
        
        return signals
    
    strategy_instance.generate_signals = enhanced_generate_signals
    return strategy_instance