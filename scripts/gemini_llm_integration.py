"""
Gemini LLM Integration for AI Trading Agent
"""

import os
import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class GeminiTradingLLM:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
    def analyze_market_sentiment(self, news_text: str, symbol: str) -> Dict[str, Any]:
        """Analyze market sentiment using Gemini."""
        
        prompt = f"""
        Analyze the market sentiment for {symbol} based on this news:
        
        {news_text}
        
        Provide a JSON response with:
        - sentiment_score: number between -1 (very bearish) and 1 (very bullish)
        - key_factors: list of important factors
        - recommendation: "bullish", "bearish", or "neutral"
        - confidence: number between 0 and 1
        - reasoning: brief explanation
        
        Respond only with valid JSON.
        """
        
        try:
            response = self._call_gemini(prompt)
            return json.loads(response)
        except:
            return {
                'sentiment_score': 0.0,
                'key_factors': ['analysis unavailable'],
                'recommendation': 'neutral',
                'confidence': 0.5,
                'reasoning': 'Gemini API unavailable'
            }
    
    def explain_trading_decision(self, decision: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Generate human-readable explanation using Gemini."""
        
        prompt = f"""
        Explain this trading decision in simple terms:
        
        Action: {decision.get('action', 'hold')}
        Symbol: {decision.get('symbol', 'UNKNOWN')}
        Confidence: {decision.get('confidence', 0):.1%}
        Price: ${market_data.get('close', 0):.2f}
        
        Provide a clear, concise explanation that a trader would understand.
        Include the reasoning and what this means for the position.
        """
        
        try:
            return self._call_gemini(prompt)
        except:
            action = decision.get('action', 'hold')
            symbol = decision.get('symbol', 'UNKNOWN')
            return f"Decision: {action.upper()} {symbol} based on technical analysis."
    
    def generate_market_insights(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate market insights using Gemini."""
        
        prompt = f"""
        Provide market insights for this portfolio:
        
        Portfolio Value: ${portfolio_data.get('total_value', 0):,.2f}
        Daily P&L: ${portfolio_data.get('daily_pnl', 0):+,.2f}
        Positions: {len(portfolio_data.get('positions', []))}
        Win Rate: {portfolio_data.get('win_rate', 0):.1%}
        Sharpe Ratio: {portfolio_data.get('sharpe_ratio', 0):.2f}
        
        Provide:
        1. Performance assessment
        2. Risk evaluation
        3. Specific recommendations
        4. Market outlook
        
        Keep it concise and actionable.
        """
        
        try:
            return self._call_gemini(prompt)
        except:
            return "Market insights unavailable - Gemini API error"
    
    def _call_gemini(self, prompt: str) -> str:
        """Make API call to Gemini."""
        
        if not self.api_key:
            raise Exception("Gemini API key not found")
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception(f"Gemini API error: {response.status_code}")

# Integration with existing strategies
def add_gemini_to_strategy(strategy_instance):
    """Add Gemini LLM to existing strategy."""
    
    gemini = GeminiTradingLLM()
    original_generate_signals = strategy_instance.generate_signals
    
    def enhanced_generate_signals(data):
        signals = original_generate_signals(data)
        
        # Add Gemini analysis
        if 'news_text' in data:
            sentiment = gemini.analyze_market_sentiment(
                data['news_text'], 
                data.get('symbol', 'UNKNOWN')
            )
            signals['gemini_sentiment'] = sentiment
        
        # Add explanation
        signals['explanation'] = gemini.explain_trading_decision(signals, data)
        
        return signals
    
    strategy_instance.generate_signals = enhanced_generate_signals
    return strategy_instance