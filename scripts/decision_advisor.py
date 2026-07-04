"""
Decision Advisor - Alerts when human intervention is needed.
"""

import logging
import smtplib
import os
from typing import Dict, Any, List
from datetime import datetime
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

class DecisionAdvisor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_thresholds = {
            'high_risk_trade': 0.1,  # 10% of portfolio
            'unusual_market_move': 0.05,  # 5% price change
            'strategy_conflict': 0.7,  # 70% disagreement
            'max_drawdown': 0.15,  # 15% drawdown
            'api_failure': True,
            'large_loss': 0.03  # 3% daily loss
        }
        
    def analyze_decision(self, decision_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if human intervention is needed."""
        
        alerts = []
        
        if decision_type == "trade_execution":
            alerts.extend(self._check_trade_risks(data))
        elif decision_type == "market_conditions":
            alerts.extend(self._check_market_conditions(data))
        elif decision_type == "strategy_performance":
            alerts.extend(self._check_strategy_performance(data))
        elif decision_type == "system_health":
            alerts.extend(self._check_system_health(data))
            
        if alerts:
            self._send_alerts(alerts, decision_type, data)
            
        return {
            'needs_human': len(alerts) > 0,
            'alerts': alerts,
            'recommendation': self._get_recommendation(alerts, data),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _check_trade_risks(self, data: Dict[str, Any]) -> List[str]:
        """Check for high-risk trading decisions."""
        alerts = []
        
        position_size = data.get('position_size_pct', 0)
        if position_size > self.alert_thresholds['high_risk_trade']:
            alerts.append(f"🚨 HIGH RISK: Position size {position_size:.1%} exceeds {self.alert_thresholds['high_risk_trade']:.1%}")
            
        strategy_agreement = data.get('strategy_agreement', 1.0)
        if strategy_agreement < self.alert_thresholds['strategy_conflict']:
            alerts.append(f"⚠️ CONFLICT: Strategies only {strategy_agreement:.1%} in agreement")
            
        return alerts
    
    def _check_market_conditions(self, data: Dict[str, Any]) -> List[str]:
        """Check for unusual market conditions."""
        alerts = []
        
        price_change = abs(data.get('price_change_pct', 0))
        if price_change > self.alert_thresholds['unusual_market_move']:
            alerts.append(f"📈 UNUSUAL: {data.get('symbol', 'Market')} moved {price_change:.1%}")
            
        volatility = data.get('volatility', 0)
        if volatility > 0.3:  # 30% volatility
            alerts.append(f"🌪️ HIGH VOLATILITY: {volatility:.1%} detected")
            
        return alerts
    
    def _check_strategy_performance(self, data: Dict[str, Any]) -> List[str]:
        """Check strategy performance issues."""
        alerts = []
        
        drawdown = data.get('max_drawdown', 0)
        if drawdown > self.alert_thresholds['max_drawdown']:
            alerts.append(f"📉 DRAWDOWN: {drawdown:.1%} exceeds limit {self.alert_thresholds['max_drawdown']:.1%}")
            
        daily_loss = abs(data.get('daily_pnl_pct', 0)) if data.get('daily_pnl_pct', 0) < 0 else 0
        if daily_loss > self.alert_thresholds['large_loss']:
            alerts.append(f"💸 LARGE LOSS: Daily loss {daily_loss:.1%}")
            
        return alerts
    
    def _check_system_health(self, data: Dict[str, Any]) -> List[str]:
        """Check system health issues."""
        alerts = []
        
        if data.get('api_failures', 0) > 0:
            alerts.append(f"🔌 API FAILURE: {data['api_failures']} failed connections")
            
        if data.get('strategy_errors', 0) > 3:
            alerts.append(f"🐛 STRATEGY ERRORS: {data['strategy_errors']} errors detected")
            
        return alerts
    
    def _get_recommendation(self, alerts: List[str], data: Dict[str, Any]) -> str:
        """Generate human-readable recommendation."""
        if not alerts:
            return "✅ All systems normal - continue automated trading"
            
        if len(alerts) >= 3:
            return "🛑 STOP TRADING - Multiple critical issues detected"
        elif any("HIGH RISK" in alert for alert in alerts):
            return "⏸️ PAUSE - Review position sizing before continuing"
        elif any("DRAWDOWN" in alert for alert in alerts):
            return "📊 REVIEW - Analyze strategy performance"
        else:
            return "👀 MONITOR - Increased supervision recommended"
    
    def _send_alerts(self, alerts: List[str], decision_type: str, data: Dict[str, Any]):
        """Send alerts via multiple channels."""
        
        # Console alert
        logger.warning(f"HUMAN INTERVENTION NEEDED - {decision_type.upper()}")
        for alert in alerts:
            logger.warning(alert)
            
        # Email alert (if configured)
        email = os.getenv('TRADING_ALERT_EMAIL')
        if email:
            self._send_email_alert(email, alerts, decision_type, data)
            
        # File alert
        self._write_alert_file(alerts, decision_type, data)
    
    def _send_email_alert(self, email: str, alerts: List[str], decision_type: str, data: Dict[str, Any]):
        """Send email alert."""
        try:
            msg = MIMEText(f"""
TRADING ALERT - Human Intervention Required

Decision Type: {decision_type}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Alerts:
{chr(10).join(alerts)}

Recommendation: {self._get_recommendation(alerts, data)}

Data: {data}

Please review and take appropriate action.
            """)
            
            msg['Subject'] = f'🚨 Trading Alert - {decision_type}'
            msg['From'] = 'trading-agent@localhost'
            msg['To'] = email
            
            # Use local SMTP or configure with your email provider
            # server = smtplib.SMTP('localhost')
            # server.send_message(msg)
            # server.quit()
            
            logger.info(f"Email alert sent to {email}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
    
    def _write_alert_file(self, alerts: List[str], decision_type: str, data: Dict[str, Any]):
        """Write alert to file for monitoring."""
        try:
            os.makedirs('alerts', exist_ok=True)
            filename = f"alerts/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"TRADING ALERT - {decision_type.upper()}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n\n")
                f.write("Alerts:\n")
                for alert in alerts:
                    f.write(f"- {alert}\n")
                f.write(f"\nRecommendation: {self._get_recommendation(alerts, data)}\n")
                f.write(f"\nData: {data}\n")
                
            logger.info(f"Alert written to {filename}")
        except Exception as e:
            logger.error(f"Failed to write alert file: {str(e)}")

class KnowledgeTranslator:
    """Translates complex trading decisions into human-readable explanations."""
    
    def explain_decision(self, decision: Dict[str, Any]) -> str:
        """Convert technical decision into plain English."""
        
        action = decision.get('action', 'hold')
        symbol = decision.get('symbol', 'UNKNOWN')
        confidence = decision.get('confidence', 0)
        strategy = decision.get('strategy', 'unknown')
        
        explanation = f"💡 DECISION EXPLANATION:\n\n"
        
        if action == 'buy':
            explanation += f"🟢 BUYING {symbol}\n"
            explanation += f"Confidence: {confidence:.1%}\n"
            explanation += f"Strategy: {strategy.title()}\n\n"
            
            if strategy == 'momentum':
                explanation += "📈 Reason: Stock price is trending upward with strong momentum"
            elif strategy == 'mean_reversion':
                explanation += "🔄 Reason: Stock is oversold and likely to bounce back"
            elif strategy == 'sentiment':
                explanation += "📰 Reason: Positive news sentiment detected"
            else:
                explanation += f"🤖 Reason: {strategy} strategy signals positive opportunity"
                
        elif action == 'sell':
            explanation += f"🔴 SELLING {symbol}\n"
            explanation += f"Confidence: {confidence:.1%}\n"
            explanation += f"Strategy: {strategy.title()}\n\n"
            
            if strategy == 'momentum':
                explanation += "📉 Reason: Downward momentum detected, cutting losses"
            elif strategy == 'mean_reversion':
                explanation += "⬆️ Reason: Stock is overbought, taking profits"
            elif strategy == 'sentiment':
                explanation += "📰 Reason: Negative news sentiment detected"
            else:
                explanation += f"🤖 Reason: {strategy} strategy signals exit opportunity"
        else:
            explanation += f"⏸️ HOLDING {symbol}\n"
            explanation += "Reason: No clear signal or waiting for better opportunity"
            
        return explanation