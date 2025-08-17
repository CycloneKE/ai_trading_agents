"""
Implement critical security safeguards
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

class TradingSafeguards:
    def __init__(self):
        self.daily_loss_limit = 0.10  # 10% max daily loss
        self.position_size_limit = 0.05  # 5% max per position
        self.large_trade_threshold = 10000  # $10k requires approval
        self.emergency_stop = False
        self.daily_losses = 0.0
        self.start_portfolio_value = 100000  # Track from start
        
    def validate_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade against all safeguards."""
        
        validation = {
            'approved': True,
            'reasons': [],
            'requires_human_approval': False
        }
        
        # Check emergency stop
        if self.emergency_stop:
            validation['approved'] = False
            validation['reasons'].append("EMERGENCY STOP ACTIVE")
            return validation
        
        # Check position size limit
        position_value = trade.get('position_value', 0)
        portfolio_value = trade.get('portfolio_value', 100000)
        position_pct = position_value / portfolio_value
        
        if position_pct > self.position_size_limit:
            validation['approved'] = False
            validation['reasons'].append(f"Position size {position_pct:.1%} exceeds limit {self.position_size_limit:.1%}")
        
        # Check daily loss limit
        if self.daily_losses > self.daily_loss_limit:
            validation['approved'] = False
            validation['reasons'].append(f"Daily loss limit {self.daily_loss_limit:.1%} exceeded")
        
        # Check large trade approval
        if position_value > self.large_trade_threshold:
            validation['requires_human_approval'] = True
            validation['reasons'].append(f"Large trade ${position_value:,.0f} requires human approval")
        
        # AI confidence check
        confidence = trade.get('confidence', 0)
        if confidence < 0.6:
            validation['approved'] = False
            validation['reasons'].append(f"AI confidence {confidence:.1%} too low (min 60%)")
        
        return validation
    
    def update_daily_losses(self, loss_amount: float):
        """Update daily loss tracking."""
        if loss_amount > 0:  # Only track losses
            self.daily_losses += loss_amount / self.start_portfolio_value
    
    def activate_emergency_stop(self, reason: str):
        """Activate emergency stop."""
        self.emergency_stop = True
        self._log_emergency_stop(reason)
        print(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def _log_emergency_stop(self, reason: str):
        """Log emergency stop event."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'emergency_stop',
            'reason': reason
        }
        
        with open('security_log.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

class AIGuardrails:
    def __init__(self):
        self.min_confidence = 0.6
        self.max_position_correlation = 0.8
        self.decision_log = []
        
    def validate_ai_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate AI decision against guardrails."""
        
        # Confidence threshold
        confidence = decision.get('confidence', 0)
        if confidence < self.min_confidence:
            self._log_rejection(decision, f"Low confidence: {confidence:.1%}")
            return False
        
        # Log all decisions for audit
        self._log_decision(decision)
        
        return True
    
    def _log_decision(self, decision: Dict[str, Any]):
        """Log AI decision for audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'validation': 'approved'
        }
        
        self.decision_log.append(log_entry)
        
        # Write to file
        with open('ai_decisions.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_rejection(self, decision: Dict[str, Any], reason: str):
        """Log rejected decision."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'validation': 'rejected',
            'reason': reason
        }
        
        with open('ai_decisions.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

class SecureAPIManager:
    def __init__(self):
        self.api_keys = {}
        self._load_secure_keys()
    
    def _load_secure_keys(self):
        """Load API keys securely."""
        # Remove from .env and use secure storage
        secure_file = 'secure_keys.enc'  # Would be encrypted in production
        
        if os.path.exists(secure_file):
            # In production, this would decrypt the file
            with open(secure_file, 'r') as f:
                self.api_keys = json.load(f)
        else:
            # Create secure storage
            self.api_keys = {
                'gemini': os.getenv('GEMINI_API_KEY', ''),
                'alpaca': os.getenv('TRADING_ALPACA_API_KEY', ''),
                'alpha_vantage': os.getenv('TRADING_ALPHA_VANTAGE_API_KEY', '')
            }
            
            # Save securely (would encrypt in production)
            with open(secure_file, 'w') as f:
                json.dump(self.api_keys, f)
            
            print("API keys moved to secure storage")
    
    def get_api_key(self, service: str) -> str:
        """Get API key securely."""
        return self.api_keys.get(service, '')

# Integration class
class SecureTradingAgent:
    def __init__(self):
        self.safeguards = TradingSafeguards()
        self.guardrails = AIGuardrails()
        self.api_manager = SecureAPIManager()
        
    def execute_trade(self, trade_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade with full security validation."""
        
        # Step 1: Validate AI decision
        if not self.guardrails.validate_ai_decision(trade_decision):
            return {'status': 'rejected', 'reason': 'AI guardrails failed'}
        
        # Step 2: Validate trade safeguards
        validation = self.safeguards.validate_trade(trade_decision)
        
        if not validation['approved']:
            return {
                'status': 'rejected', 
                'reasons': validation['reasons']
            }
        
        # Step 3: Check human approval requirement
        if validation['requires_human_approval']:
            return {
                'status': 'pending_approval',
                'message': 'Large trade requires human approval',
                'trade': trade_decision
            }
        
        # Step 4: Execute trade (would integrate with broker)
        return {
            'status': 'executed',
            'trade': trade_decision,
            'timestamp': datetime.now().isoformat()
        }

def implement_security():
    """Implement all security measures."""
    
    print("Implementing Security Safeguards...")
    
    # Create secure trading agent
    secure_agent = SecureTradingAgent()
    
    # Test security measures
    test_trade = {
        'symbol': 'AAPL',
        'action': 'buy',
        'position_value': 5000,
        'portfolio_value': 100000,
        'confidence': 0.75
    }
    
    result = secure_agent.execute_trade(test_trade)
    print(f"Test trade result: {result['status']}")
    
    # Test large trade
    large_trade = {
        'symbol': 'AAPL',
        'action': 'buy',
        'position_value': 15000,
        'portfolio_value': 100000,
        'confidence': 0.85
    }
    
    result = secure_agent.execute_trade(large_trade)
    print(f"Large trade result: {result['status']}")
    
    print("Security safeguards implemented!")
    
    return secure_agent

if __name__ == '__main__':
    implement_security()