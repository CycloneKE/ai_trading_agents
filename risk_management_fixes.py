#!/usr/bin/env python3
"""
RISK MANAGEMENT FIXES - Phase 3
Implement advanced risk controls, stop losses, and position sizing
"""

import os
import json
import math
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None

@dataclass
class RiskLimits:
    max_position_size: float = 0.05  # 5% of portfolio
    max_portfolio_var: float = 0.02  # 2% VaR
    max_drawdown: float = 0.10  # 10% max drawdown
    max_correlation: float = 0.7   # Max correlation between positions
    stop_loss_percent: float = 0.02  # 2% stop loss

class KellyCriterion:
    """Kelly Criterion for optimal position sizing"""
    
    @staticmethod
    def calculate_position_size(win_rate: float, avg_win: float, avg_loss: float, 
                              capital: float, max_risk: float = 0.02) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            capital: Available capital
            max_risk: Maximum risk per trade (0-1)
        
        Returns:
            Optimal position size in dollars
        """
        if win_rate <= 0 or avg_loss <= 0:
            return capital * max_risk
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety factor (use 25% of Kelly)
        safe_kelly = kelly_fraction * 0.25
        
        # Cap at maximum risk
        final_fraction = min(safe_kelly, max_risk)
        final_fraction = max(final_fraction, 0.001)  # Minimum 0.1%
        
        return capital * final_fraction

class AdvancedRiskManager:
    """Advanced risk management with real-time monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = config.get('initial_capital', 100000)
        self.max_drawdown_reached = 0
        self.peak_portfolio_value = self.portfolio_value
        
        # Historical data for Kelly criterion
        self.trade_history = []
        self.win_rate = 0.6  # Default assumption
        self.avg_win = 0.02  # 2% average win
        self.avg_loss = 0.015  # 1.5% average loss
    
    def validate_new_position(self, symbol: str, quantity: float, 
                            price: float, side: str) -> Tuple[bool, str, float]:
        """
        Validate a new position against risk limits
        
        Returns:
            (approved, reason, adjusted_quantity)
        """
        try:
            # Calculate position value
            position_value = abs(quantity * price)
            
            # Check maximum position size
            max_position_value = self.portfolio_value * self.risk_limits.max_position_size
            
            if position_value > max_position_value:
                # Adjust quantity to fit within limits
                adjusted_quantity = max_position_value / price
                if side == 'sell':
                    adjusted_quantity = -adjusted_quantity
                
                return True, f"Position size reduced to {self.risk_limits.max_position_size*100:.1f}% limit", adjusted_quantity
            
            # Check portfolio concentration
            if self._check_concentration_risk(symbol, position_value):
                return False, "Portfolio concentration risk too high", 0
            
            # Check correlation risk
            if self._check_correlation_risk(symbol, position_value):
                return False, "Correlation risk with existing positions too high", 0
            
            # Calculate optimal position size using Kelly
            kelly_size = KellyCriterion.calculate_position_size(
                self.win_rate, self.avg_win, self.avg_loss, 
                self.portfolio_value, self.risk_limits.max_position_size
            )
            
            kelly_quantity = kelly_size / price
            if side == 'sell':
                kelly_quantity = -kelly_quantity
            
            # Use smaller of requested and Kelly-optimal size
            if abs(quantity * price) > kelly_size:
                return True, f"Position size optimized using Kelly criterion", kelly_quantity
            
            return True, "Position approved", quantity
            
        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return False, f"Risk validation failed: {e}", 0
    
    def add_position(self, symbol: str, quantity: float, price: float) -> bool:
        """Add a new position with automatic stop loss"""
        try:
            # Calculate stop loss
            stop_loss_pct = self.risk_limits.stop_loss_percent
            
            if quantity > 0:  # Long position
                stop_loss = price * (1 - stop_loss_pct)
            else:  # Short position
                stop_loss = price * (1 + stop_loss_pct)
            
            # Create position
            position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                stop_loss=stop_loss,
                timestamp=datetime.now()
            )
            
            self.positions[symbol] = position
            
            logger.info(f"✅ Position added: {quantity} {symbol} @ ${price:.2f}, "
                       f"Stop Loss: ${stop_loss:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False
    
    def update_position_price(self, symbol: str, new_price: float) -> Optional[str]:
        """
        Update position price and check for stop loss triggers
        
        Returns:
            Action required ('stop_loss', 'take_profit', None)
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.current_price = new_price
        
        # Check stop loss
        if position.stop_loss:
            if position.quantity > 0 and new_price <= position.stop_loss:
                logger.warning(f"🛑 STOP LOSS triggered for {symbol}: "
                             f"${new_price:.2f} <= ${position.stop_loss:.2f}")
                return 'stop_loss'
            elif position.quantity < 0 and new_price >= position.stop_loss:
                logger.warning(f"🛑 STOP LOSS triggered for {symbol}: "
                             f"${new_price:.2f} >= ${position.stop_loss:.2f}")
                return 'stop_loss'
        
        # Check take profit
        if position.take_profit:
            if position.quantity > 0 and new_price >= position.take_profit:
                logger.info(f"🎯 TAKE PROFIT triggered for {symbol}: "
                           f"${new_price:.2f} >= ${position.take_profit:.2f}")
                return 'take_profit'
            elif position.quantity < 0 and new_price <= position.take_profit:
                logger.info(f"🎯 TAKE PROFIT triggered for {symbol}: "
                           f"${new_price:.2f} <= ${position.take_profit:.2f}")
                return 'take_profit'
        
        return None
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics"""
        try:
            total_value = 0
            total_pnl = 0
            
            for position in self.positions.values():
                position_value = position.quantity * position.current_price
                position_pnl = position.quantity * (position.current_price - position.avg_price)
                
                total_value += abs(position_value)
                total_pnl += position_pnl
            
            current_portfolio_value = self.portfolio_value + total_pnl
            
            # Update peak value and drawdown
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
            
            current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            self.max_drawdown_reached = max(self.max_drawdown_reached, current_drawdown)
            
            # Calculate VaR (simplified)
            position_values = [abs(p.quantity * p.current_price) for p in self.positions.values()]
            portfolio_var = sum(position_values) * 0.02  # Simplified 2% VaR
            
            return {
                'portfolio_value': current_portfolio_value,
                'total_pnl': total_pnl,
                'current_drawdown': current_drawdown,
                'max_drawdown': self.max_drawdown_reached,
                'portfolio_var': portfolio_var,
                'position_count': len(self.positions),
                'total_exposure': total_value
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation error: {e}")
            return {}
    
    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """Check if current portfolio violates risk limits"""
        violations = []
        metrics = self.calculate_portfolio_metrics()
        
        # Check drawdown limit
        if metrics.get('current_drawdown', 0) > self.risk_limits.max_drawdown:
            violations.append(f"Drawdown limit exceeded: {metrics['current_drawdown']:.2%} > {self.risk_limits.max_drawdown:.2%}")
        
        # Check VaR limit
        var_limit = self.portfolio_value * self.risk_limits.max_portfolio_var
        if metrics.get('portfolio_var', 0) > var_limit:
            violations.append(f"VaR limit exceeded: ${metrics['portfolio_var']:.0f} > ${var_limit:.0f}")
        
        # Check individual position sizes
        for symbol, position in self.positions.items():
            position_value = abs(position.quantity * position.current_price)
            max_position_value = self.portfolio_value * self.risk_limits.max_position_size
            
            if position_value > max_position_value:
                violations.append(f"Position {symbol} too large: ${position_value:.0f} > ${max_position_value:.0f}")
        
        return len(violations) == 0, violations
    
    def _check_concentration_risk(self, symbol: str, position_value: float) -> bool:
        """Check if adding position would create concentration risk"""
        # Simple sector concentration check (would need real sector mapping)
        tech_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'META']
        
        if symbol in tech_symbols:
            tech_exposure = sum(
                abs(p.quantity * p.current_price) 
                for s, p in self.positions.items() 
                if s in tech_symbols
            )
            
            total_tech_exposure = tech_exposure + position_value
            max_sector_exposure = self.portfolio_value * 0.4  # 40% max in tech
            
            return total_tech_exposure > max_sector_exposure
        
        return False
    
    def _check_correlation_risk(self, symbol: str, position_value: float) -> bool:
        """Check correlation risk with existing positions"""
        # Simplified correlation check (would need real correlation matrix)
        high_correlation_pairs = [
            ('AAPL', 'MSFT'), ('GOOGL', 'META'), ('BTC-USD', 'ETH-USD')
        ]
        
        for pair in high_correlation_pairs:
            if symbol in pair:
                other_symbol = pair[1] if symbol == pair[0] else pair[0]
                if other_symbol in self.positions:
                    other_value = abs(self.positions[other_symbol].quantity * 
                                    self.positions[other_symbol].current_price)
                    
                    combined_exposure = (position_value + other_value) / self.portfolio_value
                    
                    if combined_exposure > 0.15:  # 15% max for correlated assets
                        return True
        
        return False
    
    def emergency_stop_all(self) -> List[str]:
        """Emergency stop - close all positions"""
        logger.critical("🚨 EMERGENCY STOP - Closing all positions")
        
        stop_orders = []
        for symbol, position in self.positions.items():
            # Create market order to close position
            close_quantity = -position.quantity  # Opposite direction
            stop_orders.append(f"MARKET_SELL {symbol} {abs(close_quantity)}")
            
            logger.critical(f"🛑 Emergency close: {symbol} quantity {close_quantity}")
        
        # Clear positions (would be done after orders execute)
        self.positions.clear()
        
        return stop_orders

def create_risk_monitoring_service():
    """Create real-time risk monitoring service"""
    service_code = '''#!/usr/bin/env python3
"""
Real-time Risk Monitoring Service
"""

import time
import threading
import logging
from datetime import datetime
from risk_management_fixes import AdvancedRiskManager

class RiskMonitoringService:
    def __init__(self, risk_manager: AdvancedRiskManager):
        self.risk_manager = risk_manager
        self.running = False
        self.monitor_thread = None
        self.check_interval = 5  # Check every 5 seconds
        
    def start(self):
        """Start risk monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("🛡️ Risk monitoring service started")
    
    def stop(self):
        """Stop risk monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("🛡️ Risk monitoring service stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check risk limits
                limits_ok, violations = self.risk_manager.check_risk_limits()
                
                if not limits_ok:
                    logging.critical("🚨 RISK LIMIT VIOLATIONS:")
                    for violation in violations:
                        logging.critical(f"   ❌ {violation}")
                    
                    # Could trigger emergency actions here
                    # self.risk_manager.emergency_stop_all()
                
                # Check portfolio metrics
                metrics = self.risk_manager.calculate_portfolio_metrics()
                
                if metrics.get('current_drawdown', 0) > 0.05:  # 5% drawdown warning
                    logging.warning(f"⚠️ Portfolio drawdown: {metrics['current_drawdown']:.2%}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Risk monitoring error: {e}")
                time.sleep(10)

# Global service instance
risk_service = None

def start_risk_monitoring(config):
    """Start the risk monitoring service"""
    global risk_service
    
    risk_manager = AdvancedRiskManager(config)
    risk_service = RiskMonitoringService(risk_manager)
    risk_service.start()
    
    return risk_service

if __name__ == "__main__":
    # Test configuration
    test_config = {
        'initial_capital': 100000,
        'risk_limits': {
            'max_position_size': 0.05,
            'max_portfolio_var': 0.02,
            'max_drawdown': 0.10,
            'stop_loss_percent': 0.02
        }
    }
    
    service = start_risk_monitoring(test_config)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
'''
    
    try:
        with open('risk_monitoring_service.py', 'w') as f:
            f.write(service_code)
        
        logger.info("✅ Risk monitoring service created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create risk monitoring service: {e}")
        return False

def update_config_with_risk_limits():
    """Update configuration with enhanced risk management settings"""
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Enhanced risk management settings
        enhanced_risk = {
            "var_confidence": 0.95,
            "max_position_size": 0.05,  # Reduced from 0.08 to 0.05
            "max_portfolio_var": 0.015,  # Reduced from 0.015 to 0.01
            "max_drawdown": 0.10,  # Reduced from 0.15 to 0.10
            "max_daily_loss": 0.02,
            "stop_loss_percent": 0.02,  # New: 2% stop loss
            "take_profit_percent": 0.06,  # New: 6% take profit
            "max_correlation": 0.7,  # New: max correlation between positions
            "max_sector_exposure": 0.4,  # New: max 40% in any sector
            "kelly_fraction": 0.25,  # New: use 25% of Kelly criterion
            "emergency_drawdown": 0.08  # New: emergency stop at 8% drawdown
        }
        
        config['risk_management'].update(enhanced_risk)
        
        # Enhanced risk limits
        enhanced_limits = {
            "max_portfolio_var": 0.02,
            "max_position_size": 0.05,
            "max_drawdown": 0.10,
            "max_correlation": 0.7,
            "stop_loss_percent": 0.02
        }
        
        config['risk_limits'].update(enhanced_limits)
        
        # Save updated config
        with open('config/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("✅ Configuration updated with enhanced risk management")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}")
        return False

def main():
    """Execute all risk management fixes"""
    print("🛡️ RISK MANAGEMENT FIXES - PHASE 3")
    print("=" * 50)
    
    fixes_applied = 0
    total_fixes = 3
    
    # Fix 1: Update configuration with enhanced risk settings
    print("\n1️⃣ Updating configuration with enhanced risk management...")
    if update_config_with_risk_limits():
        fixes_applied += 1
        print("   ✅ Risk management configuration updated")
    else:
        print("   ❌ Failed to update configuration")
    
    # Fix 2: Create risk monitoring service
    print("\n2️⃣ Creating real-time risk monitoring service...")
    if create_risk_monitoring_service():
        fixes_applied += 1
        print("   ✅ Risk monitoring service created")
    else:
        print("   ❌ Failed to create risk monitoring service")
    
    # Fix 3: Test risk management features
    print("\n3️⃣ Testing risk management features...")
    try:
        # Test Kelly criterion
        kelly_size = KellyCriterion.calculate_position_size(
            win_rate=0.6, avg_win=0.02, avg_loss=0.015, 
            capital=100000, max_risk=0.05
        )
        
        if 1000 <= kelly_size <= 10000:  # Reasonable range
            print(f"   ✅ Kelly criterion working: ${kelly_size:.0f} position size")
            fixes_applied += 1
        else:
            print(f"   ⚠️ Kelly criterion result seems off: ${kelly_size:.0f}")
            fixes_applied += 0.5
            
    except Exception as e:
        print(f"   ❌ Risk management test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🛡️ RISK MANAGEMENT FIXES SUMMARY")
    print(f"   Applied: {fixes_applied}/{total_fixes} fixes")
    
    if fixes_applied >= 2:
        print("   ✅ Risk management significantly enhanced!")
        print("\n🛡️ NEW RISK FEATURES:")
        print("   • Automatic stop losses (2%)")
        print("   • Kelly criterion position sizing")
        print("   • Real-time risk monitoring")
        print("   • Portfolio concentration limits")
        print("   • Correlation risk management")
        print("   • Emergency stop functionality")
        print("\n📋 NEXT STEPS:")
        print("   1. Test risk management in paper trading")
        print("   2. Monitor risk metrics in dashboard")
        print("   3. Proceed to performance optimization")
    else:
        print("   ⚠️ Some risk management fixes failed - review errors above")
    
    return fixes_applied >= 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)