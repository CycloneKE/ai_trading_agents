"""
Real-time Risk Management Engine
Monitors and controls trading risk in real-time with automatic position sizing and limits
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    portfolio_var: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation: float
    concentration_risk: float
    leverage: float

@dataclass
class RiskLimit:
    name: str
    limit_type: str  # 'portfolio_var', 'position_size', 'drawdown', etc.
    threshold: float
    action: str  # 'warn', 'reduce', 'stop'
    enabled: bool = True

class RealTimeRiskManager:
    """Real-time risk management with automatic controls"""
    
    def __init__(self, config: Dict[str, Any], database_manager=None):
        self.config = config
        self.database = database_manager
        
        # Risk limits
        self.risk_limits = self._initialize_risk_limits()
        
        # Risk metrics tracking
        self.current_metrics = RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        self.risk_history = []
        
        # Position tracking
        self.positions = {}
        self.portfolio_value = 0.0
        self.cash = 0.0
        
        # Risk parameters
        self.lookback_period = config.get('lookback_period', 252)  # 1 year
        self.confidence_level = config.get('confidence_level', 0.95)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        
        # Emergency controls
        self.emergency_stop = False
        self.risk_alerts = []
        # Prometheus counters (optional)
        try:
            from prometheus_client import Counter
            self.trade_attempts_counter = Counter('trade_attempts_total', 'Total trade attempts')
            self.trade_success_counter = Counter('trade_success_total', 'Total successful trades')
        except Exception:
            # prometheus_client not installed or failed to import; use no-op placeholders
            self.trade_attempts_counter = None
            self.trade_success_counter = None
        
    def _initialize_risk_limits(self) -> List[RiskLimit]:
        """Initialize risk limits from configuration"""
        limits_config = self.config.get('risk_limits', {})
        
        default_limits = [
            RiskLimit('max_portfolio_var', 'portfolio_var', 0.02, 'reduce'),
            RiskLimit('max_position_size', 'position_size', 0.1, 'reduce'),
            RiskLimit('max_drawdown', 'drawdown', 0.15, 'stop'),
            RiskLimit('max_leverage', 'leverage', 2.0, 'reduce'),
            RiskLimit('max_concentration', 'concentration', 0.3, 'warn'),
            RiskLimit('min_sharpe', 'sharpe_ratio', 0.5, 'warn'),
        ]
        
        # Override with config values
        for limit in default_limits:
            if limit.name in limits_config:
                limit.threshold = limits_config[limit.name]
        
        return default_limits
    
    def pre_trade_risk_check(self, symbol: str, side: str, quantity: float, price: float) -> Tuple[bool, str, float]:
        """
        Perform pre-trade risk check
        
        Returns:
            (approved, reason, adjusted_quantity)
        """
        try:
            # Calculate proposed position value
            position_value = quantity * price
            
            # Check if emergency stop is active
            if self.emergency_stop:
                return False, "Emergency stop active", 0.0
            
            # For sells, never allow disposing of more than is currently held —
            # an oversell silently corrupts the position/P&L downstream.
            if side == 'sell':
                held_quantity = self.positions.get(symbol, {}).get('quantity', 0)
                if quantity > held_quantity:
                    if held_quantity <= 0:
                        return False, f"Cannot sell {symbol}: no position held", 0.0
                    return True, f"Sell reduced to held quantity {held_quantity}", held_quantity

            # Check position size limits against the PROJECTED total position
            # (existing holdings + this order), not the order in isolation —
            # otherwise repeated sub-limit buys accumulate past the limit.
            max_position_limit = self._get_limit_threshold('max_position_size')
            max_position_value = self.portfolio_value * max_position_limit
            if side == 'buy':
                existing_value = self.positions.get(symbol, {}).get('market_value', 0)
                projected_position_value = existing_value + position_value
                if projected_position_value > max_position_value:
                    # Maximum ADDITIONAL quantity that keeps the total within the limit
                    remaining_value = max(0.0, max_position_value - existing_value)
                    max_quantity = remaining_value / price
                    if max_quantity < quantity * 0.1:  # Less than 10% of intended
                        return False, f"Position too large: ${projected_position_value:.2f} > ${max_position_value:.2f}", 0.0
                    else:
                        return True, "Position size reduced due to limits", max_quantity
            
            # Check portfolio VaR impact
            projected_var = self._calculate_projected_var(symbol, side, quantity, price)
            max_var_limit = self._get_limit_threshold('max_portfolio_var')
            
            if projected_var > max_var_limit:
                # Calculate quantity that keeps VaR within limits
                safe_quantity = self._calculate_safe_quantity(symbol, side, price, max_var_limit)
                if safe_quantity < quantity * 0.1:
                    return False, f"VaR impact too high: {projected_var:.4f} > {max_var_limit:.4f}", 0.0
                else:
                    return True, f"Quantity reduced to maintain VaR limits", safe_quantity
            
            # Check concentration risk
            current_concentration = self._calculate_concentration_risk(symbol, quantity, price)
            max_concentration = self._get_limit_threshold('max_concentration')
            
            if current_concentration > max_concentration:
                return False, f"Concentration risk too high: {current_concentration:.2f} > {max_concentration:.2f}", 0.0
            
            # Check leverage
            projected_leverage = self._calculate_projected_leverage(quantity, price)
            max_leverage = self._get_limit_threshold('max_leverage')
            
            if projected_leverage > max_leverage:
                return False, f"Leverage too high: {projected_leverage:.2f} > {max_leverage:.2f}", 0.0
            
            # All checks passed
            return True, "Risk check passed", quantity
            
        except Exception as e:
            logger.error(f"Pre-trade risk check error: {e}")
            return False, f"Risk check error: {str(e)}", 0.0
    
    def post_trade_risk_update(self, symbol: str, side: str, quantity: float, price: float):
        """Update risk metrics after trade execution"""
        try:
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0, 'avg_price': 0, 'market_value': 0, 'high_watermark': 0.0}
            
            position = self.positions[symbol]
            
            if side == 'buy':
                new_quantity = position['quantity'] + quantity
                if new_quantity != 0:
                    position['avg_price'] = ((position['quantity'] * position['avg_price']) + 
                                           (quantity * price)) / new_quantity
                position['quantity'] = new_quantity
            else:  # sell
                if quantity > position['quantity']:
                    logger.warning(
                        f"Sell quantity {quantity} exceeds held {position['quantity']} "
                        f"for {symbol}; clamping to held quantity."
                    )
                    quantity = position['quantity']
                position['quantity'] -= quantity
                if position['quantity'] <= 0:
                    position['quantity'] = 0
                    position['avg_price'] = 0
                    position['high_watermark'] = 0.0
            
            # Update high watermark on buy
            if position['quantity'] > 0 and position['avg_price'] > 0:
                if position['high_watermark'] == 0.0:
                    position['high_watermark'] = position['avg_price']
            
            # Update market value (would need current market price in real implementation)
            position['market_value'] = position['quantity'] * price
            
            # Recalculate portfolio metrics
            self._update_portfolio_metrics()
            
            # Check for risk limit breaches
            self._check_risk_limits()
            
        except Exception as e:
            logger.error(f"Post-trade risk update error: {e}")

    def record_trade_attempt(self) -> None:
        """Record that a trade was attempted (increments Prometheus counter if available)."""
        try:
            counter = getattr(self, 'trade_attempts_counter', None)
            inc = getattr(counter, 'inc', None)
            if callable(inc):
                inc()
        except Exception:
            # Safe no-op on any failure
            pass

    def record_trade_success(self) -> None:
        """Record that a trade completed successfully (increments Prometheus counter if available)."""
        try:
            counter = getattr(self, 'trade_success_counter', None)
            inc = getattr(counter, 'inc', None)
            if callable(inc):
                inc()
        except Exception:
            pass
    
    def _calculate_projected_var(self, symbol: str, side: str, quantity: float, price: float) -> float:
        """Calculate projected portfolio VaR after trade"""
        try:
            # Simplified VaR calculation
            # In production, this would use historical returns and correlations
            
            position_value = quantity * price
            portfolio_value = self.portfolio_value or 100000  # Default if not set
            
            # Estimate volatility (simplified)
            estimated_volatility = 0.02  # 2% daily volatility
            
            # Calculate position weight
            if side == 'buy':
                new_portfolio_value = portfolio_value + position_value
                position_weight = position_value / new_portfolio_value
            else:
                position_weight = -position_value / portfolio_value
            
            # Simplified VaR calculation
            z_score = 1.645  # 95% confidence level
            projected_var = abs(position_weight) * estimated_volatility * z_score
            
            return projected_var
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0
    
    def _calculate_safe_quantity(self, symbol: str, side: str, price: float, max_var: float) -> float:
        """Calculate maximum safe quantity to stay within VaR limits"""
        try:
            estimated_volatility = 0.02
            z_score = 1.645
            
            # Calculate maximum position weight that keeps VaR within limits
            max_position_weight = max_var / (estimated_volatility * z_score)
            
            # Convert to quantity
            max_position_value = max_position_weight * self.portfolio_value
            safe_quantity = max_position_value / price
            
            return max(0, safe_quantity)
            
        except Exception as e:
            logger.error(f"Safe quantity calculation error: {e}")
            return 0.0
    
    def _calculate_concentration_risk(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate concentration risk for symbol"""
        try:
            position_value = quantity * price
            current_position_value = self.positions.get(symbol, {}).get('market_value', 0)
            total_position_value = current_position_value + position_value
            
            concentration = total_position_value / (self.portfolio_value or 1)
            return concentration
            
        except Exception as e:
            logger.error(f"Concentration risk calculation error: {e}")
            return 0.0
    
    def _calculate_projected_leverage(self, quantity: float, price: float) -> float:
        """Calculate projected leverage after trade"""
        try:
            position_value = quantity * price
            total_position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
            projected_total = total_position_value + position_value
            
            leverage = projected_total / (self.cash or 1)
            return leverage
            
        except Exception as e:
            logger.error(f"Leverage calculation error: {e}")
            return 1.0
    
    def _update_portfolio_metrics(self):
        """Update current portfolio risk metrics"""
        try:
            # Calculate portfolio value
            self.portfolio_value = sum(pos.get('market_value', 0) for pos in self.positions.values()) + self.cash
            
            # Calculate basic metrics (simplified)
            total_position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
            
            # Portfolio VaR (simplified)
            portfolio_var = 0.02 * (total_position_value / self.portfolio_value) if self.portfolio_value > 0 else 0
            
            # Concentration risk
            max_position_value = max((pos.get('market_value', 0) for pos in self.positions.values()), default=0)
            concentration_risk = max_position_value / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Leverage
            leverage = total_position_value / self.cash if self.cash > 0 else 1.0
            
            # Update current metrics
            self.current_metrics = RiskMetrics(
                portfolio_var=portfolio_var,
                max_drawdown=0.0,  # Would need historical data
                sharpe_ratio=0.0,  # Would need return history
                volatility=0.02,   # Simplified
                beta=1.0,          # Simplified
                correlation=0.0,   # Simplified
                concentration_risk=concentration_risk,
                leverage=leverage
            )
            
            # Store in history
            self.risk_history.append({
                'timestamp': datetime.utcnow(),
                'metrics': self.current_metrics
            })
            
            # Keep only recent history
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
                
        except Exception as e:
            logger.error(f"Portfolio metrics update error: {e}")
    
    def _check_risk_limits(self):
        """Check current metrics against risk limits"""
        try:
            alerts = []
            
            for limit in self.risk_limits:
                if not limit.enabled:
                    continue
                
                current_value = self._get_current_metric_value(limit.limit_type)
                
                if current_value > limit.threshold:
                    alert = {
                        'timestamp': datetime.utcnow(),
                        'limit_name': limit.name,
                        'limit_type': limit.limit_type,
                        'current_value': current_value,
                        'threshold': limit.threshold,
                        'action': limit.action,
                        'severity': self._calculate_severity(current_value, limit.threshold)
                    }
                    
                    alerts.append(alert)
                    
                    # Take action based on limit
                    if limit.action == 'stop':
                        self.emergency_stop = True
                        logger.critical(f"EMERGENCY STOP: {limit.name} breach - {current_value} > {limit.threshold}")
                    elif limit.action == 'reduce':
                        logger.warning(f"Risk limit breach: {limit.name} - {current_value} > {limit.threshold}")
                    elif limit.action == 'warn':
                        logger.info(f"Risk warning: {limit.name} - {current_value} > {limit.threshold}")
            
            # Update alerts
            self.risk_alerts.extend(alerts)
            
            # Keep only recent alerts
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.risk_alerts = [a for a in self.risk_alerts if a['timestamp'] > cutoff_time]
            
        except Exception as e:
            logger.error(f"Risk limit check error: {e}")
    
    def _get_current_metric_value(self, metric_type: str) -> float:
        """Get current value for a specific metric type"""
        metric_map = {
            'portfolio_var': self.current_metrics.portfolio_var,
            'position_size': max((pos.get('market_value', 0) / self.portfolio_value 
                                for pos in self.positions.values()), default=0),
            'drawdown': self.current_metrics.max_drawdown,
            'leverage': self.current_metrics.leverage,
            'concentration': self.current_metrics.concentration_risk,
            'sharpe_ratio': self.current_metrics.sharpe_ratio,
        }
        
        return metric_map.get(metric_type, 0.0)
    
    def _get_limit_threshold(self, limit_name: str) -> float:
        """Get threshold for a specific limit"""
        for limit in self.risk_limits:
            if limit.name == limit_name:
                return limit.threshold
        return 0.0
    
    def _calculate_severity(self, current_value: float, threshold: float) -> RiskLevel:
        """Calculate risk severity based on how much threshold is exceeded"""
        ratio = current_value / threshold if threshold > 0 else 1.0
        
        if ratio >= 2.0:
            return RiskLevel.CRITICAL
        elif ratio >= 1.5:
            return RiskLevel.HIGH
        elif ratio >= 1.2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def update_market_data(self, market_data: Dict[str, Any]):
        """Update risk calculations with new market data"""
        try:
            # Update position market values and high watermarks
            for symbol, position in self.positions.items():
                if symbol in market_data and position['quantity'] > 0:
                    current_price = market_data[symbol].get('price', position['avg_price'])
                    position['market_value'] = position['quantity'] * current_price
                    
                    if current_price > position['high_watermark']:
                        position['high_watermark'] = current_price
            
            # Recalculate metrics
            self._update_portfolio_metrics()
            self._check_risk_limits()
            
        except Exception as e:
            logger.error(f"Market data update error: {e}")
            
    def check_trailing_stops(self, trailing_stop_pct: float) -> List[Dict[str, Any]]:
        """
        Check if any positions have triggered their trailing stop loss.
        Returns a list of symbols and actions to close.
        """
        stop_actions = []
        try:
            for symbol, position in self.positions.items():
                if position['quantity'] > 0 and position['market_value'] > 0:
                    current_price = position['market_value'] / position['quantity']
                    high = position['high_watermark']
                    
                    if high > 0 and current_price < high * (1.0 - trailing_stop_pct):
                        logger.warning(
                            f"TRAILING STOP TRIGGERED for {symbol}: "
                            f"Current={current_price:.2f}, HighWM={high:.2f}, Drop={(high - current_price)/high:.2%}"
                        )
                        stop_actions.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'quantity': position['quantity'],
                            'reason': 'trailing_stop_loss'
                        })
                        
            return stop_actions
        except Exception as e:
            logger.error(f"Trailing stop check error: {e}")
            return []
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            # Calculate additional metrics
            total_alerts = len(self.risk_alerts)
            critical_alerts = len([a for a in self.risk_alerts if a['severity'] == RiskLevel.CRITICAL])
            
            # Position breakdown
            position_breakdown = []
            for symbol, position in self.positions.items():
                if position['quantity'] > 0:
                    weight = position['market_value'] / self.portfolio_value if self.portfolio_value > 0 else 0
                    position_breakdown.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'market_value': position['market_value'],
                        'weight': weight,
                        'avg_price': position['avg_price']
                    })
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'emergency_stop': self.emergency_stop,
                'current_metrics': {
                    'portfolio_var': self.current_metrics.portfolio_var,
                    'max_drawdown': self.current_metrics.max_drawdown,
                    'sharpe_ratio': self.current_metrics.sharpe_ratio,
                    'volatility': self.current_metrics.volatility,
                    'leverage': self.current_metrics.leverage,
                    'concentration_risk': self.current_metrics.concentration_risk
                },
                'risk_limits': [
                    {
                        'name': limit.name,
                        'threshold': limit.threshold,
                        'current_value': self._get_current_metric_value(limit.limit_type),
                        'utilization': self._get_current_metric_value(limit.limit_type) / limit.threshold if limit.threshold > 0 else 0,
                        'status': 'breach' if self._get_current_metric_value(limit.limit_type) > limit.threshold else 'ok'
                    }
                    for limit in self.risk_limits
                ],
                'alerts': {
                    'total': total_alerts,
                    'critical': critical_alerts,
                    'recent': self.risk_alerts[-10:] if self.risk_alerts else []
                },
                'positions': position_breakdown
            }
            
        except Exception as e:
            logger.error(f"Risk report generation error: {e}")
            return {'error': str(e)}
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop after manual review"""
        try:
            # Check if conditions are safe to resume
            critical_breaches = [a for a in self.risk_alerts 
                               if a['severity'] == RiskLevel.CRITICAL and 
                               a['timestamp'] > datetime.utcnow() - timedelta(minutes=30)]
            
            if critical_breaches:
                logger.warning("Cannot reset emergency stop: critical risk breaches still active")
                return False
            
            self.emergency_stop = False
            logger.info("Emergency stop reset")
            return True
            
        except Exception as e:
            logger.error(f"Emergency stop reset error: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk manager status"""
        status = {
            'emergency_stop': self.emergency_stop,
            'active_positions': len([p for p in self.positions.values() if p['quantity'] > 0]),
            'portfolio_value': self.portfolio_value,
            'recent_alerts': len([a for a in self.risk_alerts 
                                if a['timestamp'] > datetime.utcnow() - timedelta(hours=1)]),
            'risk_limits_count': len(self.risk_limits),
            'current_var': self.current_metrics.portfolio_var,
            'current_leverage': self.current_metrics.leverage
        }

        # Expose simple counters if available (safe to call)
        try:
            if getattr(self, 'trade_attempts_counter', None) is not None:
                # prometheus client Counter has _value.get() internals; access safely if present
                status['trade_attempts'] = float(getattr(self.trade_attempts_counter, '_value').get())
        except Exception:
            # best-effort; ignore if internals differ
            pass

        try:
            if getattr(self, 'trade_success_counter', None) is not None:
                status['trade_successes'] = float(getattr(self.trade_success_counter, '_value').get())
        except Exception:
            pass

        return status