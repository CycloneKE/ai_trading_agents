"""
Event-driven Risk Manager for real-time risk assessment.
Monitors market events, news, and portfolio changes for automatic risk adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time

from .risk_calculator import RiskCalculator

logger = logging.getLogger(__name__)


class EventType(Enum):
    MARKET_DATA = "market_data"
    NEWS_EVENT = "news_event"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_BREACH = "risk_breach"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_CHANGE = "correlation_change"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskEvent:
    """Risk event representation."""
    event_type: EventType
    risk_level: RiskLevel
    symbol: Optional[str]
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    event_id: str


class EventRiskManager:
    """
    Event-driven risk manager for real-time risk assessment and response.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize risk calculator
        self.risk_calculator = RiskCalculator(config.get('risk_calculator', {}))
        
        # Event processing
        self.event_queue = queue.Queue()
        self.risk_events = []
        self.event_handlers = {}
        self.is_running = False
        self.processing_thread = None
        
        # Risk monitoring parameters
        self.volatility_threshold = config.get('volatility_threshold', 0.05)  # 5% daily volatility
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.news_sentiment_threshold = config.get('news_sentiment_threshold', -0.3)
        self.portfolio_var_threshold = config.get('portfolio_var_threshold', 0.02)  # 2% daily VaR
        
        # Market state tracking
        self.current_volatilities = {}
        self.current_correlations = {}
        self.current_portfolio = {}
        self.market_regime = "normal"  # normal, volatile, crisis
        
        # Risk actions
        self.risk_actions = []
        self.auto_hedge_enabled = config.get('auto_hedge_enabled', False)
        self.position_scaling_enabled = config.get('position_scaling_enabled', True)
        
        # Threading
        self.lock = threading.Lock()
        
        # Register default event handlers
        self._register_default_handlers()
        
        logger.info("Event risk manager initialized")
    
    def _register_default_handlers(self):
        """
        Register default event handlers.
        """
        self.register_event_handler(EventType.MARKET_DATA, self._handle_market_data_event)
        self.register_event_handler(EventType.NEWS_EVENT, self._handle_news_event)
        self.register_event_handler(EventType.PORTFOLIO_UPDATE, self._handle_portfolio_update_event)
        self.register_event_handler(EventType.VOLATILITY_SPIKE, self._handle_volatility_spike_event)
        self.register_event_handler(EventType.CORRELATION_CHANGE, self._handle_correlation_change_event)
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """
        Register an event handler for a specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def start(self):
        """
        Start the event risk manager.
        """
        try:
            if self.is_running:
                logger.warning("Event risk manager is already running")
                return
            
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
            self.processing_thread.start()
            
            logger.info("Event risk manager started")
            
        except Exception as e:
            logger.error(f"Error starting event risk manager: {str(e)}")
    
    def stop(self):
        """
        Stop the event risk manager.
        """
        try:
            self.is_running = False
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            
            logger.info("Event risk manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping event risk manager: {str(e)}")
    
    def add_event(self, event: RiskEvent):
        """
        Add a risk event to the processing queue.
        
        Args:
            event: Risk event to process
        """
        try:
            self.event_queue.put(event)
            
        except Exception as e:
            logger.error(f"Error adding event: {str(e)}")
    
    def _process_events(self):
        """
        Process events from the queue.
        """
        while self.is_running:
            try:
                # Get event with timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process event
                self._handle_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
    
    def _handle_event(self, event: RiskEvent):
        """
        Handle a risk event.
        
        Args:
            event: Risk event to handle
        """
        try:
            # Store event
            with self.lock:
                self.risk_events.append(event)
                
                # Keep only recent events
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.risk_events = [
                    e for e in self.risk_events
                    if e.timestamp > cutoff_time
                ]
            
            # Call registered handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")
            
            # Log high-risk events
            if event.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                logger.warning(f"High-risk event: {event.message}")
            
        except Exception as e:
            logger.error(f"Error handling event: {str(e)}")
    
    def _handle_market_data_event(self, event: RiskEvent):
        """
        Handle market data events.
        
        Args:
            event: Market data event
        """
        try:
            data = event.data
            symbol = event.symbol
            
            if not symbol or 'price_data' not in data:
                return
            
            price_data = data['price_data']
            
            # Calculate volatility
            if isinstance(price_data, pd.DataFrame) and 'close' in price_data.columns:
                returns = price_data['close'].pct_change().dropna()
                
                if len(returns) >= 20:  # Need minimum data
                    current_volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                    
                    with self.lock:
                        self.current_volatilities[symbol] = current_volatility
                    
                    # Check for volatility spike
                    if current_volatility > self.volatility_threshold:
                        volatility_event = RiskEvent(
                            event_type=EventType.VOLATILITY_SPIKE,
                            risk_level=RiskLevel.HIGH if current_volatility > self.volatility_threshold * 2 else RiskLevel.MEDIUM,
                            symbol=symbol,
                            message=f"Volatility spike detected for {symbol}: {current_volatility:.3f}",
                            data={'volatility': current_volatility, 'threshold': self.volatility_threshold},
                            timestamp=datetime.utcnow(),
                            event_id=f"vol_spike_{symbol}_{int(time.time())}"
                        )
                        self.add_event(volatility_event)
            
        except Exception as e:
            logger.error(f"Error handling market data event: {str(e)}")
    
    def _handle_news_event(self, event: RiskEvent):
        """
        Handle news events.
        
        Args:
            event: News event
        """
        try:
            data = event.data
            sentiment = data.get('sentiment', {})
            
            # Check sentiment impact
            overall_sentiment = sentiment.get('overall', 0)
            confidence = sentiment.get('confidence', 0)
            
            if overall_sentiment < self.news_sentiment_threshold and confidence > 0.7:
                # Negative news with high confidence
                risk_level = RiskLevel.HIGH if overall_sentiment < -0.5 else RiskLevel.MEDIUM
                
                # Generate risk action
                risk_action = {
                    'action_type': 'reduce_exposure',
                    'symbol': event.symbol,
                    'reason': 'negative_news_sentiment',
                    'sentiment_score': overall_sentiment,
                    'confidence': confidence,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                with self.lock:
                    self.risk_actions.append(risk_action)
                
                logger.warning(f"Negative news sentiment for {event.symbol}: {overall_sentiment:.3f}")
            
        except Exception as e:
            logger.error(f"Error handling news event: {str(e)}")
    
    def _handle_portfolio_update_event(self, event: RiskEvent):
        """
        Handle portfolio update events.
        
        Args:
            event: Portfolio update event
        """
        try:
            data = event.data
            portfolio = data.get('portfolio', {})
            
            with self.lock:
                self.current_portfolio = portfolio.copy()
            
            # Calculate portfolio risk metrics
            if 'positions' in portfolio and 'price_data' in data:
                positions = portfolio['positions']
                price_data = data['price_data']
                
                portfolio_risk = self.risk_calculator.calculate_portfolio_risk(positions, price_data)
                
                # Check for risk breaches
                var_95 = portfolio_risk.get('var_metrics', {}).get('var_95', 0)
                
                if abs(var_95) > self.portfolio_var_threshold:
                    risk_breach_event = RiskEvent(
                        event_type=EventType.RISK_BREACH,
                        risk_level=RiskLevel.CRITICAL,
                        symbol=None,
                        message=f"Portfolio VaR breach: {abs(var_95):.3f} > {self.portfolio_var_threshold:.3f}",
                        data={'var_95': var_95, 'threshold': self.portfolio_var_threshold, 'portfolio_risk': portfolio_risk},
                        timestamp=datetime.utcnow(),
                        event_id=f"var_breach_{int(time.time())}"
                    )
                    self.add_event(risk_breach_event)
            
        except Exception as e:
            logger.error(f"Error handling portfolio update event: {str(e)}")
    
    def _handle_volatility_spike_event(self, event: RiskEvent):
        """
        Handle volatility spike events.
        
        Args:
            event: Volatility spike event
        """
        try:
            symbol = event.symbol
            volatility = event.data.get('volatility', 0)
            
            if self.position_scaling_enabled:
                # Generate position scaling action
                scaling_factor = min(0.5, self.volatility_threshold / volatility)  # Reduce position
                
                risk_action = {
                    'action_type': 'scale_position',
                    'symbol': symbol,
                    'scaling_factor': scaling_factor,
                    'reason': 'volatility_spike',
                    'volatility': volatility,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                with self.lock:
                    self.risk_actions.append(risk_action)
                
                logger.info(f"Generated position scaling action for {symbol}: scale by {scaling_factor:.3f}")
            
            # Update market regime
            high_vol_symbols = sum(1 for vol in self.current_volatilities.values() if vol > self.volatility_threshold)
            total_symbols = len(self.current_volatilities)
            
            if total_symbols > 0:
                vol_ratio = high_vol_symbols / total_symbols
                
                if vol_ratio > 0.5:
                    self.market_regime = "volatile"
                elif vol_ratio > 0.8:
                    self.market_regime = "crisis"
                else:
                    self.market_regime = "normal"
            
        except Exception as e:
            logger.error(f"Error handling volatility spike event: {str(e)}")
    
    def _handle_correlation_change_event(self, event: RiskEvent):
        """
        Handle correlation change events.
        
        Args:
            event: Correlation change event
        """
        try:
            data = event.data
            correlation_matrix = data.get('correlation_matrix', [])
            
            if correlation_matrix:
                # Check for high correlations
                correlation_array = np.array(correlation_matrix)
                
                # Get upper triangle (excluding diagonal)
                upper_triangle = correlation_array[np.triu_indices_from(correlation_array, k=1)]
                max_correlation = np.max(upper_triangle) if len(upper_triangle) > 0 else 0
                
                if max_correlation > self.correlation_threshold:
                    risk_action = {
                        'action_type': 'diversify_portfolio',
                        'reason': 'high_correlation',
                        'max_correlation': max_correlation,
                        'threshold': self.correlation_threshold,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    with self.lock:
                        self.risk_actions.append(risk_action)
                    
                    logger.warning(f"High correlation detected: {max_correlation:.3f}")
            
        except Exception as e:
            logger.error(f"Error handling correlation change event: {str(e)}")
    
    def process_market_data(self, symbol: str, price_data: pd.DataFrame):
        """
        Process new market data and generate events if needed.
        
        Args:
            symbol: Trading symbol
            price_data: Price data DataFrame
        """
        try:
            event = RiskEvent(
                event_type=EventType.MARKET_DATA,
                risk_level=RiskLevel.LOW,
                symbol=symbol,
                message=f"Market data update for {symbol}",
                data={'price_data': price_data},
                timestamp=datetime.utcnow(),
                event_id=f"market_data_{symbol}_{int(time.time())}"
            )
            
            self.add_event(event)
            
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
    
    def process_news_sentiment(self, symbol: str, sentiment_data: Dict[str, Any]):
        """
        Process news sentiment and generate events if needed.
        
        Args:
            symbol: Trading symbol
            sentiment_data: Sentiment analysis results
        """
        try:
            overall_sentiment = sentiment_data.get('combined', {}).get('overall', 0)
            confidence = sentiment_data.get('combined', {}).get('confidence', 0)
            
            risk_level = RiskLevel.LOW
            if abs(overall_sentiment) > 0.3 and confidence > 0.7:
                risk_level = RiskLevel.MEDIUM
            if abs(overall_sentiment) > 0.5 and confidence > 0.8:
                risk_level = RiskLevel.HIGH
            
            event = RiskEvent(
                event_type=EventType.NEWS_EVENT,
                risk_level=risk_level,
                symbol=symbol,
                message=f"News sentiment update for {symbol}: {overall_sentiment:.3f}",
                data={'sentiment': sentiment_data},
                timestamp=datetime.utcnow(),
                event_id=f"news_{symbol}_{int(time.time())}"
            )
            
            self.add_event(event)
            
        except Exception as e:
            logger.error(f"Error processing news sentiment: {str(e)}")
    
    def process_portfolio_update(self, portfolio: Dict[str, Any], price_data: Dict[str, pd.DataFrame]):
        """
        Process portfolio update and generate events if needed.
        
        Args:
            portfolio: Portfolio state
            price_data: Price data for all symbols
        """
        try:
            event = RiskEvent(
                event_type=EventType.PORTFOLIO_UPDATE,
                risk_level=RiskLevel.LOW,
                symbol=None,
                message="Portfolio update",
                data={'portfolio': portfolio, 'price_data': price_data},
                timestamp=datetime.utcnow(),
                event_id=f"portfolio_update_{int(time.time())}"
            )
            
            self.add_event(event)
            
        except Exception as e:
            logger.error(f"Error processing portfolio update: {str(e)}")
    
    def get_risk_actions(self, clear_after_read: bool = True) -> List[Dict[str, Any]]:
        """
        Get pending risk actions.
        
        Args:
            clear_after_read: Whether to clear actions after reading
            
        Returns:
            List of risk actions
        """
        with self.lock:
            actions = self.risk_actions.copy()
            if clear_after_read:
                self.risk_actions.clear()
        
        return actions
    
    def get_risk_events(self, event_type: Optional[EventType] = None, 
                       risk_level: Optional[RiskLevel] = None,
                       hours_back: int = 24) -> List[RiskEvent]:
        """
        Get risk events with optional filtering.
        
        Args:
            event_type: Filter by event type
            risk_level: Filter by risk level
            hours_back: Hours to look back
            
        Returns:
            List of filtered risk events
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        with self.lock:
            events = [e for e in self.risk_events if e.timestamp > cutoff_time]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if risk_level:
            events = [e for e in events if e.risk_level == risk_level]
        
        return events
    
    def get_market_regime(self) -> str:
        """
        Get current market regime assessment.
        
        Returns:
            Market regime string
        """
        return self.market_regime
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get event risk manager status.
        
        Returns:
            Status information
        """
        with self.lock:
            return {
                'is_running': self.is_running,
                'market_regime': self.market_regime,
                'pending_events': self.event_queue.qsize(),
                'total_events_24h': len(self.get_risk_events(hours_back=24)),
                'pending_actions': len(self.risk_actions),
                'monitored_symbols': len(self.current_volatilities),
                'high_volatility_symbols': len([s for s, v in self.current_volatilities.items() 
                                              if v > self.volatility_threshold]),
                'risk_thresholds': {
                    'volatility_threshold': self.volatility_threshold,
                    'correlation_threshold': self.correlation_threshold,
                    'news_sentiment_threshold': self.news_sentiment_threshold,
                    'portfolio_var_threshold': self.portfolio_var_threshold
                }
            }

