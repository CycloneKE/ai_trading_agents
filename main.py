def validate_config(config: dict) -> bool:
    """
    Validate the configuration structure and required fields.
    Returns True if valid, False otherwise. Logs errors for missing/invalid fields.
    """
    import logging
    required_top_keys = [
        'data_manager', 'strategies', 'risk_management', 'risk_limits',
        'brokers', 'trading', 'monitoring', 'security', 'logging'
    ]
    valid = True
    for key in required_top_keys:
        if key not in config:
            logging.error(f"Missing required config section: {key}")
            valid = False
    # Check critical subkeys
    dm = config.get('data_manager', {})
    if not dm.get('symbols') or not isinstance(dm['symbols'], list):
        logging.error("data_manager.symbols must be a non-empty list")
        valid = False
    if not dm.get('connectors') or not isinstance(dm['connectors'], dict):
        logging.error("data_manager.connectors must be a dict")
        valid = False
    brokers = config.get('brokers', {})
    if not brokers or not isinstance(brokers, dict):
        logging.error("brokers must be a dict with at least one broker defined")
        valid = False
    trading = config.get('trading', {})
    if 'initial_capital' not in trading:
        logging.error("trading.initial_capital is required")
        valid = False
    # Add more checks as needed for production
    if not valid:
        logging.error("Configuration validation failed. See errors above.")
    return valid
#!/usr/bin/env python3
"""
AI Trading Agent - Main Application Entry Point

This is the main entry point for the AI Trading Agent system.
It initializes all components and starts the trading engine.
"""

import sys
import os
import argparse
import json
import logging
import signal
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging first
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import local modules
from config_validator import load_config
from secure_config import SecureConfigManager
from database_manager import DatabaseManager
from realtime_data_feed import RealTimeDataFeed
from monitoring import get_monitoring_service
from data_manager import DataManager
from strategy_manager import StrategyManager
from broker_manager import BrokerManager
from order_execution_engine import OrderExecutionEngine
from realtime_risk_manager import RealTimeRiskManager
from performance_analytics import PerformanceAnalytics
try:
    from api_server import TradingAPI
    API_AVAILABLE = True
except ImportError:
    logger.warning("Flask not available - API server disabled")
    API_AVAILABLE = False
    TradingAPI = None

# These modules need to be implemented
class RiskCalculator:
    def __init__(self, config):
        self.config = config
    
    def assess_portfolio_risk(self, market_data, signals):
        return {'portfolio_var': 0.01, 'position_sizes': {}, 'current_drawdown': 0.0}
    
    def get_status(self):
        return {'status': 'ok'}

class EventRiskManager:
    def __init__(self, config):
        self.config = config
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def get_status(self):
        return {'status': 'ok'}

class NLPManager:
    def __init__(self, config):
        self.config = config
    
    def get_status(self):
        return {'status': 'ok'}

class AutoMLOptimizer:
    def __init__(self, config):
        self.config = config
    
    def get_status(self):
        return {'status': 'ok'}

class PortfolioOptimizer:
    def __init__(self, config):
        self.config = config
    
    def optimize_portfolio(self, market_data, optimization_method='mean_variance'):
        return {'weights': {}, 'expected_return': 0.0, 'expected_risk': 0.0}
    
    def get_status(self):
        return {'status': 'ok'}




class TradingAgent:
    """
    Main AI Trading Agent application.
    """
    
    def __init__(self, config_path: str):
        self._validate_secrets()
        self.config_path = config_path
        self.config = load_config(self.config_path)
        # Validate configuration before proceeding
        if not validate_config(self.config):
            print("ERROR: Invalid configuration. See logs for details.")
            sys.exit(1)
        self.running = False
        self.components = {}
        self.secure_config = SecureConfigManager()
        # Initialize database (optional)
        try:
            db_config = self.secure_config.get_database_config()
            self.database = DatabaseManager(db_config)
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            self.database = None
        # Initialize monitoring service
        self.monitoring_service = get_monitoring_service(self.config)
        # Initialize components
        self._initialize_components()
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("AI Trading Agent initialized successfully")

    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing system components...")
            
            # Start monitoring service first for health checks
            if self.monitoring_service:
                monitoring_port = self.config.get('monitoring', {}).get('port', 8080)
                self.monitoring_service.start(port=monitoring_port)
                logger.info(f"Monitoring service started on port {monitoring_port}")
            
            # Broker Integration (initialize first)
            self.components['broker_manager'] = BrokerManager(self.config)
            
            # Data Management
            self.components['data_manager'] = DataManager(
                self.config.get('data_manager', {})
            )
            
            # Real-time Data Feed
            self.components['realtime_feed'] = RealTimeDataFeed(
                self.config.get('realtime_data', {})
            )
            
            # Order Execution Engine
            self.execution_engine = OrderExecutionEngine(
                self.config.get('execution', {}),
                self.components['broker_manager']
            )
            
            # Real-time Risk Manager
            self.risk_manager = RealTimeRiskManager(
                self.config.get('risk_management', {}),
                self.database
            )
            
            # Performance Analytics
            self.performance_analytics = PerformanceAnalytics(
                self.config.get('analytics', {}),
                self.database
            )
            
            # API Server (conditional)
            if API_AVAILABLE and TradingAPI is not None:
                self.api_server = TradingAPI(
                    self,
                    self.config.get('api', {})
                )
            else:
                self.api_server = None
                logger.info("API server disabled - Flask not available")
            
            # Strategy Management
            self.components['strategy_manager'] = StrategyManager(
                self.config
            )
            
            # Risk Management
            self.components['risk_calculator'] = RiskCalculator(
                self.config.get('risk_management', {})
            )
            
            self.components['event_risk_manager'] = EventRiskManager(
                self.config.get('event_risk', {})
            )
            

            
            # NLP Engine
            self.components['nlp_manager'] = NLPManager(
                self.config.get('nlp', {})
            )
            
            # Advanced Features
            self.components['automl_optimizer'] = AutoMLOptimizer(
                self.config.get('automl', {})
            )
            
            self.components['portfolio_optimizer'] = PortfolioOptimizer(
                self.config.get('portfolio_optimization', {})
            )
            
            # Register health checks with monitoring service
            if self.monitoring_service:
                for name, component in self.components.items():
                    if hasattr(component, 'get_status'):
                        self.monitoring_service.register_health_check(
                            name, component.get_status
                        )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        logger.info("Trading agent stopped.")

    def stop(self):
        """Stop the trading agent and clean up resources."""
        self.running = False
        logger.info("Trading agent stopping...")
        # Attempt to gracefully stop threads (if any custom threads are tracked)
        # Example: if hasattr(self, 'broker_health_thread'): self.broker_health_thread.join(timeout=5)
        # Stop data manager and event risk manager if they have stop methods
        try:
            if 'data_manager' in self.components and hasattr(self.components['data_manager'], 'stop'):
                self.components['data_manager'].stop()
            if 'event_risk_manager' in self.components and hasattr(self.components['event_risk_manager'], 'stop'):
                self.components['event_risk_manager'].stop()
        except Exception as e:
            logger.warning(f"Error stopping components: {e}")
        # Close database connection if present
        try:
            if hasattr(self, 'database') and self.database:
                if hasattr(self.database, 'close'):
                    self.database.close()
        except Exception as e:
            logger.warning(f"Error closing database: {e}")
        logger.info("Trading agent stopped and resources cleaned up.")
    
    def start(self):
        """Start the trading agent with full automation."""
        logger.info("Starting AI Trading Agent...")
        self.running = True

        # Connect to all brokers and run health checks
        broker_manager = self.components['broker_manager']
        connection_results = broker_manager.connect_all()
        for broker_name, connected in connection_results.items():
            if connected:
                logger.info(f"Connected to {broker_name}")
            else:
                logger.error(f"✗ Failed to connect to {broker_name}")

        # Health check loop (threaded)
        def broker_health_loop():
            while self.running:
                for name, broker in broker_manager.brokers.items():
                    if hasattr(broker, 'get_status'):
                        status = broker.get_status()
                        if not status.get('is_connected', False):
                            logger.warning(f"Broker {name} disconnected, attempting reconnect...")
                            broker.connect()
                time.sleep(60)
        threading.Thread(target=broker_health_loop, daemon=True).start()

        # Failover logic: switch to backup broker if primary fails
        def failover_monitor():
            while self.running:
                primary = broker_manager.get_broker()
                if primary and not primary.is_connected:
                    for name, broker in broker_manager.brokers.items():
                        if broker.is_connected:
                            broker_manager.set_primary_broker(name)
                            logger.info(f"Failover: switched primary broker to {name}")
                            break
                time.sleep(30)
        threading.Thread(target=failover_monitor, daemon=True).start()

            # Start data ingestion
        logger.info("Starting data ingestion...")
        self.components['data_manager'].start()

        # Start strategy manager
        logger.info("Starting strategy manager...")
        # self.components['strategy_manager'].start()

        # Start risk monitoring
        logger.info("Starting risk monitoring...")
        self.components['event_risk_manager'].start()

        # Start API server (if available)
        if self.api_server:
            logger.info("Starting API server...")
            self.api_server.start()

        logger.info("AI Trading Agent initialized successfully")
        
        # Start the main trading loop
        self.run_main_loop()

    def _validate_secrets(self):
        """Check for required secrets and log warnings/errors."""
        required_env = [
            'COINBASE_API_KEY', 'COINBASE_API_SECRET', 'COINBASE_PASSPHRASE',
            'OANDA_API_KEY', 'OANDA_ACCOUNT_ID',
            'TRADING_ALPHA_VANTAGE_API_KEY', 'TRADING_FMP_API_KEY', 'TRADING_FINNHUB_API_KEY'
        ]
        missing = []
        for var in required_env:
            if not os.getenv(var):
                missing.append(var)
        if missing:
            logger.error(f"Missing required secrets: {', '.join(missing)}. Trading agent will not start.")
            print(f"ERROR: Missing required secrets: {', '.join(missing)}. See logs for details.")
            sys.exit(1)
        else:
            logger.info("All required secrets are present.")

    def run_main_loop(self):
        logger.info("Starting main trading loop...")
        loop_interval = self.config.get('trading_loop_interval', 60)  # seconds
        symbols = self.config.get('data_manager', {}).get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        
        while self.running:
            try:
                loop_start_time = time.time()
                # Get latest market data
                market_data = self.components['data_manager'].get_latest_data()
                
                if market_data:
                    # Process each symbol individually
                    all_signals = {}
                    
                    for symbol in symbols:
                        try:
                            # Extract symbol-specific data
                            symbol_data = self._extract_symbol_data(market_data, symbol)
                            
                            if symbol_data:
                                # Generate trading signals for this symbol
                                symbol_signals = self.components['strategy_manager'].generate_signals(symbol_data)
                                
                                if symbol_signals and symbol_signals.get('action') != 'hold':
                                    all_signals[symbol] = symbol_signals
                                    
                        except Exception as e:
                            logger.error(f"Error processing symbol {symbol}: {str(e)}")
                            continue
                    
                    # Process signals if any were generated
                    if all_signals:
                        # Assess risk for all signals
                        risk_assessment = self.components['risk_calculator'].assess_portfolio_risk(
                            market_data, all_signals
                        )
                        
                        # Check risk limits
                        if self._check_risk_limits(risk_assessment):
                            # Execute trades
                            self._execute_trades(all_signals, risk_assessment)
                        else:
                            logger.warning("Risk limits exceeded, skipping trade execution")
                    
                    # Update portfolio optimization
                    self._update_portfolio_optimization(market_data)
                
                # Calculate sleep time to maintain consistent loop interval
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, loop_interval - loop_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(loop_interval)  # Wait before retrying
    
    def _check_risk_limits(self, risk_assessment: Dict[str, Any]) -> bool:
        """
        Check if risk assessment is within acceptable limits.
        
        Args:
            risk_assessment: Risk assessment results
            
        Returns:
            True if within limits, False otherwise
        """
        try:
            risk_limits = self.config.get('risk_limits', {})
            
            # Check portfolio VaR
            max_var = risk_limits.get('max_portfolio_var', 0.05)
            if risk_assessment.get('portfolio_var', 0) > max_var:
                logger.warning(f"Portfolio VaR exceeds limit: {risk_assessment['portfolio_var']:.4f} > {max_var:.4f}")
                return False
            
            # Check maximum position size
            max_position = risk_limits.get('max_position_size', 0.1)
            for position_size in risk_assessment.get('position_sizes', {}).values():
                if position_size > max_position:
                    logger.warning(f"Position size exceeds limit: {position_size:.4f} > {max_position:.4f}")
                    return False
            
            # Check maximum drawdown
            max_drawdown = risk_limits.get('max_drawdown', 0.2)
            if risk_assessment.get('current_drawdown', 0) > max_drawdown:
                logger.warning(f"Current drawdown exceeds limit: {risk_assessment['current_drawdown']:.4f} > {max_drawdown:.4f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False
    
    def _extract_symbol_data(self, market_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """
        Extract data for a specific symbol from the market data.
        
        Args:
            market_data: Full market data from data manager
            symbol: Symbol to extract data for
            
        Returns:
            Symbol-specific data or None if not found
        """
        try:
            # Check different data sources
            for data_type, sources in market_data.items():
                if data_type == 'market_data':
                    for source, source_data in sources.items():
                        data = source_data.get('data', {})
                        
                        # Check if symbol data exists
                        if symbol in data:
                            symbol_data = data[symbol].copy()
                            symbol_data['symbol'] = symbol
                            symbol_data['source'] = source
                            symbol_data['timestamp'] = source_data.get('timestamp')
                            return symbol_data
                        
                        # Check if data is for this specific symbol
                        if data.get('symbol') == symbol:
                            symbol_data = data.copy()
                            symbol_data['symbol'] = symbol
                            symbol_data['source'] = source
                            symbol_data['timestamp'] = source_data.get('timestamp')
                            return symbol_data
            
            # If no data found, return None
            logger.debug(f"No market data found for symbol {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting data for symbol {symbol}: {str(e)}")
            return None
    
    def _execute_trades(self, signals: Dict[str, Dict[str, Any]], risk_assessment: Dict[str, Any]):
        """
        Execute trades based on signals and risk assessment with unified order routing.
        
        Args:
            signals: Dict of symbol -> signal data
            risk_assessment: Risk assessment results
        """
        try:
            broker_manager = self.components['broker_manager']

            for symbol, signal_data in signals.items():
                if isinstance(signal_data, dict):
                    action = signal_data.get('action', 'hold')
                    confidence = signal_data.get('confidence', 0.0)
                    position_size = signal_data.get('position_size', 0.0)
                    
                    if action != 'hold' and confidence > 0.1 and position_size > 0:
                        # Determine asset type for routing
                        asset_type = 'stock'
                        if symbol.endswith('-USD') or symbol in ['BTC-USD', 'ETH-USD']:
                            asset_type = 'crypto'
                        elif '_' in symbol:
                            asset_type = 'forex'

                        # Route to appropriate broker
                        broker = None
                        if asset_type == 'crypto':
                            broker = broker_manager.get_broker('coinbase_broker')
                        elif asset_type == 'forex':
                            broker = broker_manager.get_broker('oanda_broker')
                        else:
                            broker = broker_manager.get_broker('paper_broker')
                            
                        if not broker or not broker.is_connected:
                            logger.warning(f"No connected broker for {symbol} ({asset_type})")
                            continue

                        # Calculate quantity
                        initial_capital = self.config.get('trading', {}).get('initial_capital', 100000)
                        quantity = int(position_size * initial_capital / 100)
                        price = signal_data.get('price', 100.0)

                        logger.info(f"Placing {action} order for {symbol}: {quantity} units (confidence: {confidence:.4f}) via {broker.broker_name}")
                        # order_result = broker.place_order(order)

                        # Record metrics if monitoring is enabled
                        if self.monitoring_service:
                            strategy_name = signal_data.get('strategy', 'ensemble')
                            self.monitoring_service.record_trade(
                                action=action,
                                symbol=symbol,
                                strategy=strategy_name,
                                quantity=quantity,
                                price=price
                            )
                            self.monitoring_service.update_risk_metrics({
                                'portfolio_var': risk_assessment.get('portfolio_var', 0),
                                'max_drawdown': risk_assessment.get('max_drawdown', 0),
                                'sharpe_ratio': risk_assessment.get('sharpe_ratio', 0)
                            })
                            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
    
    def _check_risk_limits_enhanced(self, signals: Dict[str, Any], risk_assessment: Dict[str, Any]) -> bool:
        """Enhanced risk checking using Phase 2 risk manager"""
        try:
            # Use real-time risk manager if available
            if hasattr(self, 'risk_manager'):
                # Check each signal for risk compliance
                for symbol, signal_data in signals.items():
                    if isinstance(signal_data, dict):
                        action = signal_data.get('action', 'hold')
                        if action != 'hold':
                            side = 'buy' if action == 'buy' else 'sell'
                            quantity = signal_data.get('position_size', 0) * 1000  # Convert to shares
                            price = signal_data.get('price', 100)  # Default price
                            
                            approved, reason, adjusted_qty = self.risk_manager.pre_trade_risk_check(
                                symbol, side, quantity, price
                            )
                            
                            if not approved:
                                logger.warning(f"Risk check failed for {symbol}: {reason}")
                                return False
                
                return True
            else:
                # Fallback to original risk checking
                return self._check_risk_limits(risk_assessment)
                
        except Exception as e:
            logger.error(f"Enhanced risk check error: {e}")
            return False
    
    def _execute_trades_enhanced(self, signals: Dict[str, Any], risk_assessment: Dict[str, Any]):
        """Enhanced trade execution using Phase 2 execution engine"""
        try:
            if not hasattr(self, 'execution_engine'):
                # Fallback to original execution
                return self._execute_trades(signals, risk_assessment)
            
            from order_execution_engine import OrderRequest, OrderType
            
            for symbol, signal_data in signals.items():
                if isinstance(signal_data, dict):
                    action = signal_data.get('action', 'hold')
                    if action != 'hold':
                        confidence = signal_data.get('confidence', 0)
                        position_size = signal_data.get('position_size', 0)
                        
                        if confidence > 0.1 and position_size > 0:  # Minimum thresholds
                            # Create order request
                            order_request = OrderRequest(
                                symbol=symbol,
                                side=action,
                                quantity=position_size * 1000,  # Convert to shares
                                order_type=OrderType.LIMIT,
                                price=signal_data.get('price', 100),
                                strategy=signal_data.get('strategy', 'adaptive'),
                                metadata={
                                    'confidence': confidence,
                                    'timestamp': datetime.utcnow().isoformat()
                                }
                            )
                            
                            # Execute through execution engine
                            order_id = self.execution_engine.execute_order(order_request)
                            
                            if order_id:
                                logger.info(f"Order submitted: {order_id} for {symbol} {action}")
                                
                                # Record in performance analytics
                                if hasattr(self, 'performance_analytics'):
                                    self.performance_analytics.record_trade(
                                        symbol=symbol,
                                        side=action,
                                        quantity=order_request.quantity,
                                        price=order_request.price if order_request.price is not None else 0.0,
                                        strategy=order_request.strategy
                                    )
                            else:
                                logger.warning(f"Order execution failed for {symbol}")
                
        except Exception as e:
            logger.error(f"Enhanced trade execution error: {e}")
    
    def _update_portfolio_optimization(self, market_data):
        """
        Update portfolio optimization based on current market conditions.
        
        Args:
            market_data: Current market data
        """
        try:
            # Run portfolio optimization periodically
            optimization_frequency = self.config.get('portfolio_optimization', {}).get('frequency', 'daily')
            
            if optimization_frequency == 'daily':
                # Check if we should run optimization (e.g., once per day)
                current_hour = datetime.now().hour
                if current_hour == 9:  # Run at market open
                    logger.info("Running daily portfolio optimization...")
                    
                    optimizer = self.components['portfolio_optimizer']
                    optimization_result = optimizer.optimize_portfolio(
                        market_data,
                        optimization_method='mean_variance'
                    )
                    
                    if optimization_result:
                        logger.info("Portfolio optimization completed successfully")
                        # Store results for later use
                        self._store_optimization_results(optimization_result)
            
        except Exception as e:
            logger.error(f"Error updating portfolio optimization: {str(e)}")
    
    def _store_optimization_results(self, optimization_result: Dict[str, Any]):
        """
        Store portfolio optimization results.
        
        Args:
            optimization_result: Optimization results
        """
        try:
            # Store results to file or database
            results_file = f"data/optimization_results_{datetime.now().strftime('%Y%m%d')}.json"
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(optimization_result, f, indent=2, default=str)
            
            logger.info(f"Optimization results stored to {results_file}")
            
        except Exception as e:
            logger.error(f"Error storing optimization results: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status information
        """
        try:
            status = {
                'running': self.running,
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }
            
            # Get component statuses
            for component_name, component in self.components.items():
                if hasattr(component, 'get_status'):
                    status['components'][component_name] = component.get_status()
                else:
                    status['components'][component_name] = {'initialized': True}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}


def create_default_config():
    """Create a default configuration file."""
    default_config = {
        "data_manager": {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
            "update_interval": 60,
            "max_retries": 3,
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            }
        },
        "strategies": {
            "momentum": {
                "type": "supervised_learning",
                "enabled": True,
                "lookback_period": 20,
                "threshold": 0.02,
                "weight": 1.0
            },
            "mean_reversion": {
                "type": "supervised_learning",
                "enabled": True,
                "lookback_period": 10,
                "z_score_threshold": 2.0,
                "weight": 1.0
            },
            "sentiment": {
                "type": "nlp",
                "enabled": True,
                "sentiment_threshold": 0.6,
                "weight": 0.8
            },
            "reinforcement": {
                "type": "dqn",
                "enabled": True,
                "lookback_period": 30,
                "learning_rate": 0.001,
                "exploration_rate": 0.1,
                "weight": 1.0
            }
        },
        "risk_management": {
            "var_confidence": 0.95,
            "max_position_size": 0.1,
            "max_portfolio_var": 0.02,
            "max_drawdown": 0.2,
            "max_daily_loss": 0.02
        },
        "risk_limits": {
            "max_portfolio_var": 0.05,
            "max_position_size": 0.1,
            "max_drawdown": 0.2
        },
        "brokers": {
            "paper_broker": {
                "type": "paper",
                "initial_cash": 100000,
                "commission_per_trade": 1.0,
                "primary": True
            }
        },
        "trading": {
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0005
        },
        "trading_loop_interval": 60,
        "nlp": {
            "sentiment_model": "vader",
            "text_processing": {
                "remove_stopwords": True,
                "lemmatize": True
            }
        },
        "automl": {
            "n_trials": 100,
            "cv_folds": 5
        },
        "portfolio_optimization": {
            "risk_free_rate": 0.02,
            "frequency": "daily",
            "lookback_period": 252
        },
        "monitoring": {
            "enabled": True,
            "port": 8080,
            "metrics_interval": 15,
            "system_metrics_enabled": True
        },
        "security": {
            "use_encryption": True,
            "api_key_rotation_days": 90
        },
        "execution": {
            "max_slippage": 0.005,
            "max_order_size": 10000,
            "min_order_size": 100,
            "use_twap": True,
            "twap_duration": 300,
            "slice_size": 0.1
        },
        "analytics": {
            "benchmark": "SPY",
            "risk_free_rate": 0.02,
            "max_history_days": 365,
            "max_trades_history": 10000
        },
        "api": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 5001,
            "debug": False
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/trading_agent.log",
            "max_size_mb": 10,
            "backup_count": 5
        }
    }
    
    os.makedirs('config', exist_ok=True)
    config_path = 'config/config.json'
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created at {config_path}")
    
    # Create example .env file
    env_path = '.env.example'
    with open(env_path, 'w') as f:
        f.write("# Environment variables for AI Trading Agent\n")
        f.write("TRADING_MODE=paper\n")
        f.write("TRADING_MASTER_KEY=your_secure_master_key_here\n")
        f.write("TRADING_ALPACA_API_KEY=your_alpaca_api_key_here\n")
        f.write("TRADING_ALPACA_API_SECRET=your_alpaca_api_secret_here\n")
        f.write("TRADING_ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here\n")
        f.write("TRADING_NEWS_API_KEY=your_news_api_key_here\n")
    
    print(f"Example environment file created at {env_path}")
    print("Copy this to .env and fill in your actual API keys")
    
    return config_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI Trading Agent')
    parser.add_argument('--config', default='config/config.json', help='Configuration file path')
    parser.add_argument('--mode', choices=['development', 'production'], default='development', help='Running mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Log level')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration file')
    parser.add_argument('--monitoring-port', type=int, default=8080, help='Port for monitoring HTTP server')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create default config if requested
    if args.create_config:
        config_path = create_default_config()
        return 0
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Use --create-config to create a default configuration file")
        return 1
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('secrets', exist_ok=True)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set monitoring configuration
        if not args.no_monitoring:
            if 'monitoring' not in config:
                config['monitoring'] = {}
            config['monitoring']['port'] = args.monitoring_port
            config['monitoring']['enabled'] = True
        else:
            if 'monitoring' not in config:
                config['monitoring'] = {}
            config['monitoring']['enabled'] = False
        
        # Initialize and start the trading agent
        agent = TradingAgent(args.config)
        
        if args.debug:
            # In debug mode, just show status and exit
            status = agent.get_status()
            print(json.dumps(status, indent=2, default=str))
        else:
            # Start the trading agent
            agent.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)