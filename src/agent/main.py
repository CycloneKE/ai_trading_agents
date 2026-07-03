from src.utils.config_validator import validate_config
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
from src.utils.config_validator import load_config
from src.utils.secure_config import SecureConfigManager
from src.utils.database_manager import DatabaseManager
from src.agent.realtime_data_feed import RealTimeDataFeed
from src.utils.monitoring import get_monitoring_service
from src.agent.data_manager import DataManager
from src.agent.strategy_manager import StrategyManager
from src.agent.broker_manager import BrokerManager
from src.agent.order_execution_engine import OrderExecutionEngine
from src.agent.realtime_risk_manager import RealTimeRiskManager
from src.agent.performance_analytics import PerformanceAnalytics
from src.agent.risk_calculator import RiskCalculator
from src.agent.llm_orchestrator import LLMOrchestrator
from src.agent.bias_detector import BiasDetector
from src.agent.adaptive_integration import AdaptiveStrategyIntegration
try:
    from src.agent.portfolio_optimizer import PortfolioOptimizer
except ImportError as e:
    logger.warning(f"Portfolio optimization modules disabled due to import error: {e}")
    PortfolioOptimizer = None
from src.agent.event_risk_manager import EventRiskManager
try:
    from src.agent.nlp_manager import NLPManager
except ImportError as e:
    logger.warning(f"NLP modules disabled due to import error: {e}")
    NLPManager = None
try:
    from src.connectors.nse_scraper import NSEPeriodicScraper
    NSE_SCRAPER_AVAILABLE = True
except ImportError:
    NSE_SCRAPER_AVAILABLE = False
    NSEPeriodicScraper = None
try:
    from src.agent.automl_optimizer import AutoMLOptimizer
except ImportError as e:
    logger.warning(f"AutoML modules disabled due to import error: {e}")
    AutoMLOptimizer = None

try:
    from src.api.api_server import TradingAPI
    API_AVAILABLE = True
except ImportError:
    logger.warning("Flask not available - API server disabled")
    API_AVAILABLE = False
    TradingAPI = None

# Redundant mock classes removed. Using imported system components.




class TradingAgent:
    """
    Main AI Trading Agent application.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(self.config_path)
        # Validate configuration before proceeding
        config_errors = validate_config(self.config)
        if config_errors:
            for error in config_errors:
                logger.error(f"Config validation error: {error}")
            print("ERROR: Invalid configuration. See logs for details.")
            sys.exit(1)
        # Validate required secrets AFTER config is loaded — _validate_secrets
        # derives the required env vars from the enabled config (self.config).
        self._validate_secrets()
        self.running = False
        self.trading_halted = False  # kill switch: blocks all NEW orders when True
        self.components = {}
        self.order_journal = None  # initialized in start() after brokers connect
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
            # Allow a top-level flag 'use_fallback_only' or 'test_mode' to be
            # propagated into the DataManager sub-config so CI/tests can set it
            # at the top level (or keep it under data_manager).
            dm_cfg = dict(self.config.get('data_manager', {}))
            if self.config.get('use_fallback_only') or self.config.get('test_mode'):
                dm_cfg['use_fallback_only'] = True
            # Pass database_manager to DataManager so NSEConnector can read from DB
            dm_cfg['_database_manager'] = self.database
            self.components['data_manager'] = DataManager(dm_cfg)
            
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
            
            # Orchestrator & Multi-Agent Modules
            self.components['llm_orchestrator'] = LLMOrchestrator(self.config)
            self.components['bias_detector'] = BiasDetector(self.config)
            self.components['adaptive_integration'] = AdaptiveStrategyIntegration(self.config)

            
            # NLP Engine
            if NLPManager is not None:
                self.components['nlp_manager'] = NLPManager(
                    self.config.get('nlp', {})
                )
            else:
                logger.warning("Skipping NLP Manager initialization (not available)")
            
            # AutoML Optimizer
            if AutoMLOptimizer is not None:
                self.components['automl_optimizer'] = AutoMLOptimizer(
                    self.config.get('automl', {})
                )
            else:
                logger.warning("Skipping AutoML Optimizer initialization (not available)")
            
            # Portfolio Optimizer
            if PortfolioOptimizer is not None:
                self.components['portfolio_optimizer'] = PortfolioOptimizer(
                    self.config.get('portfolio_optimization', {})
                )
            else:
                logger.warning("Skipping Portfolio Optimizer initialization (not available)")
            
            # (AutoML and Portfolio optimizers already initialized above)
            
            # Register health checks with monitoring service
            if self.monitoring_service:
                for name, component in self.components.items():
                    if hasattr(component, 'get_status'):
                        self.monitoring_service.register_health_check(
                            name, component.get_status
                        )
                # Populate the health cache immediately so /health reflects the
                # registered checks right away instead of after the first refresh.
                if hasattr(self.monitoring_service, 'refresh_health_cache'):
                    self.monitoring_service.refresh_health_cache()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals by performing a full graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def stop(self):
        """Stop the trading agent and clean up resources (idempotent)."""
        if getattr(self, '_stopped', False):
            return
        self._stopped = True
        self.running = False
        logger.info("Trading agent stopping...")
        # Attempt to gracefully stop threads (if any custom threads are tracked)
        # Example: if hasattr(self, 'broker_health_thread'): self.broker_health_thread.join(timeout=5)
        # Stop data manager and event risk manager if they have stop methods
        try:
            if 'data_manager' in self.components and hasattr(self.components['data_manager'], 'stop'):
                self.components['data_manager'].stop()
            # Stop NSE scraper
            if hasattr(self, 'nse_scraper') and self.nse_scraper:
                self.nse_scraper.stop()
            if 'event_risk_manager' in self.components and hasattr(self.components['event_risk_manager'], 'stop'):
                self.components['event_risk_manager'].stop()
        except Exception as e:
            logger.warning(f"Error stopping components: {e}")
        # Close the order journal so the SQLite handle is released cleanly.
        try:
            if getattr(self, 'order_journal', None):
                self.order_journal.close()
        except Exception as e:
            logger.warning(f"Error closing order journal: {e}")
        # Stop the monitoring service (HTTP server + metrics threads) so the
        # port is released and background threads don't outlive the agent.
        try:
            if getattr(self, 'monitoring_service', None) and hasattr(self.monitoring_service, 'stop'):
                self.monitoring_service.stop()
        except Exception as e:
            logger.warning(f"Error stopping monitoring service: {e}")
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
                # Seed performance analytics from primary broker history
                primary = broker_manager.get_broker()
                if primary and primary.broker_name == broker_name:
                    try:
                        # Get last 1 week of history for a good chart
                        history = primary.get_portfolio_history(period='1W', timeframe='1H')
                        if history and history.get('equity'):
                            self.performance_analytics.seed_history(history)
                            logger.info(f"Seeded performance history from {broker_name}")
                    except Exception as e:
                        logger.warning(f"Could not seed history from {broker_name}: {e}")
            else:
                logger.error(f"✗ Failed to connect to {broker_name}")

        # Reconcile the order journal BEFORE any trading decision: every
        # order whose outcome is unknown (crash between submit and ack) is
        # resolved against the broker, so the loop never starts blind to
        # in-flight state and never re-submits a decision it already made.
        try:
            from src.agent.order_journal import OrderJournal
            self.order_journal = OrderJournal()
            primary = broker_manager.get_broker()
            if primary and primary.is_connected:
                self.order_journal.reconcile(primary)
        except Exception as e:
            logger.error(f"Order journal init/reconcile failed: {e}")
            self.order_journal = None

        # LLM weight allocator: proposes ensemble weight tilts from realized
        # attribution on a slow cadence; hard guardrails clamp every proposal
        # and it is a no-op without an LLM API key.
        try:
            from src.agent.llm_allocator import LLMAllocator
            self.llm_allocator = LLMAllocator(
                self.components.get('llm_orchestrator'),
                self.components.get('strategy_manager'),
                self.order_journal,
                self.config.get('llm_allocator', {}))
        except Exception as e:
            logger.error(f"LLM allocator init failed: {e}")
            self.llm_allocator = None

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

        # Start NSE Kenya periodic scraper
        if NSE_SCRAPER_AVAILABLE and NSEPeriodicScraper is not None:
            try:
                nse_connector = self.components.get('data_manager').connectors.get('nse') if self.components.get('data_manager') else None
                self.nse_scraper = NSEPeriodicScraper(
                    database_manager=self.database,
                    nse_connector=nse_connector,
                    interval_minutes=30,
                )
                # Seed historical data if this is first run
                self.nse_scraper.seed_historical(days=730)
                # Start periodic background scraping
                self.nse_scraper.start()
                logger.info("NSE Kenya periodic scraper started")
            except Exception as e:
                logger.warning(f"NSE scraper startup failed: {e}")

        logger.info("AI Trading Agent initialized successfully")
        
        # Start the main trading loop
        self.run_main_loop()

    def _validate_secrets(self):
        """Check that secrets required by the *enabled* config are present.

        Required env vars are derived from the active configuration — enabled,
        non-paper brokers and enabled data connectors — by extracting their
        ``${ENV_VAR}`` placeholders. This means disabling a broker no longer
        forces its credentials to be set, and enabling one enforces them.
        """
        import re

        placeholder = re.compile(r'\$\{([A-Z0-9_]+)\}')

        def _placeholders(obj):
            """Recursively yield ${ENV_VAR} names referenced anywhere in `obj`."""
            if isinstance(obj, str):
                yield from placeholder.findall(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    yield from _placeholders(value)
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    yield from _placeholders(value)

        required = set()

        # Brokers: only those explicitly enabled (paper brokers carry no secrets).
        for broker in (self.config.get('brokers') or {}).values():
            if isinstance(broker, dict) and broker.get('enabled') and broker.get('type') != 'paper':
                required.update(_placeholders(broker))

        # Data connectors: only those explicitly enabled.
        connectors = (self.config.get('data_manager') or {}).get('connectors') or {}
        for conn in connectors.values():
            if isinstance(conn, dict) and conn.get('enabled'):
                required.update(_placeholders(conn))

        missing = sorted(var for var in required if not os.getenv(var))
        if missing:
            logger.error(f"Missing required secrets for enabled components: {', '.join(missing)}. Trading agent will not start.")
            print(f"ERROR: Missing required secrets: {', '.join(missing)}. See logs for details.")
            sys.exit(1)
        else:
            logger.info(f"All required secrets are present ({len(required)} checked).")

    def run_main_loop(self):
        logger.info("Starting main trading loop...")
        loop_interval = self.config.get('trading_loop_interval', 60)  # seconds
        symbols = self.config.get('data_manager', {}).get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        self._last_optimization_date = None  # Track daily optimization
        
        # Stop-loss configuration
        stop_loss_pct = self.config.get('risk_limits', {}).get('stop_loss_pct', 0.05)
        trailing_stop_pct = self.config.get('risk_limits', {}).get('trailing_stop_pct', 0.03)
        
        while self.running:
            try:
                loop_start_time = time.time()
                # Get latest market data
                market_data = self.components['data_manager'].get_latest_data()
                
                # Slow-cadence LLM weight rebalance (self rate-limited; no-op
                # without an LLM key or while halted).
                if getattr(self, 'llm_allocator', None) and not self.trading_halted:
                    try:
                        self.llm_allocator.maybe_rebalance()
                    except Exception as e:
                        logger.error(f"LLM allocator error: {e}")

                # Pull fill results for submitted orders into the journal so
                # attribution and duplicate-close checks see current state.
                if self.order_journal:
                    try:
                        broker_manager = self.components.get('broker_manager')
                        primary = broker_manager.get_broker() if broker_manager else None
                        if primary and primary.is_connected:
                            self.order_journal.sync_fills(primary)
                    except Exception as e:
                        logger.debug(f"Fill sync error: {e}")

                if market_data:
                    # --- Stop-loss enforcement ---
                    self._enforce_stop_losses(stop_loss_pct, trailing_stop_pct)
                    
                    # Process each symbol individually
                    all_signals = {}
                    
                    for symbol in symbols:
                        try:
                            # Extract symbol-specific data
                            symbol_data = self._extract_symbol_data(market_data, symbol)
                            
                            if symbol_data:
                                # Retrieve recent news if NLP manager is functioning and data is available
                                symbol_news = []
                                if 'news_data' in market_data:
                                    for source, source_data in market_data['news_data'].items():
                                        data = source_data.get('data', [])
                                        if isinstance(data, dict):
                                            # Connectors deliver {symbol: [items]}
                                            symbol_news.extend([n for n in (data.get(symbol) or []) if isinstance(n, dict)])
                                        else:
                                            # Flat list with per-item symbol field
                                            symbol_news.extend([n for n in data if isinstance(n, dict) and n.get('symbol') == symbol])

                                # Apply Adaptive Goals to strategy execution
                                modified_params = self.components['adaptive_integration'].get_strategy_parameters('technical')
                                
                                # Generate trading signals for this symbol
                                symbol_signals = self.components['strategy_manager'].generate_signals(symbol_data)
                                
                                if symbol_signals and symbol_signals.get('action') != 'hold':
                                    # Validate with LLM Orchestrator
                                    validated_signal = self.components['llm_orchestrator'].validate_trade(symbol, symbol_signals, symbol_data, symbol_news)
                                    
                                    # Check for Algorithmic/Cognitive Bias
                                    is_biased = self.components['bias_detector'].detect_bias(validated_signal, symbol_data, market_data)
                                    if is_biased:
                                        logger.warning(f"Bias detected for {symbol} trade: {validated_signal}. Lowering confidence.")
                                        validated_signal['confidence'] *= 0.5
                                        if validated_signal['confidence'] < 0.3:
                                            validated_signal['action'] = 'hold'
                                        
                                    if validated_signal['action'] != 'hold':
                                        all_signals[symbol] = validated_signal
                                        
                        except Exception as e:
                            logger.error(f"Error processing symbol {symbol}: {str(e)}")
                            continue
                    
                    # Kill switch: observe the market but submit nothing
                    if self.trading_halted:
                        if all_signals:
                            logger.info(f"Trading halted; dropping signals for {list(all_signals)}")
                        all_signals = {}

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
                            
                            # Provide performance feedback to the adaptive loop (actual PnL feedback)
                            perf_report = self.performance_analytics.generate_performance_report(period='1D')
                            perf_metrics = perf_report.get('metrics', {})
                            real_performance = {
                                'daily_return': perf_metrics.get('total_return', 0.0),
                                'drawdown': perf_metrics.get('max_drawdown', 0.0),
                                'sharpe': perf_metrics.get('sharpe_ratio', 0.0),
                                'win_rate': perf_metrics.get('win_rate', 0.0)
                            }
                            self.components['adaptive_integration'].update_with_performance(real_performance, market_data)
                        else:
                            logger.warning("Risk limits exceeded, skipping trade execution")
                    
                    # Update portfolio optimization (once daily, not every loop)
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

                        # Calculate quantity (Max 2% of portfolio per trade for safety)
                        portfolio_value = broker.get_account_info().equity if broker.get_account_info() else 100000
                        max_risk_per_trade = 0.02 # 2% Rule
                        
                        # Position value based on signal (confidence * position_size)
                        target_pos_value = portfolio_value * min(position_size, max_risk_per_trade)
                        
                        # Fetch price
                        from src.utils.real_price_feed import price_feed
                        price = signal_data.get('price') or price_feed.get_price(symbol)
                        if not price or price <= 0:
                            logger.warning(f"No valid price for {symbol} — skipping order")
                            continue

                        if self.trading_halted:
                            logger.info(f"Trading halted; skipping {action} {symbol}")
                            continue

                        # Small-account protection: don't let an automated
                        # exit trip the pattern-day-trader rule.
                        if self._pdt_blocks_order(broker, symbol, action):
                            continue

                        # Execution plan: one blended order (default), or in
                        # 'parallel' mode one order per agreeing strategy so
                        # each keeps an independently attributed book on the
                        # symbol. Only strategies agreeing with the validated
                        # blended direction execute — dissenters skip, so the
                        # account never trades against itself within a cycle.
                        # The symbol total stays capped at target_pos_value.
                        mode = self.config.get('strategy_execution_mode', 'blend')
                        per_strategy = signal_data.get('per_strategy') or {}
                        executions = []
                        if mode == 'parallel' and per_strategy:
                            agreeing = [
                                n for n, s in per_strategy.items()
                                if s.get('action') == action and s.get('confidence', 0) >= 0.3
                            ]
                            if agreeing:
                                share = target_pos_value / len(agreeing)
                                executions = [(n, share) for n in agreeing]
                        if not executions:
                            executions = [(signal_data.get('strategy', 'ensemble'), target_pos_value)]

                        from src.agent.position_sizing import size_order
                        from src.agent.order_journal import make_client_order_id
                        from src.connectors.base_broker import OrderRequest
                        sizing_cfg = self.config.get('trading', {})
                        cycle = int(time.time() // self.config.get('trading_loop_interval', 60))

                        for strategy_name, target_value in executions:
                            # Dollar-notional sizing with fractional shares,
                            # so small accounts get properly sized positions
                            # instead of rounding to zero.
                            sized = size_order(
                                target_value, price,
                                min_notional=sizing_cfg.get('min_notional', 5.0),
                                allow_fractional=sizing_cfg.get('allow_fractional', True),
                            )
                            if not sized:
                                continue
                            quantity = sized.quantity

                            # Idempotency: one deterministic id per decision
                            # (strategy, symbol, side, loop cycle). The journal
                            # blocks re-submission after a crash/retry, and the
                            # broker deduplicates on the same id server-side.
                            client_order_id = make_client_order_id(strategy_name, symbol, action, cycle)
                            if self.order_journal and not self.order_journal.record_intent(
                                    client_order_id, symbol, action, float(quantity),
                                    'market', strategy=strategy_name):
                                logger.warning(f"Skipping duplicate order decision: {client_order_id}")
                                continue

                            order = OrderRequest(
                                symbol=symbol,
                                quantity=float(quantity),
                                side=action,
                                order_type='market',
                                # fractional orders must be DAY (broker rule);
                                # whole-share keeps DAY too - the loop
                                # re-decides every cycle, GTC adds nothing.
                                time_in_force=sized.time_in_force,
                                client_order_id=client_order_id
                            )

                            logger.info(f"Placing {action} order for {symbol}: {quantity} units "
                                        f"[{strategy_name}] via {broker.broker_name} ({client_order_id})")

                            # Record attempt in risk manager
                            if hasattr(self, 'risk_manager') and self.risk_manager:
                                self.risk_manager.record_trade_attempt()

                            # PLACE THE REAL ORDER
                            order_result = broker.place_order(order)
                            if self.order_journal:
                                if order_result:
                                    self.order_journal.mark_submitted(
                                        client_order_id, order_result.order_id, order_result.status)
                                else:
                                    # May or may not have reached the broker; the id
                                    # stays burned and reconcile resolves it at restart.
                                    self.order_journal.mark_failed(
                                        client_order_id, 'place_order returned no response')

                            # Record metrics if monitoring is enabled
                            if self.monitoring_service:
                                self.monitoring_service.record_trade(
                                    action=action,
                                    symbol=symbol,
                                    strategy=strategy_name,
                                    quantity=quantity,
                                    price=price
                                )
                        if self.monitoring_service:
                            self.monitoring_service.update_risk_metrics({
                                'portfolio_var': risk_assessment.get('portfolio_var', 0),
                                'max_drawdown': risk_assessment.get('max_drawdown', 0),
                                'sharpe_ratio': risk_assessment.get('sharpe_ratio', 0)
                            })
                        # Record trade success metric
                        try:
                            if hasattr(self, 'risk_manager'):
                                self.risk_manager.record_trade_success(symbol)
                        except Exception:
                            pass


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
    
    def halt_trading(self, flatten: bool = False, reason: str = 'operator request') -> Dict[str, Any]:
        """Kill switch: stop all NEW order submission immediately.

        With ``flatten=True`` also cancels every open order and closes every
        position with marketable extended-hours limit orders (falling back to
        market orders when no price is available), so it works premarket too.
        """
        self.trading_halted = True
        logger.warning(f"KILL SWITCH ENGAGED ({reason}); flatten={flatten}")
        result = {'halted': True, 'canceled_orders': 0, 'close_orders': 0, 'errors': []}
        if not flatten:
            return result

        try:
            broker_manager = self.components.get('broker_manager')
            broker = broker_manager.get_broker() if broker_manager else None
            if not broker or not broker.is_connected:
                result['errors'].append('no connected broker')
                return result

            for o in broker.get_orders() or []:
                try:
                    if broker.cancel_order(o.order_id):
                        result['canceled_orders'] += 1
                except Exception as e:
                    result['errors'].append(f'cancel {o.symbol}: {e}')

            from src.connectors.base_broker import OrderRequest
            from src.agent.order_journal import make_client_order_id
            data_manager = self.components.get('data_manager')
            real = data_manager.connectors.get('real_data') if data_manager else None

            for p in broker.get_positions() or []:
                if not p.quantity:
                    continue
                side = 'sell' if p.quantity > 0 else 'buy'
                coid = make_client_order_id('killswitch', p.symbol, side,
                                            int(time.time() // 300))
                if self.order_journal and not self.order_journal.record_intent(
                        coid, p.symbol, side, abs(p.quantity), 'limit',
                        strategy='kill_switch'):
                    continue
                # Marketable extended-hours limit so flattening also works
                # pre/after-market; market GTC as last resort.
                price = None
                try:
                    quote = real.get_real_time_data(p.symbol) if real else None
                    price = (quote or {}).get('price') or p.current_price
                except Exception:
                    price = p.current_price
                if price:
                    limit = round(price * (0.995 if side == 'sell' else 1.005), 2)
                    order = OrderRequest(symbol=p.symbol, quantity=abs(p.quantity),
                                         side=side, order_type='limit',
                                         time_in_force='day', limit_price=limit,
                                         extended_hours=True, client_order_id=coid)
                else:
                    order = OrderRequest(symbol=p.symbol, quantity=abs(p.quantity),
                                         side=side, order_type='market',
                                         time_in_force='gtc', client_order_id=coid)
                resp = broker.place_order(order)
                if self.order_journal:
                    if resp:
                        self.order_journal.mark_submitted(coid, resp.order_id, resp.status)
                    else:
                        self.order_journal.mark_failed(coid, 'flatten order got no response')
                if resp:
                    result['close_orders'] += 1
                    logger.warning(f"Kill switch: closing {p.quantity} {p.symbol} ({coid})")
                else:
                    result['errors'].append(f'close {p.symbol}: no response')
        except Exception as e:
            logger.error(f"Kill switch flatten error: {e}")
            result['errors'].append(str(e))
        return result

    def resume_trading(self) -> Dict[str, Any]:
        """Release the kill switch; the loop resumes submitting orders."""
        self.trading_halted = False
        logger.warning("Kill switch released; trading resumed")
        return {'halted': False}

    PDT_EQUITY_THRESHOLD = 25_000
    PDT_MAX_DAY_TRADES = 3

    def _pdt_blocks_order(self, broker, symbol: str, side: str) -> bool:
        """US pattern-day-trader guard for small accounts.

        A margin account under $25k equity is limited to 3 day trades per 5
        business days. Selling a position bought TODAY is a day trade — block
        it when the account is already at the limit. Conservative by design:
        a blocked exit still happens tomorrow; a PDT flag freezes the account
        for 90 days.
        """
        try:
            if side != 'sell' or not self.order_journal:
                return False
            acct = broker.get_account_info()
            if not acct or acct.equity >= self.PDT_EQUITY_THRESHOLD:
                return False
            if acct.day_trade_count < self.PDT_MAX_DAY_TRADES:
                return False
            today = datetime.utcnow().date().isoformat()
            bought_today = any(
                r['symbol'] == symbol and r['side'] == 'buy'
                and r['status'] in ('submitted', 'filled')
                and r['created_at'][:10] == today
                for r in self.order_journal.recent(200))
            if bought_today:
                logger.warning(
                    f"PDT guard: blocking sell of {symbol} bought today — "
                    f"account ${acct.equity:,.0f} < $25k already has "
                    f"{acct.day_trade_count} day trades")
                return True
            return False
        except Exception as e:
            logger.error(f"PDT check error for {symbol}: {e}")
            return False

    def _has_open_close_order(self, broker, symbol: str, side: str) -> bool:
        """True if the broker already has an open order that would close
        this position — placing another would stack duplicate closes."""
        try:
            for o in broker.get_orders(symbol) or []:
                if o.symbol == symbol and o.side == side and \
                        str(o.status) not in ('filled', 'canceled', 'cancelled', 'rejected', 'expired'):
                    return True
        except Exception as e:
            logger.warning(f"Open-order check failed for {symbol}: {e}")
        return False

    def _enforce_stop_losses(self, stop_loss_pct: float, trailing_stop_pct: float):
        """
        Enforce stop-loss and trailing stop rules on all open positions.
        
        Args:
            stop_loss_pct: Maximum loss percentage before auto-close (e.g. 0.05 = 5%)
            trailing_stop_pct: Trailing stop percentage (e.g. 0.03 = 3%)
        """
        try:
            broker_manager = self.components.get('broker_manager')
            if not broker_manager:
                return
            
            primary_broker = broker_manager.get_broker()
            if not primary_broker or not primary_broker.is_connected:
                return
            
            positions = primary_broker.get_positions()
            if not positions:
                return
            
            for position in positions:
                try:
                    if position.quantity == 0 or position.cost_basis <= 0:
                        continue
                    
                    unrealized_pl_pct = position.unrealized_pl / position.cost_basis
                    
                    # Hard stop-loss
                    if unrealized_pl_pct <= -stop_loss_pct:
                        close_side = 'sell' if position.quantity > 0 else 'buy'

                        # A pending close order makes another one redundant —
                        # the position still shows until the first fills, so
                        # without this check every 60s loop stacks a new order.
                        if self._has_open_close_order(primary_broker, position.symbol, close_side):
                            logger.info(f"Stop-loss close already pending for {position.symbol}; skipping")
                            continue

                        logger.warning(
                            f"STOP-LOSS triggered for {position.symbol}: "
                            f"loss={unrealized_pl_pct:.2%} exceeds limit={-stop_loss_pct:.2%}"
                        )
                        from src.connectors.base_broker import OrderRequest
                        from src.agent.order_journal import make_client_order_id
                        # 5-minute cycle: dedups rapid re-triggers while still
                        # allowing a later re-close if the position re-opens.
                        coid = make_client_order_id('stoploss', position.symbol,
                                                    close_side, int(time.time() // 300))
                        if self.order_journal and not self.order_journal.record_intent(
                                coid, position.symbol, close_side,
                                abs(position.quantity), 'market', strategy='stop_loss'):
                            logger.info(f"Stop-loss close already journaled ({coid}); skipping")
                            continue
                        close_order = OrderRequest(
                            symbol=position.symbol,
                            quantity=abs(position.quantity),
                            side=close_side,
                            order_type='market',
                            time_in_force='gtc',
                            client_order_id=coid
                        )
                        result = primary_broker.place_order(close_order)
                        if self.order_journal:
                            if result:
                                self.order_journal.mark_submitted(coid, result.order_id, result.status)
                            else:
                                self.order_journal.mark_failed(coid, 'place_order returned no response')
                        if result:
                            logger.info(f"Stop-loss order placed for {position.symbol}: {result.order_id}")
                            if self.monitoring_service:
                                self.monitoring_service.record_trade(
                                    action='stop_loss_sell',
                                    symbol=position.symbol,
                                    strategy='stop_loss',
                                    quantity=abs(position.quantity),
                                    price=position.current_price or 0
                                )
                                
                    # Trailing Stop Loss using Risk Manager
                    if hasattr(self, 'risk_manager') and self.risk_manager:
                        trailing_stops = self.risk_manager.check_trailing_stops(trailing_stop_pct)
                        for ts in trailing_stops:
                            if ts['symbol'] == position.symbol:
                                from src.connectors.base_broker import OrderRequest
                                from src.agent.order_journal import make_client_order_id
                                if self._has_open_close_order(primary_broker, position.symbol, 'sell'):
                                    logger.info(f"Trailing-stop close already pending for {position.symbol}; skipping")
                                    continue
                                coid = make_client_order_id('trailstop', position.symbol,
                                                            'sell', int(time.time() // 300))
                                if self.order_journal and not self.order_journal.record_intent(
                                        coid, position.symbol, 'sell',
                                        abs(position.quantity), 'market', strategy='trailing_stop'):
                                    logger.info(f"Trailing-stop close already journaled ({coid}); skipping")
                                    continue
                                close_order = OrderRequest(
                                    symbol=position.symbol,
                                    quantity=abs(position.quantity),
                                    side='sell',
                                    order_type='market',
                                    time_in_force='gtc',
                                    client_order_id=coid
                                )
                                result = primary_broker.place_order(close_order)
                                if self.order_journal:
                                    if result:
                                        self.order_journal.mark_submitted(coid, result.order_id, result.status)
                                    else:
                                        self.order_journal.mark_failed(coid, 'place_order returned no response')
                                if result:
                                    logger.info(f"Trailing stop-loss order placed for {position.symbol}: {result.order_id}")
                                    if self.monitoring_service:
                                        self.monitoring_service.record_trade(
                                            action='trailing_stop_sell',
                                            symbol=position.symbol,
                                            strategy='trailing_stop',
                                            quantity=abs(position.quantity),
                                            price=position.current_price or 0
                                        )
                            
                except Exception as e:
                    logger.error(f"Error enforcing stop-loss for {position.symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in stop-loss enforcement: {str(e)}")

    def _update_portfolio_optimization(self, market_data):
        """
        Update portfolio optimization based on current market conditions.
        Runs at most once per calendar day.
        
        Args:
            market_data: Current market data
        """
        try:
            today = datetime.now().date()
            
            # Only run once per day
            if hasattr(self, '_last_optimization_date') and self._last_optimization_date == today:
                return
            
            # Only run during market hours (9 AM)
            current_hour = datetime.now().hour
            if current_hour != 9:
                return
            
            logger.info("Running daily portfolio optimization...")
            self._last_optimization_date = today
            
            optimizer = self.components['portfolio_optimizer']
            optimization_result = optimizer.optimize_portfolio(
                market_data,
                optimization_method='mean_variance'
            )
            
            if optimization_result:
                logger.info("Portfolio optimization completed successfully")
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
                'trading_halted': getattr(self, 'trading_halted', False),
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