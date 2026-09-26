from src.utils.config_validator import validate_config
from src.utils.paths import DATA_DIR
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
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
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
from src.agent.daily_bars import utc_bar_date
from src.agent.nse_paper_account import (HOLDING_RULE_REASONS, NsePaperAccount,
                                         execute_stop, order_size)
from src.agent.strategy_attribution import credit_weights
from src.connectors.nse_scraper import REAL_NSE_SOURCES
from src.agent.position_rules import (LIVE_ADD_DEFAULTS, LIVE_EXIT_DEFAULTS, plan_add,
                                      plan_exit, rule, state_from_journal)
from src.agent.cost_model import classify, costs_for
from src.agent.broker_manager import BrokerManager
from src.agent.order_execution_engine import OrderExecutionEngine
from src.agent.realtime_risk_manager import RealTimeRiskManager
from src.agent.performance_analytics import PerformanceAnalytics
from src.agent.heartbeat_monitor import HeartbeatMonitor
from src.agent.risk_calculator import RiskCalculator
from src.agent.position_sizing import (DEFAULT_RISK_PER_TRADE,
                                       volatility_scaled_value)
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
except ImportError as e:
    # Name the actual missing module. This used to read "Flask not available"
    # regardless of cause, which is how a missing torch (pulled in three levels
    # down by the sentiment analyser) presented as a Flask problem and took the
    # REST API and the whole dashboard down with it.
    logger.error(f"API server disabled - could not import TradingAPI: {e}")
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
        self.start_time = time.time()  # for /api/system-health uptime
        self.trading_halted = False  # kill switch: blocks all NEW orders when True
        self.halt_reason = None
        self.halted_at = None
        self.components = {}
        self.order_journal = None  # initialized in start() after brokers connect
        self.decision_journal = None
        self._cycle_decisions = {}
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
                # Report what happened, not what was attempted. This line used
                # to claim success unconditionally, so a service that refused
                # to start still logged "started on port 8080" and the missing
                # health endpoint looked like a network problem.
                if getattr(self.monitoring_service, 'is_running', False):
                    # Include the bind address. Naming only the port is how a
                    # loopback-only server looked healthy for a whole
                    # deployment while the proxy got connection refused.
                    logger.info(
                        "Monitoring service listening on %s:%s",
                        getattr(self.monitoring_service, 'bind_address', '?'),
                        monitoring_port)
                else:
                    logger.error(
                        "Monitoring service did NOT start; /health is "
                        "unavailable, so nothing will report whether this "
                        "agent is alive.")
            
            # Initialize Immutable Audit Journal & Decoupled Task Queue Engine
            from src.agent.audit_journal import AuditJournal
            from src.agent.task_queue import TaskQueueEngine
            self.components['audit_journal'] = AuditJournal()
            self.task_queue_engine = TaskQueueEngine(num_workers=4)
            self.task_queue_engine.start()
            self.components['task_queue_engine'] = self.task_queue_engine

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
            self._restore_halt()
            
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
                logger.error("API server disabled - see the import error above")
            
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

            # Watchlist, Escalation, Ingest, and Self-Assessment Components
            from src.agent.escalation_manager import EscalationManager
            from src.agent.broker_research_ingest import BrokerResearchIngest
            from src.agent.self_assessment import SelfAssessmentEngine
            from src.agent.swarm_state_store import SwarmStateStore
            from src.agent.sector_specialist import SectorSpecialistManager
            
            em = EscalationManager()
            self.components['escalation_manager'] = em
            self.components['research_ingest'] = BrokerResearchIngest(
                self.components['llm_orchestrator'],
                em,
                self.config
            )

            # NSE order-ticket queue: NSE has no broker API, so NSE trade
            # decisions become tickets a human keys into the AIB-AXYS portal.
            from src.agent.nse_order_queue import NseOrderQueue
            self.components['nse_order_queue'] = NseOrderQueue()
            self.components['nse_paper_account'] = NsePaperAccount(
                self.components['nse_order_queue'], self.config)
            if self.config.get('nse_auto_paper_trade', True) and \
                    not self.components['nse_paper_account'].enabled:
                logger.warning("nse_auto_paper_trade is on but nse_paper_trading."
                               "starting_capital_kes is not set: NSE tickets stay "
                               "pending for the operator instead of paper-filling")
            self._last_nse_eval = 0.0
            self._nse_last_price = {}  # symbol -> last-evaluated price (freshness guard)

            # Long-term dividend sleeve: separate NSE accumulation book,
            # isolated from trading positions via the `book` tag on tickets.
            from src.agent.sleeve.fundamentals_store import FundamentalsStore
            from src.agent.sleeve.dividend_ledger import DividendLedger
            from src.agent.sleeve.sleeve_manager import SleeveManager
            self.components['fundamentals_store'] = FundamentalsStore()
            self.components['dividend_ledger'] = DividendLedger()
            self.components['sleeve_manager'] = SleeveManager(
                self.config,
                self.components['nse_order_queue'],
                self.components['fundamentals_store'],
                self.components['dividend_ledger'],
                self.components['llm_orchestrator'],
            )
            self.components['self_assessment'] = SelfAssessmentEngine(
                self.components['llm_orchestrator'],
                em,
                self.config
            )
            
            # Swarm Infrastructure & Specialist Components
            redis_client = None
            dm = self.components.get('data_manager')
            if dm and getattr(dm, 'redis_client', None) is not None:
                redis_client = dm.redis_client
                
            self.components['swarm_state_store'] = SwarmStateStore(redis_client)
            self.components['sector_specialist'] = SectorSpecialistManager(
                self.components['llm_orchestrator']
            )

            
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
            
            # Dead-Man's Switch Heartbeat Monitor
            try:
                self.heartbeat_monitor = HeartbeatMonitor(
                    risk_manager=self.risk_manager,
                    broker_manager=self.components.get('broker_manager'),
                    audit_journal=self.components.get('audit_journal'),
                    timeout_seconds=self.config.get('heartbeat_timeout_seconds', 180),
                    on_trigger=lambda reason: self.halt_trading(reason=reason),
                )
                self.heartbeat_monitor.start_watchdog()
                logger.info("Dead-Man's Switch heartbeat monitor activated (timeout=%ds)", self.config.get('heartbeat_timeout_seconds', 180))
            except Exception as e:
                logger.error(f"Heartbeat monitor initialization failed: {e}")
                self.heartbeat_monitor = None

            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def _position_gate(self, broker, symbol: str, action: str):
        """The backtest's position rules, applied before any order.

        Returns (allowed, skip_reason, held_quantity, avg_entry_price).
        Fails closed: if the
        broker cannot say what is held or pending, no order is placed,
        because an unknown position is exactly how a held symbol gets
        bought again. Brokers exposing strict fetch_positions /
        fetch_open_orders (which raise on failure) are preferred over the
        lenient getters, which return [] on error.
        """
        try:
            fetch_pos = getattr(broker, 'fetch_positions', None)
            positions = fetch_pos() if callable(fetch_pos) else (broker.get_positions() or [])
            fetch_open = getattr(broker, 'fetch_open_orders', None)
            pending = fetch_open(symbol) if callable(fetch_open) else (broker.get_orders(symbol) or [])
        except Exception as e:
            logger.warning(f"Cannot confirm holdings for {symbol}; not trading it this cycle: {e}")
            return False, 'position_unknown', 0.0, 0.0
        mine = [p for p in positions if getattr(p, 'symbol', None) == symbol]
        held = sum(float(getattr(p, 'quantity', 0) or 0) for p in mine)
        avg_entry = float(getattr(mine[0], 'avg_entry_price', 0) or 0) if mine else 0.0
        if any(getattr(o, 'symbol', None) == symbol for o in pending):
            return False, 'order_pending', held, avg_entry
        if action == 'sell' and held <= 0:
            return False, 'no_position', held, avg_entry
        # A buy on a holding is an add: _execute_trades decides whether the
        # position has earned one (position_rules.plan_add).
        return True, None, held, avg_entry

    def _history_needed(self, symbol: Optional[str] = None, required_only: bool = True) -> int:
        """Daily bars a symbol needs before every strategy can signal.

        Per symbol when the strategy manager scopes strategies to markets:
        crypto's 200-day trend strategy does not make an NSE stock "short".
        `required_only` leaves out strategies that abstain until their
        history is deep enough (slow_strategies.py).
        """
        sm = self.components.get('strategy_manager')
        if symbol is not None and hasattr(sm, 'history_needed'):
            return sm.history_needed(symbol, required_only=required_only)
        need = [int(getattr(s, 'lookback_period', 0) or 0)
                for s in (getattr(sm, 'strategies', {}) or {}).values()
                if not (required_only and getattr(s, 'abstains', False))]
        return max(need + [1])

    def _warm_start_history(self) -> None:
        """Seed strategies, regime detector and ATR from real daily bars.

        US and crypto from yfinance (the live feed's own source); NSE from
        the scraper's CSVs, real rows only, topped up from the ~2 weeks of
        real history afx publishes for any symbol still short. Symbols left
        short of history are logged at ERROR, because they cannot signal,
        and retried hourly.
        """
        from src.agent.history_warmstart import fetch_daily_history, read_nse_history
        self._last_history_attempt = time.time()
        sm = self.components.get('strategy_manager')
        if sm is None:
            return
        dm_cfg = self.config.get('data_manager', {})
        live = [s for s in dm_cfg.get('symbols', []) if s not in self._history_seeded]
        trading = TradingAgent.nse_trading_symbols(self)
        if getattr(self, '_nse_active', None) is None:
            self._nse_active = set(trading)
        nse = [s for s in trading if s not in self._history_seeded]
        # What each symbol must have to trade, and how deep to fetch so the
        # slower strategies (a 200-day average, six-month ranking) can vote.
        need_for = {s: self._history_needed(s) for s in live + nse}
        need = max(list(need_for.values()) + [1])
        want = lambda syms: max([self._history_needed(s, required_only=False) for s in syms] + [60])

        histories = {}
        if live:
            histories.update(fetch_daily_history(live, bars=want(live)))
        if nse:
            from src.connectors.nse_connector import NSE_CSV_DIR
            nse_bars = want(nse)
            nse_history = read_nse_history(nse, NSE_CSV_DIR, bars=nse_bars)
            # afx only tops up a symbol that is short. Asking it for every
            # symbol on every start, when the stored NSE history already
            # covers them, held the loop for minutes whenever afx was
            # unreachable and tripped the heartbeat watchdog.
            short_nse = [s for s in nse
                         if len(nse_history.get(s, {}).get('close', [])) < need_for[s]]
            if short_nse:
                try:
                    from src.connectors.nse_scraper import backfill_afx_history
                    if backfill_afx_history(short_nse):
                        nse_history.update(read_nse_history(short_nse, NSE_CSV_DIR, bars=nse_bars))
                except Exception as e:
                    logger.warning(f"NSE real-history backfill failed: {e}")
            histories.update(nse_history)

        if histories:
            sm.warm_start({s: h['close'] for s, h in histories.items()})
            if getattr(self, 'volatility', None):
                for s, h in histories.items():
                    try:
                        self.volatility.warm_start(s, h['close'], h['high'], h['low'])
                    except Exception as e:
                        logger.debug(f"ATR warm-start failed for {s}: {e}")

        depth = {s: len(h['close']) for s, h in histories.items()}
        self._history_seeded |= {s for s, n in depth.items() if n >= need_for.get(s, need)}
        short = {s: depth.get(s, 0) for s in live + nse if s not in self._history_seeded}
        if short:
            logger.error(
                f"History warm-start: {len(short)} symbol(s) have fewer than {need} real "
                f"daily bars and cannot signal yet: {short}. Retrying hourly.")
        elif live or nse:
            logger.info(f"History warm-start: {len(live) + len(nse)} symbols seeded "
                        f"with at least {need} daily bars")

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
        # Close the order/decision journals so SQLite handles release cleanly.
        try:
            if getattr(self, 'order_journal', None):
                self.order_journal.close()
            if getattr(self, 'decision_journal', None):
                self.decision_journal.close()
            if 'escalation_manager' in self.components:
                self.components['escalation_manager'].close()
            if 'nse_order_queue' in self.components:
                self.components['nse_order_queue'].close()
            if 'dividend_ledger' in self.components:
                self.components['dividend_ledger'].close()
            if 'sleeve_manager' in self.components:
                self.components['sleeve_manager'].close()
            if 'self_assessment' in self.components:
                self.components['self_assessment'].close()
            if 'sector_specialist' in self.components:
                self.components['sector_specialist'].close()
        except Exception as e:
            logger.warning(f"Error closing journals and swarm components: {e}")
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

        # Per-symbol ATR, used to scale stop distances to each instrument's
        # own volatility instead of applying one flat percentage to all.
        from src.agent.volatility import VolatilityTracker
        self.volatility = VolatilityTracker(
            length=self.config.get('risk_limits', {}).get('atr_length', 14))

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

        # Decision journal: records WHY the agent acts or holds each cycle,
        # for the per-symbol drill-down. Shares the journal DB.
        try:
            from src.agent.decision_journal import DecisionJournal
            self.decision_journal = DecisionJournal()
        except Exception as e:
            logger.error(f"Decision journal init failed: {e}")
            self.decision_journal = None

        # Warm-start every indicator from real daily history so the agent can
        # signal from the first cycle instead of after fifty trading days.
        # Retried hourly from the trading loop for any symbol still missing.
        self._history_seeded = set()
        self._last_history_attempt = 0.0
        self._warm_start_history()

        # Strategy settings re-tune from results, weekly or sooner for a
        # strategy that is losing, adopting only changes that beat the
        # current settings on held-out history (strategy_tuner.py). Runs in
        # a background thread; saved settings are applied here at startup.
        try:
            from src.agent.strategy_tuner import StrategyTuner, market_history
            from src.utils.paths import DATA_DIR
            self.strategy_tuner = StrategyTuner(
                self.components.get('strategy_manager'), self.config,
                history_source=lambda: market_history(self.config),
                params_path=DATA_DIR / 'strategy_params.json')
        except Exception as e:
            logger.error(f"Strategy tuner init failed: {e}")
            self.strategy_tuner = None

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
                # Pass the configured universe. Without this the scraper
                # falls back to its own hardcoded DEFAULT_SYMBOLS and keeps
                # fetching names that were removed from config, so trimming
                # the universe saved no bandwidth and the data directory
                # filled with series nothing reads.
                nse_cfg_symbols = self.config.get('data_manager', {}).get('nse_symbols')
                self.nse_scraper = NSEPeriodicScraper(
                    database_manager=self.database,
                    nse_connector=nse_connector,
                    interval_minutes=self.config.get('data_manager', {})
                                        .get('nse_scrape_interval_minutes', 30),
                    symbols=nse_cfg_symbols or None,
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
        # Stop distances are resolved per symbol from that symbol's ATR (see
        # src/agent/volatility.py). These two remain only as the cold-start
        # fallback for a symbol with too little history to have an ATR yet.
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
                if getattr(self, 'strategy_tuner', None) and not self.trading_halted:
                    try:
                        self.strategy_tuner.maybe_tune()
                    except Exception as e:
                        logger.error(f"Strategy tuner error: {e}")

                # Pull fill results for submitted orders into the journal so
                # attribution and duplicate-close checks see current state.
                if self.order_journal:
                    try:
                        broker_manager = self.components.get('broker_manager')
                        primary = broker_manager.get_broker() if broker_manager else None
                        if primary and primary.is_connected:
                            self.order_journal.sync_fills(primary)
                            # Feed real broker positions into the risk manager
                            # so trailing stops fire and /api/risk-metrics
                            # reflects actual exposure.
                            if getattr(self, 'risk_manager', None):
                                acct = primary.get_account_info()
                                self.risk_manager.sync_broker_positions(
                                    primary.get_positions(),
                                    cash=acct.cash if acct else 0.0)
                    except Exception as e:
                        logger.debug(f"Position/fill sync error: {e}")

                # Record current equity every cycle so the performance chart
                # and consolidated-equity card reflect live state. Startup
                # only seeds history from the broker's own equity curve
                # (paper accounts return none), so without this the chart
                # and equity figure stay frozen at zero for paper sessions.
                try:
                    broker_manager = self.components.get('broker_manager')
                    primary = broker_manager.get_broker() if broker_manager else None
                    if primary and primary.is_connected:
                        acct = primary.get_account_info()
                        if acct is not None and acct.equity is not None:
                            self.performance_analytics.record_portfolio_value(float(acct.equity))
                except Exception as e:
                    logger.debug(f"Portfolio value recording error: {e}")

                # Refresh strategy performance from REAL journal attribution
                # every 5 minutes, so performance-weighted decisions and the
                # status API run on realized results, not placeholders.
                if self.order_journal and \
                        time.time() - getattr(self, '_last_perf_sync', 0) > 300:
                    self._last_perf_sync = time.time()
                    try:
                        from src.agent.strategy_attribution import compute_attribution
                        attribution = compute_attribution(self.order_journal.filled_orders())
                        if attribution:
                            self.components['strategy_manager'].update_performance_from_attribution(attribution)
                    except Exception as e:
                        logger.debug(f"Performance sync error: {e}")

                if market_data:
                    # --- Stop-loss enforcement ---
                    self._enforce_stop_losses(stop_loss_pct, trailing_stop_pct)
                    
                    # Swarm: Run parallel Sector Specialists before the symbol loop.
                    # Rate-limited: sector outlooks change slowly and each pass
                    # fires one LLM call per sector — running it every 60s cycle
                    # blew straight through free-tier rate limits. Run at most
                    # once per sector_analysis_interval cycles (~once per trading
                    # day by default); the persisted SQLite profiles carry
                    # between passes.
                    import json
                    swarm_enabled = self.config.get("swarm", {}).get("enabled", True)
                    # sector_analysis_interval is expressed in loop cycles; convert
                    # to seconds so the gate doesn't depend on the per-cycle
                    # `cycle` counter (defined later in the loop body).
                    sector_interval_s = self.config.get('sector_analysis_interval', 390) * loop_interval
                    sector_due = (time.time() - getattr(self, '_last_sector_time', 0)) >= sector_interval_s
                    if swarm_enabled and sector_due and 'sector_specialist' in self.components:
                        self._last_sector_time = time.time()
                        try:
                            sector_specialist = self.components['sector_specialist']
                            state_store = self.components['swarm_state_store']
                            
                            symbols_by_sector = {}
                            for sym in symbols:
                                sect = sector_specialist.get_sector_for_symbol(sym)
                                if sect not in symbols_by_sector:
                                    symbols_by_sector[sect] = []
                                symbols_by_sector[sect].append(sym)
                                
                            import concurrent.futures
                            
                            def run_sector_analysis_task(sect, syms):
                                sect_news = []
                                for sym in syms:
                                    if isinstance(market_data, dict) and 'news_data' in market_data:
                                        for source, source_data in market_data['news_data'].items():
                                            data = source_data.get('data', [])
                                            if isinstance(data, dict):
                                                sect_news.extend([n for n in (data.get(sym) or []) if isinstance(n, dict)])
                                            else:
                                                sect_news.extend([n for n in data if isinstance(n, dict) and n.get('symbol') == sym])
                                res = sector_specialist.run_sector_analysis(sect, sect_news)
                                state_store.set(f"swarm:sector_outlook:{sect}", json.dumps(res), ttl_seconds=900)
                                return sect, res
                                
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                futures = {
                                    executor.submit(run_sector_analysis_task, sect, syms): sect
                                    for sect, syms in symbols_by_sector.items()
                                }
                                try:
                                    for fut in concurrent.futures.as_completed(futures, timeout=25.0):
                                        sect = futures[fut]
                                        try:
                                            fut.result()
                                        except Exception as e:
                                            logger.error(f"Failed parallel sector specialist for {sect}: {e}")
                                except concurrent.futures.TimeoutError:
                                    logger.warning("Parallel sector specialist analysis timed out after 25s. Continuing cycle.")
                        except Exception as e:
                            logger.error(f"Error in parallel swarm sector execution: {e}")

                    # Process each symbol individually
                    all_signals = {}
                    # Per-cycle decision records, keyed by symbol. Populated here
                    # and mutated by _execute_trades, then flushed once below so
                    # the drill-down can answer "why did/didn't it trade X?".
                    cycle = int(time.time() // loop_interval)
                    self._cycle_decisions = {}

                    # Live quotes build one bar per trading day (daily_bars.py).
                    # Every minute's quote used to become a bar of its own, so
                    # daily-calibrated strategies measured minutes and never fired.
                    live_bar_date = utc_bar_date()

                    for symbol in symbols:
                        try:
                            # Extract symbol-specific data
                            symbol_data = self._extract_symbol_data(market_data, symbol)
                            if isinstance(symbol_data, dict):
                                symbol_data['bar_date'] = live_bar_date

                            price = (symbol_data or {}).get('price') or (symbol_data or {}).get('close')

                            # Feed the volatility tracker before any decision,
                            # so stop distances reflect the current bar. Uses
                            # high/low when the feed carries them and degrades
                            # to close-only otherwise.
                            if price and getattr(self, 'volatility', None):
                                self.volatility.update(
                                    symbol, price,
                                    (symbol_data or {}).get('high'),
                                    (symbol_data or {}).get('low'),
                                    bar_date=live_bar_date)

                            self._cycle_decisions[symbol] = {
                                'symbol': symbol, 'cycle': cycle, 'action': 'hold',
                                'skip_reason': 'hold', 'price': price,
                                'ensemble_confidence': 0.0, 'per_strategy': {},
                                'llm_verdict': {}, 'executed': False,
                            }

                            # Never place real orders on synthetic fallback
                            # prices — the fallback generator exists to keep
                            # the system alive when vendors fail, not to trade.
                            if symbol_data and symbol_data.get('source') == 'fallback':
                                logger.debug(f"Skipping {symbol}: price is synthetic fallback, not tradeable")
                                self._cycle_decisions[symbol]['skip_reason'] = 'fallback_price'
                                continue

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
                                if modified_params:
                                    # Update momentum & reversion weights if adjusted by goals/performance
                                    mw = modified_params.get('momentum_weight')
                                    mr = modified_params.get('mean_reversion_weight')
                                    if mw is not None and mr is not None:
                                        self.components['strategy_manager'].strategy_weights['momentum'] = mw
                                        self.components['strategy_manager'].strategy_weights['mean_reversion'] = mr
                                
                                # Generate trading signals for this symbol
                                symbol_signals = self.components['strategy_manager'].generate_signals(symbol_data)

                                dec = self._cycle_decisions[symbol]
                                if symbol_signals:
                                    dec['action'] = symbol_signals.get('action', 'hold')
                                    dec['ensemble_confidence'] = symbol_signals.get('confidence', 0.0)
                                    dec['per_strategy'] = {
                                        n: {'action': s.get('action'), 'confidence': s.get('confidence')}
                                        for n, s in (symbol_signals.get('per_strategy') or {}).items()
                                    }

                                if symbol_signals and symbol_signals.get('action') != 'hold':
                                    # Modify position size dynamically based on adaptive integration
                                    confidence = symbol_signals.get('confidence', 0.5)
                                    broker_manager = self.components.get('broker_manager')
                                    account_val = 100000.0
                                    if broker_manager:
                                        primary = broker_manager.get_broker()
                                        if primary:
                                            try:
                                                info = primary.get_account_info()
                                                if info:
                                                    account_val = info.equity
                                            except Exception:
                                                pass
                                    pos_value = self.components['adaptive_integration'].get_position_size(
                                        symbol, confidence, account_val
                                    )
                                    symbol_signals['position_size'] = pos_value / account_val if account_val > 0 else 0.0

                                    # Fetch active research context for this symbol if available in watchlist
                                    research_context = None
                                    em = self.components.get('escalation_manager')
                                    if em:
                                        watchlist_items = em.get_active_watchlist()
                                        for item in watchlist_items:
                                            if item.get('symbol') == symbol:
                                                research_context = item
                                                break
                                    
                                    # Retrieve sector outlook from SwarmStateStore
                                    sector_outlook = None
                                    ss = self.components.get('sector_specialist')
                                    state_store = self.components.get('swarm_state_store')
                                    if ss and state_store:
                                        sect = ss.get_sector_for_symbol(symbol)
                                        cached_out = state_store.get(f"swarm:sector_outlook:{sect}")
                                        if cached_out:
                                            try:
                                                sector_outlook = json.loads(cached_out)
                                            except Exception:
                                                pass

                                    # Compute the agent's own hit-rate track record at most once per
                                    # ~10 minutes (guarded), then feed it into the LLM validation prompt.
                                    if not hasattr(self, '_verdict_scores_at') or time.time() - self._verdict_scores_at > 600:
                                        from src.agent.verdict_scoreboard import score_decisions
                                        from src.utils.real_price_feed import price_feed
                                        journal = getattr(self, 'decision_journal', None)
                                        self._verdict_scores = score_decisions(
                                            journal.recent(None, limit=500) if journal else [],
                                            price_feed.get_price)
                                        self._verdict_scores_at = time.time()
                                    from src.agent.verdict_scoreboard import summary_line

                                    # Validate with LLM Orchestrator
                                    validated_signal = self.components['llm_orchestrator'].validate_trade(
                                        symbol, symbol_signals, symbol_data, symbol_news,
                                        research_context=research_context, sector_outlook=sector_outlook,
                                        track_record=summary_line(self._verdict_scores, symbol)
                                    )
                                    dec['llm_verdict'] = {
                                        'action': validated_signal.get('action'),
                                        'confidence': validated_signal.get('confidence'),
                                        'reasoning': validated_signal.get('reasoning'),
                                    }
                                    if validated_signal.get('action') == 'hold':
                                        dec['skip_reason'] = 'llm_veto'

                                    # Check for Algorithmic/Cognitive Bias
                                    is_biased = self.components['bias_detector'].detect_bias(validated_signal, symbol_data, market_data)
                                    if is_biased:
                                        logger.warning(f"Bias detected for {symbol} trade: {validated_signal}. Lowering confidence.")
                                        validated_signal['confidence'] *= 0.5
                                        if validated_signal['confidence'] < 0.3:
                                            validated_signal['action'] = 'hold'
                                            dec['skip_reason'] = 'bias_downgrade'

                                    if validated_signal['action'] != 'hold':
                                        all_signals[symbol] = validated_signal
                                        dec['action'] = validated_signal['action']
                                        # Reason set to None here means "candidate";
                                        # _execute_trades sets the final outcome.
                                        dec['skip_reason'] = None
                                        
                        except Exception as e:
                            logger.error(f"Error processing symbol {symbol}: {str(e)}")
                            continue
                    
                    # Kill switch: observe the market but submit nothing
                    if self.trading_halted:
                        if all_signals:
                            logger.info(f"Trading halted; dropping signals for {list(all_signals)}")
                            for s in all_signals:
                                if s in self._cycle_decisions:
                                    self._cycle_decisions[s]['skip_reason'] = 'halted'
                        all_signals = {}

                    # Process signals if any were generated
                    if all_signals:
                        # Assess risk for all signals. risk_calculator.calculate_portfolio_risk()
                        # expects historical positions+price_data (a different shape than
                        # what's available here) and its return shape doesn't match what
                        # _check_risk_limits reads either — this call was never reachable
                        # before (all_signals was always empty). Build the flat shape
                        # _check_risk_limits actually expects from data that's already
                        # flowing correctly: the live risk manager's current metrics plus
                        # the position sizes the proposed signals themselves carry.
                        live_metrics = {}
                        if getattr(self, 'risk_manager', None):
                            try:
                                live_metrics = self.risk_manager.get_risk_report().get('current_metrics', {})
                            except Exception as e:
                                logger.debug(f"Risk report unavailable for risk-limit check: {e}")
                        risk_assessment = {
                            'portfolio_var': live_metrics.get('portfolio_var', 0),
                            'current_drawdown': live_metrics.get('max_drawdown', 0),
                            'position_sizes': {sym: sig.get('position_size', 0) for sym, sig in all_signals.items()},
                        }

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
                            for s in all_signals:
                                if s in self._cycle_decisions:
                                    self._cycle_decisions[s]['skip_reason'] = 'risk_limits'

                    # Flush this cycle's decisions (change-detected inside).
                    if self.decision_journal and self._cycle_decisions:
                        for dec in self._cycle_decisions.values():
                            try:
                                self.decision_journal.record(dec)
                            except Exception as e:
                                logger.debug(f"Decision record error: {e}")

                    # Update portfolio optimization (once daily, not every loop)
                    self._update_portfolio_optimization(market_data)

                    # Self-assessment: once daily (aligned to end-of-day).
                    assessment_interval = self.config.get('self_assessment', {}).get('assessment_interval', 390)
                    if cycle > 0 and cycle % assessment_interval == 0:
                        try:
                            logger.info("Executing periodic self-assessment and self-improvement loop...")
                            engine = self.components.get('self_assessment')
                            if engine:
                                plan = engine.run_assessment(
                                    self.decision_journal,
                                    self.order_journal,
                                    self.performance_analytics,
                                    cycles_to_review=assessment_interval
                                )
                                # Auto-apply safe changes
                                auto_applied, escalated = engine.apply_improvements(plan, require_approval=False)
                                logger.info(f"Retrospective assessment completed: applied {len(auto_applied)} auto-improvements, escalated {len(escalated)} structural proposals.")

                                # Prune expired local cache entries to prevent memory leaks in local offline mode
                                if state_store:
                                    pruned = state_store.clear_expired()
                                    if pruned > 0:
                                        logger.info(f"SwarmStateStore: Pruned {pruned} expired local cache entries.")

                            # Universe scout: propose untracked NSE movers as operator escalations.
                            # This NEVER auto-trades — it only creates a pending escalation; the
                            # symbol is added to the runtime universe/watchlist solely on operator approval.
                            try:
                                from src.agent.universe_scout import propose_candidates
                                nse = self.components.get('data_manager')
                                em = self.components.get('escalation_manager')
                                from src.agent import nse_screener
                                # With the screener on, every listed stock is
                                # already screened weekly; proposing one would
                                # only ask the operator to approve a no-op.
                                if nse_screener.settings(self.config).get('enabled'):
                                    em = None
                                if em and nse:
                                    tracked = set(self.nse_trading_symbols())
                                    movers = []
                                    nse_conn = getattr(nse, 'connectors', {}).get('nse')
                                    if nse_conn:
                                        quotes = nse_conn.get_all_quotes()
                                        movers = [{'symbol': q.get('symbol'), 'change_pct': q.get('change_pct')} for q in quotes]
                                    for cand in propose_candidates(tracked, movers, [])[:3]:
                                        em.create_escalation(None, cand['symbol'], 'add_symbol', cand['reason'], 'low')
                            except Exception as e:
                                logger.warning(f"Universe scout skipped: {e}")
                        except Exception as e:
                            logger.error(f"Error in self-assessment cycle: {e}")
                
                # Retry the history warm-start hourly for any symbol still
                # without enough daily bars. A vendor blip at boot would
                # otherwise leave those symbols unable to signal for weeks.
                try:
                    configured = (set(self.config.get('data_manager', {}).get('symbols', []))
                                  | set(self.nse_trading_symbols()))
                    if (configured - getattr(self, '_history_seeded', set())
                            and time.time() - getattr(self, '_last_history_attempt', 0) > 3600):
                        self._warm_start_history()
                except Exception as e:
                    logger.error(f"History warm-start retry failed: {e}")

                # NSE Kenya decision pass — separate cadence from the US loop
                # (NSE prices only refresh on the ~30-min scrape). Internally
                # rate-limited and guarded by market hours + price freshness.
                try:
                    self._evaluate_nse_symbols()
                except Exception as e:
                    logger.error(f"NSE evaluation error: {e}")

                # Long-term dividend sleeve — separate cadence again (at most
                # once per calendar month; SleeveManager gates itself).
                try:
                    self._run_sleeve_cycle()
                except Exception as e:
                    logger.error(f"Sleeve cycle error: {e}")

                # Core of the core-satellite split: index funds, rebalanced
                # monthly (core_portfolio.py). Checked every 15 minutes.
                try:
                    self._run_core_cycle()
                except Exception as e:
                    logger.error(f"Core portfolio error: {e}")

                # Dead-Man's Switch heartbeat ping
                if getattr(self, 'heartbeat_monitor', None):
                    self.heartbeat_monitor.ping()

                # Calculate sleep time to maintain consistent loop interval
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, loop_interval - loop_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(loop_interval)  # Wait before retrying

    def _evaluate_nse_symbols(self):
        """Run the standard decision pipeline over NSE symbols and turn
        non-hold, LLM-approved signals into operator order tickets (NSE has no
        broker API to auto-execute against). Rate-limited to the scrape
        cadence and guarded by market hours + per-symbol price freshness."""
        queue = self.components.get('nse_order_queue')
        dm = self.components.get('data_manager')
        nse = dm.connectors.get('nse') if dm and hasattr(dm, 'connectors') else None
        strategy_manager = self.components.get('strategy_manager')
        if not queue or not nse or not strategy_manager:
            return

        interval = self.config.get('data_manager', {}).get('nse_eval_interval', 1800)
        now = time.time()
        if now - getattr(self, '_last_nse_eval', 0) < interval:
            return
        if not nse.is_market_open():
            return
        self._last_nse_eval = now

        nse_cfg = self.config.get('nse_order_tickets', {})
        try:
            queue.expire_stale(nse_cfg.get('max_age_hours', 24))
        except Exception as e:
            logger.debug(f"NSE ticket expiry skipped: {e}")

        llm = self.components.get('llm_orchestrator')
        paper = self.components.get('nse_paper_account')
        auto_paper = (self.config.get('nse_auto_paper_trade', True)
                      and paper is not None and paper.enabled)
        base_notional = nse_cfg.get('trade_notional_kes', 50000)
        cycle = int(now // interval)
        # This week's short list from the screen of the whole exchange, plus
        # every holding (nse_screener.py); the configured list if it is off.
        try:
            TradingAgent._refresh_nse_shortlist(self)
        except Exception as e:
            logger.error(f"NSE shortlist refresh failed; keeping the last list: {e}")
        nse_symbols = TradingAgent.nse_trading_symbols(self)
        from src.agent import nse_screener
        stale_days = int(nse_screener.settings(self.config)['stale_after_days'])
        try:
            TradingAgent._sync_nse_state(self, nse_symbols)
        except Exception as e:
            logger.error(f"NSE trading list sync failed: {e}")
        real_prices = {}  # this cycle's real quotes, for the equity snapshot

        for symbol in nse_symbols:
            try:
                quote = nse.get_quote(symbol)
                if not quote or quote.get('_stale') or not quote.get('price_kes'):
                    continue
                price = float(quote['price_kes'])
                if price <= 0:
                    continue
                if quote.get('source') in REAL_NSE_SOURCES:
                    real_prices[symbol] = price

                # Never evaluate on synthetic seed prices. The scraper seeds
                # 730 days of generated history on first run and overlays
                # real bars only when a live source answers; if every live
                # scrape fails, the newest "price" is invented. The US and
                # crypto loop already refuses synthetic prices; this path did
                # not, and it proposes tickets the operator places with real
                # money. Recorded as fallback_price (the existing reason for
                # "price is not real") so it shows on the dashboard rather
                # than disappearing, and checked before the freshness guard
                # so an invented price never enters the strategy buffers.
                if quote.get('source') == 'synthetic':
                    if self.decision_journal:
                        try:
                            self.decision_journal.record({
                                'symbol': symbol, 'cycle': cycle, 'action': 'hold',
                                'skip_reason': 'fallback_price', 'price': price,
                                'ensemble_confidence': 0.0, 'per_strategy': {},
                                'llm_verdict': {}, 'executed': False,
                            })
                        except Exception as e:
                            logger.debug(f"NSE decision record error: {e}")
                    continue

                # Never act on an old price. A suspended or untraded stock
                # keeps its last bar for weeks; a stop or a fill at that
                # price is not one the market would give.
                age = TradingAgent._nse_price_age_days(quote)
                if age is not None and age > stale_days:
                    if self.decision_journal:
                        try:
                            self.decision_journal.record({
                                'symbol': symbol, 'cycle': cycle, 'action': 'hold',
                                'skip_reason': 'stale_price', 'price': price,
                                'ensemble_confidence': 0.0, 'per_strategy': {},
                                'llm_verdict': {}, 'executed': False,
                            })
                        except Exception as e:
                            logger.debug(f"NSE decision record error: {e}")
                    continue

                # Stops before signals, and whatever the signals say: a paper
                # position below its stop-loss or trailing stop is closed now,
                # without an LLM call.
                if auto_paper:
                    tracker = getattr(self, 'volatility', None)
                    try:
                        atr_pct = tracker.atr_pct(symbol) if tracker is not None else None
                    except Exception:
                        atr_pct = None
                    hit = paper.stop_check(symbol, price, atr_pct)
                    if hit:
                        reason, qty, why = hit
                        fill = execute_stop(paper, queue, symbol, price, reason, qty, why,
                                            order_journal=self.order_journal)
                        logger.warning(f"NSE {symbol} {why}; "
                                       f"{'sold' if fill else 'could not sell'} {qty} shares")
                        if self.decision_journal:
                            try:
                                self.decision_journal.record({
                                    'symbol': symbol, 'cycle': cycle, 'action': 'sell',
                                    'skip_reason': None if fill else 'duplicate',
                                    'price': price, 'ensemble_confidence': 1.0,
                                    'per_strategy': {reason: {'action': 'sell', 'confidence': 1.0}},
                                    'llm_verdict': {}, 'executed': bool(fill),
                                })
                            except Exception as e:
                                logger.debug(f"NSE decision record error: {e}")
                        continue

                # Freshness guard: skip a symbol whose price hasn't moved since
                # last eval so we don't feed a repeated stale price into the
                # strategies' rolling SMA/RSI buffers.
                last = self._nse_last_price.get(symbol)
                if last is not None and abs(price - last) < 1e-9:
                    continue
                self._nse_last_price[symbol] = price

                symbol_data = {
                    'symbol': symbol, 'price': price, 'close': price,
                    'open': float(quote.get('open_kes') or price),
                    'volume': quote.get('volume', 0),
                    'source': quote.get('source', 'nse'), 'market': 'nse',
                    # Intraday re-scrapes update today's bar instead of each
                    # becoming a bar (daily_bars.py). The CSV warm-start is
                    # already daily.
                    'bar_date': utc_bar_date(),
                }
                signals = strategy_manager.generate_signals(symbol_data)

                dec = {
                    'symbol': symbol, 'cycle': cycle, 'action': 'hold',
                    'skip_reason': 'hold', 'price': price,
                    'ensemble_confidence': 0.0, 'per_strategy': {},
                    'llm_verdict': {}, 'executed': False,
                }
                if signals:
                    dec['action'] = signals.get('action', 'hold')
                    dec['ensemble_confidence'] = signals.get('confidence', 0.0)
                    dec['per_strategy'] = {
                        n: {'action': s.get('action'), 'confidence': s.get('confidence')}
                        for n, s in (signals.get('per_strategy') or {}).items()
                    }

                validated = signals
                proposed = (signals or {}).get('action', 'hold')
                blocked = None
                adv = TradingAgent._nse_adv(symbol, (paper.limits if auto_paper else {}).get('adv_days', 20)) \
                    if proposed in ('buy', 'sell') else None
                if proposed in ('buy', 'sell'):
                    # Holding rules first, so a trade the book cannot take (an
                    # add the holding hasn't earned, a sell of nothing) never
                    # costs an LLM call. Cash is checked once the size is known.
                    _, blocked = order_size(symbol, proposed, price, base_notional, queue, paper,
                                            confidence=float(signals.get('confidence') or 0.0),
                                            adv=adv)
                    if blocked not in HOLDING_RULE_REASONS:
                        blocked = None
                if blocked:
                    dec['skip_reason'] = blocked
                    validated = {'action': 'hold', 'confidence': 0.0}
                elif signals and proposed != 'hold' and llm:
                    validated = llm.validate_trade(symbol, signals, symbol_data, None,
                                                   research_context=TradingAgent._nse_research(self, symbol))
                    dec['llm_verdict'] = {
                        'action': validated.get('action'),
                        'confidence': validated.get('confidence'),
                        'reasoning': validated.get('reasoning'),
                    }
                    if validated.get('action') == 'hold':
                        dec['skip_reason'] = 'llm_veto'

                action = (validated or {}).get('action', 'hold')
                confidence = (validated or {}).get('confidence', 0.0)
                if action in ('buy', 'sell') and confidence > 0.1 and not self.trading_halted:
                    position_size = (validated.get('position_size')
                                     or signals.get('position_size') or 0.1)
                    notional = base_notional * min(max(position_size, 0.1), 1.0)
                    # Adds only to proven winners, no selling what isn't held,
                    # a weaker sell trims and a strong one closes, a buy fits
                    # the cash (position_rules.py).
                    qty, reason = order_size(symbol, action, price, notional, queue, paper,
                                             confidence=float(confidence or 0.0), adv=adv)
                    if reason:
                        dec['skip_reason'] = reason
                    else:
                        ticket_id = queue.create_ticket(
                            symbol, action, qty, suggested_limit_price=round(price, 2),
                            rationale=f"{action.upper()} signal (ensemble conf {confidence:.2f})",
                            ensemble_confidence=confidence,
                            llm_reasoning=(validated.get('reasoning') if validated else '') or '',
                            # Credit the strategies that voted for it, so NSE
                            # results feed the performance weighting too.
                            strategy='ensemble',
                            strategy_weights=credit_weights(
                                (signals or {}).get('per_strategy'), action) or {'ensemble': 1.0})
                        if ticket_id:
                            dec['skip_reason'] = None  # actionable signal, ticket queued
                            logger.info(
                                f"NSE order ticket #{ticket_id}: {action} {qty} {symbol} "
                                f"@ {price:.2f} KES (conf {confidence:.2f})")
                            
                            # The agent fills its own ticket as a paper trade:
                            # slippage in the price, commission as fees, against
                            # the paper account's cash (nse_paper_account.py).
                            if auto_paper:
                                ok_fill, fill_data = paper.fill(
                                    ticket_id, action, price, qty,
                                    order_journal=self.order_journal)
                                if ok_fill:
                                    dec['executed'] = True
                                    logger.info(
                                        f"NSE paper fill, ticket #{ticket_id}: {action} {qty} "
                                        f"{symbol} @ {fill_data['fill_price']:.2f} KES, fees "
                                        f"{fill_data['fees_kes']:.2f}; cash now "
                                        f"{paper.cash():,.2f} KES")
                        else:
                            dec['skip_reason'] = 'duplicate'
                elif self.trading_halted and action in ('buy', 'sell'):
                    dec['skip_reason'] = 'halted'

                if self.decision_journal:
                    try:
                        self.decision_journal.record(dec)
                    except Exception as e:
                        logger.debug(f"NSE decision record error: {e}")
            except Exception as e:
                logger.error(f"Error evaluating NSE symbol {symbol}: {e}")

        # Today's reading of the paper account's value, for its equity curve.
        # Skipped when a holding has no real quote this cycle: a gap in the
        # curve is honest, a holding valued at cost is not.
        if auto_paper:
            try:
                if all(s in real_prices for s in paper.positions()):
                    paper.record_equity(real_prices)
            except Exception as e:
                logger.debug(f"NSE paper equity snapshot failed: {e}")

    def _nse_research(self, symbol: str) -> Optional[Dict[str, Any]]:
        """What AIB-AXYS says about an NSE stock, for the AI's trade review:
        an analyst's rating from an uploaded note (the watchlist), and the
        Market Pulse's fundamentals and announcements (market_pulse.py).
        The NSE review used to be given no research at all, although the
        broker covers only NSE stocks."""
        from src.agent.market_pulse import research_context
        rated = None
        em = self.components.get('escalation_manager')
        if em is not None:
            try:
                rated = next((w for w in em.get_active_watchlist()
                              if str(w.get('symbol', '')).upper() == symbol.upper()), None)
            except Exception as e:
                logger.debug(f"Watchlist unavailable for {symbol}: {e}")
        try:
            pulse = research_context(symbol)
        except Exception as e:
            logger.debug(f"Market Pulse context unavailable for {symbol}: {e}")
            pulse = None
        if not rated:
            return pulse
        ctx = {'recommendation': rated.get('recommendation'), 'target_price': rated.get('target_price'),
               'rationale': rated.get('rationale') or ''}
        if pulse:
            ctx['rationale'] = f"{ctx['rationale']} {pulse['rationale']}".strip()
        return ctx

    @staticmethod
    def _nse_price_age_days(quote: Dict[str, Any], today=None) -> Optional[int]:
        """Calendar days since the quote's bar, or None when it has no date."""
        from src.connectors.nse_connector import EAT_OFFSET
        try:
            bar_day = datetime.fromisoformat(str(quote.get('timestamp') or '')[:10]).date()
        except ValueError:
            return None
        today = today or datetime.now(EAT_OFFSET).date()
        return (today - bar_day).days

    def _nse_held(self) -> List[str]:
        """Every NSE stock held for trading: the paper account's positions
        when it is on, otherwise the trading book's recorded fills (the
        manual AIB-AXYS workflow, where real shares are bought by ticket)."""
        try:
            paper = self.components.get('nse_paper_account')
            if paper is not None and getattr(paper, 'enabled', False):
                return [s.upper() for s in paper.positions()]
            queue = self.components.get('nse_order_queue')
            if queue is not None:
                from src.agent.nse_paper_account import BOOK
                return [s.upper() for s, p in queue.positions(book=BOOK).items()
                        if (p or {}).get('quantity', 0) > 0]
        except Exception as e:
            logger.warning(f"NSE holdings unavailable for the trading list: {e}")
        return []

    def nse_trading_symbols(self) -> List[str]:
        """The NSE stocks the agent evaluates: the screener's short list, or
        the configured list (data_manager.nse_symbols) while the screener is
        off or has not run, plus every stock held. Holdings are always in,
        whatever the list says, because their stops and exits are checked
        only for the stocks evaluated here."""
        from src.agent import nse_screener
        from src.connectors.nse_universe import register
        configured = [s.upper() for s in self.config.get('data_manager', {}).get('nse_symbols', [])]
        symbols = configured
        if nse_screener.settings(self.config).get('enabled'):
            sl = nse_screener.ShortlistStore(DATA_DIR / 'nse_shortlist.json').load()
            if sl and sl.symbols:
                symbols = list(sl.symbols)
        held = TradingAgent._nse_held(self)
        out = symbols + [h for h in held if h not in symbols]
        register(out)
        return out

    def _sync_nse_state(self, symbols: List[str]) -> None:
        """Forget stocks that left the NSE trading list and warm-start the
        ones that joined, so the strategies only ever compare the stocks
        traded now, each on a continuous history."""
        current = set(symbols)
        previous = getattr(self, '_nse_active', None)
        self._nse_active = current
        if previous is None:
            return
        gone = previous - current
        if gone:
            sm = self.components.get('strategy_manager')
            if sm is not None and hasattr(sm, 'forget'):
                sm.forget(gone)
            tracker = getattr(self, 'volatility', None)
            for symbol in gone:
                if tracker is not None:
                    tracker.forget(symbol)
                getattr(self, '_nse_last_price', {}).pop(symbol, None)
            self._history_seeded -= gone
            logger.info(f"NSE trading list: dropped {sorted(gone)}")
        joined = current - previous
        if joined - self._history_seeded:
            logger.info(f"NSE trading list: added {sorted(joined)}; loading their history")
            self._warm_start_history()

    def _refresh_nse_shortlist(self, now: Optional[datetime] = None, llm=None):
        """Rebuild the NSE short list when it is due (weekly by default)."""
        from src.agent import nse_screener
        from src.connectors.nse_connector import NSE_CSV_DIR
        from src.connectors.nse_universe import market_symbols
        cfg = nse_screener.settings(self.config)
        if not cfg.get('enabled'):
            return None
        store = nse_screener.ShortlistStore(DATA_DIR / 'nse_shortlist.json')
        if not store.due(cfg, now):
            return None
        configured = [s.upper() for s in self.config.get('data_manager', {}).get('nse_symbols', [])]
        ranked = nse_screener.screen(market_symbols(NSE_CSV_DIR, configured), NSE_CSV_DIR, cfg)
        held = TradingAgent._nse_held(self)
        sl = nse_screener.refresh(ranked, held, cfg, configured,
                                  llm or self.components.get('llm_orchestrator'), now)
        store.save(sl)
        logger.info(f"NSE short list rebuilt from {sum(m.eligible for m in ranked)} eligible of "
                    f"{len(ranked)} stocks: {', '.join(f'{s} ({sl.roles[s]})' for s in sl.symbols)}"
                    + (f"; the AI removed {[r['symbol'] for r in sl.removed]}" if sl.removed else ''))
        return sl

    @staticmethod
    def _nse_adv(symbol: str, days: int = 20) -> Optional[float]:
        """Average daily volume in shares over the last `days` real NSE bars,
        or None with fewer than five days of volume (no cap is applied then)."""
        try:
            from src.agent.chart_data import nse_bars
            from src.connectors.nse_connector import NSE_CSV_DIR
            vols = [b['volume'] for b in nse_bars(symbol, NSE_CSV_DIR, int(days)) if b.get('volume')]
            return sum(vols) / len(vols) if len(vols) >= 5 else None
        except Exception as e:
            logger.debug(f"No NSE volume for {symbol}: {e}")
            return None

    def _run_sleeve_cycle(self):
        """Monthly dividend-sleeve accumulation pass. Pulls current NSE
        quotes for the sleeve's configured universe and hands them to
        SleeveManager, which gates itself to once per calendar month."""
        sleeve = self.components.get('sleeve_manager')
        dm = self.components.get('data_manager')
        nse = dm.connectors.get('nse') if dm and hasattr(dm, 'connectors') else None
        if not sleeve or not nse:
            return

        quotes = {}
        for symbol in sleeve.universe:
            quote = nse.get_quote(symbol)
            if quote and not quote.get('_stale') and quote.get('price_kes'):
                quotes[symbol] = float(quote['price_kes'])

        results = sleeve.run_monthly_cycle(quotes)
        if results:
            logger.info(f"Sleeve cycle generated {len(results)} accumulation "
                       f"ticket(s): {[r['symbol'] for r in results]}")

    def _core(self):
        """The core-satellite settings and saved state, read once."""
        if getattr(self, '_core_cfg', None) is None:
            from src.agent.core_portfolio import CoreState, config_from_dict
            dm_cfg = self.config.get('data_manager', {})
            self._core_cfg = config_from_dict(self.config.get('core_satellite'),
                                              dm_cfg.get('symbols', []))
            self._core_state = CoreState(str(DATA_DIR / 'core_portfolio.json'))
            self._core_last_check = 0.0
            self._core_value = 0.0
        return self._core_cfg, self._core_state

    def _core_symbols(self) -> set:
        cfg, _ = TradingAgent._core(self)
        return set(cfg.symbols) if cfg.enabled else set()

    def _run_core_cycle(self, now: Optional[datetime] = None, fetch_history=None,
                        price_for=None) -> List[Dict[str, Any]]:
        """Keep the core at its target weights.

        Trades only in the US regular session, never while halted, and only
        when core_portfolio.rebalance_reason says so: the first build, the
        first session of a month, or a fund drifting past its band. Returns
        the orders placed.
        """
        from src.agent import core_portfolio as core
        cfg, state = self._core()
        if not cfg.enabled or self.trading_halted:
            return []
        if now is None and time.time() - self._core_last_check < 900:
            return []
        self._core_last_check = time.time()
        now = now or datetime.now(timezone.utc)
        bm = self.components.get('broker_manager')
        broker = bm.get_broker() if bm else None
        if not broker or not broker.is_connected:
            return []
        acct = broker.get_account_info()
        if not acct or not acct.equity:
            return []
        fetch_pos = getattr(broker, 'fetch_positions', None)
        positions = fetch_pos() if callable(fetch_pos) else (broker.get_positions() or [])
        held, values, prices = {}, {}, {}
        for p in positions:
            sym = str(getattr(p, 'symbol', '')).upper()
            if sym in cfg.symbols:
                qty = float(getattr(p, 'quantity', 0) or 0)
                px = float(getattr(p, 'current_price', 0) or 0)
                held[sym] = qty
                values[sym] = float(getattr(p, 'market_value', 0) or qty * px)
                if px > 0:
                    prices[sym] = px
        # Kept fresh around the clock: crypto sizing reads it overnight.
        self._core_value = sum(values.values())
        if not core.us_session_open(now):
            return []

        targets = state.data.get('targets') or {s: 1 / len(cfg.symbols) for s in cfg.symbols}
        reason = core.rebalance_reason(state.last_rebalance, now.date(), values, targets,
                                       cfg.drift_band)
        if not reason:
            return []
        fetch_open = getattr(broker, 'fetch_open_orders', None)
        for sym in cfg.symbols:
            pending = fetch_open(sym) if callable(fetch_open) else (broker.get_orders(sym) or [])
            if any(getattr(o, 'symbol', '').upper() == sym for o in pending):
                logger.info(f"Core rebalance waits: an order for {sym} is still working")
                return []

        if fetch_history is None:
            from src.agent.history_warmstart import fetch_daily_history
            fetch_history = fetch_daily_history
        history = fetch_history(list(cfg.symbols), bars=cfg.vol_lookback + 1)
        closes = {s: (history.get(s) or {}).get('close', []) for s in cfg.symbols}
        targets = core.inverse_vol_weights(closes, cfg.vol_lookback, cfg.max_weight)
        if price_for is None:
            from src.utils.real_price_feed import price_feed
            price_for = price_feed.get_price
        for sym in cfg.symbols:
            if sym not in prices:
                px = price_for(sym) or (closes[sym][-1] if closes[sym] else None)
                if px:
                    prices[sym] = float(px)

        core_value = float(acct.equity) * cfg.core_share
        planned = core.plan_orders(targets, core_value, held, prices, cfg.min_trade_usd)
        planned = core.fit_to_cash(planned, float(getattr(acct, 'cash', 0) or 0))
        placed = self._place_core_orders(broker, planned, prices, now)
        state.save(last_rebalance=now.date().isoformat(), targets=targets, reason=reason,
                   core_value_target=round(core_value, 2))
        logger.info(f"Core rebalance ({reason}): targets "
                    f"{ {s: round(w, 3) for s, w in targets.items()} }, "
                    f"{len(placed)} order(s)")
        return placed

    def _place_core_orders(self, broker, planned, prices, now) -> List[Dict[str, Any]]:
        from src.agent import core_portfolio as core
        from src.agent.order_journal import make_client_order_id
        from src.agent.position_sizing import size_order
        from src.connectors.base_broker import OrderRequest
        sizing_cfg = self.config.get('trading', {})
        allow_fractional = sizing_cfg.get('allow_fractional', True)
        day_key = int(now.strftime('%Y%m%d'))
        placed = []
        for o in planned:
            sym, side = o['symbol'], o['side']
            if side == 'sell':
                qty = o['quantity'] if allow_fractional else float(math.floor(o['quantity']))
                tif = 'day'
                if qty <= 0:
                    continue
            else:
                sized = size_order(o['notional'], prices[sym],
                                   min_notional=sizing_cfg.get('min_notional', 5.0),
                                   allow_fractional=allow_fractional)
                if not sized:
                    continue
                qty, tif = sized.quantity, sized.time_in_force
            coid = make_client_order_id(core.STRATEGY, sym, side, day_key)
            if self.order_journal and not self.order_journal.record_intent(
                    coid, sym, side, float(qty), 'market', strategy=core.STRATEGY,
                    strategy_weights={core.STRATEGY: 1.0}):
                logger.info(f"Core order already journaled today ({coid}); skipping")
                continue
            result = broker.place_order(OrderRequest(
                symbol=sym, quantity=float(qty), side=side, order_type='market',
                time_in_force=tif, client_order_id=coid))
            if self.order_journal:
                if result:
                    self.order_journal.mark_submitted(coid, result.order_id, result.status)
                else:
                    self.order_journal.mark_failed(coid, 'place_order returned no response')
            if result:
                placed.append({**o, 'quantity': qty, 'client_order_id': coid})
        return placed

    def _core_value_now(self) -> float:
        """Market value of the core's funds at the last core check."""
        cfg, _ = TradingAgent._core(self)
        return float(getattr(self, '_core_value', 0.0) or 0.0) if cfg.enabled else 0.0

    def _active_equity(self, equity: float) -> float:
        """The strategies' share of equity under the core-satellite split."""
        from src.agent.core_portfolio import active_equity
        cfg, _ = TradingAgent._core(self)
        return active_equity(equity, cfg)

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
        # Mark the final outcome for a symbol on this cycle's decision record.
        cycle_dec = getattr(self, '_cycle_decisions', {})

        def _note(sym, reason=None, executed=False, coid=None, target=None):
            d = cycle_dec.get(sym)
            if not d:
                return
            if reason is not None:
                d['skip_reason'] = reason
            if executed:
                d['executed'] = True
                d['skip_reason'] = None
            if coid:
                d['client_order_id'] = coid
            if target is not None:
                d['target_value'] = target

        try:
            broker_manager = self.components['broker_manager']

            for symbol, signal_data in signals.items():
                if isinstance(signal_data, dict):
                    action = signal_data.get('action', 'hold')
                    confidence = signal_data.get('confidence', 0.0)
                    position_size = signal_data.get('position_size', 0.0)

                    if action != 'hold' and confidence <= 0.1:
                        _note(symbol, 'below_confidence')

                    if action != 'hold' and confidence > 0.1 and position_size > 0:
                        # Determine asset type for routing
                        asset_type = 'stock'
                        if symbol.endswith('-USD') or symbol in ['BTC-USD', 'ETH-USD']:
                            asset_type = 'crypto'
                        elif '_' in symbol:
                            asset_type = 'forex'

                        # Route to appropriate broker. Stocks go to the
                        # PRIMARY broker (Alpaca in prod) — the same book that
                        # stop-losses, PDT guard, journal reconcile/fill-sync
                        # and /api/portfolio all read. Hardcoding paper_broker
                        # here split entries onto one book while exits fired
                        # against another.
                        broker = None
                        if asset_type == 'crypto':
                            broker = broker_manager.get_broker('coinbase_broker')
                            if not broker or not getattr(broker, 'is_connected', False):
                                broker = broker_manager.get_broker()  # Fallback to paper broker
                        elif asset_type == 'forex':
                            broker = broker_manager.get_broker('oanda_broker')
                            if not broker or not getattr(broker, 'is_connected', False):
                                broker = broker_manager.get_broker()  # Fallback to paper broker
                        else:
                            broker = broker_manager.get_broker()  # primary
                            
                        if not broker or not broker.is_connected:
                            logger.warning(f"No connected broker for {symbol} ({asset_type})")
                            _note(symbol, 'no_broker')
                            continue

                        # Trade the way the backtest that justified this did:
                        # never add to a held position, never open a short,
                        # never stack an order behind one still working. The
                        # loop re-decides every minute and the idempotency key
                        # changes every minute, so without this a buy signal
                        # that holds all day places a fresh full-size buy
                        # every cycle: 390 a session per stock.
                        allowed, why, held_qty, avg_entry = self._position_gate(broker, symbol, action)
                        if not allowed:
                            _note(symbol, why)
                            continue
                        trading_cfg = self.config.get('trading', {})
                        add_rule = rule(LIVE_ADD_DEFAULTS, trading_cfg.get('add_to_winners'))
                        exit_rule = rule(LIVE_EXIT_DEFAULTS, trading_cfg.get('exits'))
                        is_add = action == 'buy' and held_qty > 0
                        if is_add and classify(symbol, self.config) not in add_rule.get('markets', []):
                            _note(symbol, 'already_held')
                            continue

                        # Calculate quantity (Max 2% of portfolio per trade for safety).
                        # Skip the cycle if account info is unavailable rather
                        # than sizing off a phantom $100k — on a small live
                        # account that would massively over-size the order.
                        account_info = broker.get_account_info()
                        if not account_info or not account_info.equity:
                            logger.warning(f"No account info for {symbol}; skipping (won't size off a default)")
                            _note(symbol, 'no_account_info')
                            continue
                        # Under the core-satellite split the strategies size
                        # from their own share of equity, and "deployed" is
                        # measured within that share, not across the core.
                        equity = float(account_info.equity)
                        portfolio_value = TradingAgent._active_equity(self, equity)
                        cash = float(getattr(account_info, 'cash', 0) or 0)
                        active_invested = max(equity - cash - TradingAgent._core_value_now(self), 0.0)
                        deployed_pct = active_invested / portfolio_value if portfolio_value else 0.0
                        if not hasattr(self, 'cash_policy'):
                            from src.agent.cash_policy import CashDeploymentPolicy
                            self.cash_policy = CashDeploymentPolicy(self.config)

                        # Size off the stop distance, the same way the backtest
                        # does. This loop used to size as a flat fraction of
                        # equity (confidence * max_position_size, capped by the
                        # cash policy) and never looked at volatility, even
                        # though the stops right below it do. The result was
                        # the same notional in an NSE bank moving 1% a day and
                        # a crypto pair moving 6%, which is six times the risk
                        # for the same signal — and it meant the live agent
                        # sized positions by a method the backtest never
                        # measured, so backtest results did not describe it.
                        limits = self.config.get('risk_limits', {})
                        atr_pct = None
                        tracker = getattr(self, 'volatility', None)
                        if tracker is not None:
                            try:
                                atr_pct = tracker.atr_pct(symbol)
                            except Exception as e:
                                logger.debug(f"No ATR for {symbol}: {e}")

                        risk_per_trade = float(
                            self.config.get('trading', {}).get('risk_per_trade',
                                                               DEFAULT_RISK_PER_TRADE))
                        # Conviction and spare cash stretch the risk budget;
                        # they no longer compete with the position cap.
                        risk_per_trade *= self.cash_policy.risk_multiplier(
                            deployed_pct, confidence)

                        target_pos_value = volatility_scaled_value(
                            portfolio_value, atr_pct,
                            confidence=confidence,
                            risk_per_trade=risk_per_trade,
                            stop_atr_mult=float(limits.get('stop_loss_atr_mult', 2.5)),
                            max_position_pct=float(limits.get('max_position_size', 0.05)))

                        # Known risk windows (elections, FOMC) scale sizing
                        # down deterministically — cycles are treated as
                        # volatility regimes, not return predictions.
                        if not hasattr(self, 'event_calendar'):
                            from src.agent.event_calendar import EventCalendar
                            self.event_calendar = EventCalendar()
                        target_pos_value *= self.event_calendar.risk_multiplier()
                        
                        # Fetch price
                        from src.utils.real_price_feed import price_feed
                        price = signal_data.get('price') or price_feed.get_price(symbol)
                        if not price or price <= 0:
                            logger.warning(f"No valid price for {symbol} — skipping order")
                            _note(symbol, 'no_price')
                            continue

                        _note(symbol, target=target_pos_value)

                        if self.trading_halted:
                            logger.info(f"Trading halted; skipping {action} {symbol}")
                            _note(symbol, 'halted')
                            continue

                        # Small-account protection: don't let an automated
                        # exit trip the pattern-day-trader rule.
                        if self._pdt_blocks_order(broker, symbol, action):
                            _note(symbol, 'pdt_guard')
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

                        # Adds and trims need the position's history: when it
                        # opened, its entries, its last trim (position_rules).
                        state = (state_from_journal(self.order_journal, symbol, held_qty, avg_entry)
                                 if held_qty > 0 else None)
                        if is_add:
                            c = costs_for(symbol, self.config)
                            add_value, why = plan_add(
                                state, price, float(c.get('commission_pct', 0)) + float(c.get('slippage_pct', 0)),
                                portfolio_value, target_pos_value, add_rule)
                            if why:
                                _note(symbol, why)
                                continue
                            executions = [(signal_data.get('strategy', 'ensemble'), add_value)]

                        symbol_executed = False
                        from src.agent.position_sizing import size_order
                        from src.agent.order_journal import make_client_order_id
                        from src.connectors.base_broker import OrderRequest
                        sizing_cfg = self.config.get('trading', {})
                        cycle = int(time.time() // self.config.get('trading_loop_interval', 60))

                        # Long-only (see _position_gate): a sell always closes
                        # a held long, so the short-sale whole-share rule that
                        # lived here no longer applies. A sell exits the whole
                        # position in one order, exactly the held quantity,
                        # as the backtest does.
                        allow_fractional = sizing_cfg.get('allow_fractional', True)
                        sell_qty = held_qty
                        if action == 'sell':
                            # A strong sell closes the position; a weaker one
                            # trims it, at most once a day (position_rules).
                            sell_qty, why = plan_exit(state, confidence, price, portfolio_value,
                                                      datetime.utcnow().date().isoformat(), exit_rule)
                            if why:
                                _note(symbol, why)
                                continue
                            if not allow_fractional and asset_type != 'crypto':
                                sell_qty = math.floor(sell_qty) or held_qty
                            sell_qty = min(round(sell_qty, 6), held_qty)
                            executions = [(signal_data.get('strategy', 'ensemble'), sell_qty * price)]

                        for strategy_name, target_value in executions:
                            if action == 'sell':
                                # A quantity, not a dollar value: recomputing it
                                # would drift by float rounding, and the broker
                                # rejects selling more than held.
                                quantity, order_tif = sell_qty, 'day'
                            else:
                                # Dollar-notional sizing with fractional shares,
                                # so small accounts get properly sized positions
                                # instead of rounding to zero.
                                sized = size_order(
                                    target_value, price,
                                    min_notional=sizing_cfg.get('min_notional', 5.0),
                                    allow_fractional=allow_fractional,
                                )
                                if not sized:
                                    _note(symbol, 'min_notional')
                                    continue
                                quantity, order_tif = sized.quantity, sized.time_in_force

                            # Idempotency: one deterministic id per decision
                            # (strategy, symbol, side, loop cycle). The journal
                            # blocks re-submission after a crash/retry, and the
                            # broker deduplicates on the same id server-side.
                            client_order_id = make_client_order_id(strategy_name, symbol, action, cycle)
                            # Credit: in parallel mode the order is one
                            # strategy's own; blended, it belongs to the
                            # strategies that voted for it, by confidence.
                            weights = ({strategy_name: 1.0} if mode == 'parallel' and per_strategy
                                       else credit_weights(per_strategy, action) or {strategy_name: 1.0})
                            if self.order_journal and not self.order_journal.record_intent(
                                    client_order_id, symbol, action, float(quantity),
                                    'market', strategy=strategy_name, strategy_weights=weights):
                                logger.warning(f"Skipping duplicate order decision: {client_order_id}")
                                if not symbol_executed:
                                    _note(symbol, 'duplicate')
                                continue

                            order = OrderRequest(
                                symbol=symbol,
                                quantity=float(quantity),
                                side=action,
                                order_type='market',
                                # fractional orders must be DAY (broker rule);
                                # whole-share keeps DAY too - the loop
                                # re-decides every cycle, GTC adds nothing.
                                time_in_force=order_tif,
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
                                    symbol_executed = True
                                    _note(symbol, executed=True, coid=client_order_id)
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

    def halt_trading(self, flatten: bool = False, reason: str = 'operator request') -> Dict[str, Any]:
        """Kill switch: stop all NEW order submission immediately.

        With ``flatten=True`` also cancels every open order and closes every
        position with marketable extended-hours limit orders (falling back to
        market orders when no price is available), so it works premarket too.
        """
        self.trading_halted = True
        self.halt_reason = reason
        self.halted_at = datetime.utcnow().isoformat()
        # Kept in the database, so a restart stays halted until an operator
        # resumes: a halt that a redeploy quietly undid would not be a halt.
        rm = getattr(self, 'risk_manager', None)
        if rm is not None and hasattr(rm, 'set_persistent_kill_switch'):
            try:
                rm.set_persistent_kill_switch(True, reason)
            except Exception as e:
                logger.error(f"Could not persist the kill switch: {e}")
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
                        strategy='kill_switch', strategy_weights={'kill_switch': 1.0}):
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
        """Release the kill switch, in memory and in the database, and re-arm
        the heartbeat watchdog; the loop resumes submitting orders."""
        self.trading_halted = False
        self.halt_reason = None
        self.halted_at = None
        rm = getattr(self, 'risk_manager', None)
        if rm is not None and hasattr(rm, 'set_persistent_kill_switch'):
            try:
                rm.set_persistent_kill_switch(False, 'resumed by an operator')
            except Exception as e:
                logger.error(f"Could not clear the stored kill switch: {e}")
        hb = getattr(self, 'heartbeat_monitor', None)
        if hb is not None and hasattr(hb, 'reset'):
            hb.reset()
        logger.warning("Kill switch released; trading resumed")
        return {'halted': False}

    def _restore_halt(self) -> None:
        """Start halted if the kill switch was engaged before the restart."""
        rm = getattr(self, 'risk_manager', None)
        if rm is None or not getattr(rm, 'emergency_stop', False):
            return
        self.trading_halted = True
        self.halt_reason = getattr(rm, 'kill_switch_reason', None) or \
            'engaged before the last restart (reason not recorded)'
        self.halted_at = getattr(rm, 'kill_switch_at', None)
        logger.warning(f"Trading is HALTED ({self.halt_reason}, since {self.halted_at}). "
                       f"Nothing new will be ordered until an operator presses Resume.")

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

    def _stop_distances_for(self, symbol: str, fallback_stop: float,
                            fallback_trail: float) -> dict:
        """Stop distances for one symbol, scaled to its own volatility.

        Returns the configured fixed percentages when no ATR is available
        yet, so a cold start or a thin feed never leaves a position without
        a stop.
        """
        tracker = getattr(self, 'volatility', None)
        atr_pct = None
        if tracker is not None:
            try:
                atr_pct = tracker.atr_pct(symbol)
            except Exception as e:
                logger.debug(f"ATR lookup failed for {symbol}: {e}")
        if atr_pct is None:
            return {'stop_loss_pct': fallback_stop,
                    'trailing_stop_pct': fallback_trail, 'source': 'fixed'}
        from src.agent.volatility import stop_distances
        return stop_distances(self.config, atr_pct)

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
            
            core_symbols = TradingAgent._core_symbols(self)
            for position in positions:
                try:
                    if position.quantity == 0 or position.cost_basis <= 0:
                        continue
                    # The core's index funds are held through drawdowns by
                    # design; a stop would defeat the point of holding them.
                    if str(position.symbol).upper() in core_symbols:
                        continue

                    unrealized_pl_pct = position.unrealized_pl / position.cost_basis

                    # Widen both stops to a multiple of this symbol's own ATR.
                    # Falls back to the passed-in fixed percentages when the
                    # symbol has too little history for an ATR reading.
                    dist = self._stop_distances_for(
                        position.symbol, stop_loss_pct, trailing_stop_pct)
                    sym_stop_pct = dist['stop_loss_pct']
                    sym_trail_pct = dist['trailing_stop_pct']

                    # Hard stop-loss
                    if unrealized_pl_pct <= -sym_stop_pct:
                        close_side = 'sell' if position.quantity > 0 else 'buy'

                        # A pending close order makes another one redundant —
                        # the position still shows until the first fills, so
                        # without this check every 60s loop stacks a new order.
                        if self._has_open_close_order(primary_broker, position.symbol, close_side):
                            logger.info(f"Stop-loss close already pending for {position.symbol}; skipping")
                            continue

                        logger.warning(
                            f"STOP-LOSS triggered for {position.symbol}: "
                            f"loss={unrealized_pl_pct:.2%} exceeds limit={-sym_stop_pct:.2%} "
                            f"({dist['source']})"
                        )
                        from src.connectors.base_broker import OrderRequest
                        from src.agent.order_journal import make_client_order_id
                        # 5-minute cycle: dedups rapid re-triggers while still
                        # allowing a later re-close if the position re-opens.
                        coid = make_client_order_id('stoploss', position.symbol,
                                                    close_side, int(time.time() // 300))
                        if self.order_journal and not self.order_journal.record_intent(
                                coid, position.symbol, close_side,
                                abs(position.quantity), 'market', strategy='stop_loss',
                                strategy_weights={'stop_loss': 1.0}):
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
                        trailing_stops = self.risk_manager.check_trailing_stops(
                            lambda sym: self._stop_distances_for(
                                sym, stop_loss_pct, trailing_stop_pct)['trailing_stop_pct'])
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
                                        abs(position.quantity), 'market', strategy='trailing_stop',
                                        strategy_weights={'trailing_stop': 1.0}):
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
            # Anchor to US Eastern (exchange time), not the host clock. The
            # Docker image pins TZ=UTC, so datetime.now().hour would gate on
            # 09:00 UTC (~04:00-05:00 ET, premarket) and never at the open.
            from zoneinfo import ZoneInfo
            et_now = datetime.now(ZoneInfo('America/New_York'))
            today = et_now.date()

            # Only run once per day
            if hasattr(self, '_last_optimization_date') and self._last_optimization_date == today:
                return

            # Only run in the first market hour (09:00-09:59 ET)
            if et_now.hour != 9:
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
            results_file = str(DATA_DIR / f"optimization_results_{datetime.now().strftime('%Y%m%d')}.json")
            
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
                'halt_reason': getattr(self, 'halt_reason', None),
                'halted_at': getattr(self, 'halted_at', None),
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