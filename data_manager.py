"""
Data Manager for coordinating all data sources and real-time data ingestion.
Handles data aggregation, validation, and distribution to other modules.
"""

import threading
import time
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


try:
    from market_data_connector import FinnhubConnector
    from news_connector import FinnhubNewsConnector, RedditConnector
    from coinbase_connector import CoinbaseConnector
    from oanda_connector import OandaConnector
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False
    FinnhubConnector = None
    FinnhubNewsConnector = None
    RedditConnector = None
    CoinbaseConnector = None
    OandaConnector = None

try:
    from real_data_connector import RealDataConnector
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False
    RealDataConnector = None

logger = logging.getLogger(__name__)


class DataManager:
    """
    Central data manager for coordinating all data sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors = {}
        self.is_running = False
        self.data_queue = queue.Queue()
        self.subscribers = []
        self.redis_client = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.data_thread = None
        self.processing_thread = None
        
        # Data storage
        self.latest_data = {}
        self.data_lock = threading.Lock()
        
        # Initialize Redis for real-time data storage
        if REDIS_AVAILABLE and redis is not None:
            self._init_redis()
        
        # Initialize connectors
        self._init_connectors()
    
    def _init_connectors(self):
        """Initialize data connectors"""
        # Initialize real data connector (priority)
        if REAL_DATA_AVAILABLE and RealDataConnector is not None:
            try:
                self.connectors['real_data'] = RealDataConnector(self.config)
                logger.info("Initialized real data connector")
                if hasattr(self.connectors['real_data'], 'connect'):
                    self.connectors['real_data'].connect()
            except Exception as e:
                logger.warning(f"Failed to initialize real data connector: {str(e)}")
        
        # Initialize legacy connectors as backup
        if CONNECTORS_AVAILABLE:
            connector_configs = self.config.get('connectors', {})
            import os
            # Finnhub
            if 'finnhub' not in connector_configs:
                connector_configs['finnhub'] = {}
            if not connector_configs['finnhub'].get('api_key'):
                connector_configs['finnhub']['api_key'] = os.getenv('FINNHUB_API_KEY', '')
            if FinnhubConnector is not None:
                try:
                    self.connectors['finnhub_market'] = FinnhubConnector(connector_configs['finnhub'])
                    logger.info("Initialized Finnhub market data connector")
                except Exception as e:
                    logger.error(f"Failed to initialize Finnhub market data connector: {str(e)}")
            if FinnhubNewsConnector is not None:
                try:
                    self.connectors['finnhub_news'] = FinnhubNewsConnector(connector_configs['finnhub'])
                    logger.info("Initialized Finnhub news connector")
                except Exception as e:
                    logger.error(f"Failed to initialize Finnhub news connector: {str(e)}")
            # Optionally keep Reddit connector
            if 'reddit' in connector_configs and RedditConnector is not None:
                try:
                    self.connectors['reddit'] = RedditConnector(connector_configs['reddit'])
                    logger.info("Initialized Reddit connector")
                except Exception as e:
                    logger.error(f"Failed to initialize Reddit connector: {str(e)}")
            # Coinbase
            if 'coinbase' in connector_configs and connector_configs['coinbase'].get('enabled', False) and CoinbaseConnector is not None and hasattr(CoinbaseConnector, 'connect') and hasattr(CoinbaseConnector, 'disconnect'):
                try:
                    self.connectors['coinbase'] = CoinbaseConnector(connector_configs['coinbase'])
                    logger.info("Initialized Coinbase crypto data connector")
                except Exception as e:
                    logger.error(f"Failed to initialize Coinbase connector: {str(e)}")
            # OANDA
            if 'oanda' in connector_configs and connector_configs['oanda'].get('enabled', False) and OandaConnector is not None and hasattr(OandaConnector, 'connect') and hasattr(OandaConnector, 'disconnect'):
                try:
                    self.connectors['oanda'] = OandaConnector(connector_configs['oanda'])
                    logger.info("Initialized OANDA forex data connector")
                except Exception as e:
                    logger.error(f"Failed to initialize OANDA connector: {str(e)}")
        if not self.connectors:
            logger.warning("No data connectors available. System will use mock data.")
    
    def _init_redis(self):
        """
        Initialize Redis connection for real-time data storage.
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install redis-py to enable caching.")
            return
            
        try:
            redis_config = self.config.get('redis', {})
            if redis_config:
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=True
                )
                logger.info("Redis connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {str(e)}")
            self.redis_client = None
    
    def start(self):
        """
        Start the data manager and all data collection processes.
        """
        if self.is_running:
            logger.warning("Data manager is already running")
            return
        
        logger.info("Starting data manager...")
        
        # Connect all connectors
        for name, connector in self.connectors.items():
            try:
                if connector.connect():
                    logger.info(f"Connected {name}")
                else:
                    logger.error(f"Failed to connect {name}")
            except Exception as e:
                logger.error(f"Error connecting {name}: {str(e)}")
        
        self.is_running = True
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.data_thread.start()
        
        # Start data processing thread
        self.processing_thread = threading.Thread(target=self._data_processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Data manager started successfully")
    
    def stop(self):
        """
        Stop the data manager and all data collection processes.
        """
        if not self.is_running:
            logger.warning("Data manager is not running")
            return
        
        logger.info("Stopping data manager...")
        
        self.is_running = False
        
        # Disconnect all connectors
        for name, connector in self.connectors.items():
            try:
                connector.disconnect()
                logger.info(f"Disconnected {name}")
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {str(e)}")
        
        # Wait for threads to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Data manager stopped")
    
    def _data_collection_loop(self):
        """
        Main data collection loop running in a separate thread.
        """
        symbols = self.config.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        market_data_interval = self.config.get('market_data_interval', 60)  # seconds
        news_data_interval = self.config.get('news_data_interval', 300)  # seconds
        
        last_market_update = 0
        last_news_update = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Collect market data
                if current_time - last_market_update >= market_data_interval:
                    self._collect_market_data(symbols)
                    last_market_update = current_time
                
                # Collect news data
                if current_time - last_news_update >= news_data_interval:
                    self._collect_news_data(symbols)
                    last_news_update = current_time
                
                time.sleep(1)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_market_data(self, symbols: List[str]):
        """
        Collect market data from all available connectors.
        """
        market_connectors = ['real_data', 'finnhub_market']
        
        futures = []
        for connector_name in market_connectors:
            if connector_name in self.connectors:
                connector = self.connectors[connector_name]
                # Real data connector doesn't have is_connected attribute
                if connector_name == 'real_data' or (hasattr(connector, 'is_connected') and connector.is_connected):
                    future = self.executor.submit(connector.get_real_time_data, symbols)
                    futures.append((connector_name, future))
        
        # Collect results
        for connector_name, future in futures:
            try:
                data = future.result(timeout=30)
                if data:
                    self.data_queue.put({
                        'type': 'market_data',
                        'source': connector_name,
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error collecting market data from {connector_name}: {str(e)}")
    
    def _collect_news_data(self, symbols: List[str]):
        """
        Collect news data from all available connectors.
        """
        news_connectors = ['finnhub_news', 'reddit']
        
        futures = []
        for connector_name in news_connectors:
            if connector_name in self.connectors:
                connector = self.connectors[connector_name]
                if connector.is_connected:
                    future = self.executor.submit(connector.get_real_time_data, symbols)
                    futures.append((connector_name, future))
        
        # Collect results
        for connector_name, future in futures:
            try:
                data = future.result(timeout=60)
                if data:
                    self.data_queue.put({
                        'type': 'news_data',
                        'source': connector_name,
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error collecting news data from {connector_name}: {str(e)}")
    
    def _data_processing_loop(self):
        """
        Process incoming data and distribute to subscribers.
        """
        while self.is_running:
            try:
                # Get data from queue with timeout
                try:
                    data_item = self.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process and store data
                self._process_data_item(data_item)
                
                # Notify subscribers
                self._notify_subscribers(data_item)
                
                self.data_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in data processing loop: {str(e)}")
    
    def _process_data_item(self, data_item: Dict[str, Any]):
        """
        Process a single data item.
        """
        try:
            data_type = data_item.get('type')
            source = data_item.get('source')
            data = data_item.get('data')
            timestamp = data_item.get('timestamp')
            
            # Store in memory
            with self.data_lock:
                if data_type not in self.latest_data:
                    self.latest_data[data_type] = {}
                
                self.latest_data[data_type][source] = {
                    'data': data,
                    'timestamp': timestamp
                }
            
            # Store in Redis if available
            if self.redis_client:
                redis_key = f"{data_type}:{source}"
                self.redis_client.setex(
                    redis_key, 
                    3600,  # 1 hour expiration
                    json.dumps(data_item)
                )
            
            logger.debug(f"Processed {data_type} data from {source}")
            
        except Exception as e:
            logger.error(f"Error processing data item: {str(e)}")
    
    def _notify_subscribers(self, data_item: Dict[str, Any]):
        """
        Notify all subscribers of new data.
        """
        for subscriber in self.subscribers:
            try:
                subscriber(data_item)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {str(e)}")
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to real-time data updates.
        
        Args:
            callback: Function to call when new data arrives
        """
        self.subscribers.append(callback)
        logger.info(f"Added subscriber: {callback.__name__}")
    
    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Unsubscribe from real-time data updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed subscriber: {callback.__name__}")
    
    def get_latest_data(self, data_type: Optional[str] = None, 
                       source: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the latest data.
        
        Args:
            data_type: Type of data to retrieve (market_data, news_data)
            source: Specific source to retrieve data from
            
        Returns:
            Dict containing the requested data
        """
        with self.data_lock:
            if data_type and source:
                return self.latest_data.get(data_type, {}).get(source, {})
            elif data_type:
                return self.latest_data.get(data_type, {})
            else:
                return self.latest_data.copy()
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                          data_type: str = 'market_data', 
                          source: Optional[str] = None) -> Dict[str, Any]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_type: Type of data (market_data, news_data)
            source: Specific source to use
            
        Returns:
            Dict containing historical data
        """
        if data_type == 'market_data':
            connectors = ['real_data', 'finnhub_market']
        elif data_type == 'news_data':
            connectors = ['finnhub_news', 'reddit']
        else:
            logger.error(f"Unknown data type: {data_type}")
            return {}
        
        if source and source in self.connectors:
            connectors = [source]
        
        historical_data = {}
        
        for connector_name in connectors:
            if connector_name in self.connectors:
                connector = self.connectors[connector_name]
                if connector.is_connected:
                    try:
                        data = connector.get_historical_data(symbol, start_date, end_date)
                        if data:
                            historical_data[connector_name] = data
                    except Exception as e:
                        logger.error(f"Error getting historical data from {connector_name}: {str(e)}")
        
        return historical_data
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of the data manager and all connectors.
        
        Returns:
            Dict containing status information
        """
        status = {
            'is_running': self.is_running,
            'redis_connected': self.redis_client is not None,
            'data_queue_size': self.data_queue.qsize(),
            'subscribers_count': len(self.subscribers),
            'connectors': {}
        }
        
        for name, connector in self.connectors.items():
            status['connectors'][name] = connector.get_status()
        
        return status