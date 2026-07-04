"""
Base connector class for data sources.
Provides a common interface for all data connectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.last_update = None
        self.error_count = 0
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to the data source.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get real-time market data for specified symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict containing real-time data
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          interval: str = '1d') -> Dict[str, Any]:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            Dict containing historical data
        """
        pass
    
    def retry_operation(self, operation, *args, **kwargs):
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Function to retry
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)
                self.error_count = 0  # Reset error count on success
                return result
            except Exception as e:
                self.error_count += 1
                logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {self.name}")
                    return None
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate incoming data for completeness and correctness.
        
        Args:
            data: Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if not data:
            return False
            
        # Basic validation - can be extended by subclasses
        required_fields = ['symbol', 'timestamp', 'price']
        return all(field in data for field in required_fields)
    
    def normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize data to a standard format.
        
        Args:
            data: Raw data from the source
            
        Returns:
            Dict: Normalized data
        """
        # Default normalization - can be overridden by subclasses
        normalized = {
            'source': self.name,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        return normalized
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get connector status information.
        
        Returns:
            Dict containing status information
        """
        return {
            'name': self.name,
            'connected': self.is_connected,
            'last_update': self.last_update,
            'error_count': self.error_count,
            'config': {k: v for k, v in self.config.items() if 'key' not in k.lower()}
        }

