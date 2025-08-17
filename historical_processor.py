"""
Historical data processing module for analyzing large volumes of historical data.
Provides feature engineering and pattern recognition capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import talib

logger = logging.getLogger(__name__)


class HistoricalDataProcessor:
    """
    Processor for historical market data analysis and feature engineering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('db_path', 'data/historical_data.db')
        self.cache_enabled = config.get('cache_enabled', True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """
        Initialize SQLite database for historical data storage.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for historical data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, source)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, feature_name)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp ON features(symbol, timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("Historical data database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
    
    def store_historical_data(self, symbol: str, data: List[Dict[str, Any]], 
                            source: str = 'unknown') -> bool:
        """
        Store historical data in the database.
        
        Args:
            symbol: Trading symbol
            data: List of historical data points
            source: Data source name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for data_point in data:
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    data_point.get('timestamp'),
                    data_point.get('open'),
                    data_point.get('high'),
                    data_point.get('low'),
                    data_point.get('close'),
                    data_point.get('volume'),
                    source
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(data)} data points for {symbol} from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store historical data: {str(e)}")
            return False
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                          source: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve historical data from the database.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Specific data source (optional)
            
        Returns:
            DataFrame with historical data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume, source
                FROM market_data
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            '''
            params = [symbol, start_date, end_date]
            
            if source:
                query += ' AND source = ?'
                params.append(source)
            
            query += ' ORDER BY timestamp'
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve historical data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            if df.empty:
                return df
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required columns for technical indicators")
                return df
            
            # Convert to numpy arrays for TA-Lib
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            volume = df['volume'].values
            
            # Moving Averages
            df['sma_10'] = talib.SMA(close_prices, timeperiod=10)
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['ema_12'] = talib.EMA(close_prices, timeperiod=12)
            df['ema_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(close_prices)
            
            # RSI
            df['rsi'] = talib.RSI(close_prices, timeperiod=14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close_prices)
            
            # Stochastic Oscillator
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices)
            
            # Average True Range
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices)
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(volume.astype(float), timeperiod=20)
            df['ad_line'] = talib.AD(high_prices, low_prices, close_prices, volume.astype(float))
            
            # Price-based indicators
            df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
            df['cci'] = talib.CCI(high_prices, low_prices, close_prices)
            
            # Momentum indicators
            df['momentum'] = talib.MOM(close_prices, timeperiod=10)
            df['roc'] = talib.ROC(close_prices, timeperiod=10)
            
            logger.debug("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {str(e)}")
            return df
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features added
        """
        try:
            if df.empty:
                return df
            
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            
            # Log returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # High-Low spread
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            
            # Open-Close spread
            df['oc_spread'] = (df['close'] - df['open']) / df['open']
            
            # Price position within the day's range
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Rolling volatility
            df['volatility_5'] = df['log_return'].rolling(window=5).std()
            df['volatility_20'] = df['log_return'].rolling(window=20).std()
            
            # Rolling correlations (if multiple symbols)
            # This would be implemented when processing multiple symbols together
            
            logger.debug("Price features calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate price features: {str(e)}")
            return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features added
        """
        try:
            if df.empty or 'volume' not in df.columns:
                return df
            
            # Volume changes
            df['volume_change'] = df['volume'].pct_change()
            df['volume_change_abs'] = df['volume_change'].abs()
            
            # Volume moving averages
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            
            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            
            # Price-Volume relationship
            df['pv_trend'] = np.where(
                (df['price_change'] > 0) & (df['volume'] > df['volume_ma_20']), 1,
                np.where((df['price_change'] < 0) & (df['volume'] > df['volume_ma_20']), -1, 0)
            )
            
            # Volume-weighted average price (VWAP) approximation
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            logger.debug("Volume features calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate volume features: {str(e)}")
            return df
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect chart patterns and anomalies.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with pattern indicators added
        """
        try:
            if df.empty:
                return df
            
            # Candlestick patterns using TA-Lib
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            # Doji patterns
            df['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            df['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)
            
            # Hammer patterns
            df['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            df['hanging_man'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
            
            # Engulfing patterns
            df['bullish_engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            
            # Morning/Evening star
            df['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            df['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            
            # Support and resistance levels (simplified)
            window = 20
            df['local_max'] = df['high'].rolling(window=window, center=True).max() == df['high']
            df['local_min'] = df['low'].rolling(window=window, center=True).min() == df['low']
            
            # Trend detection
            df['trend_short'] = np.where(df['close'] > df['sma_10'], 1, -1)
            df['trend_medium'] = np.where(df['close'] > df['sma_20'], 1, -1)
            df['trend_long'] = np.where(df['close'] > df['sma_50'], 1, -1)
            
            # Volatility breakouts
            df['volatility_breakout'] = np.where(
                df['volatility_5'] > df['volatility_20'] * 1.5, 1, 0
            )
            
            logger.debug("Pattern detection completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to detect patterns: {str(e)}")
            return df
    
    def process_symbol_data(self, symbol: str, start_date: str, end_date: str,
                          source: Optional[str] = None) -> pd.DataFrame:
        """
        Complete processing pipeline for a single symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Specific data source (optional)
            
        Returns:
            DataFrame with all features and indicators
        """
        try:
            # Get historical data
            df = self.get_historical_data(symbol, start_date, end_date, source)
            
            if df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return df
            
            # Calculate all features
            df = self.calculate_technical_indicators(df)
            df = self.calculate_price_features(df)
            df = self.calculate_volume_features(df)
            df = self.detect_patterns(df)
            
            # Add symbol column
            df['symbol'] = symbol
            
            logger.info(f"Processed {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to process data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def process_multiple_symbols(self, symbols: List[str], start_date: str, 
                               end_date: str, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Process multiple symbols in parallel.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_workers: Maximum number of worker threads
            
        Returns:
            Dict mapping symbols to processed DataFrames
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.process_symbol_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {str(e)}")
        
        logger.info(f"Processed {len(results)} out of {len(symbols)} symbols")
        return results
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'price_change') -> pd.Series:
        """
        Calculate feature importance using correlation analysis.
        
        Args:
            df: DataFrame with features
            target_col: Target column for importance calculation
            
        Returns:
            Series with feature importance scores
        """
        try:
            if df.empty or target_col not in df.columns:
                return pd.Series()
            
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols.drop([target_col], errors='ignore')
            
            # Calculate correlations
            correlations = df[numeric_cols].corrwith(df[target_col]).abs()
            
            # Sort by importance
            feature_importance = correlations.sort_values(ascending=False)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {str(e)}")
            return pd.Series()
    
    def export_processed_data(self, data: Dict[str, pd.DataFrame], 
                            output_path: str, format: str = 'csv') -> bool:
        """
        Export processed data to files.
        
        Args:
            data: Dict mapping symbols to DataFrames
            output_path: Output directory path
            format: Export format ('csv', 'parquet', 'json')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import os
            os.makedirs(output_path, exist_ok=True)
            
            for symbol, df in data.items():
                if df.empty:
                    continue
                
                filename = f"{symbol}_processed.{format}"
                filepath = os.path.join(output_path, filename)
                
                if format == 'csv':
                    df.to_csv(filepath)
                elif format == 'parquet':
                    df.to_parquet(filepath)
                elif format == 'json':
                    df.to_json(filepath, orient='records', date_format='iso')
                else:
                    logger.error(f"Unsupported format: {format}")
                    return False
            
            logger.info(f"Exported {len(data)} files to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            return False

