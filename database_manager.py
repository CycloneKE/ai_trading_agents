"""
Database Manager for Trading Agent
Handles all database operations with proper connection pooling
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    import psycopg2
    from psycopg2 import pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    pool = None

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager with connection pooling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_pool = None
        self.enabled = PSYCOPG2_AVAILABLE
        
        if not self.enabled:
            logger.warning("PostgreSQL not available. Install psycopg2-binary to enable database features.")
            return
            
        self._initialize_pool()
        self._create_tables()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        if not self.enabled:
            return
            
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['username'],
                password=self.config['password'],
                sslmode=self.config['ssl_mode']
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            self.enabled = False
    
    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)
    
    def _create_tables(self):
        """Create necessary tables"""
        if not self.enabled or not self.connection_pool:
            logger.info("Skipping table creation because database is not enabled or pool not initialized")
            return

        try:
            conn = self.get_connection()
            if not conn:
                logger.error("No database connection available to create tables")
                return
            cursor = conn.cursor()
            
            # Market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price DECIMAL(10,4),
                    high_price DECIMAL(10,4),
                    low_price DECIMAL(10,4),
                    close_price DECIMAL(10,4),
                    volume BIGINT,
                    data JSONB
                );
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
                ON market_data(symbol, timestamp);
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    action VARCHAR(10) NOT NULL,
                    quantity DECIMAL(15,6) NOT NULL,
                    price DECIMAL(10,4) NOT NULL,
                    strategy VARCHAR(50),
                    confidence DECIMAL(5,4),
                    metadata JSONB
                );
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_time 
                ON trades(symbol, timestamp);
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    total_return DECIMAL(10,6),
                    sharpe_ratio DECIMAL(8,4),
                    max_drawdown DECIMAL(8,4),
                    win_rate DECIMAL(5,4),
                    metrics JSONB
                );
            """)
            
            conn.commit()
            logger.info("Database tables created/verified")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create tables: {e}")
            raise
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                self.return_connection(conn)
            except Exception:
                pass
    
    def store_market_data(self, symbol: str, data: Dict[str, Any]):
        """Store market data"""
        if not self.enabled:
            return
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_data (symbol, timestamp, open_price, high_price, 
                                       low_price, close_price, volume, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                datetime.utcnow(),
                data.get('open'),
                data.get('high'),
                data.get('low'),
                data.get('close'),
                data.get('volume'),
                json.dumps(data)
            ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store market data: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def store_trade(self, trade_data: Dict[str, Any]):
        """Store trade execution"""
        if not self.enabled:
            return
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (symbol, timestamp, action, quantity, price, 
                                  strategy, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                trade_data['symbol'],
                datetime.utcnow(),
                trade_data['action'],
                trade_data['quantity'],
                trade_data['price'],
                trade_data.get('strategy'),
                trade_data.get('confidence'),
                json.dumps(trade_data.get('metadata', {}))
            ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store trade: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def get_recent_market_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent market data"""
        if not self.enabled:
            return []
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM market_data 
                WHERE symbol = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (symbol, limit))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def store_performance_metrics(self, metrics: Dict[str, Any]):
        """Store performance metrics"""
        if not self.enabled:
            return
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (timestamp, total_return, sharpe_ratio, 
                                               max_drawdown, win_rate, metrics)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                datetime.utcnow(),
                metrics.get('total_return'),
                metrics.get('sharpe_ratio'),
                metrics.get('max_drawdown'),
                metrics.get('win_rate'),
                json.dumps(metrics)
            ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store performance metrics: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)