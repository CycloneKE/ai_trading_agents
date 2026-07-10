"""
Swarm State Store:
Unified key-value state store for caching LLM responses and sharing swarm data.
Features a thread-safe local memory store fallback if Redis is unavailable.
"""
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SwarmStateStore:
    def __init__(self, redis_client: Optional[Any] = None):
        """
        Initializes the state store. If redis_client is provided, it will be used
        for operations. Otherwise, falls back to local in-memory storage.
        """
        self.redis = redis_client
        self.local_store = {}
        self.lock = threading.Lock()
        
        if self.redis:
            try:
                self.redis.ping()
                logger.info("SwarmStateStore: Connected to Redis storage.")
            except Exception as e:
                logger.warning(f"SwarmStateStore: Redis ping failed, falling back to local memory. Error: {e}")
                self.redis = None
        else:
            logger.info("SwarmStateStore: Redis connection absent. Using local memory fallback.")

    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        """Sets a key with a value and a time-to-live (TTL) in seconds."""
        if self.redis:
            try:
                self.redis.setex(key, ttl_seconds, value)
                return
            except Exception as e:
                logger.warning(f"SwarmStateStore: Redis setex failed, falling back to local memory: {e}")
                
        with self.lock:
            self.local_store[key] = {
                "value": value,
                "expires_at": time.time() + ttl_seconds
            }

    def get(self, key: str) -> Optional[str]:
        """Gets a value for a key. Returns None if not found or expired."""
        if self.redis:
            try:
                val = self.redis.get(key)
                if val is not None:
                    return val.decode('utf-8') if isinstance(val, bytes) else str(val)
                return None
            except Exception as e:
                logger.warning(f"SwarmStateStore: Redis get failed, attempting local fallback: {e}")
                
        with self.lock:
            entry = self.local_store.get(key)
            if not entry:
                return None
            
            # Check expiration
            if time.time() > entry["expires_at"]:
                del self.local_store[key]
                return None
                
            return entry["value"]

    def delete(self, key: str) -> bool:
        """Deletes a key. Returns True if key existed and was deleted."""
        if self.redis:
            try:
                return bool(self.redis.delete(key))
            except Exception as e:
                logger.warning(f"SwarmStateStore: Redis delete failed, falling back: {e}")
                
        with self.lock:
            if key in self.local_store:
                del self.local_store[key]
                return True
            return False

    def clear_expired(self) -> int:
        """Prunes expired keys from the local store (memory leak safety)."""
        if self.redis:
            return 0 # Redis handles this automatically
            
        with self.lock:
            now = time.time()
            expired_keys = [k for k, v in self.local_store.items() if now > v["expires_at"]]
            for k in expired_keys:
                del self.local_store[k]
            return len(expired_keys)
