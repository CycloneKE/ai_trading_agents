"""
Unit tests for swarm_state_store.py.
"""
import pytest
import time
from src.agent.swarm_state_store import SwarmStateStore


def test_swarm_state_store_local_memory():
    """Verify local memory fallback store sets, gets, deletes, and respects TTL."""
    store = SwarmStateStore(redis_client=None)
    assert store.redis is None
    
    # Test setting and getting
    store.set("test_key", "test_value", ttl_seconds=2)
    assert store.get("test_key") == "test_value"
    
    # Test deleting
    assert store.delete("test_key") is True
    assert store.get("test_key") is None
    assert store.delete("test_key") is False
    
    # Test TTL expiration
    store.set("temp_key", "temp_value", ttl_seconds=1)
    assert store.get("temp_key") == "temp_value"
    
    # Wait for TTL to expire
    time.sleep(1.2)
    assert store.get("temp_key") is None
    
    # Test pruning
    store.set("prune_key", "prune_value", ttl_seconds=0) # instant expire
    assert store.clear_expired() >= 0
