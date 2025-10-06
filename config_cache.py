import json
from functools import lru_cache


@lru_cache(maxsize=1)
def load_cached_config():
    """Load configuration with caching"""
    with open('config/config.json', 'r') as f:
        return json.load(f)

def get_config():
    """Get configuration (cached)"""
    return load_cached_config().copy()
