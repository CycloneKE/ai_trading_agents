import os
import secrets
from typing import Dict, Optional

API_KEY_FILE = "api_keys.json"

def generate_api_key() -> str:
    """Generates a secure, random API key."""
    return secrets.token_hex(32)

def save_api_key(user_id: str, api_key: str):
    """Saves an API key for a given user."""
    keys = load_api_keys()
    keys[user_id] = api_key
    with open(API_KEY_FILE, "w") as f:
        import json
        json.dump(keys, f, indent=4)

def load_api_keys() -> Dict[str, str]:
    """Loads all API keys from the storage file."""
    if not os.path.exists(API_KEY_FILE):
        return {}
    with open(API_KEY_FILE, "r") as f:
        import json
        return json.load(f)

def get_user_by_api_key(api_key: str) -> Optional[str]:
    """Finds a user ID associated with a given API key."""
    keys = load_api_keys()
    for user_id, key in keys.items():
        if key == api_key:
            return user_id
    return None
