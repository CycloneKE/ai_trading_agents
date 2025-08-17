"""
Secure Configuration Manager
Replaces hardcoded credentials with environment variables and secure storage
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SecureConfigManager:
    """Secure configuration management with encryption"""
    
    def __init__(self):
        load_dotenv()
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = 'secrets/.key'
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs('secrets', exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment variables"""
        env_var = f"TRADING_{service.upper()}_API_KEY"
        return os.getenv(env_var)
    
    def get_api_secret(self, service: str) -> Optional[str]:
        """Get API secret from environment variables"""
        env_var = f"TRADING_{service.upper()}_API_SECRET"
        return os.getenv(env_var)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'trading_agent'),
            'username': os.getenv('DB_USER', 'trading_user'),
            'password': os.getenv('DB_PASSWORD', ''),
            'ssl_mode': os.getenv('DB_SSL_MODE', 'prefer')
        }