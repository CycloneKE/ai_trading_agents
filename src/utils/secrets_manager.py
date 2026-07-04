"""
Secrets Manager for AI Trading Agent
Now uses SecureConfigManager for better security
"""

import logging
from typing import Dict, Any, Optional
from secure_config import SecureConfigManager

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Legacy wrapper for SecureConfigManager
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.secure_config = SecureConfigManager()
        logger.info("Secrets manager initialized with secure config")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value from environment variables"""
        import os
        return os.getenv(key, default)
    
    def get_api_credentials(self, service: str) -> Dict[str, Optional[str]]:
        """Get API credentials for a service"""
        return {
            'api_key': self.secure_config.get_api_key(service),
            'api_secret': self.secure_config.get_api_secret(service)
        }
    
    def validate_credentials(self, service: str) -> bool:
        """Validate that credentials exist for a service"""
        credentials = self.get_api_credentials(service)
        return bool(credentials['api_key'])
    
    def list_available_services(self) -> list:
        """List services with available credentials"""
        services = []
        common_services = ['alpaca', 'alpha_vantage', 'polygon', 'news_api']
        
        for service in common_services:
            if self.validate_credentials(service):
                services.append(service)
        
        return services
    
    def get_status(self) -> Dict[str, Any]:
        """Get secrets manager status"""
        return {
            'initialized': True,
            'available_services': self.list_available_services(),
            'encryption_enabled': True
        }

# Global secrets manager instance
_secrets_manager = None

def get_secrets_manager(config: Optional[Dict[str, Any]] = None) -> SecretsManager:
    """Get the global secrets manager instance"""
    global _secrets_manager
    
    if _secrets_manager is None:
        if config is None:
            config = {}
        _secrets_manager = SecretsManager(config)
    
    return _secrets_manager