"""
Configuration validator for the AI Trading Agent.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration.
    
    Args:
        config: Configuration dict
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required sections
    required_sections = ['data_manager', 'strategies', 'risk_management', 'brokers']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate data_manager section
    if 'data_manager' in config:
        data_manager = config['data_manager']
        
        if 'symbols' not in data_manager:
            errors.append("Missing 'symbols' in data_manager section")
        elif not isinstance(data_manager['symbols'], list):
            errors.append("'symbols' in data_manager must be a list")
        
        if 'update_interval' in data_manager and not isinstance(data_manager['update_interval'], (int, float)):
            errors.append("'update_interval' in data_manager must be a number")
    
    # Validate strategies section
    if 'strategies' in config:
        strategies = config['strategies']
        
        if not strategies:
            errors.append("No strategies defined in 'strategies' section")
        
        for name, strategy in strategies.items():
            if 'type' not in strategy:
                errors.append(f"Missing 'type' in strategy '{name}'")
            
            if 'enabled' in strategy and not isinstance(strategy['enabled'], bool):
                errors.append(f"'enabled' in strategy '{name}' must be a boolean")
            
            if 'weight' in strategy and not isinstance(strategy['weight'], (int, float)):
                errors.append(f"'weight' in strategy '{name}' must be a number")
    
    # Validate risk_management section
    if 'risk_management' in config:
        risk = config['risk_management']
        
        if 'max_position_size' in risk and not isinstance(risk['max_position_size'], (int, float)):
            errors.append("'max_position_size' in risk_management must be a number")
        
        if 'max_portfolio_var' in risk and not isinstance(risk['max_portfolio_var'], (int, float)):
            errors.append("'max_portfolio_var' in risk_management must be a number")
    
    # Validate brokers section
    if 'brokers' in config:
        brokers = config['brokers']
        
        if not brokers:
            errors.append("No brokers defined in 'brokers' section")
        
        primary_count = 0
        for name, broker in brokers.items():
            if 'type' not in broker:
                errors.append(f"Missing 'type' in broker '{name}'")
            
            if 'primary' in broker and broker['primary']:
                primary_count += 1
        
        if primary_count > 1:
            errors.append("Multiple primary brokers defined (only one allowed)")
    
    return errors

def create_default_config(config_path: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to save configuration
    """
    default_config = {
        "data_manager": {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
            "update_interval": 60,
            "max_retries": 3,
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "connectors": {
                "yahoo_finance": {
                    "enabled": True
                },
                "alpha_vantage": {
                    "enabled": False,
                    "api_key": "your_api_key_here"
                }
            }
        },
        "strategies": {
            "momentum": {
                "type": "supervised_learning",
                "enabled": True,
                "lookback_period": 20,
                "threshold": 0.02,
                "weight": 1.0,
                "features": ["close", "volume", "rsi", "macd", "bb_upper", "bb_lower"]
            },
            "mean_reversion": {
                "type": "supervised_learning",
                "enabled": True,
                "lookback_period": 10,
                "threshold": 0.01,
                "weight": 1.0,
                "features": ["close", "rsi", "bb_upper", "bb_lower"]
            },
            "sentiment": {
                "type": "nlp",
                "enabled": False,
                "sentiment_threshold": 0.6,
                "weight": 0.8
            },
            "reinforcement": {
                "type": "dqn",
                "enabled": False,
                "lookback_period": 30,
                "learning_rate": 0.001,
                "exploration_rate": 0.1,
                "weight": 1.0
            }
        },
        "risk_management": {
            "var_confidence": 0.95,
            "max_position_size": 0.1,
            "max_portfolio_var": 0.02,
            "max_drawdown": 0.2,
            "max_daily_loss": 0.02
        },
        "risk_limits": {
            "max_portfolio_var": 0.05,
            "max_position_size": 0.1,
            "max_drawdown": 0.2
        },
        "brokers": {
            "paper_broker": {
                "type": "paper",
                "initial_cash": 100000,
                "commission_per_trade": 1.0,
                "primary": True
            },
            "alpaca": {
                "type": "alpaca",
                "enabled": False,
                "api_key": "your_api_key_here",
                "api_secret": "your_api_secret_here",
                "base_url": "https://paper-api.alpaca.markets",
                "primary": False
            }
        },
        "trading": {
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0005
        },
        "trading_loop_interval": 60,
        "nlp": {
            "sentiment_model": "vader",
            "text_processing": {
                "remove_stopwords": True,
                "lemmatize": True
            }
        },
        "automl": {
            "n_trials": 100,
            "cv_folds": 5
        },
        "portfolio_optimization": {
            "risk_free_rate": 0.02,
            "frequency": "daily",
            "lookback_period": 252
        },
        "monitoring": {
            "enabled": True,
            "port": 8080,
            "metrics_interval": 15,
            "system_metrics_enabled": True
        },
        "security": {
            "use_encryption": True,
            "api_key_rotation_days": 90
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/trading_agent.log",
            "max_size_mb": 10,
            "backup_count": 5
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Created default configuration at {config_path}")