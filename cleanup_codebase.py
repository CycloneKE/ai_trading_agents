#!/usr/bin/env python3
"""
Codebase Cleanup Script

This script removes unnecessary files and keeps only the essential components
for the AI trading agent with adaptive capabilities.
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Essential files to keep
ESSENTIAL_FILES = {
    # Core system files
    'main.py',
    'base_strategy.py',
    'supervised_learning.py',
    'config_validator.py',
    'secrets_manager.py',
    'monitoring.py',
    'data_manager.py',
    'strategy_manager.py',
    'broker_manager.py',
    
    # Adaptive system files
    'src/adaptive_agent.py',
    'src/goal_manager.py',
    'src/adaptive_integration.py',
    
    # Configuration and setup
    'config/config.json',
    '.env.example',
    'requirements.txt',
    'README.md',
    'docker-compose.yml',
    'Dockerfile',
    
    # Essential directories (keep structure)
    'logs/',
    'data/',
    'secrets/',
    'config/',
}

# Files and directories to remove
REMOVE_PATTERNS = [
    # Test files
    'test_*.py',
    
    # Demo and example files
    'demo_*.py',
    '*_demo.py',
    'example_*.py',
    '*_example.py',
    
    # Deployment scripts (keep only docker)
    'deploy_*.bat',
    'deploy_*.sh',
    '*_deployment.py',
    '*_startup.sh',
    '*_startup.bat',
    'alibaba_*',
    'aws_*',
    'azure_*',
    
    # Fix and setup scripts
    'fix_*.py',
    'fix_*.bat',
    'fix_*.sh',
    'setup_*.py',
    'setup_*.bat',
    'setup_*.sh',
    'simple_*.py',
    'simple_*.bat',
    'simple_*.sh',
    
    # Dashboard and frontend (keep minimal)
    'enhanced_*.py',
    'advanced_*.py',
    'genius_*.py',
    'intelligent_*.py',
    'unified_*.py',
    'working_*.py',
    'webhook_*.py',
    'alert_*.py',
    
    # Monitoring and metrics (keep core only)
    'generate_*.py',
    'inject_*.py',
    'serve_*.py',
    'update_*.py',
    'force_*.py',
    'direct_*.py',
    'final_*.py',
    
    # Start scripts (keep minimal)
    'start_*.py',
    'start_*.bat',
    'start_*.sh',
    'run_*.py',
    'run_*.bat',
    
    # Upgrade and install scripts
    'upgrade_*.py',
    'upgrade_*.bat',
    'install_*.py',
    'install_*.bat',
    'install_*.sh',
    'enable_*.py',
    'enable_*.bat',
    'enable_*.sh',
    'use_*.py',
    'use_*.bat',
    'use_*.sh',
    
    # Documentation (keep essential only)
    '*.md',  # Will selectively keep README.md
    
    # Data files (keep structure, remove old data)
    'data/optimization_results_*.json',
    'data/paper_trading_state.json',
    '*.json',  # Will selectively keep config files
    '*.db',
    '*.enc',
    '*.key',
    '*.prom',
    
    # Directories to remove entirely
    'ai-trading-agent/',
    'alerts/',
    'frontend/',
    'grafana/',
    'k8s/',
    'learning/',
    'test_metrics/',
    '-p/',
    '.github/',
    'src/advanced_learning/',
    'src/advanced_tools/',
]

def should_keep_file(filepath):
    """Check if a file should be kept"""
    # Always keep essential files
    if filepath in ESSENTIAL_FILES:
        return True
    
    # Keep essential directories
    for essential_dir in ESSENTIAL_FILES:
        if essential_dir.endswith('/') and filepath.startswith(essential_dir):
            return True
    
    # Check if file matches removal patterns
    filename = os.path.basename(filepath)
    for pattern in REMOVE_PATTERNS:
        if pattern.endswith('/'):
            # Directory pattern
            if filepath.startswith(pattern):
                return False
        else:
            # File pattern
            import fnmatch
            if fnmatch.fnmatch(filename, pattern):
                # Special cases to keep
                if filename == 'README.md':
                    return True
                if filename == 'config.json' and 'config/' in filepath:
                    return True
                if filename == 'requirements.txt':
                    return True
                return False
    
    return True

def cleanup_codebase():
    """Clean up the codebase"""
    root_dir = os.getcwd()
    removed_count = 0
    kept_count = 0
    
    logger.info("Starting codebase cleanup...")
    
    # Walk through all files and directories
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, root_dir).replace('\\', '/')
            
            if should_keep_file(relative_path):
                kept_count += 1
                logger.debug(f"Keeping: {relative_path}")
            else:
                try:
                    os.remove(filepath)
                    removed_count += 1
                    logger.info(f"Removed: {relative_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {relative_path}: {e}")
    
    # Remove empty directories
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            relative_path = os.path.relpath(dir_path, root_dir).replace('\\', '/')
            
            # Skip essential directories
            if any(relative_path.startswith(essential.rstrip('/')) for essential in ESSENTIAL_FILES if essential.endswith('/')):
                continue
            
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    logger.info(f"Removed empty directory: {relative_path}")
            except Exception as e:
                logger.debug(f"Could not remove directory {relative_path}: {e}")
    
    logger.info(f"Cleanup complete! Removed {removed_count} files, kept {kept_count} files")

def create_minimal_requirements():
    """Create minimal requirements.txt with only essential dependencies"""
    minimal_requirements = """# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0

# Data and API
requests>=2.25.0
redis>=4.0.0
python-dotenv>=0.19.0

# Monitoring
prometheus-client>=0.12.0
flask>=2.0.0

# Logging and utilities
colorama>=0.4.4
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(minimal_requirements)
    
    logger.info("Created minimal requirements.txt")

def create_clean_readme():
    """Create a clean README.md focused on the adaptive agent"""
    readme_content = """# AI Trading Agent with Adaptive Capabilities

A self-adaptive, goal-oriented AI trading system that learns and evolves based on market conditions and performance feedback.

## Key Features

- **Self-Adaptive Agent**: Automatically adjusts strategies based on performance
- **Goal-Oriented Behavior**: Sets and pursues dynamic trading goals
- **Supervised Learning**: ML-based trading strategies with continuous learning
- **Risk Management**: Comprehensive risk assessment and position sizing
- **Real-time Monitoring**: Performance tracking and health checks

## Core Components

- `main.py` - Main application entry point
- `supervised_learning.py` - Enhanced ML strategy with adaptive capabilities
- `src/adaptive_agent.py` - Self-adaptive agent core
- `src/goal_manager.py` - Dynamic goal management
- `src/adaptive_integration.py` - Integration layer

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create configuration:
   ```bash
   python main.py --create-config
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the agent:
   ```bash
   python main.py
   ```

## Configuration

Edit `config/config.json` to customize:
- Trading strategies and parameters
- Risk management settings
- Data sources and symbols
- Adaptive agent behavior

## Adaptive Features

The agent automatically:
- Adjusts position sizes based on market volatility
- Modifies strategy parameters based on performance
- Sets and pursues dynamic goals
- Adapts to changing market conditions

## Monitoring

Access the monitoring dashboard at http://localhost:8080 when running.

## License

MIT License
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    logger.info("Created clean README.md")

if __name__ == '__main__':
    cleanup_codebase()
    create_minimal_requirements()
    create_clean_readme()
    
    logger.info("Codebase cleanup complete!")
    logger.info("Essential files preserved:")
    logger.info("- Core trading system with adaptive capabilities")
    logger.info("- Configuration and setup files")
    logger.info("- Docker deployment files")
    logger.info("- Essential documentation")