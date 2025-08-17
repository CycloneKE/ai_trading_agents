#!/usr/bin/env python3
"""
Phase 1 Setup Script
Sets up critical security fixes and infrastructure
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install updated dependencies"""
    logger.info("Installing updated dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment file"""
    logger.info("Setting up environment file...")
    
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            logger.info("Created .env file from .env.example")
            logger.warning("Please edit .env file with your actual API keys and database credentials")
        else:
            logger.error(".env.example not found")
            return False
    else:
        logger.info(".env file already exists")
    
    return True

def setup_database():
    """Set up database using Docker"""
    logger.info("Setting up database...")
    
    try:
        # Check if Docker is available
        subprocess.check_call(["docker", "--version"], stdout=subprocess.DEVNULL)
        
        # Start database services
        subprocess.check_call(["docker-compose", "up", "-d", "postgres", "redis"])
        logger.info("Database services started")
        
        # Wait a moment for services to start
        import time
        time.sleep(10)
        
        return True
    except subprocess.CalledProcessError:
        logger.warning("Docker not available. Please set up PostgreSQL and Redis manually")
        return False
    except FileNotFoundError:
        logger.warning("Docker or docker-compose not found. Please install Docker")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    directories = [
        'data',
        'logs',
        'secrets',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def test_secure_config():
    """Test secure configuration"""
    logger.info("Testing secure configuration...")
    
    try:
        from secure_config import SecureConfigManager
        config_manager = SecureConfigManager()
        logger.info("Secure configuration manager initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize secure configuration: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting Phase 1 setup...")
    
    steps = [
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Installing dependencies", install_dependencies),
        ("Testing secure configuration", test_secure_config),
        ("Setting up database", setup_database),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"Running: {step_name}")
        if not step_func():
            failed_steps.append(step_name)
            logger.error(f"Failed: {step_name}")
        else:
            logger.info(f"Completed: {step_name}")
    
    if failed_steps:
        logger.error(f"Setup completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            logger.error(f"  - {step}")
        logger.info("Please address the failed steps manually")
    else:
        logger.info("Phase 1 setup completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Edit .env file with your API keys and database credentials")
        logger.info("2. Run: python main.py --create-config")
        logger.info("3. Run: python main.py")

if __name__ == "__main__":
    main()