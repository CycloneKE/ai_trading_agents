#!/usr/bin/env python3
"""
AI Trading Agent - Root entry point stub
"""
import sys
import os

# Ensure the root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent.main import main

if __name__ == "__main__":
    main()
