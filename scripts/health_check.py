#!/usr/bin/env python3
"""
Health check script for the AI Trading Agent.
Checks the health of all components and reports any issues.
"""

import sys
import os
import argparse
import json
import logging
import requests
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_health(host: str = 'localhost', port: int = 8080) -> Dict[str, Any]:
    """
    Check the health of the trading agent.
    
    Args:
        host: Host name or IP address
        port: Port number
        
    Returns:
        Health status
    """
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Health check failed with status code {response.status_code}")
            return {
                'status': 'error',
                'error': f"HTTP {response.status_code}",
                'message': response.text
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check request failed: {str(e)}")
        return {
            'status': 'error',
            'error': 'connection_error',
            'message': str(e)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'error',
            'error': 'unknown_error',
            'message': str(e)
        }


def check_detailed_status(host: str = 'localhost', port: int = 8080) -> Dict[str, Any]:
    """
    Get detailed system status.
    
    Args:
        host: Host name or IP address
        port: Port number
        
    Returns:
        Detailed status
    """
    try:
        url = f"http://{host}:{port}/status"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Status check failed with status code {response.status_code}")
            return {
                'status': 'error',
                'error': f"HTTP {response.status_code}",
                'message': response.text
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Status check request failed: {str(e)}")
        return {
            'status': 'error',
            'error': 'connection_error',
            'message': str(e)
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {
            'status': 'error',
            'error': 'unknown_error',
            'message': str(e)
        }


def print_health_report(health_status: Dict[str, Any], detailed: bool = False):
    """
    Print a health report.
    
    Args:
        health_status: Health status
        detailed: Whether to print detailed information
    """
    status = health_status.get('status', 'unknown')
    
    if status == 'ok':
        print("✅ System health: OK")
    elif status == 'degraded':
        print("⚠️ System health: DEGRADED")
    else:
        print("❌ System health: ERROR")
    
    # Print component status
    components = health_status.get('components', {})
    if components:
        print("\nComponent Status:")
        for name, component in components.items():
            healthy = component.get('healthy', False)
            status_text = component.get('status', 'unknown')
            
            if healthy:
                print(f"  ✅ {name}: {status_text}")
            else:
                print(f"  ❌ {name}: {status_text}")
                if 'error' in component:
                    print(f"     Error: {component['error']}")
    
    # Print any errors
    if 'error' in health_status:
        print(f"\nError: {health_status['error']}")
        if 'message' in health_status:
            print(f"Message: {health_status['message']}")
    
    # Print detailed information if requested
    if detailed and 'system' in health_status:
        system = health_status['system']
        print("\nSystem Metrics:")
        print(f"  CPU Usage: {system.get('cpu_percent', 'N/A')}%")
        
        memory = system.get('memory', {})
        if memory:
            total_gb = memory.get('total', 0) / (1024 ** 3)
            available_gb = memory.get('available', 0) / (1024 ** 3)
            print(f"  Memory: {memory.get('percent', 'N/A')}% used ({available_gb:.1f} GB free of {total_gb:.1f} GB)")
        
        disk = system.get('disk', {})
        if disk:
            total_gb = disk.get('total', 0) / (1024 ** 3)
            free_gb = disk.get('free', 0) / (1024 ** 3)
            print(f"  Disk: {disk.get('percent', 'N/A')}% used ({free_gb:.1f} GB free of {total_gb:.1f} GB)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI Trading Agent Health Check')
    parser.add_argument('--host', default='localhost', help='Host name or IP address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')
    parser.add_argument('--detailed', action='store_true', help='Show detailed status')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    try:
        # Get health status
        health_status = check_health(args.host, args.port)
        
        # Get detailed status if requested
        if args.detailed:
            detailed_status = check_detailed_status(args.host, args.port)
            # Merge detailed status into health status
            health_status.update(detailed_status)
        
        # Output results
        if args.json:
            print(json.dumps(health_status, indent=2))
        else:
            print_health_report(health_status, args.detailed)
        
        # Return exit code based on health status
        if health_status.get('status') == 'ok':
            return 0
        elif health_status.get('status') == 'degraded':
            return 1
        else:
            return 2
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        if args.json:
            print(json.dumps({
                'status': 'error',
                'error': 'execution_error',
                'message': str(e)
            }, indent=2))
        else:
            print(f"❌ Health check failed: {str(e)}")
        return 3


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)