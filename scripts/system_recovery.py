#!/usr/bin/env python3
"""
System Recovery and Startup Manager
Handles system initialization, error recovery, and service management
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Optional
import requests
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemRecovery:
    def __init__(self):
        self.services = {
            'api_server': {'port': 5001, 'process': None, 'status': 'stopped'},
            'trading_agent': {'port': None, 'process': None, 'status': 'stopped'},
            'monitoring': {'port': 8080, 'process': None, 'status': 'stopped'}
        }
        self.recovery_attempts = {}
        
    def check_port_available(self, port: int) -> bool:
        """Check if port is available"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    return False
            return True
        except:
            return True
    
    def kill_process_on_port(self, port: int) -> bool:
        """Kill process running on specific port"""
        try:
            # Windows-compatible approach
            import subprocess
            result = subprocess.run(
                ['netstat', '-ano'], 
                capture_output=True, text=True, shell=True
            )
            
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', pid], 
                                         capture_output=True, shell=True)
                            logger.info(f"Killed process {pid} on port {port}")
                            return True
                        except:
                            continue
            return False
        except Exception as e:
            logger.error(f"Error killing process on port {port}: {e}")
            return False
    
    def start_api_server(self) -> bool:
        """Start the API server"""
        try:
            port = self.services['api_server']['port']
            
            # Kill existing process if needed
            if not self.check_port_available(port):
                self.kill_process_on_port(port)
                time.sleep(2)
            
            # Try simple API server first, fallback to production
            api_script = 'simple_api_server.py' if os.path.exists('simple_api_server.py') else 'production_ready_api.py'
            cmd = [sys.executable, api_script]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait for startup
            time.sleep(3)
            
            # Check if server is responding
            if self.check_service_health('api_server'):
                self.services['api_server']['process'] = process
                self.services['api_server']['status'] = 'running'
                logger.info("API server started successfully")
                return True
            else:
                process.terminate()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_trading_agent(self) -> bool:
        """Start the main trading agent"""
        try:
            cmd = [sys.executable, 'main.py', '--config', 'config/config.json']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            time.sleep(5)  # Allow startup time
            
            if process.poll() is None:  # Process still running
                self.services['trading_agent']['process'] = process
                self.services['trading_agent']['status'] = 'running'
                logger.info("Trading agent started successfully")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to start trading agent: {e}")
            return False
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if service is healthy"""
        try:
            if service_name == 'api_server':
                response = requests.get('http://localhost:5001/api/status', timeout=5)
                return response.status_code == 200
            elif service_name == 'trading_agent':
                # Check if process is running
                process = self.services[service_name]['process']
                return process and process.poll() is None
            return False
        except:
            return False
    
    def recover_service(self, service_name: str) -> bool:
        """Attempt to recover a failed service"""
        attempt_key = f"{service_name}_{datetime.now().strftime('%H')}"
        attempts = self.recovery_attempts.get(attempt_key, 0)
        
        if attempts >= 3:
            logger.error(f"Max recovery attempts reached for {service_name}")
            return False
        
        self.recovery_attempts[attempt_key] = attempts + 1
        logger.info(f"Attempting recovery for {service_name} (attempt {attempts + 1})")
        
        # Stop existing process
        self.stop_service(service_name)
        time.sleep(2)
        
        # Restart service
        if service_name == 'api_server':
            return self.start_api_server()
        elif service_name == 'trading_agent':
            return self.start_trading_agent()
        
        return False
    
    def stop_service(self, service_name: str):
        """Stop a service"""
        try:
            process = self.services[service_name]['process']
            if process:
                process.terminate()
                process.wait(timeout=10)
                self.services[service_name]['process'] = None
                self.services[service_name]['status'] = 'stopped'
                logger.info(f"Stopped {service_name}")
        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
    
    def start_all_services(self) -> Dict[str, bool]:
        """Start all services"""
        results = {}
        
        # Start API server first
        results['api_server'] = self.start_api_server()
        
        # Start trading agent
        results['trading_agent'] = self.start_trading_agent()
        
        return results
    
    def monitor_services(self):
        """Monitor services and recover if needed"""
        logger.info("Starting service monitoring...")
        
        while True:
            try:
                for service_name in ['api_server', 'trading_agent']:
                    if not self.check_service_health(service_name):
                        if self.services[service_name]['status'] == 'running':
                            logger.warning(f"{service_name} is unhealthy, attempting recovery")
                            self.recover_service(service_name)
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                time.sleep(10)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
            }
        }
        
        for service_name, service_info in self.services.items():
            status['services'][service_name] = {
                'status': service_info['status'],
                'healthy': self.check_service_health(service_name),
                'port': service_info.get('port')
            }
        
        return status

def main():
    """Main recovery function"""
    recovery = SystemRecovery()
    
    print("AI Trading System Recovery")
    print("=" * 40)
    
    # Check current status
    status = recovery.get_system_status()
    print(f"System Status at {status['timestamp']}")
    print(f"CPU: {status['system']['cpu_percent']}%")
    print(f"Memory: {status['system']['memory_percent']}%")
    print()
    
    # Start services
    print("Starting services...")
    results = recovery.start_all_services()
    
    for service, success in results.items():
        status_icon = "✓" if success else "✗"
        print(f"{status_icon} {service}: {'Started' if success else 'Failed'}")
    
    # Check final status
    print("\nFinal Status:")
    final_status = recovery.get_system_status()
    for service, info in final_status['services'].items():
        health_icon = "✓" if info['healthy'] else "✗"
        print(f"{health_icon} {service}: {info['status']} (Port: {info['port']})")
    
    # Start monitoring if requested
    if '--monitor' in sys.argv:
        print("\nStarting continuous monitoring...")
        recovery.monitor_services()

if __name__ == '__main__':
    main()