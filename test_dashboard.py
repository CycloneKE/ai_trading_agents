#!/usr/bin/env python3
"""
Dashboard Testing Script
Tests all dashboard endpoints for accuracy and functionality
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8080"

def test_endpoint(endpoint, expected_status=200):
    """Test a single endpoint"""
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        if response.status_code == expected_status:
            print(f"[PASS] {endpoint} - Status: {response.status_code}")
            return response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        else:
            print(f"[FAIL] {endpoint} - Expected: {expected_status}, Got: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] {endpoint} - {str(e)}")
        return None

def test_health_endpoint():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    data = test_endpoint("/health")
    if data:
        assert data.get('status') in ['ok', 'degraded'], "Invalid health status"
        assert 'timestamp' in data, "Missing timestamp"
        assert 'checks' in data, "Missing health checks"
        print(f"  Health Status: {data['status']}")
        print(f"  Components: {len(data['checks'])}")

def test_metrics_endpoint():
    """Test metrics endpoint"""
    print("\n=== Testing Metrics Endpoint ===")
    data = test_endpoint("/metrics")
    if data:
        print(f"  Metrics data length: {len(str(data))}")

def test_status_endpoint():
    """Test status endpoint"""
    print("\n=== Testing Status Endpoint ===")
    data = test_endpoint("/status")
    if data:
        print(f"  Status data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        if isinstance(data, dict):
            if 'running' in data:
                print(f"  System Running: {data['running']}")
            if 'timestamp' in data:
                print(f"  Timestamp: {data['timestamp']}")

def test_portfolio_endpoint():
    """Test portfolio endpoint"""
    print("\n=== Testing Portfolio Endpoint ===")
    data = test_endpoint("/portfolio", expected_status=404)  # May not exist
    if data is None:
        test_endpoint("/portfolio")  # Try with default status

def test_trades_endpoint():
    """Test trades endpoint"""
    print("\n=== Testing Trades Endpoint ===")
    data = test_endpoint("/trades", expected_status=404)  # May not exist
    if data is None:
        test_endpoint("/trades")  # Try with default status

def test_performance_endpoint():
    """Test performance endpoint"""
    print("\n=== Testing Performance Endpoint ===")
    data = test_endpoint("/performance", expected_status=404)  # May not exist
    if data is None:
        test_endpoint("/performance")  # Try with default status

def test_dashboard_load():
    """Test dashboard page load"""
    print("\n=== Testing Dashboard Page ===")
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            print("[PASS] Dashboard page loads successfully")
            print(f"  Content length: {len(response.text)}")
        else:
            print(f"[FAIL] Dashboard page - Status: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Dashboard page - {str(e)}")

def stress_test_endpoints():
    """Stress test endpoints with multiple requests"""
    print("\n=== Stress Testing Endpoints ===")
    endpoints = ["/health", "/status", "/metrics"]
    
    for endpoint in endpoints:
        print(f"Testing {endpoint} with 10 rapid requests...")
        success_count = 0
        start_time = time.time()
        
        for i in range(10):
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
                if response.status_code == 200:
                    success_count += 1
            except:
                pass
        
        end_time = time.time()
        print(f"  {endpoint}: {success_count}/10 successful, {end_time-start_time:.2f}s total")

def test_data_accuracy():
    """Test data accuracy and consistency"""
    print("\n=== Testing Data Accuracy ===")
    
    # Get health data
    health_data = test_endpoint("/health")
    if health_data:
        # Check broker status consistency
        broker_checks = health_data.get('checks', {}).get('broker_manager', {})
        if broker_checks:
            total_brokers = broker_checks.get('total_brokers', 0)
            connected_brokers = broker_checks.get('connected_brokers', 0)
            print(f"  Brokers: {connected_brokers}/{total_brokers} connected")
            
            if connected_brokers > total_brokers:
                print("[FAIL] More connected brokers than total brokers")
            else:
                print("[PASS] Broker counts are consistent")
        
        # Check data manager status
        data_checks = health_data.get('checks', {}).get('data_manager', {})
        if data_checks:
            is_running = data_checks.get('is_running', False)
            connectors = data_checks.get('connectors', {})
            print(f"  Data Manager Running: {is_running}")
            print(f"  Active Connectors: {len(connectors)}")

def run_all_tests():
    """Run all dashboard tests"""
    print("Dashboard Testing Suite")
    print("=" * 50)
    
    # Basic endpoint tests
    test_health_endpoint()
    test_status_endpoint()
    test_metrics_endpoint()
    test_portfolio_endpoint()
    test_trades_endpoint()
    test_performance_endpoint()
    test_dashboard_load()
    
    # Advanced tests
    test_data_accuracy()
    stress_test_endpoints()
    
    print("\n=== Test Summary ===")
    print("Dashboard testing completed")
    print("Check output above for any [FAIL] or [ERROR] messages")

if __name__ == "__main__":
    run_all_tests()