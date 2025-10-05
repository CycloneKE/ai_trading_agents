#!/usr/bin/env python3
"""
Dashboard UI Testing Script
Tests the actual dashboard interface and data display
"""

import requests
import json
import re
from bs4 import BeautifulSoup

BASE_URL = "http://localhost:8080"

def test_dashboard_html():
    """Test dashboard HTML structure"""
    print("=== Testing Dashboard HTML Structure ===")
    
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code != 200:
            print(f"[FAIL] Dashboard not accessible: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for essential HTML elements
        title = soup.find('title')
        if title:
            print(f"[PASS] Page title: {title.text}")
        else:
            print("[FAIL] No page title found")
        
        # Check for dashboard sections
        sections = ['system-status', 'trading-metrics', 'portfolio-summary']
        for section in sections:
            element = soup.find(id=section) or soup.find(class_=section)
            if element:
                print(f"[PASS] Found section: {section}")
            else:
                print(f"[WARN] Section not found: {section}")
        
        # Check for JavaScript/CSS
        scripts = soup.find_all('script')
        styles = soup.find_all('style') + soup.find_all('link', rel='stylesheet')
        
        print(f"[INFO] Scripts found: {len(scripts)}")
        print(f"[INFO] Stylesheets found: {len(styles)}")
        
    except Exception as e:
        print(f"[ERROR] HTML test failed: {e}")

def test_real_time_updates():
    """Test if dashboard shows real-time data"""
    print("\n=== Testing Real-time Data Updates ===")
    
    # Get initial health data
    try:
        response1 = requests.get(f"{BASE_URL}/health", timeout=5)
        if response1.status_code == 200:
            data1 = response1.json()
            timestamp1 = data1.get('timestamp')
            
            # Wait and get second reading
            import time
            time.sleep(2)
            
            response2 = requests.get(f"{BASE_URL}/health", timeout=5)
            if response2.status_code == 200:
                data2 = response2.json()
                timestamp2 = data2.get('timestamp')
                
                if timestamp1 != timestamp2:
                    print("[PASS] Timestamps are updating (real-time data)")
                    print(f"  First: {timestamp1}")
                    print(f"  Second: {timestamp2}")
                else:
                    print("[WARN] Timestamps identical (may be cached)")
            else:
                print("[FAIL] Second health check failed")
        else:
            print("[FAIL] Initial health check failed")
            
    except Exception as e:
        print(f"[ERROR] Real-time test failed: {e}")

def test_data_consistency():
    """Test data consistency across endpoints"""
    print("\n=== Testing Data Consistency ===")
    
    try:
        # Get health data
        health_resp = requests.get(f"{BASE_URL}/health", timeout=5)
        status_resp = requests.get(f"{BASE_URL}/status", timeout=5)
        
        if health_resp.status_code == 200 and status_resp.status_code == 200:
            health_data = health_resp.json()
            status_data = status_resp.json()
            
            # Check timestamp consistency (should be close)
            health_time = health_data.get('timestamp')
            status_time = status_data.get('timestamp')
            
            if health_time and status_time:
                print(f"[PASS] Both endpoints return timestamps")
                print(f"  Health: {health_time}")
                print(f"  Status: {status_time}")
            else:
                print("[FAIL] Missing timestamps")
            
            # Check system status consistency
            health_status = health_data.get('status')
            if health_status:
                print(f"[PASS] Health status: {health_status}")
            
        else:
            print("[FAIL] Could not get both health and status data")
            
    except Exception as e:
        print(f"[ERROR] Consistency test failed: {e}")

def test_error_handling():
    """Test error handling for invalid endpoints"""
    print("\n=== Testing Error Handling ===")
    
    invalid_endpoints = ['/invalid', '/nonexistent', '/api/fake']
    
    for endpoint in invalid_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if response.status_code == 404:
                print(f"[PASS] {endpoint} returns 404 as expected")
            else:
                print(f"[WARN] {endpoint} returns {response.status_code} (expected 404)")
        except Exception as e:
            print(f"[ERROR] {endpoint} test failed: {e}")

def run_ui_tests():
    """Run all UI tests"""
    print("Dashboard UI Testing Suite")
    print("=" * 50)
    
    test_dashboard_html()
    test_real_time_updates()
    test_data_consistency()
    test_error_handling()
    
    print("\n=== UI Test Summary ===")
    print("Dashboard UI testing completed")

if __name__ == "__main__":
    run_ui_tests()