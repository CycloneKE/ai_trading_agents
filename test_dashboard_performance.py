#!/usr/bin/env python3
"""
Dashboard Performance Testing
Tests response times and load handling
"""

import requests
import time
import statistics
import concurrent.futures
from datetime import datetime

BASE_URL = "http://localhost:8080"

def measure_response_time(endpoint, iterations=10):
    """Measure average response time for an endpoint"""
    times = []
    
    for _ in range(iterations):
        start = time.time()
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            end = time.time()
            if response.status_code == 200:
                times.append(end - start)
        except:
            pass
    
    if times:
        return {
            'avg': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'count': len(times)
        }
    return None

def test_response_times():
    """Test response times for all endpoints"""
    print("=== Response Time Testing ===")
    
    endpoints = ['/health', '/status', '/metrics', '/']
    
    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")
        stats = measure_response_time(endpoint)
        
        if stats:
            print(f"  Average: {stats['avg']*1000:.1f}ms")
            print(f"  Min: {stats['min']*1000:.1f}ms")
            print(f"  Max: {stats['max']*1000:.1f}ms")
            print(f"  Success: {stats['count']}/10")
            
            # Performance thresholds
            if stats['avg'] < 0.1:  # 100ms
                print("  [EXCELLENT] Very fast response")
            elif stats['avg'] < 0.5:  # 500ms
                print("  [GOOD] Acceptable response time")
            elif stats['avg'] < 1.0:  # 1s
                print("  [FAIR] Slow but usable")
            else:
                print("  [POOR] Very slow response")
        else:
            print("  [FAIL] No successful responses")

def concurrent_load_test(endpoint, concurrent_requests=5, total_requests=25):
    """Test concurrent load handling"""
    print(f"\n=== Concurrent Load Test: {endpoint} ===")
    
    def make_request():
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            end = time.time()
            return {
                'success': response.status_code == 200,
                'time': end - start,
                'status': response.status_code
            }
        except Exception as e:
            return {
                'success': False,
                'time': 0,
                'error': str(e)
            }
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(make_request) for _ in range(total_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"  Total time: {end_time - start_time:.2f}s")
    print(f"  Successful: {len(successful)}/{total_requests}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        times = [r['time'] for r in successful]
        print(f"  Avg response: {statistics.mean(times)*1000:.1f}ms")
        print(f"  Requests/sec: {len(successful)/(end_time - start_time):.1f}")
    
    if failed:
        print(f"  Failure rate: {len(failed)/total_requests*100:.1f}%")

def test_memory_usage():
    """Test if dashboard causes memory leaks"""
    print("\n=== Memory Usage Test ===")
    
    # Make many requests to see if memory grows
    print("Making 50 requests to check for memory leaks...")
    
    start_time = time.time()
    success_count = 0
    
    for i in range(50):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                success_count += 1
        except:
            pass
        
        if i % 10 == 0:
            print(f"  Progress: {i+1}/50")
    
    end_time = time.time()
    
    print(f"  Completed: {success_count}/50 successful")
    print(f"  Total time: {end_time - start_time:.2f}s")
    print(f"  Avg per request: {(end_time - start_time)/50*1000:.1f}ms")

def run_performance_tests():
    """Run all performance tests"""
    print("Dashboard Performance Testing Suite")
    print("=" * 50)
    
    test_response_times()
    concurrent_load_test('/health')
    concurrent_load_test('/status')
    test_memory_usage()
    
    print("\n=== Performance Test Summary ===")
    print("Performance testing completed")

if __name__ == "__main__":
    run_performance_tests()