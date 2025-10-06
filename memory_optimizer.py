import gc
import psutil
import threading
import time
import logging

class MemoryOptimizer:
    def __init__(self, threshold_mb=1000, check_interval=60):
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Memory optimizer started")
    
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        while self.running:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > self.threshold_mb:
                    logging.warning(f"High memory usage: {memory_mb:.1f}MB")
                    self._optimize_memory()
                
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Memory monitor error: {e}")
                time.sleep(10)
    
    def _optimize_memory(self):
        """Perform memory optimization"""
        before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        collected = gc.collect()
        
        after = psutil.Process().memory_info().rss / 1024 / 1024
        freed = before - after
        
        logging.info(f"Memory optimization: {collected} objects collected, {freed:.1f}MB freed")

# Global memory optimizer
memory_optimizer = MemoryOptimizer()
