import numpy as np
from typing import Dict, List, Tuple, Any
import json

class StrategyOptimizer:
    def __init__(self):
        self.strategies = {}
        self.optimization_history = []
    
    def register_strategy(self, name: str, parameters: Dict[str, Any], 
                         performance_func: callable):
        """Register a strategy for optimization"""
        self.strategies[name] = {
            'parameters': parameters,
            'performance_func': performance_func,
            'best_params': parameters.copy(),
            'best_performance': 0
        }
    
    def optimize_strategy(self, strategy_name: str, param_ranges: Dict[str, Tuple], 
                         iterations: int = 100) -> Dict[str, Any]:
        """Optimize strategy parameters using random search"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not registered")
        
        strategy = self.strategies[strategy_name]
        best_params = strategy['parameters'].copy()
        best_performance = float('-inf')
        
        results = []
        
        for i in range(iterations):
            # Generate random parameters within ranges
            test_params = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    test_params[param] = np.random.randint(min_val, max_val + 1)
                else:
                    test_params[param] = np.random.uniform(min_val, max_val)
            
            # Test performance
            try:
                performance = strategy['performance_func'](test_params)
                
                results.append({
                    'iteration': i,
                    'parameters': test_params.copy(),
                    'performance': performance
                })
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = test_params.copy()
                    
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                continue
        
        # Update strategy with best parameters
        strategy['best_params'] = best_params
        strategy['best_performance'] = best_performance
        
        optimization_result = {
            'strategy_name': strategy_name,
            'best_parameters': best_params,
            'best_performance': best_performance,
            'iterations': iterations,
            'improvement': best_performance - strategy.get('baseline_performance', 0),
            'all_results': results
        }
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations"""
        return {
            'total_optimizations': len(self.optimization_history),
            'strategies_optimized': list(self.strategies.keys()),
            'recent_optimizations': self.optimization_history[-5:],
            'best_performers': self._get_best_performers()
        }
    
    def _get_best_performers(self) -> List[Dict]:
        """Get best performing strategies"""
        performers = []
        
        for name, strategy in self.strategies.items():
            performers.append({
                'name': name,
                'performance': strategy['best_performance'],
                'parameters': strategy['best_params']
            })
        
        return sorted(performers, key=lambda x: x['performance'], reverse=True)

# Example strategy performance functions
def momentum_strategy_performance(params):
    """Example momentum strategy performance function"""
    lookback = params.get('lookback_period', 20)
    threshold = params.get('threshold', 0.02)
    
    # Simulate performance based on parameters
    # In reality, this would run backtests
    base_performance = 0.1
    lookback_factor = max(0, 1 - abs(lookback - 15) / 50)
    threshold_factor = max(0, 1 - abs(threshold - 0.015) / 0.1)
    
    return base_performance * lookback_factor * threshold_factor + np.random.normal(0, 0.02)

def mean_reversion_strategy_performance(params):
    """Example mean reversion strategy performance function"""
    window = params.get('window', 10)
    z_threshold = params.get('z_threshold', 2.0)
    
    base_performance = 0.08
    window_factor = max(0, 1 - abs(window - 12) / 30)
    z_factor = max(0, 1 - abs(z_threshold - 1.8) / 5)
    
    return base_performance * window_factor * z_factor + np.random.normal(0, 0.015)

# Global optimizer instance
strategy_optimizer = StrategyOptimizer()

# Register example strategies
strategy_optimizer.register_strategy(
    'momentum', 
    {'lookback_period': 20, 'threshold': 0.02},
    momentum_strategy_performance
)

strategy_optimizer.register_strategy(
    'mean_reversion',
    {'window': 10, 'z_threshold': 2.0},
    mean_reversion_strategy_performance
)
