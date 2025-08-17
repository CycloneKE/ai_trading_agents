"""
Paper Trading Test Suite
Comprehensive testing for live trading with paper money
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_data_connector import RealDataConnector
from api_key_setup import APIKeyValidator

logger = logging.getLogger(__name__)

class PaperTradingTest:
    """Paper trading test environment"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = self._load_config(config_path)
        self.data_connector = RealDataConnector(self.config)
        self.api_validator = APIKeyValidator()
        
        # Paper trading state
        self.portfolio = {
            'cash': 100000.0,
            'positions': {},
            'total_value': 100000.0,
            'trades': [],
            'performance': {
                'total_return': 0.0,
                'daily_returns': [],
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        }
        
        self.test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.test_results = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive paper trading test"""
        print("\n" + "="*60)
        print("PAPER TRADING TEST SUITE")
        print("="*60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'api_validation': {},
            'data_connectivity': {},
            'trading_simulation': {},
            'performance_metrics': {},
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # 1. Validate API keys
            print("\n1. Validating API Keys...")
            results['api_validation'] = self._test_api_keys()
            
            # 2. Test data connectivity
            print("\n2. Testing Data Connectivity...")
            results['data_connectivity'] = self._test_data_connectivity()
            
            # 3. Run trading simulation
            print("\n3. Running Trading Simulation...")
            results['trading_simulation'] = self._run_trading_simulation()
            
            # 4. Calculate performance metrics
            print("\n4. Calculating Performance Metrics...")
            results['performance_metrics'] = self._calculate_performance_metrics()
            
            # 5. Determine overall status
            results['overall_status'] = self._determine_overall_status(results)
            
            # 6. Generate report
            self._generate_test_report(results)
            
        except Exception as e:
            logger.error(f"Test suite error: {e}")
            results['error'] = str(e)
            results['overall_status'] = 'FAILED'
        
        return results
    
    def _test_api_keys(self) -> Dict[str, Any]:
        """Test API key validation"""
        try:
            validation_results = self.api_validator.validate_all_keys()
            
            valid_count = sum(validation_results.values())
            total_count = len(validation_results)
            
            status = 'PASS' if valid_count > 0 else 'FAIL'
            
            print(f"   API Keys: {valid_count}/{total_count} valid - {status}")
            
            return {
                'status': status,
                'valid_keys': valid_count,
                'total_keys': total_count,
                'details': validation_results
            }
            
        except Exception as e:
            logger.error(f"API key test error: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _test_data_connectivity(self) -> Dict[str, Any]:
        """Test real-time data connectivity"""
        try:
            connectivity_results = {}
            successful_symbols = 0
            
            for symbol in self.test_symbols:
                try:
                    data = self.data_connector.get_real_time_data(symbol)
                    
                    if data and 'price' in data and data['price'] > 0:
                        connectivity_results[symbol] = {
                            'status': 'SUCCESS',
                            'price': data['price'],
                            'source': data.get('source', 'unknown')
                        }
                        successful_symbols += 1
                    else:
                        connectivity_results[symbol] = {
                            'status': 'FAILED',
                            'error': 'Invalid data received'
                        }
                        
                except Exception as e:
                    connectivity_results[symbol] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
            
            success_rate = successful_symbols / len(self.test_symbols)
            status = 'PASS' if success_rate >= 0.6 else 'FAIL'
            
            print(f"   Data Connectivity: {successful_symbols}/{len(self.test_symbols)} symbols - {status}")
            
            return {
                'status': status,
                'success_rate': success_rate,
                'successful_symbols': successful_symbols,
                'total_symbols': len(self.test_symbols),
                'details': connectivity_results
            }
            
        except Exception as e:
            logger.error(f"Data connectivity test error: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _run_trading_simulation(self) -> Dict[str, Any]:
        """Run paper trading simulation"""
        try:
            simulation_results = {
                'trades_executed': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'total_pnl': 0.0,
                'trade_details': []
            }
            
            # Simulate 10 trades
            for i in range(10):
                symbol = self.test_symbols[i % len(self.test_symbols)]
                
                try:
                    # Get current market data
                    market_data = self.data_connector.get_real_time_data(symbol)
                    
                    if not market_data or 'price' in market_data:
                        continue
                    
                    # Simulate trading decision (simple momentum strategy)
                    action = self._simulate_trading_decision(symbol, market_data)
                    
                    if action != 'hold':
                        trade_result = self._execute_paper_trade(symbol, action, market_data)
                        simulation_results['trade_details'].append(trade_result)
                        simulation_results['trades_executed'] += 1
                        
                        if trade_result['status'] == 'SUCCESS':
                            simulation_results['successful_trades'] += 1
                            simulation_results['total_pnl'] += trade_result.get('pnl', 0)
                        else:
                            simulation_results['failed_trades'] += 1
                    
                    # Small delay between trades
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Trade simulation error for {symbol}: {e}")
                    simulation_results['failed_trades'] += 1
            
            success_rate = (simulation_results['successful_trades'] / 
                          max(simulation_results['trades_executed'], 1))
            
            status = 'PASS' if success_rate >= 0.8 else 'FAIL'
            
            print(f"   Trading Simulation: {simulation_results['successful_trades']}/{simulation_results['trades_executed']} trades - {status}")
            
            simulation_results['status'] = status
            simulation_results['success_rate'] = success_rate
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Trading simulation error: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _simulate_trading_decision(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Simulate a simple trading decision"""
        try:
            # Simple momentum strategy
            change_percent = market_data.get('change_percent', 0)
            
            if change_percent > 2.0:
                return 'buy'
            elif change_percent < -2.0:
                return 'sell'
            else:
                return 'hold'
                
        except Exception:
            return 'hold'
    
    def _execute_paper_trade(self, symbol: str, action: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a paper trade"""
        try:
            price = market_data['price']
            quantity = 10  # Fixed quantity for testing
            
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'price': price,
                'quantity': quantity,
                'value': price * quantity,
                'status': 'SUCCESS',
                'pnl': 0.0
            }
            
            # Update paper portfolio
            if action == 'buy':
                if self.portfolio['cash'] >= trade['value']:
                    self.portfolio['cash'] -= trade['value']
                    if symbol in self.portfolio['positions']:
                        self.portfolio['positions'][symbol] += quantity
                    else:
                        self.portfolio['positions'][symbol] = quantity
                else:
                    trade['status'] = 'FAILED'
                    trade['error'] = 'Insufficient cash'
            
            elif action == 'sell':
                if symbol in self.portfolio['positions'] and self.portfolio['positions'][symbol] >= quantity:
                    self.portfolio['cash'] += trade['value']
                    self.portfolio['positions'][symbol] -= quantity
                    if self.portfolio['positions'][symbol] == 0:
                        del self.portfolio['positions'][symbol]
                else:
                    trade['status'] = 'FAILED'
                    trade['error'] = 'Insufficient position'
            
            # Calculate simple P&L (mock)
            if trade['status'] == 'SUCCESS':
                trade['pnl'] = (price - 100) * quantity * 0.01  # Mock P&L calculation
            
            self.portfolio['trades'].append(trade)
            return trade
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            # Calculate portfolio value
            portfolio_value = self.portfolio['cash']
            
            for symbol, quantity in self.portfolio['positions'].items():
                try:
                    market_data = self.data_connector.get_real_time_data(symbol)
                    if market_data and 'price' in market_data:
                        portfolio_value += market_data['price'] * quantity
                except Exception:
                    portfolio_value += 100 * quantity  # Use default price
            
            self.portfolio['total_value'] = portfolio_value
            
            # Calculate returns
            initial_value = 100000.0
            total_return = (portfolio_value - initial_value) / initial_value
            
            # Calculate other metrics (simplified)
            successful_trades = len([t for t in self.portfolio['trades'] if t['status'] == 'SUCCESS'])
            total_trades = len(self.portfolio['trades'])
            
            metrics = {
                'portfolio_value': portfolio_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'successful_trades': successful_trades,
                'total_trades': total_trades,
                'win_rate': successful_trades / max(total_trades, 1),
                'cash_remaining': self.portfolio['cash'],
                'positions_count': len(self.portfolio['positions'])
            }
            
            status = 'PASS' if total_return > -0.05 else 'FAIL'  # Allow 5% loss
            metrics['status'] = status
            
            print(f"   Performance: {total_return*100:.2f}% return, {successful_trades}/{total_trades} trades - {status}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance calculation error: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """Determine overall test status"""
        try:
            statuses = []
            
            for category, result in results.items():
                if isinstance(result, dict) and 'status' in result:
                    statuses.append(result['status'])
            
            if 'ERROR' in statuses:
                return 'ERROR'
            elif 'FAIL' in statuses:
                return 'PARTIAL'
            elif 'PASS' in statuses:
                return 'PASS'
            else:
                return 'UNKNOWN'
                
        except Exception:
            return 'ERROR'
    
    def _generate_test_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("PAPER TRADING TEST REPORT")
        print("="*60)
        
        print(f"Test Date: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status']}")
        
        print(f"\nAPI Validation: {results['api_validation'].get('status', 'UNKNOWN')}")
        print(f"Data Connectivity: {results['data_connectivity'].get('status', 'UNKNOWN')}")
        print(f"Trading Simulation: {results['trading_simulation'].get('status', 'UNKNOWN')}")
        print(f"Performance Metrics: {results['performance_metrics'].get('status', 'UNKNOWN')}")
        
        # Portfolio summary
        if 'performance_metrics' in results:
            perf = results['performance_metrics']
            print(f"\nPortfolio Summary:")
            print(f"  Total Value: ${perf.get('portfolio_value', 0):,.2f}")
            print(f"  Total Return: {perf.get('total_return_pct', 0):.2f}%")
            print(f"  Trades: {perf.get('successful_trades', 0)}/{perf.get('total_trades', 0)}")
            print(f"  Win Rate: {perf.get('win_rate', 0)*100:.1f}%")
        
        # Recommendations
        print(f"\nRecommendations:")
        if results['overall_status'] == 'PASS':
            print("  ✓ System ready for live paper trading")
            print("  ✓ All components functioning properly")
        elif results['overall_status'] == 'PARTIAL':
            print("  ⚠ Some components need attention")
            print("  ⚠ Review failed tests before live trading")
        else:
            print("  ✗ System not ready for live trading")
            print("  ✗ Fix critical issues before proceeding")
        
        # Save report to file
        try:
            report_file = f"paper_trading_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading Test Suite')
    parser.add_argument('--config', default='config/config.json', help='Config file path')
    parser.add_argument('--quick', action='store_true', help='Run quick test (fewer trades)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_suite = PaperTradingTest(args.config)
    results = test_suite.run_comprehensive_test()
    
    return results['overall_status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)