#!/usr/bin/env python3
"""
Comprehensive Testing and Debugging System

Tests all components and identifies bugs and errors in the AI trading system.
"""

import sys
import os
import traceback
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemTester:
    """Main system testing class"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
    def run_all_tests(self):
        """Run comprehensive system tests"""
        logger.info("Starting comprehensive system tests...")
        
        # Test basic imports
        self.test_imports()
        
        # Test configuration
        self.test_configuration()
        
        # Test core components
        self.test_core_components()
        
        # Test Phase 2 components
        self.test_phase2_components()
        
        # Test Phase 3 components
        self.test_phase3_components()
        
        # Test Phase 4 components
        self.test_phase4_components()
        
        # Test main application
        self.test_main_application()
        
        # Generate report
        self.generate_test_report()
    
    def test_imports(self):
        """Test all module imports"""
        logger.info("Testing imports...")
        
        modules_to_test = [
            'config_validator',
            'secure_config',
            'database_manager',
            'realtime_data_feed',
            'monitoring',
            'data_manager',
            'strategy_manager',
            'broker_manager',
            'supervised_learning',
            'order_execution_engine',
            'realtime_risk_manager',
            'performance_analytics',
            'api_server',
            'reinforcement_learning_agent',
            'ensemble_strategy_manager',
            'market_regime_detector',
            'advanced_portfolio_optimizer',
            'quantum_optimizer',
            'alternative_data_engine',
            'explainable_ai_engine',
            'federated_learning_system'
        ]
        
        import_results = {}
        
        for module in modules_to_test:
            try:
                __import__(module)
                import_results[module] = 'SUCCESS'
                logger.info(f"[PASS] {module} imported successfully")
            except Exception as e:
                import_results[module] = f'ERROR: {str(e)}'
                self.errors.append(f"Import error in {module}: {str(e)}")
                logger.error(f"[FAIL] {module} import failed: {str(e)}")
        
        self.test_results['imports'] = import_results
    
    def test_configuration(self):
        """Test configuration loading"""
        logger.info("Testing configuration...")
        
        try:
            from config_validator import load_config, validate_config
            
            # Test default config creation
            if not os.path.exists('config/config.json'):
                from main import create_default_config
                create_default_config()
            
            # Test config loading
            config = load_config('config/config.json')
            self.test_results['config_load'] = 'SUCCESS'
            logger.info("[PASS] Configuration loaded successfully")
            
            # Test config validation
            validation_result = validate_config(config)
            self.test_results['config_validation'] = 'SUCCESS' if validation_result else 'WARNING'
            
        except Exception as e:
            self.test_results['config_load'] = f'ERROR: {str(e)}'
            self.errors.append(f"Configuration error: {str(e)}")
            logger.error(f"[FAIL] Configuration test failed: {str(e)}")
    
    def test_core_components(self):
        """Test core Phase 1 components"""
        logger.info("Testing core components...")
        
        # Test SecureConfigManager
        try:
            from secure_config import SecureConfigManager
            secure_config = SecureConfigManager()
            self.test_results['secure_config'] = 'SUCCESS'
            logger.info("[PASS] SecureConfigManager works")
        except Exception as e:
            self.test_results['secure_config'] = f'ERROR: {str(e)}'
            self.errors.append(f"SecureConfigManager error: {str(e)}")
        
        # Test DatabaseManager
        try:
            from database_manager import DatabaseManager
            db_config = {'host': 'localhost', 'port': 5432, 'database': 'test'}
            db_manager = DatabaseManager(db_config)
            self.test_results['database_manager'] = 'SUCCESS'
            logger.info("[PASS] DatabaseManager initialized")
        except Exception as e:
            self.test_results['database_manager'] = f'WARNING: {str(e)}'
            self.warnings.append(f"DatabaseManager warning: {str(e)}")
        
        # Test DataManager
        try:
            from data_manager import DataManager
            data_manager = DataManager({})
            self.test_results['data_manager'] = 'SUCCESS'
            logger.info("[PASS] DataManager initialized")
        except Exception as e:
            self.test_results['data_manager'] = f'ERROR: {str(e)}'
            self.errors.append(f"DataManager error: {str(e)}")
    
    def test_phase2_components(self):
        """Test Phase 2 components"""
        logger.info("Testing Phase 2 components...")
        
        # Test OrderExecutionEngine
        try:
            from order_execution_engine import OrderExecutionEngine
            from broker_manager import BrokerManager
            
            broker_manager = BrokerManager({})
            execution_engine = OrderExecutionEngine({}, broker_manager)
            self.test_results['order_execution'] = 'SUCCESS'
            logger.info("[PASS] OrderExecutionEngine initialized")
        except Exception as e:
            self.test_results['order_execution'] = f'ERROR: {str(e)}'
            self.errors.append(f"OrderExecutionEngine error: {str(e)}")
        
        # Test RealTimeRiskManager
        try:
            from realtime_risk_manager import RealTimeRiskManager
            risk_manager = RealTimeRiskManager({}, None)
            self.test_results['risk_manager'] = 'SUCCESS'
            logger.info("[PASS] RealTimeRiskManager initialized")
        except Exception as e:
            self.test_results['risk_manager'] = f'ERROR: {str(e)}'
            self.errors.append(f"RealTimeRiskManager error: {str(e)}")
        
        # Test PerformanceAnalytics
        try:
            from performance_analytics import PerformanceAnalytics
            analytics = PerformanceAnalytics({}, None)
            self.test_results['performance_analytics'] = 'SUCCESS'
            logger.info("[PASS] PerformanceAnalytics initialized")
        except Exception as e:
            self.test_results['performance_analytics'] = f'ERROR: {str(e)}'
            self.errors.append(f"PerformanceAnalytics error: {str(e)}")
    
    def test_phase3_components(self):
        """Test Phase 3 components"""
        logger.info("Testing Phase 3 components...")
        
        # Test ReinforcementLearningAgent
        try:
            from reinforcement_learning_agent import RLTradingAgent
            rl_agent = RLTradingAgent({})
            self.test_results['rl_agent'] = 'SUCCESS'
            logger.info("[PASS] RLTradingAgent initialized")
        except Exception as e:
            self.test_results['rl_agent'] = f'ERROR: {str(e)}'
            self.errors.append(f"RLTradingAgent error: {str(e)}")
        
        # Test EnsembleStrategyManager
        try:
            from ensemble_strategy_manager import EnsembleStrategyManager
            ensemble = EnsembleStrategyManager({})
            self.test_results['ensemble_manager'] = 'SUCCESS'
            logger.info("[PASS] EnsembleStrategyManager initialized")
        except Exception as e:
            self.test_results['ensemble_manager'] = f'ERROR: {str(e)}'
            self.errors.append(f"EnsembleStrategyManager error: {str(e)}")
        
        # Test MarketRegimeDetector
        try:
            from market_regime_detector import MarketRegimeDetector
            regime_detector = MarketRegimeDetector({})
            self.test_results['regime_detector'] = 'SUCCESS'
            logger.info("[PASS] MarketRegimeDetector initialized")
        except Exception as e:
            self.test_results['regime_detector'] = f'ERROR: {str(e)}'
            self.errors.append(f"MarketRegimeDetector error: {str(e)}")
    
    def test_phase4_components(self):
        """Test Phase 4 components"""
        logger.info("Testing Phase 4 components...")
        
        # Test QuantumOptimizer
        try:
            from quantum_optimizer import QuantumPortfolioOptimizer
            quantum_opt = QuantumPortfolioOptimizer({})
            self.test_results['quantum_optimizer'] = 'SUCCESS'
            logger.info("[PASS] QuantumPortfolioOptimizer initialized")
        except Exception as e:
            self.test_results['quantum_optimizer'] = f'WARNING: {str(e)}'
            self.warnings.append(f"QuantumOptimizer warning: {str(e)}")
        
        # Test AlternativeDataEngine
        try:
            from alternative_data_engine import AlternativeDataEngine
            alt_data = AlternativeDataEngine({})
            self.test_results['alternative_data'] = 'SUCCESS'
            logger.info("[PASS] AlternativeDataEngine initialized")
        except Exception as e:
            self.test_results['alternative_data'] = f'ERROR: {str(e)}'
            self.errors.append(f"AlternativeDataEngine error: {str(e)}")
        
        # Test ExplainableAIEngine
        try:
            from explainable_ai_engine import ExplainableAIEngine
            explainable_ai = ExplainableAIEngine({})
            self.test_results['explainable_ai'] = 'SUCCESS'
            logger.info("[PASS] ExplainableAIEngine initialized")
        except Exception as e:
            self.test_results['explainable_ai'] = f'ERROR: {str(e)}'
            self.errors.append(f"ExplainableAIEngine error: {str(e)}")
        
        # Test FederatedLearningSystem
        try:
            from federated_learning_system import FederatedLearningSystem
            federated = FederatedLearningSystem({})
            self.test_results['federated_learning'] = 'SUCCESS'
            logger.info("[PASS] FederatedLearningSystem initialized")
        except Exception as e:
            self.test_results['federated_learning'] = f'ERROR: {str(e)}'
            self.errors.append(f"FederatedLearningSystem error: {str(e)}")
    
    def test_main_application(self):
        """Test main application initialization"""
        logger.info("Testing main application...")
        
        try:
            # Test config creation
            if not os.path.exists('config/config.json'):
                from main import create_default_config
                create_default_config()
            
            # Test TradingAgent initialization
            from main import TradingAgent
            agent = TradingAgent('config/config.json')
            self.test_results['main_application'] = 'SUCCESS'
            logger.info("[PASS] Main application initialized successfully")
            
        except Exception as e:
            self.test_results['main_application'] = f'ERROR: {str(e)}'
            self.errors.append(f"Main application error: {str(e)}")
            logger.error(f"[FAIL] Main application test failed: {str(e)}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results.values() if r == 'SUCCESS']),
                'warnings': len([r for r in self.test_results.values() if 'WARNING' in str(r)]),
                'errors': len([r for r in self.test_results.values() if 'ERROR' in str(r)])
            },
            'test_results': self.test_results,
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        # Save report
        os.makedirs('test_reports', exist_ok=True)
        report_file = f"test_reports/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("SYSTEM TEST REPORT")
        print("="*60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"[PASS] Passed: {report['summary']['passed']}")
        print(f"[WARN] Warnings: {report['summary']['warnings']}")
        print(f"[FAIL] Errors: {report['summary']['errors']}")
        print(f"\nReport saved to: {report_file}")
        
        if self.errors:
            print("\nCRITICAL ERRORS:")
            for error in self.errors:
                print(f"  [FAIL] {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  [WARN] {warning}")
        
        print("="*60)
        
        return report


def run_quick_test():
    """Run a quick test of the main application"""
    try:
        print("Running quick test...")
        
        # Test basic imports
        from main import TradingAgent, create_default_config
        
        # Create config if needed
        if not os.path.exists('config/config.json'):
            create_default_config()
        
        # Test agent initialization
        agent = TradingAgent('config/config.json')
        
        # Test status
        status = agent.get_status()
        print(f"Agent status: {status.get('running', 'unknown')}")
        
        print("[PASS] Quick test passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Quick test failed: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_quick_test()
    else:
        tester = SystemTester()
        tester.run_all_tests()