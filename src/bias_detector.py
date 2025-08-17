"""
Algorithmic Bias Detection and Mitigation for Trading AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BiasDetector:
    """Detects and mitigates algorithmic bias in trading decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bias_threshold = config.get('bias_threshold', 0.15)  # 15% bias threshold
        self.lookback_days = config.get('bias_lookback_days', 30)
        
        # Bias tracking
        self.decision_history = []
        self.performance_by_sector = {}
        self.performance_by_market_cap = {}
        self.performance_by_volatility = {}
        
    def detect_sector_bias(self, decisions: List[Dict]) -> Dict[str, float]:
        """Detect bias towards specific sectors"""
        sector_counts = {}
        sector_performance = {}
        
        for decision in decisions:
            sector = decision.get('sector', 'unknown')
            action = decision.get('action', 'hold')
            returns = decision.get('returns', 0.0)
            
            if sector not in sector_counts:
                sector_counts[sector] = {'buy': 0, 'sell': 0, 'hold': 0}
                sector_performance[sector] = []
            
            sector_counts[sector][action] += 1
            sector_performance[sector].append(returns)
        
        # Calculate bias metrics
        bias_scores = {}
        total_decisions = len(decisions)
        
        for sector, counts in sector_counts.items():
            total_sector = sum(counts.values())
            sector_ratio = total_sector / total_decisions
            
            # Check for over/under representation
            expected_ratio = 1.0 / len(sector_counts)  # Equal distribution expected
            bias_score = abs(sector_ratio - expected_ratio) / expected_ratio
            
            avg_performance = np.mean(sector_performance[sector]) if sector_performance[sector] else 0.0
            
            bias_scores[sector] = {
                'representation_bias': bias_score,
                'avg_performance': avg_performance,
                'decision_count': total_sector
            }
        
        return bias_scores
    
    def detect_market_cap_bias(self, decisions: List[Dict]) -> Dict[str, float]:
        """Detect bias towards large/small cap stocks"""
        cap_buckets = {'small': [], 'mid': [], 'large': []}
        
        for decision in decisions:
            market_cap = decision.get('market_cap', 0)
            returns = decision.get('returns', 0.0)
            
            if market_cap < 2e9:  # < $2B
                bucket = 'small'
            elif market_cap < 10e9:  # $2B - $10B
                bucket = 'mid'
            else:  # > $10B
                bucket = 'large'
            
            cap_buckets[bucket].append({
                'returns': returns,
                'action': decision.get('action', 'hold')
            })
        
        bias_metrics = {}
        total_decisions = sum(len(bucket) for bucket in cap_buckets.values())
        
        for cap_type, decisions_list in cap_buckets.items():
            if not decisions_list:
                continue
                
            count = len(decisions_list)
            ratio = count / total_decisions
            expected_ratio = 1.0 / 3  # Equal distribution expected
            
            avg_returns = np.mean([d['returns'] for d in decisions_list])
            buy_ratio = sum(1 for d in decisions_list if d['action'] == 'buy') / count
            
            bias_metrics[cap_type] = {
                'representation_bias': abs(ratio - expected_ratio) / expected_ratio,
                'avg_performance': avg_returns,
                'buy_ratio': buy_ratio,
                'count': count
            }
        
        return bias_metrics
    
    def detect_temporal_bias(self, decisions: List[Dict]) -> Dict[str, float]:
        """Detect bias in timing of decisions"""
        hourly_decisions = {}
        daily_decisions = {}
        
        for decision in decisions:
            timestamp = decision.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            hour = timestamp.hour
            day = timestamp.strftime('%A')
            
            if hour not in hourly_decisions:
                hourly_decisions[hour] = []
            if day not in daily_decisions:
                daily_decisions[day] = []
            
            hourly_decisions[hour].append(decision)
            daily_decisions[day].append(decision)
        
        # Analyze hourly bias
        hourly_bias = {}
        total_hours = len(hourly_decisions)
        expected_per_hour = len(decisions) / 24 if total_hours > 0 else 0
        
        for hour, hour_decisions in hourly_decisions.items():
            count = len(hour_decisions)
            bias_score = abs(count - expected_per_hour) / (expected_per_hour + 1e-10)
            avg_returns = np.mean([d.get('returns', 0) for d in hour_decisions])
            
            hourly_bias[hour] = {
                'count': count,
                'bias_score': bias_score,
                'avg_returns': avg_returns
            }
        
        return {
            'hourly_bias': hourly_bias,
            'peak_hour': max(hourly_decisions.keys(), key=lambda h: len(hourly_decisions[h])) if hourly_decisions else None
        }
    
    def detect_feature_bias(self, features: Dict[str, List[float]], decisions: List[str]) -> Dict[str, float]:
        """Detect bias in feature usage"""
        feature_importance = {}
        
        # Calculate correlation between features and decisions
        decision_numeric = [1 if d == 'buy' else (-1 if d == 'sell' else 0) for d in decisions]
        
        for feature_name, feature_values in features.items():
            if len(feature_values) == len(decision_numeric):
                correlation = np.corrcoef(feature_values, decision_numeric)[0, 1]
                feature_importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        # Detect over-reliance on single features
        max_importance = max(feature_importance.values()) if feature_importance else 0.0
        bias_threshold = 0.8  # 80% reliance on single feature is concerning
        
        return {
            'feature_importance': feature_importance,
            'max_feature_bias': max_importance,
            'has_feature_bias': max_importance > bias_threshold
        }
    
    def generate_bias_report(self, decisions: List[Dict], features: Dict[str, List[float]] = None) -> Dict[str, Any]:
        """Generate comprehensive bias report"""
        if not decisions:
            return {'error': 'No decisions to analyze'}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_decisions': len(decisions),
            'analysis_period': self.lookback_days
        }
        
        # Sector bias analysis
        try:
            sector_bias = self.detect_sector_bias(decisions)
            report['sector_bias'] = sector_bias
            report['sector_bias_detected'] = any(
                metrics['representation_bias'] > self.bias_threshold 
                for metrics in sector_bias.values()
            )
        except Exception as e:
            logger.warning(f"Sector bias analysis failed: {e}")
            report['sector_bias'] = {}
        
        # Market cap bias analysis
        try:
            cap_bias = self.detect_market_cap_bias(decisions)
            report['market_cap_bias'] = cap_bias
            report['cap_bias_detected'] = any(
                metrics['representation_bias'] > self.bias_threshold 
                for metrics in cap_bias.values()
            )
        except Exception as e:
            logger.warning(f"Market cap bias analysis failed: {e}")
            report['market_cap_bias'] = {}
        
        # Temporal bias analysis
        try:
            temporal_bias = self.detect_temporal_bias(decisions)
            report['temporal_bias'] = temporal_bias
        except Exception as e:
            logger.warning(f"Temporal bias analysis failed: {e}")
            report['temporal_bias'] = {}
        
        # Feature bias analysis
        if features:
            try:
                feature_bias = self.detect_feature_bias(features, [d.get('action', 'hold') for d in decisions])
                report['feature_bias'] = feature_bias
                report['feature_bias_detected'] = feature_bias.get('has_feature_bias', False)
            except Exception as e:
                logger.warning(f"Feature bias analysis failed: {e}")
                report['feature_bias'] = {}
        
        # Overall bias score
        bias_flags = [
            report.get('sector_bias_detected', False),
            report.get('cap_bias_detected', False),
            report.get('feature_bias_detected', False)
        ]
        report['overall_bias_score'] = sum(bias_flags) / len(bias_flags)
        report['bias_detected'] = report['overall_bias_score'] > 0.3
        
        return report
    
    def suggest_mitigation(self, bias_report: Dict[str, Any]) -> List[str]:
        """Suggest bias mitigation strategies"""
        suggestions = []
        
        if bias_report.get('sector_bias_detected'):
            suggestions.append("Implement sector-balanced portfolio allocation")
            suggestions.append("Add sector diversification constraints")
        
        if bias_report.get('cap_bias_detected'):
            suggestions.append("Balance exposure across market cap segments")
            suggestions.append("Add market cap diversity scoring")
        
        if bias_report.get('feature_bias_detected'):
            suggestions.append("Regularize model to reduce single-feature dependence")
            suggestions.append("Implement feature importance monitoring")
            suggestions.append("Use ensemble methods to diversify feature usage")
        
        if bias_report.get('overall_bias_score', 0) > 0.5:
            suggestions.append("Implement bias-aware position sizing")
            suggestions.append("Add fairness constraints to optimization")
            suggestions.append("Regular bias auditing and model retraining")
        
        return suggestions
    
    def apply_bias_correction(self, decision: Dict[str, Any], bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bias correction to trading decision"""
        corrected_decision = decision.copy()
        
        # Reduce position size if bias detected
        if bias_report.get('bias_detected', False):
            original_size = decision.get('position_size', 0.0)
            bias_penalty = bias_report.get('overall_bias_score', 0.0)
            corrected_size = original_size * (1.0 - bias_penalty * 0.5)  # Max 50% reduction
            
            corrected_decision['position_size'] = max(corrected_size, 0.0)
            corrected_decision['bias_adjusted'] = True
            corrected_decision['bias_penalty'] = bias_penalty
            
            logger.info(f"Applied bias correction: {original_size:.3f} -> {corrected_size:.3f}")
        
        return corrected_decision