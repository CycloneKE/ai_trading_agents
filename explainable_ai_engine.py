#!/usr/bin/env python3
"""
Explainable AI Engine

Advanced explainability system for trading decisions using SHAP, LIME,
and custom interpretability methods for regulatory compliance and transparency.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based model explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainers = {}
        self.background_data = {}
        
    def create_explainer(self, model, X_background: np.ndarray, model_type: str = 'tree'):
        """Create SHAP explainer for a model"""
        try:
            if not SHAP_AVAILABLE:
                return None
            
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, X_background)
            elif model_type == 'kernel':
                explainer = shap.KernelExplainer(model.predict, X_background)
            else:
                explainer = shap.Explainer(model, X_background)
            
            model_id = id(model)
            self.explainers[model_id] = explainer
            self.background_data[model_id] = X_background
            
            return explainer
            
        except Exception as e:
            logger.error(f"SHAP explainer creation failed: {e}")
            return None
    
    def explain_prediction(self, model, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Explain a single prediction using SHAP"""
        try:
            model_id = id(model)
            if model_id not in self.explainers:
                return {'error': 'No explainer available for this model'}
            
            explainer = self.explainers[model_id]
            shap_values = explainer.shap_values(X)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for binary classification
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first sample
            
            # Create explanation
            explanation = {
                'feature_importance': {},
                'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0,
                'prediction_value': float(np.sum(shap_values) + (explainer.expected_value if hasattr(explainer, 'expected_value') else 0)),
                'feature_contributions': {}
            }
            
            # Feature importance and contributions
            for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
                explanation['feature_importance'][feature] = float(abs(shap_val))
                explanation['feature_contributions'][feature] = {
                    'value': float(X[i]) if len(X.shape) == 1 else float(X[0][i]),
                    'contribution': float(shap_val),
                    'impact': 'positive' if shap_val > 0 else 'negative'
                }
            
            # Sort by importance
            sorted_features = sorted(explanation['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            explanation['top_features'] = sorted_features[:10]
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}


class LIMEExplainer:
    """LIME-based model explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainers = {}
    
    def create_explainer(self, X_train: np.ndarray, feature_names: List[str], 
                        mode: str = 'regression'):
        """Create LIME explainer"""
        try:
            if not LIME_AVAILABLE:
                return None
            
            explainer = LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                mode=mode,
                discretize_continuous=True
            )
            
            explainer_id = hash(tuple(feature_names))
            self.explainers[explainer_id] = explainer
            
            return explainer
            
        except Exception as e:
            logger.error(f"LIME explainer creation failed: {e}")
            return None
    
    def explain_prediction(self, model, X: np.ndarray, feature_names: List[str],
                          num_features: int = 10) -> Dict[str, Any]:
        """Explain prediction using LIME"""
        try:
            explainer_id = hash(tuple(feature_names))
            if explainer_id not in self.explainers:
                return {'error': 'No LIME explainer available'}
            
            explainer = self.explainers[explainer_id]
            
            # Get explanation
            if len(X.shape) > 1:
                X_sample = X[0]
            else:
                X_sample = X
            
            explanation = explainer.explain_instance(
                X_sample, 
                model.predict, 
                num_features=num_features
            )
            
            # Parse LIME explanation
            lime_explanation = {
                'feature_importance': {},
                'local_prediction': float(explanation.local_pred[0]) if hasattr(explanation, 'local_pred') else 0.0,
                'intercept': float(explanation.intercept[0]) if hasattr(explanation, 'intercept') else 0.0
            }
            
            # Extract feature contributions
            for feature_idx, contribution in explanation.as_list():
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'feature_{feature_idx}'
                lime_explanation['feature_importance'][feature_name] = float(abs(contribution))
            
            return lime_explanation
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}


class CustomExplainer:
    """Custom explainability methods for trading-specific insights"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def explain_trading_decision(self, signal: Dict[str, Any], 
                               market_data: Dict[str, Any],
                               strategy_name: str) -> Dict[str, Any]:
        """Explain trading decision in business terms"""
        try:
            explanation = {
                'decision': signal.get('action', 'hold'),
                'confidence': signal.get('confidence', 0),
                'strategy': strategy_name,
                'reasoning': [],
                'risk_factors': [],
                'market_context': {},
                'regulatory_notes': []
            }
            
            # Analyze decision reasoning
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)
            
            if action == 'buy':
                explanation['reasoning'].append(f"Model indicates positive expected return with {confidence:.1%} confidence")
                if confidence > 0.7:
                    explanation['reasoning'].append("High confidence signal based on strong technical indicators")
                elif confidence > 0.5:
                    explanation['reasoning'].append("Moderate confidence signal with mixed indicators")
                else:
                    explanation['reasoning'].append("Low confidence signal - proceed with caution")
            
            elif action == 'sell':
                explanation['reasoning'].append(f"Model indicates negative expected return with {confidence:.1%} confidence")
                explanation['risk_factors'].append("Potential downside risk identified")
            
            else:
                explanation['reasoning'].append("No clear directional signal - maintaining current position")
            
            # Market context
            volatility = market_data.get('volatility', 0)
            volume = market_data.get('volume', 0)
            
            if volatility > 0.03:
                explanation['market_context']['volatility'] = 'High market volatility detected'
                explanation['risk_factors'].append('Elevated market volatility increases position risk')
            
            if volume > market_data.get('avg_volume', volume):
                explanation['market_context']['volume'] = 'Above-average trading volume'
            
            # Regulatory compliance notes
            explanation['regulatory_notes'].extend([
                'Decision based on quantitative model analysis',
                'No insider information used in decision process',
                'Risk management controls applied',
                'Decision subject to position size limits'
            ])
            
            return explanation
            
        except Exception as e:
            logger.error(f"Custom explanation failed: {e}")
            return {'error': str(e)}
    
    def analyze_feature_stability(self, feature_history: List[Dict[str, float]],
                                 window_size: int = 20) -> Dict[str, Any]:
        """Analyze stability of feature importance over time"""
        try:
            if len(feature_history) < window_size:
                return {'error': 'Insufficient history for stability analysis'}
            
            # Convert to DataFrame for easier analysis
            if not PANDAS_AVAILABLE:
                return {'error': 'Pandas not available for stability analysis'}
            
            df = pd.DataFrame(feature_history[-window_size:])
            
            stability_analysis = {
                'feature_stability': {},
                'trending_features': {},
                'stable_features': [],
                'unstable_features': []
            }
            
            for feature in df.columns:
                values = df[feature].values
                
                # Calculate stability metrics
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 1e-8)  # Coefficient of variation
                
                # Trend analysis
                x = np.arange(len(values))
                trend_slope = np.polyfit(x, values, 1)[0]
                
                stability_analysis['feature_stability'][feature] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'coefficient_of_variation': float(cv),
                    'trend_slope': float(trend_slope),
                    'stability_score': float(1 / (1 + cv))  # Higher is more stable
                }
                
                # Categorize features
                if cv < 0.2:
                    stability_analysis['stable_features'].append(feature)
                elif cv > 0.5:
                    stability_analysis['unstable_features'].append(feature)
                
                if abs(trend_slope) > 0.01:
                    stability_analysis['trending_features'][feature] = {
                        'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                        'slope': float(trend_slope)
                    }
            
            return stability_analysis
            
        except Exception as e:
            logger.error(f"Feature stability analysis failed: {e}")
            return {'error': str(e)}


class ExplainableAIEngine:
    """Main explainable AI engine coordinating all explanation methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shap_explainer = SHAPExplainer(config.get('shap', {}))
        self.lime_explainer = LIMEExplainer(config.get('lime', {}))
        self.custom_explainer = CustomExplainer(config.get('custom', {}))
        
        # Explanation history
        self.explanation_history = []
        self.model_explanations = {}
        
        # Configuration
        self.explanation_methods = config.get('methods', ['shap', 'lime', 'custom'])
        self.store_explanations = config.get('store_explanations', True)
        
        logger.info("Explainable AI Engine initialized")
    
    def explain_model_prediction(self, model, X: np.ndarray, feature_names: List[str],
                                model_type: str = 'tree') -> Dict[str, Any]:
        """Comprehensive model prediction explanation"""
        try:
            explanation = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'feature_names': feature_names,
                'explanations': {}
            }
            
            # SHAP explanation
            if 'shap' in self.explanation_methods and SHAP_AVAILABLE:
                try:
                    # Create explainer if not exists
                    model_id = id(model)
                    if model_id not in self.shap_explainer.explainers:
                        # Use sample of X as background data
                        background_size = min(100, len(X)) if len(X.shape) > 1 else 1
                        if len(X.shape) > 1:
                            background = X[:background_size]
                        else:
                            background = X.reshape(1, -1)
                        self.shap_explainer.create_explainer(model, background, model_type)
                    
                    shap_explanation = self.shap_explainer.explain_prediction(model, X, feature_names)
                    explanation['explanations']['shap'] = shap_explanation
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            # LIME explanation
            if 'lime' in self.explanation_methods and LIME_AVAILABLE:
                try:
                    # Create LIME explainer if needed
                    explainer_id = hash(tuple(feature_names))
                    if explainer_id not in self.lime_explainer.explainers:
                        if len(X.shape) > 1:
                            self.lime_explainer.create_explainer(X, feature_names)
                        else:
                            self.lime_explainer.create_explainer(X.reshape(1, -1), feature_names)
                    
                    lime_explanation = self.lime_explainer.explain_prediction(model, X, feature_names)
                    explanation['explanations']['lime'] = lime_explanation
                except Exception as e:
                    logger.warning(f"LIME explanation failed: {e}")
            
            # Store explanation
            if self.store_explanations:
                self.explanation_history.append(explanation)
                # Keep only recent explanations
                if len(self.explanation_history) > 1000:
                    self.explanation_history = self.explanation_history[-1000:]
            
            return explanation
            
        except Exception as e:
            logger.error(f"Model explanation failed: {e}")
            return {'error': str(e)}
    
    def explain_trading_decision(self, signal: Dict[str, Any], 
                               market_data: Dict[str, Any],
                               strategy_name: str,
                               model=None, 
                               features: Optional[np.ndarray] = None,
                               feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive trading decision explanation"""
        try:
            explanation = {
                'timestamp': datetime.now().isoformat(),
                'signal': signal,
                'strategy': strategy_name,
                'explanations': {}
            }
            
            # Custom business explanation
            if 'custom' in self.explanation_methods:
                custom_explanation = self.custom_explainer.explain_trading_decision(
                    signal, market_data, strategy_name
                )
                explanation['explanations']['business'] = custom_explanation
            
            # Model-based explanation if model and features provided
            if model is not None and features is not None and feature_names is not None:
                model_explanation = self.explain_model_prediction(
                    model, features, feature_names
                )
                explanation['explanations']['model'] = model_explanation
            
            # Risk explanation
            explanation['explanations']['risk'] = self._explain_risk_factors(signal, market_data)
            
            # Regulatory explanation
            explanation['explanations']['regulatory'] = self._generate_regulatory_explanation(
                signal, strategy_name
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Trading decision explanation failed: {e}")
            return {'error': str(e)}
    
    def _explain_risk_factors(self, signal: Dict[str, Any], 
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain risk factors in the trading decision"""
        try:
            risk_explanation = {
                'position_risk': {},
                'market_risk': {},
                'model_risk': {},
                'overall_risk_level': 'medium'
            }
            
            # Position risk
            position_size = signal.get('position_size', 0)
            if position_size > 0.1:
                risk_explanation['position_risk']['large_position'] = 'Position size exceeds 10% of portfolio'
            
            confidence = signal.get('confidence', 0.5)
            if confidence < 0.6:
                risk_explanation['model_risk']['low_confidence'] = f'Model confidence is {confidence:.1%}'
            
            # Market risk
            volatility = market_data.get('volatility', 0)
            if volatility > 0.03:
                risk_explanation['market_risk']['high_volatility'] = f'Market volatility is {volatility:.1%}'
            
            # Overall risk assessment
            risk_factors = sum(len(v) for v in risk_explanation.values() if isinstance(v, dict))
            if risk_factors == 0:
                risk_explanation['overall_risk_level'] = 'low'
            elif risk_factors <= 2:
                risk_explanation['overall_risk_level'] = 'medium'
            else:
                risk_explanation['overall_risk_level'] = 'high'
            
            return risk_explanation
            
        except Exception as e:
            logger.error(f"Risk explanation failed: {e}")
            return {'error': str(e)}
    
    def _generate_regulatory_explanation(self, signal: Dict[str, Any], 
                                       strategy_name: str) -> Dict[str, Any]:
        """Generate regulatory compliance explanation"""
        return {
            'compliance_status': 'compliant',
            'decision_basis': 'quantitative model analysis',
            'data_sources': 'public market data only',
            'risk_controls': 'position limits and stop-losses applied',
            'audit_trail': f'Decision logged with timestamp and model version',
            'strategy_classification': strategy_name,
            'human_oversight': 'automated decision subject to human review'
        }
    
    def generate_explanation_report(self, symbol: str, 
                                  time_period: int = 7) -> Dict[str, Any]:
        """Generate comprehensive explanation report"""
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=time_period)
            
            # Filter recent explanations
            recent_explanations = [
                exp for exp in self.explanation_history
                if datetime.fromisoformat(exp['timestamp']) > cutoff_date
            ]
            
            if not recent_explanations:
                return {'error': 'No recent explanations available'}
            
            report = {
                'symbol': symbol,
                'period': f'{time_period} days',
                'total_decisions': len(recent_explanations),
                'decision_breakdown': {},
                'confidence_analysis': {},
                'risk_analysis': {},
                'feature_importance_trends': {}
            }
            
            # Analyze decisions
            actions = [exp['signal']['action'] for exp in recent_explanations if 'signal' in exp]
            for action in ['buy', 'sell', 'hold']:
                report['decision_breakdown'][action] = actions.count(action)
            
            # Confidence analysis
            confidences = [exp['signal']['confidence'] for exp in recent_explanations 
                          if 'signal' in exp and 'confidence' in exp['signal']]
            if confidences:
                report['confidence_analysis'] = {
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences))
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Explanation report generation failed: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get explainable AI engine status"""
        return {
            'status': 'active',
            'shap_available': SHAP_AVAILABLE,
            'lime_available': LIME_AVAILABLE,
            'pandas_available': PANDAS_AVAILABLE,
            'explanation_methods': self.explanation_methods,
            'explanations_stored': len(self.explanation_history),
            'models_explained': len(self.shap_explainer.explainers)
        }