#!/usr/bin/env python3
"""
Market Regime Detection System

Advanced system for detecting and adapting to different market regimes
using multiple statistical and machine learning approaches.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Advanced market regime detection using multiple approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_period = config.get('lookback_period', 252)  # 1 year
        self.n_regimes = config.get('n_regimes', 4)
        self.update_frequency = config.get('update_frequency', 5)  # days
        
        # Data storage
        self.price_history = deque(maxlen=self.lookback_period * 2)
        self.volume_history = deque(maxlen=self.lookback_period * 2)
        self.volatility_history = deque(maxlen=self.lookback_period)
        self.correlation_history = deque(maxlen=self.lookback_period)
        
        # Models
        self.regime_models = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.pca = PCA(n_components=5) if SKLEARN_AVAILABLE else None
        
        # Current state
        self.current_regime = 'unknown'
        self.regime_probabilities = {}
        self.regime_history = deque(maxlen=100)
        self.last_update = datetime.now() - timedelta(days=999)
        
        # Regime definitions
        self.regime_definitions = {
            'bull_market': {'volatility': 'low', 'trend': 'up', 'volume': 'normal'},
            'bear_market': {'volatility': 'high', 'trend': 'down', 'volume': 'high'},
            'sideways': {'volatility': 'low', 'trend': 'flat', 'volume': 'low'},
            'crisis': {'volatility': 'very_high', 'trend': 'down', 'volume': 'very_high'}
        }
        
        self._initialize_models()
        logger.info("Market Regime Detector initialized")
    
    def _initialize_models(self):
        """Initialize regime detection models"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available - using simplified regime detection")
            return
        
        try:
            # K-means clustering for regime identification
            self.regime_models['kmeans'] = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
            
            # Gaussian Mixture Model for probabilistic regimes
            self.regime_models['gmm'] = GaussianMixture(
                n_components=self.n_regimes,
                random_state=42,
                covariance_type='full'
            )
            
            logger.info("Regime detection models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing regime models: {e}")
    
    def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data for regime detection"""
        try:
            # Extract relevant data
            price = market_data.get('close', 0)
            volume = market_data.get('volume', 0)
            high = market_data.get('high', price)
            low = market_data.get('low', price)
            
            # Store data
            self.price_history.append(price)
            self.volume_history.append(volume)
            
            # Calculate derived metrics
            if len(self.price_history) >= 2:
                returns = np.diff(list(self.price_history))
                if len(returns) >= 20:
                    volatility = np.std(returns[-20:]) * np.sqrt(252)  # Annualized
                    self.volatility_history.append(volatility)
            
            # Update regime detection if needed
            if self._should_update_regime():
                self._detect_regime()
                self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _should_update_regime(self) -> bool:
        """Check if regime should be updated"""
        days_since_update = (datetime.now() - self.last_update).days
        return (days_since_update >= self.update_frequency and 
                len(self.price_history) >= 50)
    
    def _extract_regime_features(self) -> Optional[np.ndarray]:
        """Extract features for regime detection"""
        try:
            if len(self.price_history) < 50:
                return None
            
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history))
            
            features = []
            
            # Price-based features
            returns = np.diff(prices)
            if len(returns) >= 20:
                # Volatility measures
                features.append(np.std(returns[-20:]))  # Short-term volatility
                features.append(np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns))  # Long-term volatility
                
                # Trend measures
                features.append(np.mean(returns[-20:]))  # Short-term trend
                features.append(np.mean(returns[-60:]) if len(returns) >= 60 else np.mean(returns))  # Long-term trend
                
                # Momentum measures
                features.append(returns[-1])  # Latest return
                features.append(np.sum(returns[-5:]))  # 5-day momentum
                features.append(np.sum(returns[-20:]))  # 20-day momentum
                
                # Skewness and kurtosis
                if len(returns) >= 30:
                    from scipy import stats
                    features.append(stats.skew(returns[-30:]))
                    features.append(stats.kurtosis(returns[-30:]))
                else:
                    features.extend([0, 0])
            
            # Volume-based features
            if len(volumes) >= 20:
                volume_ma = np.mean(volumes[-20:])
                features.append(volumes[-1] / volume_ma if volume_ma > 0 else 1)  # Volume ratio
                features.append(np.std(volumes[-20:]) / volume_ma if volume_ma > 0 else 0)  # Volume volatility
            else:
                features.extend([1, 0])
            
            # Price level features
            if len(prices) >= 50:
                sma_20 = np.mean(prices[-20:])
                sma_50 = np.mean(prices[-50:])
                features.append((prices[-1] - sma_20) / sma_20)  # Distance from SMA20
                features.append((sma_20 - sma_50) / sma_50)  # SMA20 vs SMA50
                
                # Support/Resistance
                recent_high = np.max(prices[-20:])
                recent_low = np.min(prices[-20:])
                features.append((prices[-1] - recent_low) / (recent_high - recent_low + 1e-8))  # Position in range
            else:
                features.extend([0, 0, 0.5])
            
            # Market structure features
            if len(returns) >= 10:
                # Autocorrelation
                features.append(np.corrcoef(returns[-10:-1], returns[-9:])[0, 1] if len(returns) >= 10 else 0)
                
                # Drawdown
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                features.append(np.min(drawdown[-20:]) if len(drawdown) >= 20 else 0)
            else:
                features.extend([0, 0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
            return None
    
    def _detect_regime(self):
        """Detect current market regime"""
        try:
            features = self._extract_regime_features()
            if features is None:
                return
            
            # Ensure we have enough data
            if len(self.price_history) < 100:
                self.current_regime = 'insufficient_data'
                return
            
            # Rule-based regime detection (always available)
            rule_based_regime = self._rule_based_regime_detection(features)
            
            if SKLEARN_AVAILABLE and len(self.price_history) >= 200:
                # ML-based regime detection
                ml_regime = self._ml_based_regime_detection(features)
                
                # Combine rule-based and ML-based results
                self.current_regime = self._combine_regime_predictions(rule_based_regime, ml_regime)
            else:
                self.current_regime = rule_based_regime
            
            # Store regime history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': self.current_regime,
                'probabilities': self.regime_probabilities.copy(),
                'features': features.tolist()
            })
            
            logger.info(f"Market regime detected: {self.current_regime}")
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            self.current_regime = 'error'
    
    def _rule_based_regime_detection(self, features: np.ndarray) -> str:
        """Rule-based regime detection using feature thresholds"""
        try:
            if len(features) < 10:
                return 'unknown'
            
            volatility = features[0]  # Short-term volatility
            trend = features[2]  # Short-term trend
            volume_ratio = features[9] if len(features) > 9 else 1.0
            
            # Define thresholds (these would be calibrated based on historical data)
            high_vol_threshold = 0.02  # 2% daily volatility
            very_high_vol_threshold = 0.04  # 4% daily volatility
            trend_threshold = 0.001  # 0.1% daily trend
            high_volume_threshold = 1.5
            
            # Crisis regime (highest priority)
            if volatility > very_high_vol_threshold and trend < -trend_threshold:
                return 'crisis'
            
            # Bear market
            elif volatility > high_vol_threshold and trend < -trend_threshold:
                return 'bear_market'
            
            # Bull market
            elif volatility < high_vol_threshold and trend > trend_threshold:
                return 'bull_market'
            
            # Sideways market
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Rule-based regime detection error: {e}")
            return 'unknown'
    
    def _ml_based_regime_detection(self, features: np.ndarray) -> str:
        """ML-based regime detection using clustering"""
        try:
            if not SKLEARN_AVAILABLE:
                return 'unknown'
            
            # Prepare historical features for training
            historical_features = self._get_historical_features()
            if historical_features is None or len(historical_features) < 50:
                return 'unknown'
            
            # Scale features
            scaled_features = self.scaler.fit_transform(historical_features)
            current_scaled = self.scaler.transform([features])
            
            # Fit clustering models
            kmeans_labels = self.regime_models['kmeans'].fit_predict(scaled_features)
            current_kmeans = self.regime_models['kmeans'].predict(current_scaled)[0]
            
            # Fit GMM for probabilities
            self.regime_models['gmm'].fit(scaled_features)
            regime_probs = self.regime_models['gmm'].predict_proba(current_scaled)[0]
            current_gmm = np.argmax(regime_probs)
            
            # Store probabilities
            self.regime_probabilities = {
                f'regime_{i}': float(prob) for i, prob in enumerate(regime_probs)
            }
            
            # Map cluster to regime name
            regime_mapping = self._map_clusters_to_regimes(kmeans_labels, historical_features)
            ml_regime = regime_mapping.get(current_kmeans, 'unknown')
            
            return ml_regime
            
        except Exception as e:
            logger.error(f"ML-based regime detection error: {e}")
            return 'unknown'
    
    def _get_historical_features(self) -> Optional[np.ndarray]:
        """Get historical features for model training"""
        try:
            if len(self.price_history) < 200:
                return None
            
            features_list = []
            prices = list(self.price_history)
            
            # Generate features for different time windows
            for i in range(50, len(prices) - 20, 5):  # Every 5 days
                window_prices = prices[i-50:i+20]
                window_volumes = list(self.volume_history)[i-50:i+20] if len(self.volume_history) > i+20 else [1] * 70
                
                # Create temporary market data for feature extraction
                temp_detector = MarketRegimeDetector({'lookback_period': 100})
                temp_detector.price_history = deque(window_prices, maxlen=100)
                temp_detector.volume_history = deque(window_volumes, maxlen=100)
                
                window_features = temp_detector._extract_regime_features()
                if window_features is not None:
                    features_list.append(window_features)
            
            return np.array(features_list) if features_list else None
            
        except Exception as e:
            logger.error(f"Error getting historical features: {e}")
            return None
    
    def _map_clusters_to_regimes(self, labels: np.ndarray, features: np.ndarray) -> Dict[int, str]:
        """Map cluster labels to regime names based on cluster characteristics"""
        try:
            mapping = {}
            
            for cluster_id in range(self.n_regimes):
                cluster_mask = labels == cluster_id
                if not np.any(cluster_mask):
                    continue
                
                cluster_features = features[cluster_mask]
                
                # Analyze cluster characteristics
                avg_volatility = np.mean(cluster_features[:, 0])  # Short-term volatility
                avg_trend = np.mean(cluster_features[:, 2])  # Short-term trend
                
                # Map based on characteristics
                if avg_volatility > 0.03 and avg_trend < -0.001:
                    mapping[cluster_id] = 'crisis'
                elif avg_volatility > 0.02 and avg_trend < -0.0005:
                    mapping[cluster_id] = 'bear_market'
                elif avg_volatility < 0.015 and avg_trend > 0.0005:
                    mapping[cluster_id] = 'bull_market'
                else:
                    mapping[cluster_id] = 'sideways'
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error mapping clusters to regimes: {e}")
            return {}
    
    def _combine_regime_predictions(self, rule_based: str, ml_based: str) -> str:
        """Combine rule-based and ML-based regime predictions"""
        try:
            # If both agree, use the prediction
            if rule_based == ml_based:
                return rule_based
            
            # Priority rules for disagreement
            if rule_based == 'crisis' or ml_based == 'crisis':
                return 'crisis'  # Crisis has highest priority
            
            if rule_based in ['bear_market', 'bull_market'] and ml_based in ['bear_market', 'bull_market']:
                # Both detect directional markets, use rule-based (more conservative)
                return rule_based
            
            # Default to rule-based for other cases
            return rule_based
            
        except Exception as e:
            logger.error(f"Error combining regime predictions: {e}")
            return rule_based
    
    def get_current_regime(self) -> Dict[str, Any]:
        """Get current market regime information"""
        try:
            regime_info = {
                'regime': self.current_regime,
                'confidence': self._calculate_regime_confidence(),
                'probabilities': self.regime_probabilities.copy(),
                'last_update': self.last_update.isoformat(),
                'regime_definition': self.regime_definitions.get(self.current_regime, {}),
                'data_points': len(self.price_history)
            }
            
            # Add regime-specific recommendations
            regime_info['recommendations'] = self._get_regime_recommendations()
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error getting current regime: {e}")
            return {'regime': 'error', 'error': str(e)}
    
    def _calculate_regime_confidence(self) -> float:
        """Calculate confidence in current regime prediction"""
        try:
            if not self.regime_probabilities:
                return 0.5
            
            # Use entropy-based confidence
            probs = list(self.regime_probabilities.values())
            if len(probs) == 0:
                return 0.5
            
            # Normalize probabilities
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
            
            # Calculate entropy
            entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
            max_entropy = np.log(len(probs))
            
            # Convert to confidence (1 - normalized_entropy)
            confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 0.5
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def _get_regime_recommendations(self) -> Dict[str, Any]:
        """Get trading recommendations based on current regime"""
        recommendations = {
            'bull_market': {
                'strategy_preference': ['momentum', 'growth'],
                'risk_level': 'moderate',
                'position_sizing': 'normal',
                'rebalancing_frequency': 'low'
            },
            'bear_market': {
                'strategy_preference': ['defensive', 'short_selling'],
                'risk_level': 'low',
                'position_sizing': 'reduced',
                'rebalancing_frequency': 'high'
            },
            'sideways': {
                'strategy_preference': ['mean_reversion', 'range_trading'],
                'risk_level': 'moderate',
                'position_sizing': 'normal',
                'rebalancing_frequency': 'moderate'
            },
            'crisis': {
                'strategy_preference': ['defensive', 'cash'],
                'risk_level': 'very_low',
                'position_sizing': 'minimal',
                'rebalancing_frequency': 'very_high'
            }
        }
        
        return recommendations.get(self.current_regime, {
            'strategy_preference': ['balanced'],
            'risk_level': 'moderate',
            'position_sizing': 'normal',
            'rebalancing_frequency': 'moderate'
        })
    
    def get_regime_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get regime history for the specified number of days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            filtered_history = [
                entry for entry in self.regime_history
                if entry['timestamp'] > cutoff_date
            ]
            
            return filtered_history
            
        except Exception as e:
            logger.error(f"Error getting regime history: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get regime detector status"""
        return {
            'status': 'active',
            'current_regime': self.current_regime,
            'sklearn_available': SKLEARN_AVAILABLE,
            'data_points': len(self.price_history),
            'last_update': self.last_update.isoformat(),
            'regimes_detected': len(set(entry['regime'] for entry in self.regime_history))
        }