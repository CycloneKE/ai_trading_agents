"""
Supervised Learning Strategy implementation.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

import os

from base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SupervisedLearningStrategy(BaseStrategy):
    """
    Strategy using supervised learning models for prediction.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the supervised learning strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Strategy specific configuration
        self.lookback_period = config.get('lookback_period', 20)
        self.threshold = config.get('threshold', 0.02)
        self.model_type = config.get('model_type', 'random_forest')
        self.features = config.get('features', ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower'])
        
        # Model objects
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        
        # Historical data
        self.historical_data = {}
        self.predictions = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model."""
        # Initial model setup
        try:
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            else:
                logger.warning(f"Unknown model type: {self.model_type}, using random forest")
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            self.scaler = StandardScaler()
            logger.info(f"Initialized {self.model_type} model for {self.name}")
        except Exception as e:
            logger.error(f"Error initializing model for {self.name}: {str(e)}")

        # AutoML optimization (run after initial setup)
        try:
            from automl_optimizer import AutoMLOptimizer
            parameter_space = {
                'model_type': {'type': 'categorical', 'choices': ['random_forest', 'gradient_boosting']},
                'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2}
            }
            dummy_data = pd.DataFrame({
                'close': np.random.rand(100),
                'volume': np.random.rand(100),
                'rsi': np.random.rand(100),
                'macd': np.random.rand(100),
                'bb_upper': np.random.rand(100),
                'bb_lower': np.random.rand(100),
                'volatility': np.random.rand(100),
                'sentiment_score': np.random.rand(100)
            })
            def strategy_func(params, market_data):
                if params['model_type'] == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=params.get('n_estimators', 100),
                        max_depth=params.get('max_depth', 10),
                        random_state=42
                    )
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=params.get('n_estimators', 100),
                        learning_rate=params.get('learning_rate', 0.1),
                        max_depth=params.get('max_depth', 5),
                        random_state=42
                    )
                X = market_data.dropna().values
                y = np.random.randint(0, 2, size=X.shape[0])
                model.fit(X, y)
                preds = model.predict(X)
                returns = np.diff(preds)
                if len(returns) > 1:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                else:
                    sharpe = 0.0
                return sharpe
            automl = AutoMLOptimizer(self.config)
            best_params = automl.optimize_strategy_parameters(
                strategy_func=strategy_func,
                parameter_space=parameter_space,
                market_data=dummy_data,
                evaluation_metric='sharpe_ratio'
            )
            logger.info(f"AutoML best parameters: {best_params}")
            params = best_params.get('best_params', {})
            self.model_type = params.get('model_type', self.model_type)
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', 10)
            learning_rate = params.get('learning_rate', 0.1)
            # Re-initialize model with optimized parameters
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
        except Exception as e:
            logger.warning(f"AutoML optimization failed: {str(e)}")
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals from market data.
        
        Args:
            data: Market data and features
            
        Returns:
            Dict containing trading signals
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Update historical data
            self._update_historical_data(symbol, data)
            
            # Check if we have enough data
            if len(self.historical_data.get(symbol, [])) < self.lookback_period:
                logger.warning(f"Not enough historical data for {symbol} ({len(self.historical_data.get(symbol, []))} < {self.lookback_period})")
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0
                }
            
            # Extract features
            features = self._extract_features(symbol)
            
            if features is None or len(features) == 0:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0
                }
            
            # Make prediction if model is trained
            if self.model is not None and hasattr(self.model, 'predict'):
                # Scale features
                if self.scaler is not None:
                    features_scaled = self.scaler.transform([features])
                else:
                    features_scaled = np.array([features])
                
                # Get prediction
                if self.model_type == 'random_forest' and hasattr(self.model, 'predict_proba'):
                    # Classification model
                    proba = self.model.predict_proba(features_scaled)[0]
                    if len(proba) >= 3:
                        # Multi-class: sell, hold, buy
                        sell_prob, hold_prob, buy_prob = proba
                    else:
                        # Binary: sell, buy
                        sell_prob, buy_prob = proba
                        hold_prob = 0.0
                    
                    # Determine action
                    if buy_prob > sell_prob and buy_prob > self.threshold:
                        action = 'buy'
                        confidence = buy_prob
                    elif sell_prob > buy_prob and sell_prob > self.threshold:
                        action = 'sell'
                        confidence = sell_prob
                    else:
                        action = 'hold'
                        confidence = hold_prob
                else:
                    # Regression model
                    prediction = self.model.predict(features_scaled)[0]
                    self.predictions[symbol] = prediction
                    
                    # Convert regression to action
                    if prediction > self.threshold:
                        action = 'buy'
                        confidence = min(prediction, 1.0)
                    elif prediction < -self.threshold:
                        action = 'sell'
                        confidence = min(abs(prediction), 1.0)
                    else:
                        action = 'hold'
                        confidence = 1.0 - abs(prediction)
                
                # Calculate position size based on confidence
                position_size = confidence * self.config.get('max_position_size', 0.1)
                
                return {
                    'action': action,
                    'confidence': float(confidence),
                    'position_size': float(position_size),
                    'prediction': float(prediction) if 'prediction' in locals() else 0.0,
                    'features': {f: float(v) for f, v in zip(self.features, features)}
                }
            else:
                logger.warning(f"Model not trained for {self.name}")
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error generating signals for {self.name}: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0,
                'error': str(e)
            }
    
    def update_model(self, data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the strategy model with new data.
        
        Args:
            data: New market data
            feedback: Optional performance feedback
            
        Returns:
            bool: True if update successful
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Update historical data
            self._update_historical_data(symbol, data)
            
            # Check if we have enough data to train
            if len(self.historical_data.get(symbol, [])) < self.lookback_period * 5:
                logger.debug(f"Not enough data to train model for {symbol}")
                return False
            
            # Check if we have feedback
            if feedback is None:
                return False
            
            # Extract training data
            X, y = self._prepare_training_data(symbol, feedback)
            
            if X is None or y is None or len(X) == 0 or len(y) == 0:
                logger.warning(f"No training data available for {symbol}")
                return False


            win_rate = feedback.get('win_rate', None)
            if win_rate is not None:
                old_threshold = self.threshold
                # If win rate drops, make strategy more conservative
                if win_rate < 0.5:
                    self.threshold = min(self.threshold * 1.1, 0.1)
                elif win_rate > 0.7:
                    self.threshold = max(self.threshold * 0.9, 0.01)
                if self.threshold != old_threshold:
                    logger.info(f"Adapted threshold for {self.name} from {old_threshold:.4f} to {self.threshold:.4f} based on win rate {win_rate:.2f}")


            sharpe_ratio = feedback.get('sharpe_ratio', None)
            if sharpe_ratio is not None and sharpe_ratio < 0.5:
                logger.info(f"Sharpe ratio low ({sharpe_ratio:.2f}), consider retraining or tuning model for {self.name}")
                # Optionally, trigger AutoML optimization or retraining here
            
            # Scale features
            X_scaled = self.scaler.fit_transform(np.array(X))
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Update feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = {
                    feature: float(importance)
                    for feature, importance in zip(self.features, self.model.feature_importances_)
                }
            
            self.last_update = datetime.utcnow()
            logger.info(f"Updated model for {self.name} with {len(X)} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating model for {self.name}: {str(e)}")
            return False
    
    def _update_historical_data(self, symbol: str, data: Dict[str, Any]):
        """
        Update historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: New market data
        """
        try:
            # Initialize if needed
            if symbol not in self.historical_data:
                self.historical_data[symbol] = []
            
            # Add new data point
            self.historical_data[symbol].append(data)
            
            # Keep only lookback_period * 10 data points
            max_history = self.lookback_period * 10
            if len(self.historical_data[symbol]) > max_history:
                self.historical_data[symbol] = self.historical_data[symbol][-max_history:]
                
        except Exception as e:
            logger.error(f"Error updating historical data for {symbol}: {str(e)}")
    
    def _extract_features(self, symbol: str) -> Optional[List[float]]:
        """
        Extract features for prediction.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of feature values or None if error
        """
        try:
            if symbol not in self.historical_data:
                return None
            
            # Get historical data
            history = self.historical_data[symbol]
            if len(history) < self.lookback_period:
                return None
            
            # Get latest data point
            latest = history[-1]
            
            # Check if we have required price data
            if 'close' not in latest and 'price' not in latest:
                logger.error(f"No price data available for {symbol}")
                return None
            
            # Use 'close' or fallback to 'price'
            close_price = latest.get('close', latest.get('price', 0.0))
            
            # Extract basic features
            features = []
            
            # Price features
            if 'close' in self.features:
                features.append(close_price)
            
            if 'open' in self.features:
                features.append(latest.get('open', 0.0))
            
            if 'high' in self.features:
                features.append(latest.get('high', 0.0))
            
            if 'low' in self.features:
                features.append(latest.get('low', 0.0))
            
            if 'volume' in self.features:
                features.append(latest.get('volume', 0.0))
            
            # Calculate technical indicators
            # Ensure all data points have 'close' field
            processed_history = []
            for h in history[-self.lookback_period:]:
                processed_h = h.copy()
                if 'close' not in processed_h and 'price' in processed_h:
                    processed_h['close'] = processed_h['price']
                elif 'close' not in processed_h:
                    processed_h['close'] = 100.0  # Default fallback
                processed_history.append(processed_h)
            
            df = pd.DataFrame(processed_history)
            
            # RSI
            if 'rsi' in self.features:
                if 'rsi' in latest:
                    features.append(latest['rsi'])
                else:
                    # Calculate RSI
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs)).iloc[-1]
                    features.append(rsi)
            
            # MACD
            if 'macd' in self.features:
                if 'macd' in latest:
                    features.append(latest['macd'])
                else:
                    # Calculate MACD
                    ema12 = df['close'].ewm(span=12).mean()
                    ema26 = df['close'].ewm(span=26).mean()
                    macd = (ema12 - ema26).iloc[-1]
                    features.append(macd)
            
            # Bollinger Bands
            if 'bb_upper' in self.features or 'bb_lower' in self.features:
                if 'bb_upper' in latest and 'bb_lower' in latest:
                    if 'bb_upper' in self.features:
                        features.append(latest['bb_upper'])
                    if 'bb_lower' in self.features:
                        features.append(latest['bb_lower'])
                else:
                    # Calculate Bollinger Bands
                    sma = df['close'].rolling(window=20).mean()
                    std = df['close'].rolling(window=20).std()
                    bb_upper = (sma + (std * 2)).iloc[-1]
                    bb_lower = (sma - (std * 2)).iloc[-1]
                    
                    if 'bb_upper' in self.features:
                        features.append(bb_upper)
                    if 'bb_lower' in self.features:
                        features.append(bb_lower)
            
            # Moving Averages
            if 'sma_50' in self.features:
                if 'sma_50' in latest:
                    features.append(latest['sma_50'])
                else:
                    sma_50 = df['close'].rolling(window=min(50, len(df))).mean().iloc[-1]
                    features.append(sma_50)
            
            if 'sma_200' in self.features:
                if 'sma_200' in latest:
                    features.append(latest['sma_200'])
                else:
                    sma_200 = df['close'].rolling(window=min(200, len(df))).mean().iloc[-1]
                    features.append(sma_200)
            
            # Volatility (standard deviation of returns)
            if 'volatility' in self.features:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() if not returns.empty else 0.0
                features.append(volatility)
            
            # Sentiment Score (from latest data)
            if 'sentiment_score' in self.features:
                features.append(latest.get('sentiment_score', 0.0))
            
            # Ensure all features are valid numbers
            features = [float(f) if f is not None else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {str(e)}")
            return None
    
    def _prepare_training_data(self, symbol: str, feedback: Dict[str, Any]) -> Tuple[List[List[float]], List[float]]:
        """
        Prepare training data from historical data and feedback.
        
        Args:
            symbol: Trading symbol
            feedback: Performance feedback
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            if symbol not in self.historical_data:
                return None, None
            
            # Get historical data
            history = self.historical_data[symbol]
            if len(history) < self.lookback_period:
                return None, None
            
            # Extract features and labels
            X = []
            y = []
            
            # Use feedback to create labels
            returns = feedback.get('returns', [])
            
            # Skip if no returns data
            if not returns:
                return None, None
            
            # Create training data
            for i in range(self.lookback_period, len(history) - 1):
                # Extract features from historical data
                data_point = history[i]
                features = []
                
                # Extract basic features
                for feature in self.features:
                    if feature in data_point:
                        features.append(data_point[feature])
                    else:
                        # Skip this data point if missing features
                        features = None
                        break
                
                if features is None:
                    continue
                
                # Get corresponding return
                if i - self.lookback_period < len(returns):
                    future_return = returns[i - self.lookback_period]
                    
                    # Create label based on return
                    if self.model_type == 'random_forest':
                        # Classification: -1 (sell), 0 (hold), 1 (buy)
                        if future_return > self.threshold:
                            label = 1  # buy
                        elif future_return < -self.threshold:
                            label = -1  # sell
                        else:
                            label = 0  # hold
                    else:
                        # Regression: actual return
                        label = future_return
                    
                    X.append(features)
                    y.append(label)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data for {symbol}: {str(e)}")
            return None, None
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the strategy model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: True if save successful
        """
        try:
            if self.model is None:
                logger.warning(f"No model to save for {self.name}")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'config': self.config,
                'features': self.features,
                'last_update': self.last_update
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved model for {self.name} to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {self.name}: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load the strategy model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            bool: True if load successful
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
            
            # Load model data
            model_data = joblib.load(filepath)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_importance = model_data.get('feature_importance', {})
            self.features = model_data.get('features', self.features)
            self.last_update = model_data.get('last_update')
            
            logger.info(f"Loaded model for {self.name} from {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {self.name}: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get strategy status.
        
        Returns:
            Dict containing status information
        """
        status = super().get_status()
        
        # Add strategy specific information
        status.update({
            'model_type': self.model_type,
            'lookback_period': self.lookback_period,
            'threshold': self.threshold,
            'features': self.features,
            'feature_importance': self.feature_importance,
            'model_trained': self.model is not None and hasattr(self.model, 'predict'),
            'historical_data_count': {
                symbol: len(data) for symbol, data in self.historical_data.items()
            }
        })
        
        return status