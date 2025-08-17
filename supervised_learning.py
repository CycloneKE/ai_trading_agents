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
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import os

from base_strategy import BaseStrategy
from src.adaptive_agent import SelfAdaptiveAgent
from src.goal_manager import GoalManager
from src.fmp_provider import FMPProvider
from src.bias_detector import BiasDetector

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
        self.threshold = config.get('threshold', 0.01)  # Lowered from 0.02 to 0.01
        self.model_type = config.get('model_type', 'random_forest')
        self.features = config.get('features', ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower'])
        
        # Stop-loss and risk management
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 2.0)
        self.trailing_stop_enabled = config.get('trailing_stop_enabled', True)
        
        # Trend following configuration
        self.trend_following_enabled = config.get('trend_following_enabled', True)
        self.ema_fast = config.get('ema_fast', 12)
        self.ema_slow = config.get('ema_slow', 26)
        self.adx_threshold = config.get('adx_threshold', 25)
        
        # Position tracking for stop-losses
        self.active_positions = {}  # {symbol: {'entry_price': float, 'stop_price': float, 'highest_price': float}}
        
        # Model objects
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        
        # Historical data
        self.historical_data = {}
        self.predictions = {}
        
        # Initialize model
        self._initialize_model()
        
        # Initialize adaptive components
        self.adaptive_agent = SelfAdaptiveAgent(config)
        self.goal_manager = GoalManager(config)
        self.original_threshold = self.threshold
        
        # Initialize FMP provider for fundamental data
        self.fmp_provider = FMPProvider()
        
        # Add fundamental features if not already present
        fundamental_features = ['pe_ratio', 'debt_ratio', 'current_ratio', 'market_cap']
        for feature in fundamental_features:
            if feature not in self.features:
                self.features.append(feature)
        
        # Add trend following features
        trend_features = ['ema_12', 'ema_26', 'adx', 'atr']
        for feature in trend_features:
            if feature not in self.features:
                self.features.append(feature)
        
        # Initialize bias detector
        self.bias_detector = BiasDetector(config)
        self.decision_history = []
    
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
            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    eval_metric='rmse'
                )
            else:
                if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
                    logger.warning("XGBoost not available, using random forest")
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
            model_choices = ['random_forest', 'gradient_boosting']
            if XGBOOST_AVAILABLE:
                model_choices.append('xgboost')
            
            parameter_space = {
                'model_type': {'type': 'categorical', 'choices': model_choices},
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
                elif params['model_type'] == 'xgboost' and XGBOOST_AVAILABLE:
                    model = xgb.XGBRegressor(
                        n_estimators=params.get('n_estimators', 100),
                        learning_rate=params.get('learning_rate', 0.1),
                        max_depth=params.get('max_depth', 6),
                        random_state=42,
                        eval_metric='rmse'
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
            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                    eval_metric='rmse'
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
            
            # Ensure model is trained before making predictions
            if self.model is not None:
                # Try to train if not already trained
                try:
                    # Test if model is trained by trying a prediction
                    test_features = np.array([features])
                    if self.scaler is None:
                        self.scaler = StandardScaler()
                        self.scaler.fit(test_features)
                    
                    if not hasattr(self.scaler, 'scale_'):
                        self.scaler.fit(test_features)
                    
                    features_scaled = self.scaler.transform(test_features)
                    
                    # Test if model is trained
                    if not hasattr(self.model, 'n_features_in_'):
                        # Train with simple data
                        y_simple = [0, 1] * (len(test_features) // 2 + 1)
                        y_simple = y_simple[:len(test_features)]
                        self.model.fit(features_scaled, y_simple)
                        logger.info(f"Model trained for {symbol}")
                    
                except Exception as train_error:
                    logger.warning(f"Model training failed: {train_error}")
                    return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}
                
                # Now make prediction
                features_scaled = self.scaler.transform([features])
                
                # Initialize default values
                prediction = 0.0
                action = 'hold'
                confidence = 0.0
                
                # Get prediction based on model type
                if self.model_type == 'random_forest' and hasattr(self.model, 'predict_proba'):
                    # Classification model - use predict_proba
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
                    prediction = buy_prob - sell_prob  # Set prediction for consistency
                elif self.model_type in ['gradient_boosting', 'xgboost']:
                    # Regression model - use predict only
                    prediction = self.model.predict(features_scaled)[0]
                    self.predictions[symbol] = prediction
                    
                    # Convert regression to action
                    if prediction > self.threshold:
                        action = 'buy'
                        confidence = min(abs(prediction), 1.0)
                    elif prediction < -self.threshold:
                        action = 'sell'
                        confidence = min(abs(prediction), 1.0)
                    else:
                        action = 'hold'
                        confidence = 1.0 - abs(prediction)
                else:
                    # Default case - use predict
                    prediction = self.model.predict(features_scaled)[0]
                    self.predictions[symbol] = prediction
                    
                    # Convert to action
                    if prediction > self.threshold:
                        action = 'buy'
                        confidence = min(abs(prediction), 1.0)
                    elif prediction < -self.threshold:
                        action = 'sell'
                        confidence = min(abs(prediction), 1.0)
                    else:
                        action = 'hold'
                        confidence = 1.0 - abs(prediction)
                
                # Check for stop-loss triggers first
                stop_loss_action = self._check_stop_loss(symbol, data.get('close', 0), data.get('atr', 0))
                if stop_loss_action:
                    return stop_loss_action
                
                # Apply trend following filter
                if self.trend_following_enabled and not self._trend_filter_passed(data):
                    return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}
                
                # Apply adaptive filtering
                should_trade = self._should_trade_adaptive(symbol, confidence)
                if not should_trade:
                    return {'action': 'hold', 'confidence': 0.0, 'position_size': 0.0}
                
                # Calculate adaptive position size
                position_size = self._get_adaptive_position_size(symbol, confidence, data.get('account_value', 100000))
                
                # Update position tracking for stop-losses
                if action == 'buy':
                    self._update_position_entry(symbol, data.get('close', 0), data.get('atr', 0))
                elif action == 'sell' and symbol in self.active_positions:
                    del self.active_positions[symbol]
                
                # Create decision record
                decision = {
                    'action': action,
                    'confidence': float(confidence),
                    'position_size': float(position_size),
                    'prediction': float(prediction),
                    'features': {f: float(v) for f, v in zip(self.features, features)},
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'market_cap': data.get('market_cap', 0),
                    'sector': data.get('sector', 'unknown'),
                    'stop_price': self.active_positions.get(symbol, {}).get('stop_price', 0)
                }
                
                # Apply bias detection and correction
                if len(self.decision_history) >= 10:  # Need minimum decisions for bias analysis
                    bias_report = self.bias_detector.generate_bias_report(self.decision_history[-30:])
                    decision = self.bias_detector.apply_bias_correction(decision, bias_report)
                
                # Track decision
                self.decision_history.append(decision.copy())
                if len(self.decision_history) > 100:  # Keep last 100 decisions
                    self.decision_history = self.decision_history[-100:]
                
                return decision
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
            
            # Update adaptive system
            if feedback:
                self._update_adaptive_system(symbol, data, feedback)
            
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
            if self.scaler is not None:
                X_scaled = self.scaler.fit_transform(np.array(X))
            else:
                X_scaled = np.array(X)
            
            # Train model
            if self.model is not None:
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
            
            # Extract basic features
            features = []
            
            # Price features
            if 'close' in self.features:
                features.append(latest.get('close', 0.0))
            
            if 'open' in self.features:
                features.append(latest.get('open', 0.0))
            
            if 'high' in self.features:
                features.append(latest.get('high', 0.0))
            
            if 'low' in self.features:
                features.append(latest.get('low', 0.0))
            
            if 'volume' in self.features:
                features.append(latest.get('volume', 0.0))
            
            # Calculate technical indicators
            df = pd.DataFrame([h for h in history[-self.lookback_period:]])
            
            # RSI
            if 'rsi' in self.features:
                if 'rsi' in latest:
                    features.append(latest['rsi'])
                else:
                    # Calculate RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = ((-delta).where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / (loss + 1e-10)  # Add small value to avoid division by zero
                    rsi_series = 100 - (100 / (1 + rs))
                    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
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
            
            # Trend Following Indicators
            if 'ema_12' in self.features:
                if 'ema_12' in latest:
                    features.append(latest['ema_12'])
                else:
                    ema_12 = df['close'].ewm(span=self.ema_fast).mean().iloc[-1]
                    features.append(ema_12)
            
            if 'ema_26' in self.features:
                if 'ema_26' in latest:
                    features.append(latest['ema_26'])
                else:
                    ema_26 = df['close'].ewm(span=self.ema_slow).mean().iloc[-1]
                    features.append(ema_26)
            
            if 'adx' in self.features:
                if 'adx' in latest:
                    features.append(latest['adx'])
                else:
                    adx = self._calculate_adx(df)
                    features.append(adx)
            
            if 'atr' in self.features:
                if 'atr' in latest:
                    features.append(latest['atr'])
                else:
                    atr = self._calculate_atr(df)
                    features.append(atr)
            
            # FMP Fundamental Data
            try:
                if any(f in self.features for f in ['pe_ratio', 'debt_ratio', 'current_ratio', 'market_cap']):
                    ratios = self.fmp_provider.get_financial_ratios(symbol)
                    quote = self.fmp_provider.get_quote(symbol)
                    
                    if 'pe_ratio' in self.features:
                        features.append(ratios.get('priceEarningsRatio', 0.0))
                    if 'debt_ratio' in self.features:
                        features.append(ratios.get('debtRatio', 0.0))
                    if 'current_ratio' in self.features:
                        features.append(ratios.get('currentRatio', 0.0))
                    if 'market_cap' in self.features:
                        features.append(quote.get('marketCap', 0.0))
            except Exception as e:
                logger.warning(f"FMP data unavailable for {symbol}: {str(e)}")
                # Add zeros for missing fundamental features
                for feature in ['pe_ratio', 'debt_ratio', 'current_ratio', 'market_cap']:
                    if feature in self.features:
                        features.append(0.0)
                        
            # Add zeros for missing trend features
            for feature in ['ema_12', 'ema_26', 'adx', 'atr']:
                if feature in self.features and len([f for f in self.features[:len(features)] if f == feature]) == 0:
                    features.append(0.0)
            
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
                return ([], [])
            
            # Get historical data
            history = self.historical_data[symbol]
            if len(history) < self.lookback_period:
                return ([], [])
            
            # Extract features and labels
            X = []
            y = []
            
            # Use feedback to create labels
            returns = feedback.get('returns', [])
            
            # Skip if no returns data
            if not returns:
                return ([], [])
            
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
            
            return (X, y)
            
        except Exception as e:
            logger.error(f"Error preparing training data for {symbol}: {str(e)}")
            return ([], [])
    
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
    
    def _should_trade_adaptive(self, symbol: str, signal_strength: float) -> bool:
        """Check if trade should be executed based on adaptive criteria"""
        priority_goals = self.goal_manager.get_priority_goals(1)
        
        if priority_goals:
            primary_goal = priority_goals[0]
            if primary_goal.goal_type.value == 'drawdown_control':
                threshold = 0.7  # More conservative
            elif primary_goal.goal_type.value == 'profit_target':
                threshold = 0.3  # More aggressive
            else:
                threshold = 0.5
        else:
            threshold = 0.5
        
        return abs(signal_strength) > threshold
    
    def _get_adaptive_position_size(self, symbol: str, signal_strength: float, account_value: float) -> float:
        """Calculate adaptive position size"""
        base_size = self.config.get('max_position_size', 0.1)
        risk_tolerance = self.adaptive_agent.config.get('risk_tolerance', 0.02)
        
        # Adjust based on market regime
        if self.adaptive_agent.market_regime == 'high_volatility':
            size_multiplier = 0.7
        elif self.adaptive_agent.market_regime == 'low_volatility':
            size_multiplier = 1.2
        else:
            size_multiplier = 1.0
        
        position_value = account_value * risk_tolerance * base_size * size_multiplier * abs(signal_strength)
        return min(position_value / account_value, 0.1)  # Cap at 10%
    
    def _check_stop_loss(self, symbol: str, current_price: float, atr: float) -> Optional[Dict[str, Any]]:
        """Check if stop-loss should be triggered"""
        if symbol not in self.active_positions or current_price <= 0:
            return None
        
        position = self.active_positions[symbol]
        
        # Update trailing stop if price moved favorably
        if self.trailing_stop_enabled and current_price > position['highest_price']:
            position['highest_price'] = current_price
            # Update stop price (trailing stop)
            new_stop = current_price - (atr * self.stop_loss_atr_multiplier)
            position['stop_price'] = max(position['stop_price'], new_stop)
        
        # Check if stop-loss triggered
        if current_price <= position['stop_price']:
            logger.info(f"Stop-loss triggered for {symbol} at {current_price:.2f} (stop: {position['stop_price']:.2f})")
            return {
                'action': 'sell',
                'confidence': 1.0,
                'position_size': 0.1,  # Sell all
                'stop_loss_triggered': True,
                'stop_price': position['stop_price']
            }
        
        return None
    
    def _update_position_entry(self, symbol: str, entry_price: float, atr: float):
        """Update position tracking for new entry"""
        if atr <= 0:
            atr = entry_price * 0.02  # Default 2% if ATR not available
        
        stop_price = entry_price - (atr * self.stop_loss_atr_multiplier)
        
        self.active_positions[symbol] = {
            'entry_price': entry_price,
            'stop_price': stop_price,
            'highest_price': entry_price
        }
        
        logger.info(f"Position opened for {symbol} at {entry_price:.2f}, stop at {stop_price:.2f}")
    
    def _trend_filter_passed(self, data: Dict[str, Any]) -> bool:
        """Check if trend following conditions are met"""
        try:
            # Get trend indicators
            ema_12 = data.get('ema_12', 0)
            ema_26 = data.get('ema_26', 0)
            adx = data.get('adx', 0)
            close = data.get('close', 0)
            
            # Trend strength filter (ADX > threshold)
            if adx < self.adx_threshold:
                return False
            
            # Trend direction filter (price above both EMAs for uptrend)
            if close > ema_12 > ema_26:
                return True  # Uptrend confirmed
            
            # Allow sells in downtrend
            if close < ema_12 < ema_26:
                return True  # Downtrend confirmed
            
            return False  # Sideways market, avoid trading
            
        except Exception as e:
            logger.warning(f"Trend filter error: {e}")
            return True  # Default to allow trading if error
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        try:
            if len(df) < period + 1:
                return 25.0  # Default neutral value
            
            # Simplified ADX calculation
            high = df['high'].values if 'high' in df.columns else df['close'].values
            low = df['low'].values if 'low' in df.columns else df['close'].values
            close = df['close'].values
            
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate Directional Movement
            dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            # Smooth the values
            atr = pd.Series(tr).rolling(period).mean().iloc[-1]
            di_plus = pd.Series(dm_plus).rolling(period).mean().iloc[-1] / atr * 100
            di_minus = pd.Series(dm_minus).rolling(period).mean().iloc[-1] / atr * 100
            
            # Calculate ADX
            dx = abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
            return float(dx)
            
        except Exception as e:
            logger.warning(f"ADX calculation error: {e}")
            return 25.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(df) < period:
                return df['close'].iloc[-1] * 0.02  # Default 2% of price
            
            high = df['high'].values if 'high' in df.columns else df['close'].values
            low = df['low'].values if 'low' in df.columns else df['close'].values
            close = df['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            atr = pd.Series(tr).rolling(period).mean().iloc[-1]
            return float(atr) if not pd.isna(atr) else df['close'].iloc[-1] * 0.02
            
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            return df['close'].iloc[-1] * 0.02 if len(df) > 0 else 1.0
    
    def _train_initial_model(self, symbol: str):
        """Train model with initial synthetic data to prevent not fitted errors"""
        try:
            if len(self.historical_data.get(symbol, [])) < 10:
                return False
            
            # Create simple training data from historical data
            history = self.historical_data[symbol]
            X = []
            y = []
            
            for i in range(len(history) - 1):
                features = self._extract_features_from_data(history[i])
                if features and len(features) == len(self.features):
                    # Simple label: 1 if next price > current price, 0 otherwise
                    current_price = history[i].get('close', 0)
                    next_price = history[i + 1].get('close', 0)
                    
                    if self.model_type == 'random_forest':
                        label = 1 if next_price > current_price else 0
                    else:
                        label = (next_price - current_price) / current_price if current_price > 0 else 0
                    
                    X.append(features)
                    y.append(label)
            
            if len(X) >= 5:  # Minimum samples needed
                X = np.array(X)
                y = np.array(y)
                
                # Fit scaler
                if self.scaler is None:
                    self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.model.fit(X_scaled, y)
                logger.info(f"Initial model trained for {symbol} with {len(X)} samples")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Initial model training failed for {symbol}: {e}")
            return False
    
    def _extract_features_from_data(self, data_point: Dict[str, Any]) -> Optional[List[float]]:
        """Extract features from a single data point"""
        try:
            features = []
            for feature in self.features:
                if feature in data_point:
                    features.append(float(data_point[feature]))
                else:
                    # Use default values for missing features
                    if feature in ['close', 'open', 'high', 'low']:
                        features.append(data_point.get('close', 100.0))
                    elif feature == 'volume':
                        features.append(1000000.0)
                    elif feature == 'rsi':
                        features.append(50.0)
                    else:
                        features.append(0.0)
            
            return features if len(features) == len(self.features) else None
            
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return None
    
    def _update_adaptive_system(self, symbol: str, data: Dict[str, Any], feedback: Dict[str, Any]):
        """Update adaptive agent and goal manager"""
        market_data = {
            'symbol': symbol,
            'volatility': data.get('volatility', 0.2),
            'returns': feedback.get('returns', []),
            'trend': feedback.get('trend', 0.0)
        }
        
        performance_metrics = {
            'total_return': feedback.get('total_return', 0.0),
            'sharpe_ratio': feedback.get('sharpe_ratio', 0.0),
            'max_drawdown': feedback.get('max_drawdown', 0.0),
            'daily_var': feedback.get('daily_var', 0.02),
            'volatility': market_data['volatility'],
            'win_rate': feedback.get('win_rate', 0.5)
        }
        
        self.adaptive_agent.update_performance(performance_metrics)
        self.goal_manager.update_goals(market_data, performance_metrics)
        
        # Adapt threshold based on goal progress
        priority_goals = self.goal_manager.get_priority_goals(1)
        if priority_goals and priority_goals[0].progress < 0.3:
            self.threshold = self.original_threshold * 0.8  # More aggressive
    
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
        
        # Add adaptive status
        adaptive_status = self.adaptive_agent.get_agent_state()
        goal_status = self.goal_manager.get_goal_status()
        
        status.update({
            'adaptive_enabled': True,
            'current_goals': len(goal_status['goal_details']),
            'goal_progress': goal_status['average_progress'],
            'market_regime': adaptive_status['market_regime'],
            'threshold_adapted': self.threshold != self.original_threshold,
            'stop_loss_enabled': True,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trend_following_enabled': self.trend_following_enabled,
            'active_positions': len(self.active_positions),
            'signal_threshold': self.threshold
        })
        
        # Add bias monitoring status
        if len(self.decision_history) >= 10:
            bias_report = self.bias_detector.generate_bias_report(self.decision_history[-30:])
            status.update({
                'bias_detected': bias_report.get('bias_detected', False),
                'bias_score': bias_report.get('overall_bias_score', 0.0),
                'decisions_analyzed': len(self.decision_history)
            })
        
        return status