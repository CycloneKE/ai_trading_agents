"""
Reinforcement Learning Strategy implementation.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import os
import joblib
import random
from collections import deque

from base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class DQNStrategy(BaseStrategy):
    """
    Strategy using Deep Q-Network (DQN) reinforcement learning.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the DQN strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Strategy specific configuration
        self.lookback_period = config.get('lookback_period', 20)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.discount_factor = config.get('discount_factor', 0.95)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.target_update_frequency = config.get('target_update_frequency', 10)
        self.features = config.get('features', ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower'])
        
        # Model objects
        self.model = None
        self.target_model = None
        self.memory = deque(maxlen=self.memory_size)
        self.update_counter = 0
        
        # Historical data
        self.historical_data = {}
        self.last_state = {}
        self.last_action = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the DQN model."""
        try:
            # Import here to avoid dependency if not using TensorFlow
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout
                from tensorflow.keras.optimizers import Adam
                
                # Set up the model
                input_dim = len(self.features)
                
                # Main model for training
                self.model = Sequential([
                    Dense(64, input_dim=input_dim, activation='relu'),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(3, activation='linear')  # 3 actions: buy, hold, sell
                ])
                
                self.model.compile(
                    loss='mse',
                    optimizer=Adam(learning_rate=self.learning_rate)
                )
                
                # Target model for stable Q-values
                self.target_model = Sequential([
                    Dense(64, input_dim=input_dim, activation='relu'),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(3, activation='linear')
                ])
                
                # Copy weights from main model to target model
                self.target_model.set_weights(self.model.get_weights())
                
                logger.info(f"Initialized DQN model for {self.name}")
                
            except ImportError:
                logger.error("TensorFlow not installed. Cannot initialize DQN model.")
                self.model = None
                self.target_model = None
            
        except Exception as e:
            logger.error(f"Error initializing DQN model for {self.name}: {str(e)}")
            self.model = None
            self.target_model = None
    
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
            state = self._extract_features(symbol)
            
            if state is None or len(state) == 0:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0
                }
            
            # Store current state
            self.last_state[symbol] = state
            
            # Make prediction if model is trained
            if self.model is not None:
                # Exploration vs exploitation
                if random.random() < self.exploration_rate:
                    # Exploration: random action
                    action_idx = random.randint(0, 2)  # 0: sell, 1: hold, 2: buy
                else:
                    # Exploitation: best action according to model
                    state_tensor = np.array([state])
                    q_values = self.model.predict(state_tensor, verbose=0)[0]
                    action_idx = np.argmax(q_values)
                
                # Convert action index to action
                if action_idx == 0:
                    action = 'sell'
                elif action_idx == 2:
                    action = 'buy'
                else:
                    action = 'hold'
                
                # Store last action
                self.last_action[symbol] = action_idx
                
                # Calculate confidence and position size
                if self.model is not None:
                    q_values = self.model.predict(np.array([state]), verbose=0)[0]
                    confidence = (q_values[action_idx] - min(q_values)) / (max(q_values) - min(q_values) + 1e-10)
                    confidence = max(0.0, min(1.0, confidence))
                else:
                    confidence = 0.5
                    q_values = [0.0, 0.0, 0.0]
                
                position_size = confidence * self.config.get('max_position_size', 0.1)
                
                return {
                    'action': action,
                    'confidence': float(confidence),
                    'position_size': float(position_size),
                    'q_values': {
                        'sell': float(q_values[0]),
                        'hold': float(q_values[1]),
                        'buy': float(q_values[2])
                    } if self.model is not None else {}
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
        Update the strategy model with new data and feedback.
        
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
            
            # Check if we have enough data
            if len(self.historical_data.get(symbol, [])) < self.lookback_period:
                return False
            
            # Check if we have feedback and last state/action
            if (feedback is None or 
                symbol not in self.last_state or 
                symbol not in self.last_action):
                return False
            
            # Extract current state
            current_state = self._extract_features(symbol)
            
            if current_state is None:
                return False
            
            # Get reward from feedback
            reward = self._calculate_reward(feedback, symbol)
            
            # Get last state and action
            last_state = self.last_state[symbol]
            last_action = self.last_action[symbol]
            
            # Store experience in memory
            self.memory.append((last_state, last_action, reward, current_state))
            
            # Train model if we have enough experiences
            if len(self.memory) >= self.batch_size and self.model is not None:
                self._train_model()
                
                # Update target model periodically
                self.update_counter += 1
                if self.update_counter % self.target_update_frequency == 0 and self.target_model is not None:
                    self.target_model.set_weights(self.model.get_weights())
                    logger.debug(f"Updated target model for {self.name}")
            
            self.last_update = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(f"Error updating model for {self.name}: {str(e)}")
            return False
    
    def _train_model(self):
        """Train the DQN model using experience replay."""
        try:
            if self.model is None or len(self.memory) < self.batch_size:
                return
            
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            
            states = []
            targets = []
            
            for state, action, reward, next_state in batch:
                # Current Q-values
                target = self.model.predict(np.array([state]), verbose=0)[0]
                
                # Next Q-values from target model (for stability)
                if self.target_model is not None:
                    next_q_values = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                else:
                    next_q_values = [0.0, 0.0, 0.0]
                
                # Update Q-value for the action taken
                if reward is not None:
                    target[action] = reward + self.discount_factor * np.max(next_q_values)
                
                states.append(state)
                targets.append(target)
            
            # Train model
            self.model.fit(
                np.array(states),
                np.array(targets),
                batch_size=self.batch_size,
                epochs=1,
                verbose=0
            )
            
        except Exception as e:
            logger.error(f"Error training DQN model: {str(e)}")
    
    def _calculate_reward(self, feedback: Dict[str, Any], symbol: str) -> float:
        """
        Calculate reward from feedback.
        
        Args:
            feedback: Performance feedback
            symbol: Trading symbol
            
        Returns:
            Reward value
        """
        try:
            # Get action
            action = self.last_action.get(symbol)
            
            if action is None:
                return 0.0
            
            # Get return
            returns = feedback.get('returns', {}).get(symbol, 0.0)
            
            # Calculate reward based on action and return
            if action == 0:  # sell
                reward = -returns  # Negative return is good for sell
            elif action == 2:  # buy
                reward = returns  # Positive return is good for buy
            else:  # hold
                reward = abs(returns) * 0.1  # Small reward for stability
            
            # Scale reward
            reward = np.clip(reward * 10, -1.0, 1.0)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
    
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
        Extract features for the DQN model.
        
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
                    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
                    loss = (-delta).where(delta < 0, 0.0).rolling(window=14).mean()
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
            
            # Volatility (standard deviation of returns)
            if 'volatility' in self.features:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() if not returns.empty else 0.0
                features.append(volatility)
            
            # Sentiment Score (from latest data)
            if 'sentiment_score' in self.features:
                features.append(latest.get('sentiment_score', 0.0))
            
            if 'sma_200' in self.features:
                if 'sma_200' in latest:
                    features.append(latest['sma_200'])
                else:
                    sma_200 = df['close'].rolling(window=min(200, len(df))).mean().iloc[-1]
                    features.append(sma_200)
            
            # Ensure all features are valid numbers
            features = [float(f) if f is not None else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {str(e)}")
            return None
    
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
            
            # Save model weights
            model_path = filepath + ".h5"
            target_model_path = filepath + "_target.h5"
            
            if self.model is not None:
                self.model.save_weights(model_path)
            if self.target_model is not None:
                self.target_model.save_weights(target_model_path)
            
            # Save other data
            data_path = filepath + "_data.pkl"
            model_data = {
                'config': self.config,
                'features': self.features,
                'last_update': self.last_update,
                'exploration_rate': self.exploration_rate,
                'update_counter': self.update_counter
            }
            
            joblib.dump(model_data, data_path)
            
            logger.info(f"Saved DQN model for {self.name} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving DQN model for {self.name}: {str(e)}")
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
            model_path = filepath + ".h5"
            target_model_path = filepath + "_target.h5"
            data_path = filepath + "_data.pkl"
            
            if not os.path.exists(model_path) or not os.path.exists(data_path):
                logger.warning(f"Model files not found: {filepath}")
                return False
            
            # Initialize model if needed
            if self.model is None:
                self._initialize_model()
                
            if self.model is None:
                logger.error("Failed to initialize model")
                return False
            
            # Load model weights
            if self.model is not None:
                self.model.load_weights(model_path)
            
            if os.path.exists(target_model_path) and self.target_model is not None:
                self.target_model.load_weights(target_model_path)
            elif self.target_model is not None and self.model is not None:
                self.target_model.set_weights(self.model.get_weights())
            
            # Load other data
            model_data = joblib.load(data_path)
            
            self.features = model_data.get('features', self.features)
            self.last_update = model_data.get('last_update')
            self.exploration_rate = model_data.get('exploration_rate', self.exploration_rate)
            self.update_counter = model_data.get('update_counter', 0)
            
            logger.info(f"Loaded DQN model for {self.name} from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading DQN model for {self.name}: {str(e)}")
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
            'model_type': 'dqn',
            'lookback_period': self.lookback_period,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'discount_factor': self.discount_factor,
            'memory_size': len(self.memory),
            'update_counter': self.update_counter,
            'features': self.features,
            'model_trained': self.model is not None,
            'historical_data_count': {
                symbol: len(data) for symbol, data in self.historical_data.items()
            }
        })
        
        return status