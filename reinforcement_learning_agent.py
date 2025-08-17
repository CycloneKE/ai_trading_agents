#!/usr/bin/env python3
"""
Reinforcement Learning Trading Agent

Advanced RL agent using Deep Q-Network (DQN) for autonomous trading decisions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import json
import os
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class RLTradingAgent:
    """Reinforcement Learning Trading Agent using DQN"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_size = config.get('state_size', 20)
        self.action_size = 3  # 0: sell, 1: hold, 2: buy
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.update_target_freq = config.get('update_target_freq', 100)
        
        self.memory = ReplayBuffer(config.get('memory_size', 10000))
        self.step_count = 0
        self.episode_count = 0
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            self.update_target_network()
        else:
            logger.warning("PyTorch not available - RL agent disabled")
            self.q_network = None
        
        self.portfolio_value_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        logger.info(f"RL Trading Agent initialized with state_size={self.state_size}")
    
    def update_target_network(self):
        """Update target network with current network weights"""
        if self.q_network:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Convert market data to state vector"""
        try:
            # Extract features for state representation
            features = []
            
            # Price features
            features.extend([
                market_data.get('close', 0),
                market_data.get('volume', 0),
                market_data.get('high', 0),
                market_data.get('low', 0),
                market_data.get('open', 0)
            ])
            
            # Technical indicators
            features.extend([
                market_data.get('rsi', 50),
                market_data.get('macd', 0),
                market_data.get('bb_upper', 0),
                market_data.get('bb_lower', 0),
                market_data.get('sma_20', 0),
                market_data.get('ema_12', 0),
                market_data.get('volatility', 0)
            ])
            
            # Portfolio features
            features.extend([
                market_data.get('portfolio_value', 100000),
                market_data.get('cash', 50000),
                market_data.get('position', 0),
                market_data.get('unrealized_pnl', 0),
                market_data.get('realized_pnl', 0)
            ])
            
            # Market sentiment
            features.extend([
                market_data.get('sentiment_score', 0),
                market_data.get('news_sentiment', 0),
                market_data.get('market_regime', 0)
            ])
            
            # Normalize features
            state = np.array(features[:self.state_size], dtype=np.float32)
            state = np.nan_to_num(state, 0)
            
            # Simple normalization
            state = np.clip(state / (np.abs(state).max() + 1e-8), -1, 1)
            
            return state
            
        except Exception as e:
            logger.error(f"Error creating state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if not TORCH_AVAILABLE or self.q_network is None:
            return 1  # Default to hold
        
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if not TORCH_AVAILABLE or len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()
    
    def calculate_reward(self, action: int, market_data: Dict[str, Any], 
                        prev_portfolio_value: float) -> float:
        """Calculate reward based on action and market outcome"""
        try:
            current_portfolio_value = market_data.get('portfolio_value', prev_portfolio_value)
            
            # Base reward from portfolio change
            portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            reward = portfolio_return * 100  # Scale reward
            
            # Action-specific rewards
            price_change = market_data.get('price_change_pct', 0)
            
            if action == 2:  # Buy
                if price_change > 0:
                    reward += 0.1  # Reward for buying before price increase
                else:
                    reward -= 0.05  # Penalty for buying before price decrease
            elif action == 0:  # Sell
                if price_change < 0:
                    reward += 0.1  # Reward for selling before price decrease
                else:
                    reward -= 0.05  # Penalty for selling before price increase
            
            # Risk penalty
            volatility = market_data.get('volatility', 0)
            if volatility > 0.02:  # High volatility
                reward -= 0.02
            
            # Transaction cost penalty
            if action != 1:  # Not hold
                reward -= 0.001  # Small transaction cost
            
            return np.clip(reward, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals using RL agent"""
        try:
            state = self.get_state(market_data)
            action = self.act(state, training=False)
            
            # Convert action to trading signal
            action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            signal_action = action_map[action]
            
            # Calculate confidence based on Q-values
            confidence = 0.5
            if TORCH_AVAILABLE and self.q_network:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    q_values_np = q_values.cpu().numpy()[0]
                    confidence = float(np.max(q_values_np) - np.mean(q_values_np))
                    confidence = np.clip(confidence, 0, 1)
            
            # Position sizing based on confidence
            position_size = confidence * 0.1 if signal_action != 'hold' else 0.0
            
            return {
                'action': signal_action,
                'confidence': float(confidence),
                'position_size': float(position_size),
                'q_values': q_values_np.tolist() if TORCH_AVAILABLE and self.q_network else [0, 0, 0],
                'epsilon': self.epsilon,
                'strategy': 'reinforcement_learning'
            }
            
        except Exception as e:
            logger.error(f"Error generating RL signals: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0,
                'error': str(e)
            }
    
    def train_step(self, market_data: Dict[str, Any], prev_market_data: Optional[Dict[str, Any]] = None):
        """Perform one training step"""
        if not prev_market_data:
            return
        
        try:
            prev_state = self.get_state(prev_market_data)
            current_state = self.get_state(market_data)
            
            # Get last action from history
            if len(self.action_history) > 0:
                action = self.action_history[-1]
                
                # Calculate reward
                prev_portfolio_value = prev_market_data.get('portfolio_value', 100000)
                reward = self.calculate_reward(action, market_data, prev_portfolio_value)
                
                # Store experience
                done = market_data.get('market_closed', False)
                self.remember(prev_state, action, reward, current_state, done)
                
                # Train if enough experiences
                if len(self.memory) >= self.batch_size:
                    self.replay()
                
                # Store metrics
                self.reward_history.append(reward)
                
        except Exception as e:
            logger.error(f"Error in training step: {e}")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not TORCH_AVAILABLE or self.q_network is None:
            return
        
        try:
            torch.save({
                'model_state_dict': self.q_network.state_dict(),
                'target_model_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count
            }, filepath)
            logger.info(f"RL model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not TORCH_AVAILABLE or not os.path.exists(filepath):
            return
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_count = checkpoint.get('step_count', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            logger.info(f"RL model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RL agent performance metrics"""
        try:
            metrics = {
                'total_steps': self.step_count,
                'total_episodes': self.episode_count,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
                'recent_rewards': list(self.reward_history)[-10:] if self.reward_history else [],
                'torch_available': TORCH_AVAILABLE
            }
            
            if self.portfolio_value_history:
                metrics['portfolio_performance'] = {
                    'current_value': self.portfolio_value_history[-1],
                    'total_return': (self.portfolio_value_history[-1] / self.portfolio_value_history[0] - 1) * 100,
                    'volatility': np.std(self.portfolio_value_history) / np.mean(self.portfolio_value_history) * 100
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting RL metrics: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'status': 'active' if TORCH_AVAILABLE else 'disabled',
            'torch_available': TORCH_AVAILABLE,
            'model_loaded': self.q_network is not None,
            'training_steps': self.step_count,
            'epsilon': self.epsilon
        }