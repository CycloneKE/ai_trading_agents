#!/usr/bin/env python3
"""
Federated Learning System

Collaborative learning system allowing multiple trading agents to learn
together while preserving data privacy and competitive advantages.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
import asyncio
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class FederatedModel(nn.Module):
    """Federated learning model for trading strategies"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 3):
        super(FederatedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.softmax(self.fc3(x), dim=1)


class PrivacyPreservingAggregator:
    """Privacy-preserving model aggregation using differential privacy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epsilon = config.get('privacy_epsilon', 1.0)  # Privacy budget
        self.delta = config.get('privacy_delta', 1e-5)
        self.noise_multiplier = config.get('noise_multiplier', 1.0)
        
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients"""
        try:
            if not TORCH_AVAILABLE:
                return gradients
            
            noisy_gradients = {}
            
            for name, grad in gradients.items():
                # Calculate sensitivity (L2 norm of gradients)
                sensitivity = torch.norm(grad, p=2)
                
                # Add Gaussian noise for differential privacy
                noise_scale = self.noise_multiplier * sensitivity / self.epsilon
                noise = torch.normal(0, noise_scale, size=grad.shape)
                
                noisy_gradients[name] = grad + noise
            
            return noisy_gradients
            
        except Exception as e:
            logger.error(f"Noise addition failed: {e}")
            return gradients
    
    def aggregate_models(self, model_updates: List[Dict[str, torch.Tensor]], 
                        weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Aggregate model updates from multiple participants"""
        try:
            if not model_updates:
                return {}
            
            if weights is None:
                weights = [1.0 / len(model_updates)] * len(model_updates)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Initialize aggregated parameters
            aggregated = {}
            
            # Get parameter names from first model
            param_names = model_updates[0].keys()
            
            for param_name in param_names:
                # Weighted average of parameters
                weighted_params = []
                for i, update in enumerate(model_updates):
                    if param_name in update:
                        weighted_param = update[param_name] * weights[i]
                        weighted_params.append(weighted_param)
                
                if weighted_params:
                    aggregated[param_name] = torch.stack(weighted_params).sum(dim=0)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            return {}


class SecureCommunication:
    """Secure communication layer for federated learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_key = None
        
        if CRYPTO_AVAILABLE:
            # Generate or load encryption key
            key_file = config.get('encryption_key_file', 'federated_key.key')
            try:
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            except FileNotFoundError:
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
            
            self.cipher = Fernet(self.encryption_key)
        else:
            logger.warning("Cryptography not available - using unencrypted communication")
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data for secure transmission"""
        try:
            if CRYPTO_AVAILABLE and self.cipher:
                return self.cipher.encrypt(data)
            else:
                return data  # Fallback to unencrypted
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt received data"""
        try:
            if CRYPTO_AVAILABLE and self.cipher:
                return self.cipher.decrypt(encrypted_data)
            else:
                return encrypted_data  # Fallback to unencrypted
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def create_secure_hash(self, data: Dict[str, Any]) -> str:
        """Create secure hash for data integrity"""
        try:
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Hash creation failed: {e}")
            return ""


class FederatedLearningParticipant:
    """Individual participant in federated learning"""
    
    def __init__(self, participant_id: str, config: Dict[str, Any]):
        self.participant_id = participant_id
        self.config = config
        self.model = None
        self.local_data = []
        self.training_history = []
        
        # Privacy settings
        self.share_gradients_only = config.get('share_gradients_only', True)
        self.min_data_size = config.get('min_data_size', 100)
        
        # Communication
        self.secure_comm = SecureCommunication(config.get('security', {}))
        
        if TORCH_AVAILABLE:
            input_size = config.get('input_size', 20)
            self.model = FederatedModel(input_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Federated participant {participant_id} initialized")
    
    def add_local_data(self, features: np.ndarray, labels: np.ndarray):
        """Add data to local training set"""
        try:
            if len(features) != len(labels):
                raise ValueError("Features and labels must have same length")
            
            self.local_data.append({
                'features': features,
                'labels': labels,
                'timestamp': datetime.now()
            })
            
            # Keep only recent data
            max_data_points = self.config.get('max_local_data', 1000)
            if len(self.local_data) > max_data_points:
                self.local_data = self.local_data[-max_data_points:]
            
        except Exception as e:
            logger.error(f"Local data addition failed: {e}")
    
    def local_training_round(self, epochs: int = 5) -> Optional[Dict[str, torch.Tensor]]:
        """Perform local training and return model updates"""
        try:
            if not TORCH_AVAILABLE or self.model is None:
                return None
            
            if len(self.local_data) < self.min_data_size:
                logger.warning(f"Insufficient local data: {len(self.local_data)} < {self.min_data_size}")
                return None
            
            # Prepare training data
            all_features = []
            all_labels = []
            
            for data_point in self.local_data[-self.min_data_size:]:
                all_features.extend(data_point['features'])
                all_labels.extend(data_point['labels'])
            
            X = torch.FloatTensor(all_features)
            y = torch.LongTensor(all_labels)
            
            # Store initial model state
            initial_state = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Local training
            self.model.train()
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Calculate model updates (difference from initial state)
            model_updates = {}
            for name, param in self.model.named_parameters():
                model_updates[name] = param - initial_state[name]
            
            # Record training metrics
            self.training_history.append({
                'timestamp': datetime.now(),
                'epochs': epochs,
                'loss': float(loss.item()),
                'data_size': len(all_features)
            })
            
            return model_updates
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            return None
    
    def apply_global_update(self, global_update: Dict[str, torch.Tensor]):
        """Apply global model update to local model"""
        try:
            if not TORCH_AVAILABLE or self.model is None:
                return
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in global_update:
                        param.add_(global_update[name])
            
            logger.info(f"Applied global update to participant {self.participant_id}")
            
        except Exception as e:
            logger.error(f"Global update application failed: {e}")
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get local model performance metrics"""
        try:
            if not self.training_history:
                return {}
            
            recent_history = self.training_history[-10:]  # Last 10 training rounds
            
            return {
                'avg_loss': float(np.mean([h['loss'] for h in recent_history])),
                'training_rounds': len(self.training_history),
                'data_points': sum(len(data['features']) for data in self.local_data),
                'last_training': self.training_history[-1]['timestamp'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {}


class FederatedLearningCoordinator:
    """Central coordinator for federated learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.participants = {}
        self.global_model = None
        self.aggregator = PrivacyPreservingAggregator(config.get('privacy', {}))
        self.secure_comm = SecureCommunication(config.get('security', {}))
        
        # Federated learning parameters
        self.min_participants = config.get('min_participants', 3)
        self.participation_rate = config.get('participation_rate', 0.8)
        self.global_rounds = 0
        
        # History
        self.training_history = []
        
        if TORCH_AVAILABLE:
            input_size = config.get('input_size', 20)
            self.global_model = FederatedModel(input_size)
        
        logger.info("Federated Learning Coordinator initialized")
    
    def register_participant(self, participant: FederatedLearningParticipant):
        """Register a new participant"""
        try:
            self.participants[participant.participant_id] = participant
            logger.info(f"Registered participant: {participant.participant_id}")
            
            # Initialize participant with global model if available
            if self.global_model is not None and participant.model is not None:
                participant.model.load_state_dict(self.global_model.state_dict())
            
        except Exception as e:
            logger.error(f"Participant registration failed: {e}")
    
    def select_participants(self) -> List[str]:
        """Select participants for current training round"""
        try:
            available_participants = list(self.participants.keys())
            
            if len(available_participants) < self.min_participants:
                logger.warning(f"Insufficient participants: {len(available_participants)} < {self.min_participants}")
                return []
            
            # Select random subset based on participation rate
            num_selected = max(self.min_participants, 
                             int(len(available_participants) * self.participation_rate))
            
            selected = np.random.choice(available_participants, 
                                      size=min(num_selected, len(available_participants)), 
                                      replace=False)
            
            return selected.tolist()
            
        except Exception as e:
            logger.error(f"Participant selection failed: {e}")
            return []
    
    async def federated_training_round(self) -> Dict[str, Any]:
        """Execute one round of federated training"""
        try:
            # Select participants
            selected_participants = self.select_participants()
            
            if not selected_participants:
                return {'error': 'No participants available for training'}
            
            logger.info(f"Starting federated round {self.global_rounds + 1} with {len(selected_participants)} participants")
            
            # Collect local updates
            local_updates = []
            participant_weights = []
            
            for participant_id in selected_participants:
                participant = self.participants[participant_id]
                
                # Perform local training
                update = participant.local_training_round()
                
                if update is not None:
                    local_updates.append(update)
                    
                    # Weight by local data size
                    data_size = sum(len(data['features']) for data in participant.local_data)
                    participant_weights.append(data_size)
            
            if not local_updates:
                return {'error': 'No valid updates received'}
            
            # Aggregate updates with privacy preservation
            aggregated_update = self.aggregator.aggregate_models(local_updates, participant_weights)
            
            # Apply differential privacy noise
            if self.config.get('use_differential_privacy', True):
                aggregated_update = self.aggregator.add_noise_to_gradients(aggregated_update)
            
            # Update global model
            if self.global_model is not None:
                with torch.no_grad():
                    for name, param in self.global_model.named_parameters():
                        if name in aggregated_update:
                            param.add_(aggregated_update[name])
            
            # Distribute global update to all participants
            for participant in self.participants.values():
                participant.apply_global_update(aggregated_update)
            
            # Record training round
            round_info = {
                'round': self.global_rounds + 1,
                'participants': len(local_updates),
                'timestamp': datetime.now().isoformat(),
                'total_data_points': sum(participant_weights)
            }
            
            self.training_history.append(round_info)
            self.global_rounds += 1
            
            logger.info(f"Completed federated round {self.global_rounds}")
            
            return round_info
            
        except Exception as e:
            logger.error(f"Federated training round failed: {e}")
            return {'error': str(e)}
    
    def get_global_model_state(self) -> Optional[Dict[str, Any]]:
        """Get current global model state"""
        try:
            if self.global_model is None:
                return None
            
            return {
                'model_state': {name: param.detach().numpy().tolist() 
                              for name, param in self.global_model.named_parameters()},
                'rounds_completed': self.global_rounds,
                'participants': len(self.participants),
                'last_update': self.training_history[-1]['timestamp'] if self.training_history else None
            }
            
        except Exception as e:
            logger.error(f"Global model state retrieval failed: {e}")
            return None
    
    def get_federation_metrics(self) -> Dict[str, Any]:
        """Get federated learning metrics"""
        try:
            metrics = {
                'total_participants': len(self.participants),
                'active_participants': len([p for p in self.participants.values() 
                                          if len(p.local_data) >= p.min_data_size]),
                'global_rounds': self.global_rounds,
                'total_data_points': sum(sum(len(data['features']) for data in p.local_data) 
                                       for p in self.participants.values()),
                'privacy_enabled': self.config.get('use_differential_privacy', True),
                'encryption_enabled': CRYPTO_AVAILABLE
            }
            
            # Participant performance
            participant_metrics = {}
            for pid, participant in self.participants.items():
                participant_metrics[pid] = participant.get_model_performance()
            
            metrics['participant_performance'] = participant_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Federation metrics calculation failed: {e}")
            return {}


class FederatedLearningSystem:
    """Main federated learning system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coordinator = FederatedLearningCoordinator(config.get('coordinator', {}))
        self.local_participant = None
        
        # System settings
        self.auto_training = config.get('auto_training', True)
        self.training_interval = config.get('training_interval_hours', 6)
        self.last_training = datetime.now() - timedelta(days=1)
        
        logger.info("Federated Learning System initialized")
    
    def create_local_participant(self, participant_id: str) -> FederatedLearningParticipant:
        """Create and register local participant"""
        try:
            participant_config = self.config.get('participant', {})
            participant = FederatedLearningParticipant(participant_id, participant_config)
            
            self.coordinator.register_participant(participant)
            self.local_participant = participant
            
            return participant
            
        except Exception as e:
            logger.error(f"Local participant creation failed: {e}")
            return None
    
    def add_training_data(self, features: np.ndarray, labels: np.ndarray):
        """Add training data to local participant"""
        try:
            if self.local_participant is None:
                logger.warning("No local participant - creating default participant")
                self.create_local_participant("local_agent")
            
            self.local_participant.add_local_data(features, labels)
            
        except Exception as e:
            logger.error(f"Training data addition failed: {e}")
    
    async def run_training_cycle(self) -> Dict[str, Any]:
        """Run a complete federated training cycle"""
        try:
            if not self.should_train():
                return {'message': 'Training not due yet'}
            
            result = await self.coordinator.federated_training_round()
            self.last_training = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Training cycle failed: {e}")
            return {'error': str(e)}
    
    def should_train(self) -> bool:
        """Check if federated training should be performed"""
        hours_since_training = (datetime.now() - self.last_training).total_seconds() / 3600
        return hours_since_training >= self.training_interval
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get federated learning system status"""
        try:
            status = {
                'status': 'active',
                'torch_available': TORCH_AVAILABLE,
                'crypto_available': CRYPTO_AVAILABLE,
                'coordinator_active': self.coordinator is not None,
                'local_participant_active': self.local_participant is not None,
                'auto_training': self.auto_training,
                'last_training': self.last_training.isoformat(),
                'next_training': (self.last_training + timedelta(hours=self.training_interval)).isoformat()
            }
            
            # Add coordinator metrics
            if self.coordinator:
                status.update(self.coordinator.get_federation_metrics())
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {'error': str(e)}