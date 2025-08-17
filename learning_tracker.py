"""
Learning Tracker - Shows AI learning progress and model updates.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

class LearningTracker:
    def __init__(self):
        self.learning_log = []
        self.model_versions = {}
        
    def log_learning_event(self, strategy: str, event_type: str, data: Dict[str, Any]):
        """Log when the AI learns something new."""
        
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'strategy': strategy,
            'event_type': event_type,
            'data': data,
            'learning_score': self._calculate_learning_score(data)
        }
        
        self.learning_log.append(event)
        self._save_learning_log()
        
        # Log to console
        logger.info(f"🧠 LEARNING: {strategy} - {event_type}")
        logger.info(f"   Score: {event['learning_score']:.3f}")
        
    def track_model_update(self, strategy: str, old_accuracy: float, new_accuracy: float, training_samples: int):
        """Track when a model gets updated/retrained."""
        
        improvement = new_accuracy - old_accuracy
        
        self.model_versions[strategy] = {
            'version': self.model_versions.get(strategy, {}).get('version', 0) + 1,
            'accuracy': new_accuracy,
            'improvement': improvement,
            'training_samples': training_samples,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        self.log_learning_event(strategy, 'model_update', {
            'old_accuracy': old_accuracy,
            'new_accuracy': new_accuracy,
            'improvement': improvement,
            'training_samples': training_samples
        })
        
        logger.info(f"📈 MODEL UPDATE: {strategy}")
        logger.info(f"   Accuracy: {old_accuracy:.3f} → {new_accuracy:.3f} ({improvement:+.3f})")
        logger.info(f"   Samples: {training_samples}")
        
    def track_prediction_feedback(self, strategy: str, prediction: float, actual: float, symbol: str):
        """Track prediction vs actual results for learning."""
        
        error = abs(prediction - actual)
        accuracy = 1.0 - min(error, 1.0)  # Cap error at 100%
        
        self.log_learning_event(strategy, 'prediction_feedback', {
            'symbol': symbol,
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'accuracy': accuracy
        })
        
    def track_pattern_discovery(self, strategy: str, pattern: str, confidence: float):
        """Track when AI discovers new patterns."""
        
        self.log_learning_event(strategy, 'pattern_discovery', {
            'pattern': pattern,
            'confidence': confidence
        })
        
        logger.info(f"🔍 PATTERN FOUND: {strategy}")
        logger.info(f"   Pattern: {pattern}")
        logger.info(f"   Confidence: {confidence:.3f}")
        
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        
        if not self.learning_log:
            return {'status': 'No learning events recorded yet'}
            
        recent_events = self.learning_log[-10:]  # Last 10 events
        
        # Calculate learning metrics
        total_events = len(self.learning_log)
        avg_learning_score = sum(e['learning_score'] for e in self.learning_log) / total_events
        
        # Count event types
        event_counts = {}
        for event in self.learning_log:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        return {
            'total_learning_events': total_events,
            'average_learning_score': avg_learning_score,
            'event_types': event_counts,
            'model_versions': self.model_versions,
            'recent_events': recent_events,
            'last_learning': self.learning_log[-1]['timestamp'] if self.learning_log else None
        }
        
    def _calculate_learning_score(self, data: Dict[str, Any]) -> float:
        """Calculate how significant this learning event is."""
        
        if 'improvement' in data:
            return min(abs(data['improvement']) * 10, 1.0)  # Model improvement
        elif 'accuracy' in data:
            return data['accuracy']  # Prediction accuracy
        elif 'confidence' in data:
            return data['confidence']  # Pattern confidence
        else:
            return 0.5  # Default moderate score
            
    def _save_learning_log(self):
        """Save learning log to file."""
        try:
            os.makedirs('learning', exist_ok=True)
            with open('learning/learning_log.json', 'w') as f:
                json.dump({
                    'learning_log': self.learning_log[-1000:],  # Keep last 1000 events
                    'model_versions': self.model_versions
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learning log: {str(e)}")

# Global learning tracker instance
learning_tracker = LearningTracker()

def get_learning_tracker():
    """Get the global learning tracker instance."""
    return learning_tracker