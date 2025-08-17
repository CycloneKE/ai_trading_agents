"""
AutoML Optimizer for automatic parameter tuning and model optimization.
Uses hyperparameter optimization and automated feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import pickle
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import optuna
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    optimization_history: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    validation_metrics: Dict[str, float]


class AutoMLOptimizer:
    """
    AutoML optimizer for trading strategy parameters and model selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Optimization settings
        self.n_trials = config.get('n_trials', 100)
        self.cv_folds = config.get('cv_folds', 5)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        
        # Model selection
        self.models = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'svr': SVR
        }
        
        # Feature engineering
        self.scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'none': None
        }
        
        # Optimization history
        self.optimization_history = []
        self.best_models = {}
        
        # Study storage
        self.studies = {}
        
        logger.info("AutoML optimizer initialized")
    
    def create_features(self, price_data: pd.DataFrame, 
                       technical_indicators: bool = True,
                       lag_features: bool = True,
                       rolling_features: bool = True) -> pd.DataFrame:
        """
        Create features for model training.
        
        Args:
            price_data: OHLCV price data
            technical_indicators: Whether to include technical indicators
            lag_features: Whether to include lag features
            rolling_features: Whether to include rolling window features
            
        Returns:
            Feature DataFrame
        """
        try:
            features = price_data.copy()
            
            if 'close' not in features.columns:
                logger.error("Price data must contain 'close' column")
                return pd.DataFrame()
            
            # Basic price features
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            
            if technical_indicators:
                # Technical indicators
                features['sma_5'] = features['close'].rolling(5).mean()
                features['sma_10'] = features['close'].rolling(10).mean()
                features['sma_20'] = features['close'].rolling(20).mean()
                features['ema_12'] = features['close'].ewm(span=12).mean()
                features['ema_26'] = features['close'].ewm(span=26).mean()
                
                # MACD
                features['macd'] = features['ema_12'] - features['ema_26']
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']
                
                # RSI
                delta = features['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                features['bb_middle'] = features['close'].rolling(20).mean()
                bb_std = features['close'].rolling(20).std()
                features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
                features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
                features['bb_width'] = features['bb_upper'] - features['bb_lower']
                features['bb_position'] = (features['close'] - features['bb_lower']) / features['bb_width']
                
                # Volume indicators (if volume available)
                if 'volume' in features.columns:
                    features['volume_sma'] = features['volume'].rolling(20).mean()
                    features['volume_ratio'] = features['volume'] / features['volume_sma']
                    features['price_volume'] = features['close'] * features['volume']
            
            if lag_features:
                # Lag features
                for lag in [1, 2, 3, 5, 10]:
                    features[f'close_lag_{lag}'] = features['close'].shift(lag)
                    features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                    features[f'volume_lag_{lag}'] = features.get('volume', pd.Series()).shift(lag)
            
            if rolling_features:
                # Rolling window features
                for window in [5, 10, 20]:
                    features[f'close_mean_{window}'] = features['close'].rolling(window).mean()
                    features[f'close_std_{window}'] = features['close'].rolling(window).std()
                    features[f'close_min_{window}'] = features['close'].rolling(window).min()
                    features[f'close_max_{window}'] = features['close'].rolling(window).max()
                    features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
                    features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            
            # Time-based features
            if isinstance(features.index, pd.DatetimeIndex):
                features['hour'] = features.index.hour
                features['day_of_week'] = features.index.dayofweek
                features['month'] = features.index.month
                features['quarter'] = features.index.quarter
            
            # Drop NaN values
            features = features.dropna()
            
            logger.info(f"Created {len(features.columns)} features from price data")
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return pd.DataFrame()
    
    def optimize_model(self, X: pd.DataFrame, y: pd.Series, 
                      model_type: str = 'auto',
                      objective: str = 'regression') -> OptimizationResult:
        """
        Optimize model hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to optimize ('auto' for automatic selection)
            objective: Optimization objective ('regression' or 'classification')
            
        Returns:
            OptimizationResult with best parameters and model
        """
        try:
            if X.empty or y.empty:
                raise ValueError("Empty feature matrix or target variable")
            
            # Create study
            study_name = f"{model_type}_{objective}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(direction='maximize', study_name=study_name)
            
            # Define objective function
            def objective_func(trial):
                try:
                    if model_type == 'auto':
                        # Automatic model selection
                        model_name = trial.suggest_categorical('model', list(self.models.keys()))
                    else:
                        model_name = model_type
                    
                    # Get model class
                    model_class = self.models[model_name]
                    
                    # Suggest hyperparameters based on model type
                    params = self._suggest_hyperparameters(trial, model_name)
                    
                    # Feature scaling
                    scaler_type = trial.suggest_categorical('scaler', list(self.scalers.keys()))
                    
                    # Prepare data
                    X_scaled = X.copy()
                    if scaler_type != 'none':
                        scaler = self.scalers[scaler_type]()
                        X_scaled = pd.DataFrame(
                            scaler.fit_transform(X_scaled),
                            columns=X_scaled.columns,
                            index=X_scaled.index
                        )
                    
                    # Feature selection
                    n_features = trial.suggest_int('n_features', 
                                                 min(10, len(X_scaled.columns)), 
                                                 len(X_scaled.columns))
                    
                    if n_features < len(X_scaled.columns):
                        selector = SelectKBest(score_func=f_regression, k=n_features)
                        X_selected = selector.fit_transform(X_scaled, y)
                        X_scaled = pd.DataFrame(X_selected, index=X_scaled.index)
                    
                    # Create and train model
                    model = model_class(**params)
                    
                    # Cross-validation
                    cv = TimeSeriesSplit(n_splits=self.cv_folds)
                    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
                    
                    return np.mean(scores)
                    
                except Exception as e:
                    logger.error(f"Error in objective function: {str(e)}")
                    return -np.inf
            
            # Optimize
            study.optimize(objective_func, n_trials=self.n_trials)
            
            # Get best parameters and retrain model
            best_params = study.best_params.copy()
            model_name = best_params.pop('model', model_type)
            scaler_type = best_params.pop('scaler', 'none')
            n_features = best_params.pop('n_features', len(X.columns))
            
            # Prepare final data
            X_final = X.copy()
            scaler = None
            if scaler_type != 'none':
                scaler = self.scalers[scaler_type]()
                X_final = pd.DataFrame(
                    scaler.fit_transform(X_final),
                    columns=X_final.columns,
                    index=X_final.index
                )
            
            # Feature selection
            selector = None
            if n_features < len(X_final.columns):
                selector = SelectKBest(score_func=f_regression, k=n_features)
                X_selected = selector.fit_transform(X_final, y)
                X_final = pd.DataFrame(X_selected, index=X_final.index)
                selected_features = X.columns[selector.get_support()].tolist()
            else:
                selected_features = X.columns.tolist()
            
            # Train final model
            model_class = self.models[model_name]
            best_model = model_class(**best_params)
            best_model.fit(X_final, y)
            
            # Calculate validation metrics
            y_pred = best_model.predict(X_final)
            validation_metrics = {
                'r2_score': r2_score(y, y_pred),
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred))
            }
            
            # Feature importance
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                for i, importance in enumerate(best_model.feature_importances_):
                    if i < len(selected_features):
                        feature_importance[selected_features[i]] = float(importance)
            elif hasattr(best_model, 'coef_'):
                for i, coef in enumerate(best_model.coef_):
                    if i < len(selected_features):
                        feature_importance[selected_features[i]] = float(abs(coef))
            
            # Store optimization history
            optimization_history = []
            for trial in study.trials:
                optimization_history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                })
            
            # Create result
            result = OptimizationResult(
                best_params=study.best_params,
                best_score=study.best_value,
                best_model=best_model,
                optimization_history=optimization_history,
                feature_importance=feature_importance,
                validation_metrics=validation_metrics
            )
            
            # Store study and result
            self.studies[study_name] = study
            self.best_models[model_name] = {
                'model': best_model,
                'scaler': scaler,
                'selector': selector,
                'selected_features': selected_features,
                'result': result
            }
            
            logger.info(f"Optimization completed. Best score: {study.best_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            return OptimizationResult(
                best_params={},
                best_score=0.0,
                best_model=None,
                optimization_history=[],
                feature_importance={},
                validation_metrics={}
            )
    
    def _suggest_hyperparameters(self, trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for different model types.
        
        Args:
            trial: Optuna trial object
            model_name: Name of the model
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {}
        
        if model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
        elif model_name == 'gradient_boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state
            }
            
        elif model_name == 'ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True),
                'random_state': self.random_state
            }
            
        elif model_name == 'lasso':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'random_state': self.random_state
            }
            
        elif model_name == 'svr':
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
            
        elif model_name == 'linear_regression':
            params = {}  # Linear regression has no hyperparameters to tune
        
        return params
    
    def optimize_strategy_parameters(self, strategy_func: Callable, 
                                   parameter_space: Dict[str, Any],
                                   market_data: pd.DataFrame,
                                   evaluation_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize trading strategy parameters.
        
        Args:
            strategy_func: Strategy function to optimize
            parameter_space: Dictionary defining parameter search space
            market_data: Market data for backtesting
            evaluation_metric: Metric to optimize
            
        Returns:
            Best parameters and performance metrics
        """
        try:
            study_name = f"strategy_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(direction='maximize', study_name=study_name)
            
            def objective(trial):
                try:
                    # Suggest parameters
                    params = {}
                    for param_name, param_config in parameter_space.items():
                        if param_config['type'] == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name, 
                                param_config['low'], 
                                param_config['high']
                            )
                        elif param_config['type'] == 'float':
                            params[param_name] = trial.suggest_float(
                                param_name, 
                                param_config['low'], 
                                param_config['high']
                            )
                        elif param_config['type'] == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name, 
                                param_config['choices']
                            )
                    
                    # Run strategy with parameters
                    performance = strategy_func(market_data, params)
                    
                    # Return evaluation metric
                    return performance.get(evaluation_metric, 0.0)
                    
                except Exception as e:
                    logger.error(f"Error in strategy optimization objective: {str(e)}")
                    return -np.inf
            
            # Optimize
            study.optimize(objective, n_trials=self.n_trials)
            
            # Store study
            self.studies[study_name] = study
            
            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'optimization_history': [
                    {
                        'trial': trial.number,
                        'value': trial.value,
                        'params': trial.params
                    } for trial in study.trials
                ]
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {str(e)}")
            return {}
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save optimized model to file.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        try:
            if model_name in self.best_models:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.best_models[model_name], f)
                logger.info(f"Model {model_name} saved to {filepath}")
            else:
                logger.error(f"Model {model_name} not found")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load optimized model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model dictionary
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of all optimization runs.
        
        Returns:
            Summary of optimization results
        """
        try:
            summary = {
                'total_studies': len(self.studies),
                'total_models': len(self.best_models),
                'studies': {},
                'best_models': {}
            }
            
            # Study summaries
            for study_name, study in self.studies.items():
                summary['studies'][study_name] = {
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'n_trials': len(study.trials),
                    'study_name': study_name
                }
            
            # Model summaries
            for model_name, model_data in self.best_models.items():
                result = model_data['result']
                summary['best_models'][model_name] = {
                    'best_score': result.best_score,
                    'validation_metrics': result.validation_metrics,
                    'n_features': len(model_data['selected_features']),
                    'feature_importance': dict(list(result.feature_importance.items())[:5])  # Top 5
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {str(e)}")
            return {}

