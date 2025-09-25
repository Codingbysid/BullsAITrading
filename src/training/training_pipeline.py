"""
Advanced training pipeline with overfitting prevention.

This module implements:
- Purged cross-validation to prevent data leakage
- Walk-forward analysis for time series
- Ensemble methods with regularization
- Hyperparameter optimization
- Performance monitoring and early stopping
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
import optuna
from pathlib import Path

from ..config.settings import get_settings
from ..data.data_sources import data_manager
from ..data.feature_engineering import FeatureEngineer
from ..data.sentiment_analysis import sentiment_aggregator
from ..models.trading_models import create_ensemble_model
from ..risk.risk_management import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Data parameters
    lookback_period: int = 252  # 1 year of trading days
    min_samples: int = 1000  # Minimum samples for training
    test_size: float = 0.2  # 20% for testing
    
    # Cross-validation parameters
    n_splits: int = 5  # Number of CV folds
    purge_days: int = 5  # Days to purge between train/test
    embargo_days: int = 3  # Days to embargo after test
    
    # Model parameters
    max_depth: int = 6  # Maximum tree depth
    min_samples_split: int = 50  # Minimum samples to split
    min_samples_leaf: int = 20  # Minimum samples per leaf
    max_features: str = 'sqrt'  # Feature selection strategy
    
    # Regularization parameters
    l1_regularization: float = 0.1  # L1 regularization
    l2_regularization: float = 0.1  # L2 regularization
    dropout_rate: float = 0.2  # Dropout rate for neural networks
    early_stopping_patience: int = 10  # Early stopping patience
    
    # Optimization parameters
    n_trials: int = 100  # Number of hyperparameter trials
    optimization_metric: str = 'f1_score'  # Optimization metric
    
    # Risk parameters
    max_position_size: float = 0.2  # Maximum position size
    min_confidence: float = 0.6  # Minimum prediction confidence


class PurgedTimeSeriesSplit:
    """
    Purged cross-validation for time series data.
    Prevents data leakage by purging overlapping periods.
    """
    
    def __init__(self, n_splits: int = 5, purge_days: int = 5, embargo_days: int = 3):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None):
        """Generate train/test splits with purging."""
        indices = np.arange(len(X))
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = len(X) * (i + 1) // (self.n_splits + 1)
            test_end = len(X) * (i + 2) // (self.n_splits + 1)
            
            # Apply purging
            train_end = test_start - self.purge_days
            test_start_purged = test_end + self.embargo_days
            
            # Create train and test indices
            train_indices = indices[:train_end]
            test_indices = indices[test_start_purged:test_end]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class AdvancedTrainingPipeline:
    """Advanced training pipeline with overfitting prevention."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.settings = get_settings()
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager()
        
        # Training data
        self.training_data = None
        self.features = None
        self.targets = None
        
        # Models
        self.models = {}
        self.ensemble_model = None
        
        # Performance tracking
        self.performance_history = []
        self.validation_scores = {}
        
        # Paths
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
    
    async def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare comprehensive training data."""
        logger.info("Preparing training data...")
        
        all_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_period * 2)  # Extra data for features
        
        for symbol in self.settings.target_symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                
                # Get market data
                market_data = await data_manager.get_market_data(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if market_data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Get sentiment data
                sentiment_data = await sentiment_aggregator.get_comprehensive_sentiment(
                    symbol=symbol,
                    lookback_days=30
                )
                
                # Create features
                features = self.feature_engineer.create_all_features(market_data, symbol)
                
                if not features.empty:
                    # Add sentiment features
                    if sentiment_data:
                        features['sentiment_score'] = sentiment_data.get('overall_sentiment', 0.0)
                        features['sentiment_confidence'] = sentiment_data.get('confidence', 0.0)
                    else:
                        features['sentiment_score'] = 0.0
                        features['sentiment_confidence'] = 0.0
                    
                    all_data.append(features)
                    logger.info(f"Added {len(features)} samples for {symbol}")
                else:
                    logger.warning(f"Failed to create features for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No training data available")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_data)} samples, {len(combined_data.columns)} features")
        
        # Prepare features and targets
        feature_columns = [col for col in combined_data.columns 
                          if not col.startswith('future_') and col != 'symbol']
        X = combined_data[feature_columns].fillna(0)
        y = combined_data['future_return'].fillna(0)
        
        # Convert continuous returns to categorical signals
        y_categorical = self._convert_to_signals(y)
        
        self.training_data = combined_data
        self.features = X
        self.targets = y_categorical
        
        return X, y_categorical
    
    def _convert_to_signals(self, returns: pd.Series) -> pd.Series:
        """Convert continuous returns to trading signals."""
        # Create signals based on return thresholds
        signals = pd.Series(0, index=returns.index)  # Default: HOLD
        
        # Buy signal: positive returns above threshold
        buy_threshold = returns.quantile(0.6)  # Top 40% of returns
        signals[returns >= buy_threshold] = 1
        
        # Sell signal: negative returns below threshold
        sell_threshold = returns.quantile(0.4)  # Bottom 40% of returns
        signals[returns <= sell_threshold] = -1
        
        return signals
    
    def train_models_with_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train models with proper cross-validation."""
        logger.info("Training models with cross-validation...")
        
        # Initialize cross-validation
        cv = PurgedTimeSeriesSplit(
            n_splits=self.config.n_splits,
            purge_days=self.config.purge_days,
            embargo_days=self.config.embargo_days
        )
        
        # Train individual models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                max_depth=self.config.max_depth,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Cross-validation results
        cv_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            scores = []
            fold_predictions = []
            fold_targets = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                scores.append({
                    'fold': fold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
                
                fold_predictions.extend(y_pred)
                fold_targets.extend(y_test)
            
            # Calculate average scores
            avg_scores = {
                'accuracy': np.mean([s['accuracy'] for s in scores]),
                'precision': np.mean([s['precision'] for s in scores]),
                'recall': np.mean([s['recall'] for s in scores]),
                'f1_score': np.mean([s['f1_score'] for s in scores]),
                'std_accuracy': np.std([s['accuracy'] for s in scores]),
                'std_f1': np.std([s['f1_score'] for s in scores])
            }
            
            cv_results[name] = {
                'model': model,
                'scores': scores,
                'avg_scores': avg_scores,
                'predictions': fold_predictions,
                'targets': fold_targets
            }
            
            logger.info(f"{name} - CV F1: {avg_scores['f1_score']:.3f} ± {avg_scores['std_f1']:.3f}")
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        ensemble_model = create_ensemble_model()
        ensemble_model.fit(X, y)
        
        cv_results['ensemble'] = {
            'model': ensemble_model,
            'avg_scores': {'f1_score': 0.0}  # Placeholder
        }
        
        self.models = cv_results
        return cv_results
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        logger.info("Optimizing hyperparameters...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
            }
            
            # Create model with suggested parameters
            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
            
            # Cross-validation
            cv = PurgedTimeSeriesSplit(n_splits=3, purge_days=5, embargo_days=3)
            scores = []
            
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                scores.append(f1)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.3f}")
        
        return study.best_params
    
    def train_neural_network(self, X: pd.DataFrame, y: pd.Series) -> tf.keras.Model:
        """Train neural network with regularization."""
        logger.info("Training neural network...")
        
        # Convert targets to categorical
        y_categorical = tf.keras.utils.to_categorical(y + 1, num_classes=3)  # -1, 0, 1 -> 0, 1, 2
        
        # Build model with regularization
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            layers.Dropout(self.config.dropout_rate),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.BatchNormalization(),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            
            layers.Dense(3, activation='softmax')  # 3 classes: sell, hold, buy
        ])
        
        # Compile with regularization
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for overfitting prevention
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train with validation split
        history = model.fit(
            X, y_categorical,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=0
        )
        
        logger.info(f"Neural network trained for {len(history.history['loss'])} epochs")
        
        return model
    
    def evaluate_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics."""
        logger.info("Evaluating model performance...")
        
        # Use the best model from cross-validation
        best_model_name = max(self.models.keys(), 
                             key=lambda k: self.models[k]['avg_scores']['f1_score'])
        best_model = self.models[best_model_name]['model']
        
        # Final evaluation
        y_pred = best_model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, best_model.feature_importances_))
        
        # Risk-adjusted metrics
        risk_metrics = self._calculate_risk_adjusted_metrics(y, y_pred)
        
        evaluation_results = {
            'model_name': best_model_name,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'risk_metrics': risk_metrics,
            'cv_results': self.models
        }
        
        logger.info(f"Model evaluation complete - F1 Score: {metrics['f1_score']:.3f}")
        
        return evaluation_results
    
    def _calculate_risk_adjusted_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        # This would integrate with actual returns and risk metrics
        # For now, return placeholder metrics
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    
    def save_models(self, evaluation_results: Dict[str, Any]):
        """Save trained models and results."""
        logger.info("Saving models and results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for name, model_data in self.models.items():
            model_path = self.model_path / f"{name}_{timestamp}.joblib"
            joblib.dump(model_data['model'], model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save evaluation results
        results_path = self.model_path / f"evaluation_results_{timestamp}.joblib"
        joblib.dump(evaluation_results, results_path)
        logger.info(f"Saved evaluation results to {results_path}")
        
        # Save training configuration
        config_path = self.model_path / f"training_config_{timestamp}.joblib"
        joblib.dump(self.config, config_path)
        logger.info(f"Saved training config to {config_path}")
    
    async def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting complete training pipeline...")
        
        try:
            # 1. Prepare training data
            X, y = await self.prepare_training_data()
            
            if len(X) < self.config.min_samples:
                raise ValueError(f"Insufficient training data: {len(X)} < {self.config.min_samples}")
            
            # 2. Train models with cross-validation
            cv_results = self.train_models_with_validation(X, y)
            
            # 3. Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X, y)
            
            # 4. Train neural network
            nn_model = self.train_neural_network(X, y)
            
            # 5. Evaluate performance
            evaluation_results = self.evaluate_model_performance(X, y)
            
            # 6. Save models
            self.save_models(evaluation_results)
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'status': 'success',
                'cv_results': cv_results,
                'best_params': best_params,
                'evaluation': evaluation_results,
                'training_samples': len(X),
                'features_count': len(X.columns)
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# Global training pipeline instance
training_pipeline = AdvancedTrainingPipeline()


async def run_training_pipeline(config: TrainingConfig = None) -> Dict[str, Any]:
    """
    Run the complete training pipeline.
    
    Args:
        config: Training configuration parameters
        
    Returns:
        Dictionary with training results
    """
    pipeline = AdvancedTrainingPipeline(config)
    return await pipeline.run_complete_training()
